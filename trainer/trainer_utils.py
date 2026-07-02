import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def get_model_params(model):
    total = sum(p.numel() for p in model.parameters()) / 1e9
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    Logger(f'Model Params: {total:.3f}B, Trainable: {trainable:.3f}B')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content, level="info"):
    """带时间戳和级别的日志输出，仅主进程打印。"""
    if is_main_process():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = {"info": "INFO", "warn": "WARN", "error": "ERROR"}.get(level, "INFO")
        print(f"[{timestamp}] [{prefix}] {content}")


def get_lr(current_step, total_steps, lr, warmup_steps=0):
    """带预热的余弦退火学习率调度。"""
    if warmup_steps > 0 and current_step < warmup_steps:
        return lr * current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))



# def get_decoder_layer_class(model):
#     """自动检测模型的 DecoderLayer 类。"""
#     if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
#         return type(model.model.layers[0])
#     for name, module in model.named_modules():
#         class_name = type(module).__name__
#         if 'DecoderLayer' in class_name or 'Block' in class_name:
#             return type(module)
#     raise ValueError("无法自动检测 DecoderLayer 类。请手动传入 decoder_layer_cls 参数。")


def get_model_block_classes(model):
    """
    自动检测模型的 DecoderLayer 类和 MoE Block 类。
    返回: (decoder_layer_cls, moe_block_cls)
    """
    decoder_layer_cls = None
    moe_block_cls = None

    # 1. 快速路径：直接访问第一层（适用于绝大多数 Transformer 架构）
    if hasattr(model, 'model') and hasattr(model.model, 'layers') and len(model.model.layers) > 0:
        first_layer = model.model.layers[0]
        decoder_layer_cls = type(first_layer)

        # 尝试在第一层内部查找 MoE 组件
        # 常见命名: block_sparse_moe (Mixtral/Qwen), moe (DeepSeek)
        for attr_name in ['block_sparse_moe', 'moe']:
            module = getattr(first_layer, attr_name, None)
            if module is not None:
                # 进一步确认是 MoE 模块（通常类名包含 SparseMoe 或 MoE）
                cls_name = type(module).__name__
                if 'SparseMoe' in cls_name or 'MoE' in cls_name:
                    moe_block_cls = type(module)
                    break

    # 2. 兜底逻辑：全局遍历搜索（防止特殊架构快速路径失效）
    if decoder_layer_cls is None:
        for name, module in model.named_modules():
            cls_name = type(module).__name__

            # 检测 DecoderLayer
            if decoder_layer_cls is None and ('DecoderLayer' in cls_name or 'Block' in cls_name):
                decoder_layer_cls = type(module)

            # 检测 MoE Block
            if moe_block_cls is None and ('SparseMoe' in cls_name or 'MoE' in cls_name):
                moe_block_cls = type(module)

    if decoder_layer_cls is None:
        raise ValueError("无法自动检测 DecoderLayer 类。请手动传入 decoder_layer_cls 参数。")

    return decoder_layer_cls, moe_block_cls


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    if hasattr(torch, 'npu') and torch.npu.is_available():
        backend = "hccl"
    else:
        backend = "nccl"
    from datetime import timedelta
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=7200))
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        torch.npu.manual_seed(seed)
        torch.npu.manual_seed_all(seed)


def lm_checkpoint(action='load', optimizer=None, epoch=0, global_step=0, wandb=None, scaler=None,
                  save_dir='../checkpoints', weight_prefix=None, **kwargs):
    """
    保存或加载训练状态（不包含模型权重）。
    Args:
        action: 'save' 表示保存，'load' 表示加载。
    """
    os.makedirs(save_dir, exist_ok=True)
    prefix = weight_prefix if weight_prefix is not None else 'resume'
    ckp_path = f'{save_dir}/{prefix}_resume.pth'

    if action == 'save':
        # ================= 保存模式 =================
        wandb_id = kwargs.get('wandb_id', None)
        if wandb_id is None and wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        resume_data = {
            'optimizer': optimizer.state_dict() if optimizer else None,
            'scaler': scaler.state_dict() if scaler else None,
            'epoch': epoch,
            'global_step': global_step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id,
        }
        # 保存其他额外传入的参数
        for key, value in kwargs.items():
            if key != 'wandb_id' and value is not None:
                resume_data[key] = value

        # 原子写入
        resume_tmp = ckp_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, ckp_path)
        del resume_data
        Logger(f"成功保存训练状态至 {ckp_path}")

    elif action == 'load':
        # ================= 加载模式 =================
        if not os.path.exists(ckp_path):
            Logger(f"断点文件 {ckp_path} 不存在，无法恢复训练状态。", level="warn")
            return None

        ckp_data = torch.load(ckp_path, map_location='cpu', weights_only=False)

        # 兼容旧版本：如果没有 global_step，尝试从 step 转换
        if 'global_step' not in ckp_data and 'step' in ckp_data:
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            old_micro_step = ckp_data['step']
            scaled_micro_step = old_micro_step * saved_ws // current_ws
            ckp_data['global_step'] = scaled_micro_step
            Logger(
                f'兼容旧版 Checkpoint: GPU数量变化({saved_ws}→{current_ws})，global_step 推算为 {ckp_data["global_step"]}')

        return ckp_data
    else:
        raise ValueError(f"lm_checkpoint 不支持的 action: {action}，必须是 'save' 或 'load'")


def init_model(model_name_or_path, from_weight=None, device='cuda', use_flash_attn=False):
    """加载模型和 tokenizer。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            Logger("Using FlashAttention-2")
        except ImportError:
            Logger("FlashAttention-2 not installed, fallback to SDPA", level="warn")
            attn_impl = "sdpa"
    else:
        attn_impl = "sdpa"

    if device.startswith("cuda") or device.startswith("npu"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 所有 rank 都在 CPU 上加载预训练权重
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=True,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
    )

    # 如果提供了训练后的权重路径，覆盖加载
    if from_weight is not None and os.path.exists(from_weight):
        Logger(f"Loading custom weights from {from_weight}")
        state_dict = torch.load(from_weight, map_location='cpu', weights_only=False)
        if 'model' in state_dict and isinstance(state_dict['model'], dict):
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)

    get_model_params(model)
    return model.to(device), tokenizer

class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理尾部不足 batch_size 的数据
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # 向上取整计算总 batch 数
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """Reward 模型封装，支持批量推理。"""

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        from transformers import AutoConfig
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, 'from_legacy_cache'):
            @classmethod
            def _from_legacy_cache(cls, past_key_values): return cls()

            DynamicCache.from_legacy_cache = _from_legacy_cache
        if not hasattr(DynamicCache, 'to_legacy_cache'):
            def _to_legacy_cache(self): return ()

            DynamicCache.to_legacy_cache = _to_legacy_cache

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if 'rope_type' in config.rope_scaling:
                rope_type = config.rope_scaling['rope_type']
                if rope_type == 'default': rope_type = 'linear'
                config.rope_scaling['type'] = rope_type
            if 'factor' not in config.rope_scaling:
                config.rope_scaling['factor'] = 1.0
        config.use_cache = False
        self.model = AutoModel.from_pretrained(model_path, config=config, dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)

    @torch.no_grad()
    def batch_get_scores(self, messages_list, responses):
        scores = []
        for messages, response in zip(messages_list, responses):
            scores.append(self.get_score(messages, response))
        return scores