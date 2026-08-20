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
import torch.nn as nn
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
        # # ================= 加载模式 =================
        # if not os.path.exists(ckp_path):
        #     Logger(f"断点文件 {ckp_path} 不存在，无法恢复训练状态。", level="warn")
        #     return None
        #
        # ckp_data = torch.load(ckp_path, map_location='cpu', weights_only=False)
        #
        # # 兼容旧版本：如果没有 global_step，尝试从 step 转换
        # if 'global_step' not in ckp_data and 'step' in ckp_data:
        #     saved_ws = ckp_data.get('world_size', 1)
        #     current_ws = dist.get_world_size() if dist.is_initialized() else 1
        #     old_micro_step = ckp_data['step']
        #     scaled_micro_step = old_micro_step * saved_ws // current_ws
        #     ckp_data['global_step'] = scaled_micro_step
        #     Logger(
        #         f'兼容旧版 Checkpoint: GPU数量变化({saved_ws}→{current_ws})，global_step 推算为 {ckp_data["global_step"]}')

        # ================= 加载模式 =================
        if not os.path.exists(ckp_path):
            Logger(f"断点文件 {ckp_path} 不存在，无法恢复训练状态。", level="warn")
            return None
        ckp_data = torch.load(ckp_path, map_location='cpu', weights_only=False)

        # === 新增/修改部分：统一的步数换算逻辑 ===
        # 获取保存的步数（兼容新旧版本）
        saved_global_step = ckp_data.get('global_step', ckp_data.get('step', 0))

        # 异构环境自适应逻辑
        saved_ws = ckp_data.get('world_size', 1)
        current_ws = dist.get_world_size() if dist.is_initialized() else 1

        # 如果卡数变化，进行换算
        if saved_ws != current_ws:
            # 这里的逻辑：总样本数守恒 -> step * world_size = constant
            # 新 step = 旧 step * 旧 ws / 新 ws
            adjusted_global_step = int(saved_global_step * saved_ws / current_ws)
            Logger(
                f"[异构恢复] GPU数量变化({saved_ws} -> {current_ws})，步数换算: {saved_global_step} -> {adjusted_global_step}")
            ckp_data['global_step'] = adjusted_global_step  # 覆盖为正确的步数
        else:
            ckp_data['global_step'] = saved_global_step

        # 兼容旧版本标记清理（可选）
        if 'step' in ckp_data:
            del ckp_data['step']
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
    """Reward 模型封装，支持 InternLM2 RM 和 Skywork-Reward 等通用 RM。"""

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        from transformers.cache_utils import DynamicCache

        # ---------- DynamicCache 兼容补丁 ----------
        if not hasattr(DynamicCache, 'from_legacy_cache'):
            @classmethod
            def _from_legacy_cache(cls, past_key_values):
                return cls()
            DynamicCache.from_legacy_cache = _from_legacy_cache

        if not hasattr(DynamicCache, 'to_legacy_cache'):
            def _to_legacy_cache(self):
                return ()
            DynamicCache.to_legacy_cache = _to_legacy_cache
        # -------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if 'rope_type' in config.rope_scaling:
                rope_type = config.rope_scaling['rope_type']
                if rope_type == 'default':
                    rope_type = 'linear'
                config.rope_scaling['type'] = rope_type
            if 'factor' not in config.rope_scaling:
                config.rope_scaling['factor'] = 1.0

        config.use_cache = False
        self.model = AutoModel.from_pretrained(model_path, config=config, dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

        # ============================================================
        # 关键修复 1：处理 tok_embeddings 越界（InternLM2 RM 需要）
        # ============================================================
        self._fix_embedding_size()

        # ============================================================
        # 关键修复 2：手动加载被丢弃的 score 打分头（Skywork-Reward 需要）
        # ============================================================
        self.score_head = self._load_score_head(model_path, dtype, device)

        if self.score_head is not None:
            Logger(f"[RM] 成功加载 score 打分头: {self.score_head.weight.shape}")

    def _fix_embedding_size(self):
        """如果 tokenizer 词表 > embedding 行数，手动扩展 tok_embeddings"""
        tokenizer_vocab_size = len(self.tokenizer)

        # 自动探测 embedding 层
        emb_layer = None
        emb_attr_name = None
        for attr in ['model.tok_embeddings', 'model.embed_tokens', 'embed_tokens', 'tok_embeddings']:
            obj = self.model
            try:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
                if hasattr(obj, 'weight') and len(obj.weight.shape) == 2:
                    emb_layer = obj
                    emb_attr_name = attr
                    break
            except AttributeError:
                continue

        if emb_layer is None:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Embedding) and module.weight.shape[0] > 1000:
                    emb_layer = module
                    emb_attr_name = name
                    break

        if emb_layer is None:
            return

        old_vocab = emb_layer.weight.shape[0]
        if tokenizer_vocab_size > old_vocab:
            new_emb = nn.Embedding(
                tokenizer_vocab_size, emb_layer.weight.shape[1],
                padding_idx=emb_layer.padding_idx
            ).to(self.model.device)
            with torch.no_grad():
                new_emb.weight[:old_vocab].copy_(emb_layer.weight.data)
                mean_vec = emb_layer.weight.data.mean(dim=0, keepdim=True)
                new_emb.weight[old_vocab:].copy_(
                    mean_vec.expand(tokenizer_vocab_size - old_vocab, -1)
                )
            new_emb.weight.requires_grad = emb_layer.weight.requires_grad

            # 写回模型
            parent = self.model
            parts = emb_attr_name.split('.')
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_emb)

            self.model.config.vocab_size = tokenizer_vocab_size
            Logger(f"[RM Fix] tok_embeddings 扩展: {old_vocab} -> {tokenizer_vocab_size}")

    def _load_score_head(self, model_path, dtype, device):
        """
        手动从 checkpoint 中加载 score 打分头。
        AutoModel 加载 Qwen3Model 时会丢弃 score.weight (UNEXPECTED)，
        需要手动读取并创建对应的 Linear 层。
        """
        import os

        # 1. 找到 checkpoint 文件（支持单文件和分片）
        files_to_check = []
        for fname in ['model.safetensors', 'pytorch_model.bin']:
            fpath = os.path.join(model_path, fname)
            if os.path.exists(fpath):
                files_to_check.append(fpath)
                break

        if not files_to_check:
            # 分片文件
            for fname in sorted(os.listdir(model_path)):
                if fname.startswith('model-') and (fname.endswith('.safetensors') or fname.endswith('.bin')):
                    files_to_check.append(os.path.join(model_path, fname))

        if not files_to_check:
            Logger(f"[RM] 未找到 checkpoint 文件，跳过 score head 加载")
            return None

        # 2. 遍历所有文件查找 score 权重
        score_weight = None
        score_bias = None

        for fpath in files_to_check:
            try:
                if fpath.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    sd = load_file(fpath)
                else:
                    sd = torch.load(fpath, map_location='cpu', weights_only=False)

                for key, val in sd.items():
                    if key.endswith('score.weight') or key == 'score.weight':
                        score_weight = val
                    elif key.endswith('score.bias') or key == 'score.bias':
                        score_bias = val
                    elif key.endswith('reward_head.weight') or key == 'reward_head.weight':
                        score_weight = val
                    elif key.endswith('reward_head.bias') or key == 'reward_head.bias':
                        score_bias = val
            except:
                continue

        if score_weight is None:
            Logger(f"[RM] checkpoint 中未找到 score 权重，可能模型自带 get_score 方法")
            return None

        # 3. 创建 Linear 层
        hidden_size = score_weight.shape[1]
        num_labels = score_weight.shape[0]

        score_layer = nn.Linear(hidden_size, num_labels, bias=(score_bias is not None))
        score_layer.weight.data = score_weight.to(dtype)
        if score_bias is not None:
            score_layer.bias.data = score_bias.to(dtype)
        score_layer = score_layer.to(device)

        return score_layer

    @torch.no_grad()
    def get_score(self, messages, response):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response},
        ]

        # ---------- 方式 1：模型自带 get_score (InternLM2 RM) ----------
        if hasattr(self.model, 'get_score') and callable(getattr(self.model, 'get_score')):
            try:
                score = self.model.get_score(self.tokenizer, eval_messages)
                if isinstance(score, torch.Tensor):
                    score = score.detach().cpu().float().item()
                return max(min(score, 3.0), -3.0)
            except Exception as e:
                Logger(f"[WARN] model.get_score failed: {e}", level="warn")
                # 降级到方式 2
                pass

        # ---------- 方式 2：手动前向 + score head (Skywork-Reward 等) ----------
        try:
            text = self.tokenizer.apply_chat_template(
                eval_messages, tokenize=False, add_generation_prompt=False
            )
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取最后一个非 padding token 的 hidden state
            if hasattr(outputs, 'last_hidden_state'):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
            else:
                raise ValueError("模型输出没有 last_hidden_state 或 hidden_states")

            last_idx = attention_mask.sum(dim=1) - 1
            last_token_hidden = last_hidden[torch.arange(last_hidden.size(0)), last_idx]

            # 用 score head 打分
            if self.score_head is not None:
                score = self.score_head(last_token_hidden)
                score = score.squeeze(-1).float().item()
            elif hasattr(self.model, 'v_head'):
                score = self.model.v_head(last_token_hidden).squeeze(-1).float().item()
            elif hasattr(self.model, 'score'):
                score = self.model.score(last_token_hidden).squeeze(-1).float().item()
            else:
                raise ValueError("没有可用的 score head")

            return max(min(score, 3.0), -3.0)

        except Exception as e:
            Logger(f"[WARN] get_score forward failed: {e}", level="warn")
            return 0.0

    @torch.no_grad()
    def batch_get_scores(self, messages_list, responses):
        scores = []
        for messages, response in zip(messages_list, responses):
            scores.append(self.get_score(messages, response))
        return scores