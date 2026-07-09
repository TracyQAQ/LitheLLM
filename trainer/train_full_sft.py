import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import logging
import warnings
import math
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
    FullOptimStateDictConfig,
)
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    get_lr,
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    init_model,
    SkipBatchSampler,
    get_model_block_classes,
    # get_decoder_layer_class,
)

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def dynamic_collate_fn(batch, pad_token_id, label_pad_id=-100):
    """动态 padding 到 batch 内最大长度，并生成 attention_mask"""
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    for ids, labs in zip(input_ids, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([labs, torch.full((pad_len,), label_pad_id, dtype=torch.long)]))
        attention_masks.append(
            torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

    return torch.stack(padded_input_ids), torch.stack(padded_labels), torch.stack(attention_masks)


def train_epoch(epoch, loader, iters, args, model, optimizer, scaler, autocast_ctx, device_type, global_step,
                wandb=None, save_checkpoint_fn=None):
    """
    执行一个 Epoch 的训练。
    返回更新后的 global_step。
    """
    start_time = time.time()
    last_log_time = start_time
    warmup_steps = getattr(args, 'warmup_steps', 0)

    # 使用 math.ceil 计算，包含 flush 产生的额外步数
    steps_per_epoch = math.ceil(iters / args.accumulation_steps)
    total_global_steps = args.epochs * steps_per_epoch

    # 统计变量
    running_loss = 0.0
    running_acc = 0.0
    log_micro_steps = 0

    # 记录上次打印日志时的 global_step
    last_log_step = global_step

    # 确保开始前梯度为空
    optimizer.zero_grad(set_to_none=True)

    for micro_step, (input_ids, labels, attention_mask) in enumerate(loader):
        input_ids = input_ids.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        attention_mask = attention_mask.to(args.device, non_blocking=True)

        # 1. 前向传播
        with autocast_ctx:
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)

        # 2. 获取真实的 Loss 和 Acc
        real_loss = outputs.loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        correct = (preds == shift_labels) & mask
        acc = correct.sum().float() / (mask.sum().float() + 1e-8)

        # 3. 缩放 Loss
        scaled_loss = real_loss / args.accumulation_steps
        running_loss += real_loss.item()
        running_acc += acc.item()
        log_micro_steps += 1

        # 4. 反向传播
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # 5. 判断是否达到累积步数
        is_accumulation_step = (micro_step + 1) % args.accumulation_steps == 0
        if is_accumulation_step:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            local_step = global_step % steps_per_epoch
            if local_step == 0:
                local_step = steps_per_epoch
            safe_step = min(global_step, total_global_steps)
            lr = get_lr(safe_step, total_global_steps, args.learning_rate, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # 保证当前 Epoch 最后一步一定会被打印
            if local_step % args.log_interval == 0 or local_step == steps_per_epoch:
                time_since_last_log = time.time() - last_log_time
                last_log_time = time.time()

                # 动态计算实际走过的步数
                steps_since_last_log = global_step - last_log_step
                last_log_step = global_step

                current_loss = running_loss / log_micro_steps
                current_acc = running_acc / log_micro_steps
                current_lr = optimizer.param_groups[-1]['lr']

                # 使用实际步数计算精确速度
                steps_per_sec = steps_since_last_log / max(time_since_last_log, 1e-5)
                remaining_steps = max(0, total_global_steps - global_step)
                eta_min = (remaining_steps / steps_per_sec) / 60 if steps_per_sec > 0 else 0

                if is_main_process():
                    Logger(
                        f'Epoch:[{epoch + 1}/{args.epochs}] Step:[{local_step}/{steps_per_epoch}], '
                        f'loss: {current_loss:.4f}, acc: {current_acc:.4f}, '
                        f'lr: {current_lr:.8f}, eta: {eta_min:.1f}min'
                    )
                if wandb:
                    wandb.log({
                        "train/loss": current_loss,
                        "train/accuracy": current_acc,
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch + 1,
                        "train/local_step": local_step
                    }, step=global_step)

                running_loss = 0.0
                running_acc = 0.0
                log_micro_steps = 0

                if args.save_interval > 0 and local_step % args.save_interval == 0:
                    model.eval()
                    save_checkpoint_fn(epoch, global_step)
                    model.train()

        # 清理显存
        del input_ids, labels, attention_mask, outputs, real_loss, scaled_loss

    # ================= Epoch 结束后的兜底处理 =================
    remainder = len(loader) % args.accumulation_steps
    if remainder != 0:
        if is_main_process():
            Logger(f"Epoch {epoch + 1} end: Flushing {remainder} remaining accumulated gradients.")
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        local_step = global_step % steps_per_epoch
        if local_step == 0:
            local_step = steps_per_epoch
        safe_step = min(global_step, total_global_steps)
        lr = get_lr(safe_step, total_global_steps, args.learning_rate, warmup_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 补充 flush 后最后一步的日志打印
        if is_main_process():
            time_since_last_log = time.time() - last_log_time
            steps_since_last_log = global_step - last_log_step
            steps_per_sec = steps_since_last_log / max(time_since_last_log, 1e-5)
            remaining_steps = max(0, total_global_steps - global_step)
            eta_min = (remaining_steps / steps_per_sec) / 60 if steps_per_sec > 0 else 0

            current_loss = running_loss / log_micro_steps if log_micro_steps > 0 else 0.0
            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}] Step:[{local_step}/{steps_per_epoch}] (Flushing completed), '
                f'loss: {current_loss:.4f}, lr: {optimizer.param_groups[-1]["lr"]:.8f}, eta: {eta_min:.1f}min'
            )
            if wandb and log_micro_steps > 0:
                avg_loss = running_loss / log_micro_steps
                wandb.log({"train/loss": avg_loss, "train/global_step": global_step}, step=global_step)

    return global_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3.5 Full SFT with FSDP (CUDA/NPU)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--warmup_steps", type=int, default=100, help="学习率预热步数")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=5, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=5, help="日志打印间隔 (基于 global_step)")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔 (基于 global_step)")
    parser.add_argument('--max_seq_len', default=2048, type=int, help="训练的最大截断长度")
    parser.add_argument("--data_path", type=str, default="../dataset/new_data.jsonl")
    parser.add_argument('--model_name_or_path', default='../Qwen3.5-9B', type=str)
    parser.add_argument('--from_weight', default=None, type=str)
    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-Full-SFT")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_fsdp", default=1, type=int, choices=[0, 1])
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full",
                        choices=["full", "shard_grad_op", "no_shard"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="backward_post",
                        choices=["backward_pre", "backward_post", "no_prefetch"])
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="系统提示词的 Markdown 文件路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        if torch.cuda.is_available():
            args.device = f"cuda:{local_rank}"
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            args.device = f"npu:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('../checkpoints', exist_ok=True)

    # ========== 3. 设置混合精度 ==========
    if torch.cuda.is_available():
        device_type = "cuda"
        amp_module = torch.cuda.amp
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu"
        amp_module = torch.npu.amp
    else:
        device_type = "cpu"
        amp_module = None

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    if amp_module is not None:
        autocast_ctx = amp_module.autocast(dtype=dtype)
    else:
        autocast_ctx = nullcontext()

    # ========== 4. 配 wandb ==========
    wandb = None
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = lm_checkpoint(
            action='load',
            save_dir='../checkpoints',
            weight_prefix=args.save_weight
        )

    if args.use_wandb and is_main_process():
        try:
            import swanlab as wandb
        except ImportError:
            import wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"Qwen3-Full-SFT-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 定义模型、数据 ==========
    use_fsdp = args.use_fsdp == 1 and dist.is_initialized()

    # ------------------ 新权重加载逻辑 ------------------
    weight_path = None
    if ckp_data:
        resume_weight_path = f'{args.save_dir}/{args.save_weight}.pth'
        if os.path.exists(resume_weight_path):
            weight_path = resume_weight_path
            if is_main_process():
                Logger(f"断点续训: 将从 {resume_weight_path} 恢复模型权重")
        else:
            if args.from_weight and os.path.exists(args.from_weight):
                weight_path = args.from_weight
                if is_main_process():
                    Logger(f"警告: 未找到 SFT 权重 {resume_weight_path}，将使用 --from_weight 指定的 {weight_path}")
            else:
                if is_main_process():
                    Logger("警告: 未找到任何可加载的权重，将使用原始预训练模型")
    else:
        if args.from_weight and os.path.exists(args.from_weight):
            weight_path = args.from_weight
            if is_main_process():
                Logger(f"加载外部权重: {weight_path}")
        else:
            if is_main_process():
                Logger("从头开始训练（使用原始权重）")
    # -------------------------------------------------

    model, tokenizer = init_model(
        model_name_or_path=args.model_name_or_path,
        from_weight=weight_path,
        device=args.device if not use_fsdp else "cpu",
        use_flash_attn=args.use_flash_attn,
    )
    model.config.use_cache = False

    # ========== 读取 Markdown 系统提示词 ==========
    system_prompt_text = None
    if args.system_prompt_file and os.path.exists(args.system_prompt_file):
        with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt_text = f.read().strip()
        if is_main_process():
            Logger(f"成功从 {args.system_prompt_file} 加载系统提示词")
    # ============================================

    train_ds = SFTDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        system_prompt=system_prompt_text  # 传入读取到的内容
    )
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # ========== 6. 从 ckp 恢复训练状态 (Optimizer/Scaler/Step) ==========
    start_epoch, global_step = 0, 0
    first_epoch_skip_micro = 0

    if ckp_data:
        start_epoch = ckp_data.get('epoch', 0)
        global_step = ckp_data.get('global_step', 0)
        if is_main_process():
            Logger(f"原始恢复状态: Epoch {start_epoch + 1}, Global Step {global_step}")

    # ========== 7. FSDP / DDP 包装 ==========
    if use_fsdp:
        import functools
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        sharding_strategy = {
            "full": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }[args.fsdp_sharding_strategy]

        backward_prefetch = {
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST,
            "no_prefetch": None,
        }[args.fsdp_backward_prefetch]

        mp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        mixed_precision = MixedPrecision(
            param_dtype=mp_dtype,
            reduce_dtype=mp_dtype,
            buffer_dtype=mp_dtype,
        )
        cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None

        if device_type == "cuda":
            device_id = torch.cuda.current_device()
        elif device_type == "npu":
            device_id = torch.npu.current_device()
        else:
            device_id = None

        fsdp_kwargs = dict(
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            device_id=device_id,
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=True,
        )

        # 1. 获取 DecoderLayer 和 MoE Block 类
        decoder_layer_cls, moe_block_cls = get_model_block_classes(model)
        Logger(f"Detected decoder layer class: {decoder_layer_cls.__name__}")
        if moe_block_cls:
            Logger(f"Detected MoE block class: {moe_block_cls.__name__}")

        # 2. 配置 Activation Checkpointing
        # 注意：Checkpointing 通常加在 DecoderLayer 级别，无需改动
        check_fn = lambda submodule: isinstance(submodule, decoder_layer_cls)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=check_fn,
        )
        Logger("Activation checkpointing applied (BEFORE FSDP wrapping)")

        # 3. 配置 FSDP Auto Wrap Policy
        # 关键：如果有 MoE Block，将其加入封装集合，实现更细粒度的参数切分
        transformer_wrap_classes = {decoder_layer_cls}
        if moe_block_cls:
            transformer_wrap_classes.add(moe_block_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_wrap_classes,
        )
        Logger(f"FSDP auto wrap policy classes: {[cls.__name__ for cls in transformer_wrap_classes]}")

        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
        Logger(f"FSDP wrapping completed with auto_wrap_policy")
        if is_main_process():
            total_params = sum(p.numel() for p in model.parameters()) / 1e9
            Logger(f"FSDP model total params: {total_params:.3f}B")
    elif dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
        Logger("Using DDP (FSDP disabled)")

    # ========== 8. 创建优化器 ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler_enabled = (args.dtype == 'float16' and device_type == 'cuda')

    if device_type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    elif device_type == "npu":
        scaler = torch.npu.amp.GradScaler(enabled=False)
    else:
        scaler = None

    if ckp_data and 'optimizer' in ckp_data and ckp_data['optimizer'] is not None:
        if isinstance(model, FSDP):
            load_optim_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, None, load_optim_policy):
                optim_state = FSDP.optim_state_dict_to_load(model, optimizer, ckp_data['optimizer'])
                optimizer.load_state_dict(optim_state)
            Logger("FSDP mode: optimizer state loaded successfully via FSDP.optim_state_dict_to_load")
        else:
            optimizer.load_state_dict(ckp_data['optimizer'])

    if ckp_data and 'scaler' in ckp_data and scaler is not None and ckp_data['scaler'] is not None:
        scaler.load_state_dict(ckp_data['scaler'])


    # ========== 9. 定义 checkpoint 保存函数 ==========
    def save_fsdp_checkpoint(epoch, current_global_step, is_epoch_end=False):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            model_state = model.state_dict()

        # 【修复2】仅在主进程保存，防止并发写文件冲突
        if is_main_process():
            save_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            model_state = {k: v.to(save_dtype).cpu() for k, v in model_state.items()}
            model_ckp = f'{args.save_dir}/{args.save_weight}.pth'
            torch.save(model_state, model_ckp)
            Logger(f"成功保存模型权重至 {model_ckp}")
            del model_state

            wandb_id = None
            if wandb:
                if hasattr(wandb, 'get_run'):
                    run = wandb.get_run()
                    wandb_id = getattr(run, 'id', None) if run else None
                else:
                    wandb_id = getattr(wandb, 'id', None)

            lm_checkpoint(
                action='save',
                optimizer=None,  # 保持不保存 optimizer 状态
                epoch=epoch,
                global_step=current_global_step,
                wandb=wandb,
                scaler=scaler if scaler_enabled else None,
                save_dir='../checkpoints',
                weight_prefix=args.save_weight,
                wandb_id=wandb_id
            )

        if dist.is_initialized():
            dist.barrier()
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "npu":
            torch.npu.empty_cache()


    def save_ddp_checkpoint(epoch, current_global_step, is_epoch_end=False):
        if not is_main_process():
            return
        ckp = f'{args.save_dir}/{args.save_weight}.pth'
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        save_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        torch.save({k: v.to(save_dtype).cpu() for k, v in state_dict.items()}, ckp)
        Logger(f"成功保存模型权重至 {ckp}")
        del state_dict

        lm_checkpoint(
            action='save',
            optimizer=optimizer,
            epoch=epoch,
            global_step=current_global_step,
            wandb=wandb,
            scaler=scaler if scaler_enabled else None,
            save_dir='../checkpoints',
            weight_prefix=args.save_weight,
        )
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "npu":
            torch.npu.empty_cache()


    save_checkpoint_fn = save_fsdp_checkpoint if use_fsdp else save_ddp_checkpoint

    # ========== 10. 训练循环 ==========
    from functools import partial

    # 【修复4】修正 iters_per_epoch 计算，考虑 DistributedSampler 的 padding 机制
    if dist.is_initialized() and train_sampler:
        iters_per_epoch = math.ceil(len(train_sampler) / args.batch_size)
    else:
        iters_per_epoch = math.ceil(len(train_ds) / args.batch_size)

    steps_per_epoch = math.ceil(iters_per_epoch / args.accumulation_steps)
    if ckp_data is not None and global_step > 0:
        completed_epochs = global_step // steps_per_epoch
        remain_global_steps = global_step % steps_per_epoch
        start_epoch = completed_epochs
        first_epoch_skip_micro = remain_global_steps * args.accumulation_steps

        if is_main_process():
            Logger(
                f"调整后：start_epoch={start_epoch + 1}, global_step={global_step}, 跳过micro步数={first_epoch_skip_micro}")

    else:
        start_epoch = 0
        first_epoch_skip_micro = 0

    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)

        skip_micro_steps = first_epoch_skip_micro if epoch == start_epoch else 0
        if skip_micro_steps >= iters_per_epoch:
            if is_main_process():
                Logger(f"警告: skip_micro_steps ({skip_micro_steps}) >= iters_per_epoch ({iters_per_epoch})，强制设为 0")
            skip_micro_steps = 0

        batch_sampler = SkipBatchSampler(train_sampler or list(range(len(train_ds))), args.batch_size, skip_micro_steps)
        my_collate = partial(dynamic_collate_fn, pad_token_id=tokenizer.pad_token_id)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                            collate_fn=my_collate)

        if skip_micro_steps > 0 and is_main_process():
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: 断点续训，跳过前 {skip_micro_steps} 个 micro-steps，从 global_step {global_step + 1} 继续')

        global_step = train_epoch(
            epoch,
            loader,
            iters_per_epoch,
            args,
            model,
            optimizer,
            scaler,
            autocast_ctx,
            device_type,
            global_step=global_step,
            wandb=wandb,
            save_checkpoint_fn=save_checkpoint_fn,
        )

        # Epoch 结束时保存一次 (如果 save_interval 没触发)
        if args.save_interval <= 0 or global_step % args.save_interval != 0:
            save_checkpoint_fn(epoch, global_step, is_epoch_end=True)

    # ========== 11. 清理 ==========
    if dist.is_initialized():
        dist.destroy_process_group()
