import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import functools
import math
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
    CPUOffload,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import (
    Logger,
    is_main_process,
    lm_checkpoint,
    init_distributed_mode,
    setup_seed,
    SkipBatchSampler,
    init_model,
    LMForRewardModel,
    get_model_block_classes,
    # get_decoder_layer_class,
)
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps, _clear_model_cache
import warnings

warnings.filterwarnings('ignore')
mp.set_start_method('spawn', force=True)


def grpo_train_epoch(epoch, loader, iters, args, model, optimizer, scaler, scheduler,
                     rollout_engine, ref_model, reward_function, tokenizer,
                     save_checkpoint_fn, start_global_step=0, wandb=None):
    global_step = start_global_step
    running_loss = 0.0
    running_reward = 0.0
    optimizer.zero_grad(set_to_none=True)
    ppo_epochs = getattr(args, 'ppo_epochs', 1)

    # 恢复这一行计算：用于日志显示进度
    steps_per_epoch = math.ceil(iters / args.accumulation_steps)

    # 1. 逻辑冲突校验
    if ppo_epochs > 1 and args.accumulation_steps > 1:
        if is_main_process():
            Logger("错误: ppo_epochs > 1 时不支持 accumulation_steps > 1。", level="error")
        raise ValueError("Configuration Conflict: ppo_epochs > 1 and accumulation_steps > 1")

    for micro_step, batch in enumerate(loader):
        t_step = time.time()
        prompts = batch['prompt']
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # 2. Rollout 采样
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids.clone()
        completion_ids = rollout_result.completion_ids.clone()
        completions = rollout_result.completions
        full_attention_mask = rollout_result.attention_mask.to(args.device)
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()

        _clear_model_cache(ref_model)
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(
                ref_model, outputs, completion_ids.size(1), attention_mask=full_attention_mask
            ).detach()

        # 3. 计算奖励与优势 (Advantage)
        rewards = reward_function.calculate(prompts, completions, args.device)
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)
        advantages = ((rewards - mean_r) / (std_r + 1e-4)).detach()

        is_eos = completion_ids == tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = (
                torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(
            1)).int().detach()

        # 4. PPO 内层循环
        for ppo_epoch in range(ppo_epochs):
            _clear_model_cache(model)
            with autocast_ctx:
                res = model(outputs, attention_mask=full_attention_mask, use_cache=False)
                logits = res.logits[:, :-1, :]
                per_token_logps = F.log_softmax(logits, dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:,
                                  -completion_ids.size(1):]

                kl_div = ref_per_token_logps - per_token_logps
                per_token_kl = torch.exp(kl_div) - kl_div - 1
                ratio = torch.exp(per_token_logps - old_per_token_logps)

                clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
                per_token_loss1 = ratio * advantages.unsqueeze(1)
                per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
                per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
                policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            if scaler is not None:
                scaler.scale(policy_loss).backward()
                scaler.unscale_(optimizer)
            else:
                policy_loss.backward()

            if args.grad_clip > 0:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            running_loss += policy_loss.item()
            running_reward += rewards.mean().item()

        # 5. 全局步数推进
        scheduler.step()
        rollout_engine.update_policy(model)
        global_step += 1

        # 6. 原版格式的日志输出
        if global_step % args.log_interval == 0:
            current_loss = running_loss / (args.log_interval * ppo_epochs)
            current_reward = running_reward / (args.log_interval * ppo_epochs)
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = ((ref_per_token_logps - per_token_logps.detach()) * completion_mask).sum().item() / max(
                completion_mask.sum().item(), 1)

            # 恢复计算 local_step
            local_step = (global_step - 1) % steps_per_epoch + 1

            Logger(
                f'Epoch:[{epoch + 1}/{args.epochs}] Step:[{local_step}/{steps_per_epoch}], '
                f'Reward: {current_reward:.4f}, KL: {kl_ref_val:.4f}, '
                f'AdvStd: {advantages.std().item():.4f}, '
                f'Loss: {current_loss:.6f}, Len: {avg_len_val:.0f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.8f}, '
                f'Time: {time.time() - t_step:.1f}s')

            if wandb and is_main_process():
                wandb.log({"reward": current_reward, "kl_ref": kl_ref_val, "policy_loss": current_loss,
                           "avg_response_len": avg_len_val, "learning_rate": optimizer.param_groups[0]['lr']},
                          step=global_step)

            running_loss, running_reward = 0.0, 0.0

        if args.save_interval > 0:
            # 重新计算当前 epoch 内的局部步数
            local_step = (global_step - 1) % steps_per_epoch + 1

            # 使用 local_step 作为触发条件
            if local_step % args.save_interval == 0:
                save_checkpoint_fn(epoch, global_step)

    return global_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen GRPO with FSDP")
    parser.add_argument("--save_dir", type=str, default="../out")
    parser.add_argument('--save_weight', default='grpo', type=str)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--max_seq_len', default=2048, type=int)
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl")
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="grpo", choices=["grpo", "cispo"])
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon_high", type=float, default=5.0)
    parser.add_argument('--model_name_or_path', default='../Qwen3.5-0.8B', type=str)
    parser.add_argument('--from_weight', default='full_sft', type=str)

    parser.add_argument("--use_reward_model", type=int, default=0, choices=[0, 1],
                        help="是否启用 reward model (1启用, 0仅用规则奖励)")
    parser.add_argument("--reward_model_path", type=str, default="../internlm2-1_8b-reward")

    parser.add_argument('--from_resume', default=1, type=int, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Qwen-GRPO")
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument("--debug_interval", type=int, default=20)
    parser.add_argument("--thinking_ratio", type=float, default=0.9)
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"])
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8996")
    parser.add_argument("--sglang_model_path", type=str, default="../model")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_fsdp", default=0, type=int, choices=[0, 1])
    parser.add_argument("--fsdp_sharding_strategy", type=str, default="full",
                        choices=["full", "shard_grad_op", "no_shard"])
    parser.add_argument("--fsdp_cpu_offload", action="store_true")
    parser.add_argument("--fsdp_backward_prefetch", type=str, default="backward_post",
                        choices=["backward_pre", "backward_post", "no_prefetch"])
    parser.add_argument("--system_prompt_file", type=str, default=None,
                        help="系统提示词的 Markdown 文件路径")
    parser.add_argument("--reward_func_type", type=str, default="DefaultReward",
                        help="使用的奖励函数类型，例如: DefaultReward, MathTaskReward 等")
    parser.add_argument("--ppo_epochs", type=int, default=3, help="PPO 内层循环次数，建议 3-4 次")
    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        if torch.cuda.is_available():
            args.device = f"cuda:{local_rank}"
        elif hasattr(torch, 'npu') and torch.npu.is_available():
            args.device = f"npu:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    use_fsdp = args.use_fsdp == 1 and dist.is_initialized()

    # ========== 2. ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs('../checkpoints', exist_ok=True)
    ckp_data = None
    if args.from_resume == 1:
        ckp_data = lm_checkpoint(action='load', save_dir='../checkpoints', weight_prefix=args.save_weight)

    # ========== 3. 混合精度 ==========
    if torch.cuda.is_available():
        device_type = "cuda";
        amp_module = torch.cuda.amp
    elif hasattr(torch, 'npu') and torch.npu.is_available():
        device_type = "npu";
        amp_module = torch.npu.amp
    else:
        device_type = "cpu";
        amp_module = None
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = amp_module.autocast(dtype=dtype) if amp_module else nullcontext()

    # ========== 4. wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb

        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        wandb.init(project=args.wandb_project, name=f"GRPO-BS{args.batch_size}-LR{args.learning_rate}", id=wandb_id,
                   resume='must' if wandb_id else None)

    # ========== 5. 模型 ==========
    model_device = args.device if not use_fsdp else "cpu"

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
                    Logger(f"警告: 未找到 GRPO 权重 {resume_weight_path}，将使用 --from_weight 指定的 {weight_path}")
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

    model, tokenizer = init_model(args.model_name_or_path, weight_path, model_device, args.use_flash_attn)
    ref_model, _ = init_model(args.model_name_or_path, weight_path, model_device, args.use_flash_attn)
    ref_model = ref_model.eval().requires_grad_(False)

    model.config.use_cache = False
    ref_model.config.use_cache = False

    reward_model = None
    if args.use_reward_model == 1:
        Logger(f"Loading reward model on {args.device} with fp32...")
        reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float32)
    else:
        Logger("Reward model is disabled. Using rule-based rewards only.")

    # ========== 动态初始化奖励函数 ==========
    from trainer import rewards as rf_module

    # 直接使用传入的类名获取对应的类
    RewardClass = getattr(rf_module, args.reward_func_type, None)

    # 安全校验：确保类存在，且必须继承自 BaseRewardFunction
    if RewardClass is None or not issubclass(RewardClass, rf_module.BaseRewardFunction):
        raise ValueError(
            f"在 reward_functions.py 中找不到支持的奖励函数类: '{args.reward_func_type}'。"
            f"请确保类名拼写正确，且继承自 BaseRewardFunction。"
        )

    # 动态实例化
    reward_function = RewardClass(num_generations=args.num_generations, reward_model=reward_model)
    Logger(f"成功加载奖励函数: {args.reward_func_type}")
    # ========================================

    # ========== 6. FSDP / DDP ==========
    if use_fsdp:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing)

        sharding_strategy = {"full": ShardingStrategy.FULL_SHARD, "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
                             "no_shard": ShardingStrategy.NO_SHARD}[args.fsdp_sharding_strategy]
        backward_prefetch = \
            {"backward_pre": BackwardPrefetch.BACKWARD_PRE, "backward_post": BackwardPrefetch.BACKWARD_POST,
             "no_prefetch": None}[args.fsdp_backward_prefetch]
        mp_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        mixed_precision = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        cpu_offload = CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None
        device_id = torch.cuda.current_device() if device_type == "cuda" else (
            torch.npu.current_device() if device_type == "npu" else None)

        fsdp_kwargs = dict(sharding_strategy=sharding_strategy, mixed_precision=mixed_precision,
                           cpu_offload=cpu_offload, backward_prefetch=backward_prefetch, device_id=device_id,
                           sync_module_states=True, use_orig_params=True, limit_all_gathers=True)

        # 1. 获取 DecoderLayer 和 MoE Block 类
        decoder_layer_cls, moe_block_cls = get_model_block_classes(model)
        Logger(f"Detected decoder layer: {decoder_layer_cls.__name__}")
        if moe_block_cls:
            Logger(f"Detected MoE block class: {moe_block_cls.__name__}")

        # 2. Activation Checkpointing 配置
        # 注意：Checkpointing 只需应用在 DecoderLayer 级别，无需改动
        check_fn = lambda m: isinstance(m, decoder_layer_cls)
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=check_fn
        )
        apply_activation_checkpointing(
            ref_model,
            checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=check_fn
        )

        # 3. 构建包含 MoE 的包装策略
        transformer_wrap_classes = {decoder_layer_cls}
        if moe_block_cls:
            transformer_wrap_classes.add(moe_block_cls)

        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_wrap_classes,
        )
        Logger(f"FSDP auto wrap policy classes: {[cls.__name__ for cls in transformer_wrap_classes]}")
        # 4. 应用 FSDP
        # model 和 ref_model 使用相同的包装策略
        model = FSDP(model, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
        ref_model = FSDP(ref_model, auto_wrap_policy=auto_wrap_policy, **fsdp_kwargs)
        Logger("FSDP wrapping done (Model & RefModel)")

        if is_main_process():
            total_params = sum(p.numel() for p in model.parameters()) / 1e9
            Logger(f"FSDP model total params: {total_params:.3f}B")
    elif dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 7. 优化器与 Scheduler ==========
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler_enabled = (args.dtype == 'float16' and device_type == 'cuda')
    if device_type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    elif device_type == "npu":
        from torch.npu.amp import GradScaler

        scaler = GradScaler(enabled=False)
    else:
        scaler = None

    # ========== 8. Rollout 引擎 ==========
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine, policy_model=model, tokenizer=tokenizer, device=args.device,
        autocast_ctx=autocast_ctx, sglang_base_url=args.sglang_base_url, sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path, model_name_or_path=args.model_name_or_path,
    )

    # ========== 9. 数据 ==========
    # ========== 读取 Markdown 系统提示词 ==========
    system_prompt_text = None
    if args.system_prompt_file and os.path.exists(args.system_prompt_file):
        with open(args.system_prompt_file, 'r', encoding='utf-8') as f:
            system_prompt_text = f.read().strip()
        if is_main_process():
            Logger(f"成功从 {args.system_prompt_file} 加载系统提示词")
    # ============================================

    # 修改 RLAIFDataset 的初始化，传入 system_prompt
    train_ds = RLAIFDataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len + args.max_gen_len,
        thinking_ratio=args.thinking_ratio,
        system_prompt=system_prompt_text  # 传入读取到的内容
    )

    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    if dist.is_initialized() and train_sampler:
        iters_per_epoch = math.ceil(len(train_sampler) / args.batch_size)
    else:
        iters_per_epoch = math.ceil(len(train_ds) / args.batch_size)

    steps_per_epoch = math.ceil(iters_per_epoch / args.accumulation_steps)
    total_optimizer_steps = steps_per_epoch * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 10. 恢复训练状态 ==========
    start_epoch, global_step = 0, 0
    first_epoch_skip_micro = 0
    if ckp_data:
        start_epoch = ckp_data.get('epoch', 0)
        global_step = ckp_data.get('global_step', 0)
        if is_main_process():
            Logger(f"原始恢复状态: Epoch {start_epoch + 1}, Global Step {global_step}")

        if 'scaler' in ckp_data and scaler is not None and ckp_data['scaler'] is not None:
            scaler.load_state_dict(ckp_data['scaler'])

        if 'scheduler' in ckp_data and ckp_data['scheduler'] is not None:
            scheduler.load_state_dict(ckp_data['scheduler'])

        if global_step > 0:
            completed_epochs = global_step // steps_per_epoch
            remain_global_steps = global_step % steps_per_epoch
            start_epoch = completed_epochs
            first_epoch_skip_micro = remain_global_steps * args.accumulation_steps
            if is_main_process():
                Logger(
                    f"调整后：start_epoch={start_epoch + 1}, global_step={global_step}, 跳过micro步数={first_epoch_skip_micro}")


    # ========== 11. Checkpoint 保存函数 ==========
    def save_fsdp_checkpoint(epoch, current_global_step):
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            ms = model.state_dict()

        if is_main_process():
            sd = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
            ms = {k: v.to(sd).cpu() for k, v in ms.items()}
            model_ckp = f'{args.save_dir}/{args.save_weight}.pth'
            torch.save(ms, model_ckp)
            Logger(f"成功保存模型权重至 {model_ckp}")
            del ms

            wid = None
            if wandb:
                r = wandb.get_run() if hasattr(wandb, 'get_run') else None
                wid = getattr(r, 'id', None) if r else getattr(wandb, 'id', None)

            lm_checkpoint(
                action='save',
                optimizer=None,  # 不保存 optimizer 状态
                epoch=epoch,
                global_step=current_global_step,
                wandb=wandb,
                scaler=scaler if scaler_enabled else None,
                save_dir='../checkpoints',
                weight_prefix=args.save_weight,
                wandb_id=wid,
                scheduler=scheduler.state_dict()
            )

        if dist.is_initialized():
            dist.barrier()
        if device_type == "cuda":
            torch.cuda.empty_cache()


    def save_ddp_checkpoint(epoch, current_global_step):
        if not is_main_process():
            return
        model.eval()
        raw = model.module if isinstance(model, DistributedDataParallel) else model
        raw = getattr(raw, '_orig_mod', raw)
        sd = raw.state_dict()
        sdt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        model_ckp = f'{args.save_dir}/{args.save_weight}.pth'
        torch.save({k: v.to(sdt).cpu() for k, v in sd.items()}, model_ckp)
        Logger(f"成功保存模型权重至 {model_ckp}")

        lm_checkpoint(
            action='save',
            optimizer=None,  # 不保存 optimizer 状态
            epoch=epoch,
            global_step=current_global_step,
            wandb=wandb,
            scaler=scaler if scaler_enabled else None,
            save_dir='../checkpoints',
            weight_prefix=args.save_weight,
            scheduler=scheduler.state_dict()
        )
        model.train();
        del sd
        if device_type == "cuda":
            torch.cuda.empty_cache()


    save_checkpoint_fn = save_fsdp_checkpoint if use_fsdp else save_ddp_checkpoint

    # ========== 12. 训练 ==========
    Logger("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)

        skip_micro_steps = first_epoch_skip_micro if epoch == start_epoch else 0
        if skip_micro_steps >= iters_per_epoch:
            if is_main_process():
                Logger(f"警告: skip_micro_steps ({skip_micro_steps}) >= iters_per_epoch ({iters_per_epoch})，强制设为 0")
            skip_micro_steps = 0

        batch_sampler = SkipBatchSampler(train_sampler or torch.randperm(len(train_ds)).tolist(), args.batch_size,
                                         skip_micro_steps)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True,
                            multiprocessing_context='spawn' if args.num_workers > 0 else None)

        if skip_micro_steps > 0 and is_main_process():
            Logger(
                f'Epoch [{epoch + 1}/{args.epochs}]: 断点续训，跳过前 {skip_micro_steps} 个 micro-steps，从 global_step {global_step + 1} 继续')

        global_step = grpo_train_epoch(
            epoch, loader, iters_per_epoch, args, model, optimizer, scaler, scheduler,
            rollout_engine, ref_model, reward_function, tokenizer, save_checkpoint_fn, global_step, wandb
        )

        # Epoch 结束时保存一次 (如果 save_interval 没触发)
        final_local_step = global_step % steps_per_epoch
        if final_local_step == 0: final_local_step = steps_per_epoch  # 处理整除情况

        if args.save_interval <= 0 or final_local_step % args.save_interval != 0:
            save_checkpoint_fn(epoch, global_step)

    if dist.is_initialized():
        dist.destroy_process_group()
