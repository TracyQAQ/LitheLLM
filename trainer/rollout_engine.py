import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from contextlib import nullcontext
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM


def _clear_model_cache(model):
    """清除 transformers 5.x 模型内部的 KV 缓存，防止跨 forward 调用污染"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    if isinstance(model, FSDP):
        _clear_model_cache(model.module)
        return
    # 清除模型级别的缓存
    if hasattr(model, '_cache'):
        model._cache = None
    if hasattr(model, 'past_key_values'):
        model.past_key_values = None
    # 清除每个 decoder layer 的缓存
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            if hasattr(layer, 'self_attn'):
                if hasattr(layer.self_attn, '_cached_kvp'):
                    layer.self_attn._cached_kvp = None
                # transformers 5.x HybridCache / StaticCache
                if hasattr(layer.self_attn, 'layer_cache'):
                    layer.self_attn.layer_cache = None


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int,
                            attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    计算每个 token 的 log probability。
    关键：必须传 attention_mask 且 use_cache=False！
    - attention_mask：左填充 PAD 必须被 mask，否则垃圾 logits → NaN
    - use_cache=False：防止 KV 缓存污染后续训练 forward
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, FSDP):
        logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits
    elif isinstance(model, DistributedDataParallel):
        input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
        logits = model.module(input_ids, attention_mask=attention_mask, use_cache=False).logits
    else:
        input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
        logits = model(input_ids, attention_mask=attention_mask, use_cache=False).logits

    logits = logits[:, -n_keep - 1:-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    attention_mask: Tensor


class RolloutEngine(ABC):
    tokenizer = None

    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        pass

    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        pass


def _build_full_attention_mask(prompt_attention_mask: Tensor, completion_ids: Tensor,
                                num_generations: int, eos_token_id: int) -> Tensor:
    """
    构建 output_ids 的完整 attention_mask。
    左填充的 PAD 必须被 mask 为 0，否则模型 attend 到 PAD 产生垃圾 logits → NaN。
    """
    expanded_prompt_mask = prompt_attention_mask.repeat_interleave(num_generations, dim=0)
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    completion_attn_mask = (
        torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
        <= eos_idx.unsqueeze(1)
    ).long()
    return torch.cat([expanded_prompt_mask, completion_attn_mask], dim=1)


class TorchRolloutEngine(RolloutEngine):
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda",
                 autocast_ctx=None, model_name_or_path: str = None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
        self.model_name_or_path = model_name_or_path
        self._gen_model = None

    def _ensure_gen_model(self):
        if self._gen_model is not None:
            return
        self._gen_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    def _sync_gen_model_from_fsdp(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

        self._ensure_gen_model()
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(self.policy_model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = self.policy_model.state_dict()
        self._gen_model.load_state_dict(state_dict, assign=True)
        self._gen_model = self._gen_model.to(self.device).eval()
        del state_dict

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        is_fsdp = isinstance(self.policy_model, FSDP)

        if is_fsdp:
            if self._gen_model is None:
                self._sync_gen_model_from_fsdp()

            # ========== Step 1: 用独立非 FSDP 模型 generate（速度快） ==========
            with torch.no_grad():
                output_ids = self._gen_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_generations,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            full_attention_mask = _build_full_attention_mask(
                attention_mask, completion_ids, num_generations, self.tokenizer.eos_token_id
            )

            # ========== Step 2: 用 FSDP 模型计算 logps ==========
            # 关键：use_cache=False + 清除缓存，防止 KV 缓存污染训练 forward
            _clear_model_cache(self.policy_model)

            ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
            with ctx:
                with torch.no_grad():
                    per_token_logps = compute_per_token_logps(
                        self.policy_model, output_ids, completion_ids.size(1),
                        attention_mask=full_attention_mask
                    )

            # 再次清除缓存，确保不影响后续训练 forward
            _clear_model_cache(self.policy_model)

        else:
            model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=num_generations,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            full_attention_mask = _build_full_attention_mask(
                attention_mask, completion_ids, num_generations, self.tokenizer.eos_token_id
            )

            ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
            with ctx:
                per_token_logps = compute_per_token_logps(
                    self.policy_model, output_ids, completion_ids.size(1),
                    attention_mask=full_attention_mask
                )

        return RolloutResult(output_ids, completion_ids, per_token_logps, completions, full_attention_mask)

    def update_policy(self, model: torch.nn.Module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        self.policy_model = model
        if isinstance(model, FSDP) and self.model_name_or_path:
            self._sync_gen_model_from_fsdp()


class SGLangRolloutEngine(RolloutEngine):
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests

    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int,
                temperature: float = 0.8) -> RolloutResult:
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]

        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
                "skip_special_tokens": False,  # 新增：保留 EOS 等特殊字符，以便正确计算 mask
            },
            "return_logprob": True,
            "return_text": True,  # 确保返回文本
        }
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        results = resp.json()
        if not isinstance(results, list):
            results = [results]

        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})

            # 1. 优先尝试获取 output_ids，如果没有则从 text 重新 tokenize
            completion_ids = meta.get("output_ids", result.get("output_ids", []))

            # 如果 SGLang 没返回 output_ids，则用 text 重新编码
            if not completion_ids:
                # SGLang 可能把 text 放在顶层，或者 meta_info 里没有
                text = result.get("text", "")
                # 注意：这里不要 skip special tokens，因为我们需要 eos_token 来停止
                completion_ids = self.tokenizer.encode(text, add_special_tokens=False)

            # 2. 解析 logprobs
            raw_logprobs = meta.get("output_token_logprobs", [])
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)

            if len(logprobs) > len(completion_ids):
                logprobs = logprobs[-len(completion_ids):]
            elif len(logprobs) < len(completion_ids):
                # 用极小值补齐，防止 exp 运算溢出
                logprobs.extend([-20.0] * (len(completion_ids) - len(logprobs)))

            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)

            # 解码时去掉 special tokens 用于展示和 reward 计算
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))

        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(lp) for lp in all_logprobs)

        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

        output_lengths = [len(ids) for ids in all_output_ids]
        output_attention_mask = torch.tensor(
            [[1] * l + [0] * (max_out_len - l) for l in output_lengths], device=device
        )

        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=-20.0),
            completions=completions,
            attention_mask=output_attention_mask,
        )

    def update_policy(self, model: torch.nn.Module):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig

        abs_path = os.path.abspath(self.shared_ckpt_path)
        is_fsdp = isinstance(model, FSDP)
        is_main = not dist.is_initialized() or dist.get_rank() == 0

        if is_fsdp:
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = model.state_dict()
            if is_main:
                unwrapped = model.module
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)
            if dist.is_initialized():
                dist.barrier()
            del state_dict
        else:
            if is_main:
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped.save_pretrained(abs_path, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)

        if is_main:
            resp = self.http.post(
                f"{self.base_url}/update_weights_from_disk",
                json={"model_path": abs_path},
                timeout=self.timeout
            )
            if resp.status_code != 200:
                print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
            return resp.status_code == 200
        # 【核心修改】增加分布式屏障，确保所有进程在此同步，等待 SGLang 更新完成
        if dist.is_initialized():
            dist.barrier()
        return True

    def flush_cache(self) -> bool:
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200

    def health(self) -> bool:
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


def create_rollout_engine(
        engine_type: str = "torch",
        policy_model: torch.nn.Module = None,
        tokenizer=None,
        device: str = "cuda",
        autocast_ctx=None,
        sglang_base_url: str = None,
        sglang_model_path: str = None,
        sglang_shared_path: str = None,
        model_name_or_path: str = None,
) -> RolloutEngine:
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx, model_name_or_path)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")