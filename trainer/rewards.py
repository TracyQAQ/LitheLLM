import math
import re
import torch
from trainer.trainer_utils import Logger


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


class BaseRewardFunction:
    """奖励函数基类，新增任务时可继承此类并实现 calculate 方法"""

    def __init__(self, num_generations, reward_model=None):
        self.num_generations = num_generations
        self.reward_model = reward_model

    def calculate(self, prompts, responses, device):
        """
        计算奖励分数。
        Args:
            prompts: List[str], 输入提示词
            responses: List[str], 模型生成的回复
            device: torch.device, 计算设备
        Returns:
            torch.Tensor: 奖励分数张量
        """
        raise NotImplementedError("子类必须实现 calculate 方法")


class DefaultReward(BaseRewardFunction):
    """默认的 GRPO 奖励函数：包含长度惩罚、final_answer 格式奖励、重复惩罚和 RM 奖励"""

    def calculate(self, prompts, responses, device):
        rewards = torch.zeros(len(responses), device=device)
        nan_count = 0
        with torch.no_grad():
            batch_size = len(prompts)
            for i in range(batch_size):
                for j in range(self.num_generations):
                    response_idx = i * self.num_generations + j
                    response = responses[response_idx]
                    prompt = prompts[i]

                    pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                    matches = re.findall(pattern, prompt, re.DOTALL)
                    messages = [{"role": role, "content": content.strip()} for role, content in matches]
                    if not messages:
                        messages = [{"role": "user", "content": prompt}]

                    answer = response
                    # 1. 长度惩罚
                    rewards[response_idx] += 0.1 if 20 <= len(response.strip()) <= 800 else -0.5

                    # 2. final_answer 格式奖励 (增加渐进式引导)
                    if 'final_answer' in response:
                        thinking_content, answer_content = response.split('final_answer', 1)
                        # 完整格式奖励
                        rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                        rewards[response_idx] += 0.25 if response.count('final_answer') == 1 else -0.25
                        answer = answer_content.strip()
                        rewards[response_idx] -= rep_penalty(answer)
                    else:
                        # 【新增引导逻辑】：打破全是 0.5 分的死锁
                        # 检查模型是否输出了部分潜在的相关词汇（根据你的实际任务调整这些词）
                        if any(kw in response for kw in ['<|im_end|>', 'answer:', '答案', '结论', '最终结果']):
                            rewards[response_idx] += 0.2  # 明显的正奖励
                        else:
                            rewards[response_idx] -= 0.5  # 重罚无格式输出

                    # 3. Reward Model 奖励
                    if self.reward_model is not None:
                        try:
                            score = self.reward_model.get_score(messages, answer)
                        except Exception as e:
                            Logger(f"[WARN] reward_model.get_score failed: {e}", level="warn")
                            score = 0.0
                        if isinstance(score, torch.Tensor):
                            score = score.detach().cpu().float().item()
                        if math.isnan(score) or math.isinf(score):
                            nan_count += 1
                            score = 0.0
                        rewards[response_idx] += score

        if nan_count > 0:
            Logger(f"[WARN] {nan_count}/{batch_size * self.num_generations} reward scores were NaN/Inf", level="warn")
        return rewards

# ========== 在这里可以不断扩展新的奖励函数 ==========
# class MathTaskReward(BaseRewardFunction):
#     def calculate(self, prompts, responses, device):
#         # 自定义数学任务的奖励逻辑
#         pass