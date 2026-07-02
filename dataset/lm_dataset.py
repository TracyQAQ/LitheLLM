from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Value

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MAX_RETRIES = 100  # 防止异常样本导致无限递归


def pre_processing_chat(conversations, add_system_ratio=0.2):
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations):
        return conversations
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是一名AI机器人，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是一个AI大模型，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are a knowledgeable LLM, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are a helpful AI, a small but useful language model.",
    ]
    # 概率性添加 system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations


def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    # 以 80% 概率移除空思考标签
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(
            str(sample['text']), add_special_tokens=False,
            max_length=self.max_length - 2, truncation=True,
        ).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, system_prompt=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt  # 新增：保存外部传入的系统提示词
        features = Features({
            'conversations': [{
                'role': Value('string'),
                'content': Value('string'),
                'reasoning_content': Value('string'),
                'tools': Value('string'),
                'tool_calls': Value('string'),
            }]
        })
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 预计算 assistant 标记的 token IDs（只计算一次）
        self.assistant_start_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.assistant_end_ids = tokenizer.encode("<|im_end|>\n", add_special_tokens=False)

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        clean_messages = []
        tools = None  # 存储全局工具定义

        for message in conversations:
            role = message.get("role")
            content = message.get("content", "")

            if role is None:
                continue
            if content is None:
                content = ""
            content = str(content)

            # ========== 1. 提取全局工具定义 ==========
            # 如果 tools 字段在 system message 中，且为字符串，需解析为 List
            if role == "system" and message.get("tools"):
                try:
                    if isinstance(message["tools"], str):
                        tools = json.loads(message["tools"])
                    else:
                        tools = message["tools"]
                except Exception:
                    pass  # 解析失败忽略

            # ========== 2. 构建消息字典 ==========
            msg_dict = {"role": role, "content": content}

            # ========== 3. 透传 reasoning_content ==========
            if message.get("reasoning_content"):
                msg_dict["reasoning_content"] = message["reasoning_content"]

            # ========== 4. 修复 tool_calls 解析 ==========
            # 将 JSON 字符串转换为 Python 对象，供模板遍历
            if role == "assistant" and message.get("tool_calls"):
                raw_tool_calls = message["tool_calls"]
                try:
                    if isinstance(raw_tool_calls, str):
                        msg_dict["tool_calls"] = json.loads(raw_tool_calls)
                    else:
                        msg_dict["tool_calls"] = raw_tool_calls
                except Exception:
                    pass  # 解析失败保留原样

            # ========== 5. 修复过滤逻辑 ==========
            # 必须保留 tool 和 system 角色（即使 content 为空）
            # 其他角色（user, assistant）需有实际内容或特殊字段
            if role in ["tool", "system"]:
                clean_messages.append(msg_dict)
            elif content or msg_dict.get("reasoning_content") or msg_dict.get("tool_calls"):
                clean_messages.append(msg_dict)

        if not any(msg['role'] == 'user' for msg in clean_messages):
            raise ValueError("No user message found")

        # ========== 6. 调用模板 ==========
        # 将提取的 tools 作为独立参数传入
        return self.tokenizer.apply_chat_template(
            clean_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )

    def generate_labels(self, input_ids):
        """
        定位 assistant 回复部分，将非 assistant 部分的 label 设为 -100。
        使用 KMP-like 匹配提升效率。
        """
        labels = [-100] * len(input_ids)
        start_pattern = self.assistant_start_ids
        end_pattern = self.assistant_end_ids
        start_len = len(start_pattern)
        end_len = len(end_pattern)
        n = len(input_ids)
        i = 0
        while i <= n - start_len:
            # 检查是否匹配 assistant_start_ids
            matched = True
            for k in range(start_len):
                if input_ids[i + k] != start_pattern[k]:
                    matched = False
                    break
            if matched:
                start = i + start_len
                end = start
                # 寻找结束位置
                while end <= n - end_len:
                    end_matched = True
                    for k in range(end_len):
                        if input_ids[end + k] != end_pattern[k]:
                            end_matched = False
                            break
                    if end_matched:
                        break
                    end += 1
                # 标记 assistant 部分的 label
                label_end = min(end + end_len, self.max_length, n)
                for j in range(start, label_end):
                    labels[j] = input_ids[j]
                i = label_end
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        for retry in range(MAX_RETRIES):
            try:
                sample_idx = (index + retry) % len(self.samples)
                sample = self.samples[sample_idx]
                conversations = sample['conversations']
                # 验证：必须包含 user 角色
                if not conversations or not any(
                        msg.get('role') == 'user' and msg.get('content') for msg in conversations
                ):
                    continue

                # ============== 新增：处理系统提示词 ==============
                if self.system_prompt:
                    # 将对象转为可修改的列表形式
                    conversations = [dict(msg) for msg in conversations]
                    # 如果原对话带有 system，则替换其内容；否则在最前面插入
                    if conversations[0].get('role') == 'system':
                        conversations[0]['content'] = self.system_prompt
                    else:
                        conversations.insert(0, {'role': 'system', 'content': self.system_prompt})
                else:
                    # 如果没有外部提示词，走原有的随机添加逻辑
                    conversations = pre_processing_chat(conversations)
                # ================================================

                prompt = self.create_chat_prompt(conversations)
                prompt = post_processing_chat(prompt)
                input_ids = self.tokenizer(prompt, truncation=True, max_length=self.max_length).input_ids
                # input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
                labels = self.generate_labels(input_ids)
                return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
            except Exception:
                continue
        # 所有重试都失败，返回零填充的默认样本
        import logging
        logging.getLogger(__name__).warning(f"样本 index={index} 经 {MAX_RETRIES} 次重试仍失败，返回空样本")
        dummy = torch.zeros(self.max_length, dtype=torch.long)
        return dummy, torch.full((self.max_length,), -100, dtype=torch.long)



class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5, system_prompt=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio
        self.system_prompt = system_prompt  # 新增
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # ============== 新增：处理系统提示词 ==============
        if self.system_prompt:
            conversations = [dict(msg) for msg in conversations]
            if conversations and conversations[0].get('role') == 'system':
                conversations[0]['content'] = self.system_prompt
            else:
                conversations.insert(0, {'role': 'system', 'content': self.system_prompt})
        else:
            conversations = pre_processing_chat(conversations)
        # ================================================
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            enable_thinking=use_thinking,
            add_generation_prompt=True,
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])
        return {'prompt': prompt, 'answer': ""}


class AgentRLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}


if __name__ == "__main__":
    pass