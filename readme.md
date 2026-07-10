# LitheLLM

LitheLLM 是一个轻量、灵活且高效的大语言模型（LLM）分布式训练框架。本项目支持从预训练（Pretraining）、指令微调（SFT）到强化学习对齐（GRPO）的完整训练链路。同时支持 NVIDIA GPU 生态和华为 NPU（Ascend）环境。

## ✨ 核心特性

* **全链路训练支持**：涵盖 Pretrain、Full SFT、GRPO (Generative Reward Policy Optimization) 等主流训练范式。
* **支持分布式训练**：内置 FSDP (Fully Sharded Data Parallel) 和 DDP，支持全参数分片、梯度分片以及 CPU Offload。
* **异构算力适配**：兼容 NVIDIA CUDA (支持 FlashAttention-2) 和 华为 NPU (Ascend) 生态，底层自动切换 `torch.cuda.amp` 与 `torch.npu.amp`。
* **支持断点续训**：提供 `xxx_resume.pth` 状态保存机制，自动保存 Scaler、Scheduler 状态及 Epoch/Step 进度，可从中断处精准恢复。

## 📂 项目结构

```text
LitheLLM/
├── checkpoints/          # 存放断点续训状态文件 (如 pretrain_resume.pth, full_sft_resume.pth)
├── dataset/              # 数据集及数据处理逻辑
│   ├── pretrain.jsonl    # Pretrain 训练数据
│   ├── full_sft.jsonl    # SFT 训练数据
│   ├── rlaif.jsonl       # GRPO 训练数据
│   └── lm_dataset.py     # Torch Dataset 实现，支持动态 Padding 和 Chat Template
├── out/                  # 训练输出目录，存放最终合并的权重文件 (如 pretrain.pth, full_sft.pth)
├── trainer/              # 训练器核心代码
│   ├── rewards.py        # 奖励函数计算逻辑 (支持 Rule-based 和 Reward Model)
│   ├── rollout_engine.py # GRPO 采样引擎 (PyTorch/SGLang)
│   ├── train_pretrain.py # 预训练启动脚本
│   ├── train_full_sft.py # 全量指令微调启动脚本
│   ├── train_grpo.py     # GRPO 强化学习启动脚本
│   └── trainer_utils.py  # 训练工具类 (日志、LR 调度、环境初始化等)
├── merge_weights.py      # 将训练后的 pth 权重与官方权重合并的脚本
├── README.md
└── requirements.txt
```

## ⚙️ 环境安装

```bash
git clone [https://github.com/your-username/LitheLLM.git](https://github.com/your-username/LitheLLM.git)
cd LitheLLM
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 预训练 (Pretraining)

预训练阶段期望的数据集格式为 JSONL，每行包含一个 `text` 字段。

```bash
torchrun --nproc_per_node=8 train_pretrain.py \
    --model_name_or_path /path/to/model \
    --from_weight None \
    --from_resume 1 \
    --data_path ../dataset/pretrain.jsonl \
    --save_dir ../out \
    --save_weight pretrain \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --accumulation_steps 5 \
    --max_seq_len 2048 \
    --dtype bfloat16 \
    --log_interval 10 \
    --save_interval 100 \
    --use_fsdp 1 \
    --fsdp_sharding_strategy full \
    --fsdp_cpu_offload \
    --fsdp_backward_prefetch backward_post
```

### 2. 指令微调 (Supervised Fine-Tuning)

SFT 阶段会自动调用 `apply_chat_template` 并且只对 Assistant 的回复计算 Loss。数据集为标准的对话格式（支持 tools 和 reasoning_content）。

```bash
torchrun --nproc_per_node=8 train_full_sft.py \
    --model_name_or_path /path/to/model \
    --from_weight pretrain \
    --from_resume 1 \
    --data_path ../dataset/full_sft.jsonl \
    --save_dir ../out \
    --save_weight full_sft \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 1e-5 \
    --warmup_steps 5 \
    --accumulation_steps 5 \
    --max_seq_len 2048 \
    --dtype bfloat16 \
    --log_interval 10 \
    --save_interval 100 \
    --use_fsdp 1 \
    --fsdp_sharding_strategy full \
    --fsdp_cpu_offload \
    --fsdp_backward_prefetch backward_post \
    --system_prompt_file /path/to/prompt_file
```

### 3. GRPO 强化学习对齐

GRPO 脚本 (`train_grpo.py`) 负责在给定的 prompt 下生成多条候选回答，并通过奖励函数计算优势（Advantage），更新策略模型。

使用torch作为推理引擎，启动训练命令如下。
```bash
torchrun --nproc_per_node=8 train_grpo.py \
    --model_name_or_path /path/to/model \
    --from_weight full_sft \
    --from_resume 1 \
    --data_path ../dataset/rlaif.jsonl \
    --save_dir ../out \
    --save_weight grpo \
    --epochs 5 \
    --batch_size 1 \
    --learning_rate 5e-7 \
    --max_seq_len 2048 \
    --max_gen_len 1024 \
    --num_generations 8 \
    --use_reward_model 0 \
    --thinking_ratio 0.0 \
    --log_interval 10 \
    --save_interval 100 \
    --thinking_ratio 0.0 \
    --rollout_engine torch \
    --use_fsdp 1 \
    --fsdp_sharding_strategy full \
    --fsdp_cpu_offload \
    --fsdp_backward_prefetch backward_post \
    --system_prompt_file /path/to/prompt_file \
    --reward_func_type DefaultReward \
    --ppo_epochs 3
```

若使用sglang作为推理引擎，用以下命令启动sglang服务器。
```bash
python -m sglang.launch_server \
    --model-path path/to/rollout/model \
    --port 8996 \
    --host 0.0.0.0
```
启动训练命令如下。
```bash
torchrun --nproc_per_node=8 train_grpo.py \
    --model_name_or_path /path/to/model \
    --from_weight full_sft \
    --from_resume 1 \
    --data_path ../dataset/rlaif.jsonl \
    --save_dir ../out \
    --save_weight grpo \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 5e-7 \
    --max_seq_len 2048 \
    --max_gen_len 1024 \
    --num_generations 8 \
    --use_reward_model 0 \
    --thinking_ratio 0.0 \
    --log_interval 10 \
    --save_interval 100 \
    --thinking_ratio 0.0 \
    --rollout_engine sglang \
    --sglang_base_url http://127.0.0.1:8996 \
    --sglang_model_path path/to/rollout/model \
    --sglang_shared_path path/to/sglang_ckpt_grpo \
    --use_fsdp 1 \
    --fsdp_sharding_strategy full \
    --fsdp_cpu_offload \
    --fsdp_backward_prefetch backward_post \
    --system_prompt_file /path/to/prompt_file \
    --reward_func_type DefaultReward \
    --ppo_epochs 3

```

## 🧩 自定义奖励函数 (Rewards)

支持 GRPO 添加自定义的任务（例如数学推理、代码评测），需在 `trainer/rewards.py` 中继承 `BaseRewardFunction`：

```python
class MathTaskReward(BaseRewardFunction):
    def calculate(self, prompts, responses, device):
        rewards = torch.zeros(len(responses), device=device)
        # 实现你的打分逻辑 ...
        return rewards
```
在启动 `train_grpo.py` 时传入类名 `--reward_func_type MathTaskReward` 即可。

## 💾 权重合并与导出

框架在训练时保存的是 PyTorch 原生 `state_dict` 格式的 `.pth` 文件（在 `out/` 目录下）。训练完成后，可使用 `merge_weights.py` 脚本，将这些增量/全量更新的张量与 HuggingFace 官方格式的模型进行合并，以便用于部署。
```bash
python merge_weights.py \
    --trained_weights_path ./out/xxx.pth \
    --original_model_path /path/to/original_model \
    --output_dir /path/to/output_model
```

## 📋 数据集格式要求

内置 `lm_dataset.py` 解析器，数据集为以下格式的 `.jsonl` 文件：

* **Pretrain**: `{"text": "大语言模型是..."}`
* **SFT**: `{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！我是AI。"}]}`
* **GRPO**: 与 SFT 格式类似，但模型只会将 `user` 的内容作为 prompt，并 `rollout` 生成 assistant 的回答用于 RL。

## 📜 License
This project is licensed under the MIT License. 

## 🙏 Acknowledgements
LitheLLM is built on the shoulders of giants: HuggingFace transformers, PyTorch FSDP, and the incredible open-source LLM community.