import torch
import os
import shutil
import json
import argparse
from safetensors.torch import load_file as sf_load, save_file as sf_save
from glob import glob

def merge_shards(
        trained_weights_path,
        original_model_path,
        output_dir,
        save_dtype,
):
    # 1. 加载训练权重
    print("加载训练权重...")
    trained = torch.load(trained_weights_path, map_location="cpu")

    # 读取官方模型的配置
    with open(os.path.join(original_model_path, "config.json")) as f:
        config = json.load(f)
    text_config = config.get("text_config", config)
    tie_embeddings = text_config.get("tie_word_embeddings", False)
    print(f"tie_word_embeddings = {tie_embeddings}")

    # 构建官方键名 -> 训练权重的映射
    train_weights_map = {}
    for k, v in trained.items():
        # 清理常见前缀
        if k.startswith("_orig_mod."):
            k = k[10:]
        if k.startswith("module."):
            k = k[7:]

        # 处理 MTP 参数（官方键名以 "mtp." 开头，无额外前缀）
        if "mtp." in k:
            # 移除可能存在的 "model." 前缀，保留 "mtp." 形式
            if k.startswith("model."):
                k = k[6:]          # 去掉 "model."
            # 此时 k 应为 "mtp.layers.0.weight" 等
            official_key = k
        else:
            # 非 MTP 参数：语言部分需要加 "model.language_model."
            if not k.startswith("model."):
                k = "model." + k
            if k == "model.lm_head.weight":
                if tie_embeddings:
                    continue  # 丢弃 tied lm_head
                else:
                    official_key = "lm_head.weight"
            else:
                official_key = k.replace("model.", "model.language_model.", 1)

        train_weights_map[official_key] = v

    print(f"准备合并 {len(train_weights_map)} 个训练参数")

    # 2. 逐片处理官方模型的 safetensors 文件
    os.makedirs(output_dir, exist_ok=True)
    sf_files = sorted(glob(os.path.join(original_model_path, "*.safetensors")))
    sf_files = [f for f in sf_files if "index" not in f]
    print(f"找到 {len(sf_files)} 个分片文件")

    for sf_path in sf_files:
        print(f"处理 {os.path.basename(sf_path)}...")
        shard = sf_load(sf_path)
        replaced = 0
        for key in list(shard.keys()):
            if key in train_weights_map:
                shard[key] = train_weights_map[key].to(shard[key].dtype)
                replaced += 1
        print(f"  替换了 {replaced} 个参数")
        out_path = os.path.join(output_dir, os.path.basename(sf_path))
        sf_save({k: v.to(save_dtype) for k, v in shard.items()}, out_path)
        del shard

    # 3. 复制配置文件
    for extra in ["config.json", "preprocessor_config.json", "generation_config.json",
                  "vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
                  "chat_template.jinja", "model.safetensors.index.json"]:
        src = os.path.join(original_model_path, extra)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, extra))
            print(f"已复制 {extra}")

    print("权重合并完成！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_weights_path", required=True)
    parser.add_argument("--original_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    save_dtype = dtype_map[args.save_dtype]

    merge_shards(
        trained_weights_path=args.trained_weights_path,
        original_model_path=args.original_model_path,
        output_dir=args.output_dir,
        save_dtype=save_dtype,
    )