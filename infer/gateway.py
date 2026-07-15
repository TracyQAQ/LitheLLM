import os
import json
import asyncio
import logging
import httpx
import argparse
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from transformers import AutoTokenizer

# ==================== 命令行参数配置 ====================
parser = argparse.ArgumentParser(description="vLLM API Gateway (兼容 ChatML 与自定义 Prompt)")

parser.add_argument("--vllm-backend", type=str,
                    default="http://localhost:31039",
                    help="底层 vLLM 服务的完整地址 (例如: http://localhost:31039)")

parser.add_argument("--host", type=str,
                    default="0.0.0.0",
                    help="网关服务监听的 Host")

parser.add_argument("--port", type=int,
                    default="8081",
                    help="网关服务监听的端口")

parser.add_argument("--system-prompt-file", type=str,
                    default="",
                    help="系统提示词文件路径")

parser.add_argument("--tokenizer-path", type=str,
                    default="../vllm_model",
                    help="Tokenizer 权重目录路径，用于本地接管 Chat Template 渲染")

# 使用 parse_known_args 避免在某些 WSGI/ASGI 容器中运行时代入异常参数报错
args, _ = parser.parse_known_args()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Gateway")

app = FastAPI()

# ==================== 初始化 Tokenizer ====================
try:
    logger.info(f"正在从 {args.tokenizer_path} 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
except Exception as e:
    logger.warning(f"Tokenizer 加载失败，将启用原生的 Qwen ChatML 手动拼接。错误: {e}")
    tokenizer = None


# ==================== 加载系统提示词 ====================
def load_system_prompt() -> str:
    if os.path.exists(args.system_prompt_file):
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
            if prompt:
                logger.info(f"成功加载系统提示词: {args.system_prompt_file} ({len(prompt)} 字符)")
                return prompt
    logger.warning(f"未找到系统提示词文件 ({args.system_prompt_file})，将不使用默认系统提示词")
    return ""


SYSTEM_PROMPT = load_system_prompt()


# ==================== 核心：接管 Template 渲染 ====================
def build_raw_prompt(messages: list) -> str:
    """将标准的 messages 数组转化为模型底层的纯文本 Prompt"""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True,
                                             enable_thinking=False)  # 保留之前针对 Qwen 模板的修复

    # 兜底逻辑：手动拼接 Qwen 标准的 ChatML 格式
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ==================== 预热逻辑 ====================
async def warmup_vllm():
    """等待 vLLM 就绪并发送预热请求"""
    logger.info(f"正在等待底层 vLLM 服务就绪 ({args.vllm_backend})...")
    async with httpx.AsyncClient(timeout=300.0) as client:
        while True:
            try:
                resp = await client.get(f"{args.vllm_backend}/health")
                if resp.status_code == 200:
                    break
            except httpx.RequestError:
                pass
            await asyncio.sleep(2)

        logger.info("vLLM 已就绪，开始发送预热请求 (CUDA Graph 编译)...")

        # 构造通用预热用的 User Content
        messages = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": "你好"})

        raw_prompt = build_raw_prompt(messages)
        warmup_body = {
            "prompt": raw_prompt,
            "temperature": 0.0,
            "max_tokens": 1
        }

        try:
            await client.post(f"{args.vllm_backend}/v1/completions", json=warmup_body)
            logger.info("预热完成！网关已准备好接收高并发请求。")
        except Exception as e:
            logger.error(f"预热失败: {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(warmup_vllm())


# ==================== 路由与转发 ====================
@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # 1. 注入系统提示词 (仅在配置了 SYSTEM_PROMPT 且客户端未传 system 角色时注入)
    if SYSTEM_PROMPT and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})

    # 2. 强制转换为底层的 Completions 请求
    raw_prompt = build_raw_prompt(messages)

    # 这里可以根据通用问答需求调整 temperature 等参数
    completion_body = {
        "prompt": raw_prompt,
        "temperature": body.get("temperature", 0.7),
        "top_p": body.get("top_p", 0.9),
        "max_tokens": body.get("max_tokens", 2048),
        "stream": body.get("stream", False),
        "stop": ["<|im_end|>"]  # 提前停止标志
    }

    is_stream = completion_body["stream"]

    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0))
    req = client.build_request("POST", f"{args.vllm_backend}/v1/completions", json=completion_body)
    resp = await client.send(req, stream=is_stream)

    # 3. 响应格式适配器 (将 Completions 伪装回 Chat 格式返回给调用端)
    if is_stream:
        async def stream_adapter():
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        yield line + "\n\n"
                        continue
                    try:
                        data_json = json.loads(data_str)
                        if "choices" in data_json and len(data_json["choices"]) > 0:
                            text = data_json["choices"][0].get("text", "")
                            chat_chunk = {
                                "id": data_json.get("id", "chatcmpl-gateway"),
                                "object": "chat.completion.chunk",
                                "created": data_json.get("created", 0),
                                "model": body.get("model", "qwen3.5"),
                                "choices": [{
                                    "delta": {"content": text},
                                    "index": 0,
                                    "finish_reason": data_json["choices"][0].get("finish_reason")
                                }]
                            }
                            yield f"data: {json.dumps(chat_chunk)}\n\n"
                    except Exception as e:
                        logger.error(f"流式解析异常: {e}")
            await resp.aclose()
            await client.aclose()

        return StreamingResponse(
            stream_adapter(),
            media_type=resp.headers.get("content-type", "text/event-stream")
        )
    else:
        content = await resp.aread()
        try:
            data_json = json.loads(content)
            text = data_json["choices"][0].get("text", "")
            chat_resp = {
                "id": data_json.get("id", "chatcmpl-gateway"),
                "object": "chat.completion",
                "created": data_json.get("created", 0),
                "model": body.get("model", "qwen3.5"),
                "choices": [{
                    "message": {"role": "assistant", "content": text},
                    "index": 0,
                    "finish_reason": data_json["choices"][0].get("finish_reason")
                }]
            }
            content = json.dumps(chat_resp).encode("utf-8")
        except:
            pass

        await resp.aclose()
        await client.aclose()
        return Response(
            content=content,
            media_type=resp.headers.get("content-type", "application/json"),
            status_code=resp.status_code
        )


# ==================== 其他路径透传 ====================
@app.api_route("/{path_name:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_all(path_name: str, request: Request):
    async with httpx.AsyncClient(timeout=300.0) as client:
        url = f"{args.vllm_backend}/{path_name}"
        resp = await client.request(
            request.method, url,
            content=await request.body(),
            headers=request.headers
        )
        return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)