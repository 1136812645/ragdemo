#!/usr/bin/env python3
import argparse
import json
import sys
import urllib.error
import urllib.request


DEFAULT_MODEL = "qwen3.6:latest"
DEFAULT_URL = "http://127.0.0.1:11434/api/chat"


def chat_once(model: str, prompt: str, url: str, timeout: float) -> str:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {error.code}: {details}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            "无法连接到 Ollama。请确认 `ollama serve` 正在运行，且监听在 127.0.0.1:11434。"
        ) from error

    try:
        return body["message"]["content"].strip()
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Ollama 返回了无法识别的响应: {body}") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="与本地 Ollama 模型进行命令行对话")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"模型名称，默认值: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Ollama 接口地址，默认值: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="单次请求超时时间（秒），默认 180",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="可选：直接发送单条消息；不传则进入交互模式",
    )
    return parser


def repl(model: str, url: str, timeout: float) -> int:
    print(f"已连接到 Ollama，当前模型: {model}")
    print("输入内容开始对话，输入 exit 或 quit 退出。")

    while True:
        try:
            prompt = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return 0

        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            print("已退出。")
            return 0

        try:
            reply = chat_once(model=model, prompt=prompt, url=url, timeout=timeout)
        except RuntimeError as error:
            print(f"错误: {error}", file=sys.stderr)
            continue

        print(f"模型: {reply}\n")


def main() -> int:
    args = build_parser().parse_args()

    if args.prompt:
        try:
            print(chat_once(args.model, args.prompt, args.url, args.timeout))
        except RuntimeError as error:
            print(f"错误: {error}", file=sys.stderr)
            return 1
        return 0

    return repl(args.model, args.url, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())