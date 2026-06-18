"""简化的统一 LLM 客户端。

使用 OpenAI SDK 调用任何兼容 OpenAI 接口的服务。
相比 models.py，去掉了手写 HTTP/SSE/重试的复杂度，
代码量从 ~400 行缩减到 ~50 行。
"""

import os
from typing import Dict, List

from openai import OpenAI


class ModelClient:
    """统一的 LLM 客户端。

    基于 "Hello Agents" 的 HelloAgentsLLM 模式，
    使用 OpenAI SDK 调用任何兼容 OpenAI 接口的服务。
    """

    def __init__(
        self,
        model: str = None,
        api_key: str = None,
        base_url: str = None,
        timeout: int = None,
    ):
        self.model = model or os.getenv("LLM_MODEL_ID")
        api_key = api_key or os.getenv("LLM_API_KEY")
        base_url = base_url or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("model、api_key、base_url 必须提供或通过环境变量设置")

        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """调用模型并返回响应文本（流式输出）。"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )

        collected = []
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            collected.append(content)

        return "".join(collected)

    def complete(self, prompt: str, max_new_tokens: int, **kwargs) -> str:
        """兼容 pico runtime 的 complete() 接口。

        将字符串 prompt 包装为 messages 格式后调用 think()。
        """
        messages = [{"role": "user", "content": prompt}]
        text = self.think(messages)
        if text:
            return text
        return ""


# ── 使用示例 ──────────────────────────────────────────────────
if __name__ == "__main__":
    client = ModelClient(
        model="deepseek-chat",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    )
    result = client.think([{"role": "user", "content": "你好"}])
    print(result)
