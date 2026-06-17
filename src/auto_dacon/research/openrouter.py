from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol


class ResearchClient(Protocol):
    def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        ...


@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout: float | None = None
    temperature: float = 0.35

    def chat(self, model: str, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        from openai import OpenAI

        api_key = (self.api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required. Pass --openrouter-api-key or set the environment variable.")

        timeout = self.timeout
        if timeout is None:
            timeout = float(os.getenv("AUTO_DACON_LLM_TIMEOUT", "120"))
        client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError(f"OpenRouter model returned empty content: {model}")
        return content


def openrouter_chat(model: str, messages: list[dict[str, str]], api_key: str, max_tokens: int = 2200) -> str:
    return OpenRouterClient(api_key=api_key).chat(model, messages, max_tokens=max_tokens)
