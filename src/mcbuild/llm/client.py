"""OpenRouter chat client: tool calling + images + reasoning + usage/cost tracking."""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from io import BytesIO

from openai import OpenAI
from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cost_usd += other.cost_usd


@dataclass
class ChatResult:
    message: object
    usage: Usage
    raw: object = None


def image_to_data_url(img: Image.Image) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        return False
    return status in RETRYABLE_STATUS_CODES


class OpenRouterClient:
    """Thin wrapper over the OpenAI SDK pointed at OpenRouter."""

    def __init__(self, api_key: str | None = None, base_url: str = DEFAULT_BASE_URL):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in the environment or a .env file."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.total_usage = Usage()

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning: str = "medium",
        max_retries: int = 4,
    ) -> ChatResult:
        extra_body: dict = {"usage": {"include": True}}
        if reasoning and reasoning != "off":
            extra_body["reasoning"] = {"effort": reasoning}

        attempt = 0
        resp = None
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    extra_body=extra_body,
                )
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries or not _is_retryable(e):
                    raise
                time.sleep(min(2**attempt, 30))

        choice = resp.choices[0]
        usage_obj = getattr(resp, "usage", None)
        details = getattr(usage_obj, "completion_tokens_details", None)
        usage = Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(details, "reasoning_tokens", 0) or 0,
            cost_usd=getattr(usage_obj, "cost", 0.0) or 0.0,
        )
        self.total_usage.add(usage)
        return ChatResult(message=choice.message, usage=usage, raw=resp)

    def generate_image(self, model: str, prompt: str) -> bytes | None:
        """Best-effort concept-reference image generation. Returns None on any failure."""
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"modalities": ["image", "text"]},
            )
            choice = resp.choices[0]
            images = getattr(choice.message, "images", None)
            if not images:
                return None
            first = images[0]
            image_url = (
                first.get("image_url", {}).get("url")
                if isinstance(first, dict)
                else getattr(getattr(first, "image_url", None), "url", None)
            )
            if not image_url or not image_url.startswith("data:"):
                return None
            b64_data = image_url.split(",", 1)[1]
            return base64.b64decode(b64_data)
        except Exception:
            return None
