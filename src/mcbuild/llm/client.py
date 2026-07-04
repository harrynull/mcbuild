"""OpenRouter chat client: tool calling + images + reasoning + usage/cost tracking."""

from __future__ import annotations

import base64
import os
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from io import BytesIO

from openai import OpenAI
from PIL import Image

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# (kind, text) delta callback, kind in {"reasoning", "content"}.
OnDelta = Callable[[str, str], None]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def cache_rate(self) -> float:
        """Fraction of prompt_tokens served from cache (0.0-1.0), or 0.0 if none/unknown."""
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_tokens += other.cached_tokens
        self.cost_usd += other.cost_usd


@dataclass
class ChatResult:
    message: object
    usage: Usage
    raw: object = None


@dataclass
class _StreamFnCall:
    name: str | None
    arguments: str


@dataclass
class _StreamToolCall:
    id: str | None
    function: _StreamFnCall
    type: str = "function"


@dataclass
class StreamedMessage:
    """Duck-typed like an OpenAI ChatCompletionMessage: content/tool_calls/reasoning*."""

    content: str | None = None
    reasoning: str | None = None
    reasoning_details: list | None = None
    tool_calls: list[_StreamToolCall] | None = None


def _rd_field(piece, key):
    if isinstance(piece, dict):
        return piece.get(key)
    return getattr(piece, key, None)


def _merge_reasoning_detail(acc: dict[int, dict], fragment) -> None:
    """Merge one streamed reasoning_details fragment into its per-index block.

    Streaming delivers a reasoning block in pieces: `text`/`data` arrive across several
    fragments and the `signature` usually lands in a later one, all keyed by `index`.
    We reassemble complete blocks so the signature stays attached to its full text — a
    naive concat would send Anthropic a thinking block with a mismatched/missing
    signature and the request would be rejected.
    """
    idx = _rd_field(fragment, "index")
    idx = int(idx) if idx is not None else 0
    entry = acc.setdefault(idx, {})
    for key in ("type", "format", "id"):
        val = _rd_field(fragment, key)
        if val and not entry.get(key):
            entry[key] = val
    for key in ("text", "data", "summary"):
        val = _rd_field(fragment, key)
        if val:
            entry[key] = entry.get(key, "") + val
    sig = _rd_field(fragment, "signature")
    if sig:  # last non-empty wins; the signature covers the whole reassembled block
        entry["signature"] = sig


def consume_stream(chunks: Iterable, on_delta: OnDelta | None = None) -> tuple[StreamedMessage, object | None]:
    """Accumulate a stream of chat-completion-chunk objects into a full message.

    Works against any iterable of duck-typed chunks (real OpenAI SDK stream chunks,
    or plain objects/SimpleNamespaces in tests) each exposing `.choices[0].delta`
    (with optional `.content`, `.reasoning`, `.reasoning_details`, `.tool_calls`)
    and an optional `.usage` (populated on the final chunk when requested).
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    reasoning_acc: dict[int, dict] = {}
    tool_call_acc: dict[int, dict] = {}
    usage_obj = None

    for chunk in chunks:
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage_obj = chunk_usage

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        delta = choices[0].delta

        content_piece = getattr(delta, "content", None)
        if content_piece:
            content_parts.append(content_piece)
            if on_delta:
                on_delta("content", content_piece)

        reasoning_piece = getattr(delta, "reasoning", None)
        if reasoning_piece:
            reasoning_parts.append(reasoning_piece)
            if on_delta:
                on_delta("reasoning", reasoning_piece)

        details_piece = getattr(delta, "reasoning_details", None)
        if details_piece:
            for fragment in details_piece:
                _merge_reasoning_detail(reasoning_acc, fragment)

        delta_tool_calls = getattr(delta, "tool_calls", None)
        if delta_tool_calls:
            for tc in delta_tool_calls:
                idx = getattr(tc, "index", 0)
                entry = tool_call_acc.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        entry["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        entry["arguments"] += fn.arguments

    tool_calls = [
        _StreamToolCall(id=v["id"], function=_StreamFnCall(name=v["name"], arguments=v["arguments"]))
        for _, v in sorted(tool_call_acc.items())
    ] or None

    reasoning_details = [reasoning_acc[i] for i in sorted(reasoning_acc)] or None
    message = StreamedMessage(
        content="".join(content_parts) or None,
        reasoning="".join(reasoning_parts) or None,
        reasoning_details=reasoning_details,
        tool_calls=tool_calls,
    )
    return message, usage_obj


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

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        session_id: str | None = None,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in the environment or a .env file."
            )
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.total_usage = Usage()
        # A stable `user` id, sent with every request in this run, lets OpenRouter route
        # repeat calls to the same upstream/provider instance — without it, requests can
        # bounce between backends and lose the prompt-cache hits _with_prompt_caching relies on.
        self.session_id = session_id or uuid.uuid4().hex

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        reasoning: str = "medium",
        max_retries: int = 4,
        stream: bool = False,
        on_delta: OnDelta | None = None,
    ) -> ChatResult:
        if stream:
            return self._chat_streaming(model, messages, tools, reasoning, max_retries, on_delta)
        return self._chat_blocking(model, messages, tools, reasoning, max_retries)

    def _extra_body(self, reasoning: str) -> dict:
        extra_body: dict = {"usage": {"include": True}}
        if reasoning and reasoning != "off":
            extra_body["reasoning"] = {"effort": reasoning}
        return extra_body

    def _usage_from_obj(self, usage_obj: object | None) -> Usage:
        completion_details = getattr(usage_obj, "completion_tokens_details", None)
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        return Usage(
            prompt_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            cost_usd=getattr(usage_obj, "cost", 0.0) or 0.0,
        )

    def _chat_blocking(
        self, model: str, messages: list[dict], tools: list[dict] | None, reasoning: str, max_retries: int
    ) -> ChatResult:
        extra_body = self._extra_body(reasoning)

        attempt = 0
        resp = None
        while True:
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    user=self.session_id,
                    extra_body=extra_body,
                )
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries or not _is_retryable(e):
                    raise
                time.sleep(min(2**attempt, 30))

        choice = resp.choices[0]
        usage = self._usage_from_obj(getattr(resp, "usage", None))
        self.total_usage.add(usage)
        return ChatResult(message=choice.message, usage=usage, raw=resp)

    def _chat_streaming(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None,
        reasoning: str,
        max_retries: int,
        on_delta: OnDelta | None,
    ) -> ChatResult:
        extra_body = self._extra_body(reasoning)

        attempt = 0
        while True:
            try:
                stream = self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    user=self.session_id,
                    extra_body=extra_body,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                message, usage_obj = consume_stream(stream, on_delta)
                break
            except Exception as e:
                attempt += 1
                if attempt > max_retries or not _is_retryable(e):
                    raise
                time.sleep(min(2**attempt, 30))

        usage = self._usage_from_obj(usage_obj)
        self.total_usage.add(usage)
        return ChatResult(message=message, usage=usage, raw=None)

    def generate_image(self, model: str, prompt: str) -> bytes | None:
        """Best-effort concept-reference image generation. Returns None on any failure."""
        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                user=self.session_id,
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
