"""Scripted offline LLM: design brief -> broken blueprint -> fixed blueprint -> verify -> finish.

Used by `mcbuild --fake-llm` and the offline integration test so the full agent
loop (including error-driven self-repair) can be exercised without network access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mcbuild.llm.client import ChatResult, Usage


@dataclass
class _FnCall:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    id: str
    function: _FnCall
    type: str = "function"


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list | None = None
    reasoning_details: list | None = None
    reasoning: str | None = None


class FakeLLM:
    """A minimal duck-typed stand-in for OpenRouterClient."""

    def __init__(self) -> None:
        self.total_usage = Usage()
        self._step = 0
        self._script = [
            self._design_brief,
            self._broken_submit,
            self._fixed_submit,
            self._verify_query,
            self._verify_inspect,
            self._finish,
        ]

    def generate_image(self, model: str, prompt: str) -> bytes | None:
        """No-op stand-in: --fake-llm never requests a reference image."""
        return None

    def chat(self, model: str, messages: list[dict], tools=None, reasoning: str = "off", **kwargs) -> ChatResult:
        step = self._script[min(self._step, len(self._script) - 1)]
        self._step += 1
        message = step()
        usage = Usage(prompt_tokens=10, completion_tokens=10)
        self.total_usage.add(usage)
        return ChatResult(message=message, usage=usage, raw=None)

    def _design_brief(self) -> _FakeMessage:
        return _FakeMessage(content="Design brief: a small 5x5 stone hut with a doorway.")

    def _broken_submit(self) -> _FakeMessage:
        code = "walls(0, 0, 4, 4, 0, 2, 'stoen')\nfloor(0, 0, 4, 4, 0, 'stone')"
        return _FakeMessage(
            tool_calls=[
                _ToolCall(
                    id="call_1",
                    function=_FnCall(
                        name="submit_blueprint",
                        arguments=json.dumps({"code": code, "design_notes": "first attempt"}),
                    ),
                )
            ]
        )

    def _fixed_submit(self) -> _FakeMessage:
        code = "walls(0, 0, 4, 4, 0, 2, 'stone')\nfloor(0, 0, 4, 4, 0, 'stone')\nclear(2, 1, 0, 2, 2, 0)"
        return _FakeMessage(
            content="The typo was in the block name — 'stone' is correct. Adding a doorway too.",
            reasoning="The previous error said 'stoen' isn't a known block; closest match is 'stone'.",
            tool_calls=[
                _ToolCall(
                    id="call_2",
                    function=_FnCall(
                        name="submit_blueprint",
                        arguments=json.dumps({"code": code, "design_notes": "fixed block name, added doorway"}),
                    ),
                )
            ],
        )

    def _verify_query(self) -> _FakeMessage:
        return _FakeMessage(
            tool_calls=[
                _ToolCall(
                    id="call_verify_query",
                    function=_FnCall(
                        name="query", arguments=json.dumps({"mode": "slice", "slice_axis": "y", "slice_at": 0})
                    ),
                )
            ]
        )

    def _verify_inspect(self) -> _FakeMessage:
        return _FakeMessage(
            tool_calls=[
                _ToolCall(
                    id="call_verify_inspect",
                    function=_FnCall(name="inspect", arguments=json.dumps({"yaw": 2, "cutaway": "x"})),
                )
            ]
        )

    def _finish(self) -> _FakeMessage:
        return _FakeMessage(
            tool_calls=[
                _ToolCall(
                    id="call_3",
                    function=_FnCall(
                        name="finish",
                        arguments=json.dumps(
                            {
                                "summary": "Built a small stone hut with a doorway.",
                                "completed_interior_check": True,
                            }
                        ),
                    ),
                )
            ]
        )
