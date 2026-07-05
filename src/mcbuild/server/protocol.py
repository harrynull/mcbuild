"""JSON message shapes sent to a connected mod client over the WebSocket.

This is the wire contract with the Forge mod's Gson POJOs (`net/ProtocolMessages.java`) —
keep the two in sync when changing shapes here.

    {"type": "hello", "run_dir": str}
    {"type": "chat", "text": str}
    {"type": "blocks", "iteration": int, "changes": [{"x": int, "y": int, "z": int, "block": str|None}]}
    {"type": "finish", "summary": str}
    {"type": "error", "reason": str}
"""

from __future__ import annotations

import json


def hello(run_dir: str) -> str:
    return json.dumps({"type": "hello", "run_dir": run_dir})


def chat(text: str) -> str:
    return json.dumps({"type": "chat", "text": text})


def blocks(iteration: int, changes: list[dict]) -> str:
    return json.dumps({"type": "blocks", "iteration": iteration, "changes": changes})


def finish(summary: str) -> str:
    return json.dumps({"type": "finish", "summary": summary})


def error(reason: str) -> str:
    return json.dumps({"type": "error", "reason": reason})
