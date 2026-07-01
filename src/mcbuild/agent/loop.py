"""Agent orchestrator: tool dispatch, iteration state, context pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from mcbuild.agent import prompts, tools
from mcbuild.config import Config
from mcbuild.dsl import sandbox
from mcbuild.dsl.errors import BlueprintError
from mcbuild.llm.client import Usage, image_to_data_url
from mcbuild.render import views
from mcbuild.render.iso import render_iso
from mcbuild.rundir import RunDir
from mcbuild.voxel import VoxelGrid

EventCallback = Callable[[str, dict], None]


@dataclass
class AgentResult:
    finished: bool
    summary: str
    grid: VoxelGrid | None
    stats: dict | None
    iterations: int
    usage: Usage


def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s) if s else {}
    except json.JSONDecodeError:
        return {}


def _message_to_dict(msg: Any) -> dict:
    d: dict[str, Any] = {"role": "assistant", "content": getattr(msg, "content", None) or ""}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ]
    reasoning_details = getattr(msg, "reasoning_details", None)
    if reasoning_details:
        d["reasoning_details"] = reasoning_details
    return d


def _tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _format_stats_for_tool(stats: dict) -> str:
    dims = stats["dims"]
    dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}" if dims else "empty"
    mats = ", ".join(f"{n} x{c}" for n, c in stats["top_materials"]) or "(none)"
    return f"Build succeeded. dimensions={dims_str} blocks={stats['block_count']} top_materials=[{mats}]"


def _prune_old_images(messages: list[dict], marks: list[int], keep_last: int = 2) -> None:
    stale = marks[:-keep_last] if len(marks) > keep_last else []
    for idx in stale:
        msg = messages[idx]
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                new_content.append({"type": "text", "text": "[render pruned to save tokens]"})
            else:
                new_content.append(part)
        msg["content"] = new_content


def _sanitize_messages_for_json(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m2 = dict(m)
        content = m2.get("content")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    new_content.append({"type": "image_url", "image_url": {"url": f"<{len(url)} chars omitted>"}})
                else:
                    new_content.append(part)
            m2["content"] = new_content
        out.append(m2)
    return out


def run_agent(
    prompt: str,
    llm: Any,
    config: Config,
    rundir: RunDir,
    reference_image: Image.Image | None = None,
    on_event: EventCallback | None = None,
) -> AgentResult:
    def emit(event_type: str, **data: Any) -> None:
        if on_event:
            on_event(event_type, data)

    messages: list[dict] = [{"role": "system", "content": prompts.build_system_prompt()}]
    user_content: list[dict] = [
        {"type": "text", "text": prompts.build_user_prompt(prompt, config.seed, reference_image is not None)}
    ]
    if reference_image is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(reference_image)}})
    messages.append({"role": "user", "content": user_content})

    best_grid: VoxelGrid | None = None
    best_stats: dict | None = None
    iteration = 0
    consecutive_failures = 0
    image_marks: list[int] = []

    def finalize(finished: bool, summary: str) -> AgentResult:
        rundir.write_json("session.json", _sanitize_messages_for_json(messages))
        return AgentResult(finished, summary, best_grid, best_stats, iteration, llm.total_usage)

    llm_turns = 0
    max_llm_turns = config.max_iters * 4 + 4  # generous cap so text-only turns can't loop forever

    while iteration < config.max_iters and llm_turns < max_llm_turns:
        llm_turns += 1
        result = llm.chat(model=config.model, messages=messages, tools=tools.ALL_TOOLS, reasoning=config.reasoning)
        msg = result.message
        messages.append(_message_to_dict(msg))

        if config.cost_ceiling is not None and llm.total_usage.cost_usd >= config.cost_ceiling:
            emit("abort", reason=f"cost ceiling of ${config.cost_ceiling:.2f} reached")
            return finalize(False, f"Aborted: cost ceiling of ${config.cost_ceiling:.2f} reached.")

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            text = getattr(msg, "content", "") or ""
            emit("assistant_text", text=text)
            messages.append(
                {"role": "user", "content": "Please proceed: call submit_blueprint with your design, or finish() if already done."}
            )
            continue

        finished = False
        finish_summary = ""

        for tc in tool_calls:
            name = tc.function.name
            args = _safe_json_loads(tc.function.arguments)

            if name == "finish":
                finished = True
                finish_summary = args.get("summary", "")
                messages.append(_tool_result(tc.id, "Build finished."))
                emit("finish", summary=finish_summary)
                break

            if name == "submit_blueprint":
                iteration += 1
                code = args.get("code", "")
                design_notes = args.get("design_notes", "")
                iter_dir = rundir.iter_dir(iteration)
                (iter_dir / "blueprint.py").write_text(code, encoding="utf-8")
                emit("submit_blueprint", iteration=iteration, design_notes=design_notes, code=code)

                grid = VoxelGrid()
                try:
                    sandbox.run_blueprint(code, grid, seed=config.seed)
                except BlueprintError as e:
                    consecutive_failures += 1
                    emit("blueprint_error", iteration=iteration, error=str(e))
                    messages.append(_tool_result(tc.id, f"Blueprint failed:\n{e}"))
                    if consecutive_failures >= config.max_consecutive_failures:
                        emit("abort", reason="too many consecutive blueprint failures")
                        return finalize(False, "Aborted after repeated blueprint failures.")
                    continue

                consecutive_failures = 0
                sheet, stats = views.build_contact_sheet(grid)
                rundir.save_image(f"iter_{iteration:02d}/render.png", sheet)
                (iter_dir / "stats.json").write_text(json.dumps(stats, indent=2))
                best_grid, best_stats = grid, stats
                emit("render", iteration=iteration, stats=stats, image=sheet)

                messages.append(_tool_result(tc.id, _format_stats_for_tool(stats)))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompts.build_critique_nudge()},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(sheet)}},
                        ],
                    }
                )
                image_marks.append(len(messages) - 1)
                _prune_old_images(messages, image_marks)
                continue

            if name == "inspect":
                if best_grid is None:
                    messages.append(_tool_result(tc.id, "Nothing has been built yet; call submit_blueprint first."))
                    continue
                yaw = int(args.get("yaw", 0) or 0)
                cutaway = args.get("cutaway", "none")
                clip = None if cutaway in (None, "none") else cutaway
                img = render_iso(best_grid, yaw=yaw, clip=clip)
                emit("inspect", yaw=yaw, cutaway=cutaway, image=img)
                messages.append(_tool_result(tc.id, "Inspection render attached in the next message."))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Inspection view (yaw={yaw}, cutaway={cutaway}):"},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                        ],
                    }
                )
                image_marks.append(len(messages) - 1)
                _prune_old_images(messages, image_marks)
                continue

            messages.append(_tool_result(tc.id, f"Unknown tool '{name}'."))

        rundir.write_json("session.json", _sanitize_messages_for_json(messages))
        if finished:
            return finalize(True, finish_summary)

    return finalize(False, "Reached max iterations without finishing.")
