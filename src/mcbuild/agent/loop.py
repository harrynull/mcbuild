"""Agent orchestrator: tool dispatch, iteration state, context pruning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from PIL import Image

from mcbuild.agent import prompts, query, tools
from mcbuild.config import Config
from mcbuild.dsl import sandbox
from mcbuild.dsl.errors import BlueprintError
from mcbuild.export.schem import export_schem
from mcbuild.llm.client import Usage, image_to_data_url
from mcbuild.render import views
from mcbuild.render.camera import Camera, CameraRenderError, render_from_camera
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


def _clear_region(grid: VoxelGrid, region) -> None:
    """Remove every cell inside the inclusive bbox [x1,y1,z1,x2,y2,z2] from the grid."""
    x1, y1, z1, x2, y2, z2 = (int(round(v)) for v in region)
    xlo, xhi = sorted((x1, x2))
    ylo, yhi = sorted((y1, y2))
    zlo, zhi = sorted((z1, z2))
    for x in range(xlo, xhi + 1):
        for y in range(ylo, yhi + 1):
            for z in range(zlo, zhi + 1):
                grid.clear_block(x, y, z)


def _extract_reasoning_text(msg: Any) -> str:
    """Best-effort plain-text reasoning summary, for CLI display (not sent back to the API)."""
    text = getattr(msg, "reasoning", None)
    if text:
        return str(text)
    details = getattr(msg, "reasoning_details", None) or []
    parts: list[str] = []
    for block in details:
        if isinstance(block, dict):
            piece = block.get("text") or block.get("summary")
        else:
            piece = getattr(block, "text", None) or getattr(block, "summary", None)
        if piece:
            parts.append(str(piece))
    return "\n".join(parts)


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
    cumulative_source = ""  # full replayable program reflecting best_grid's construction
    iteration = 0
    consecutive_failures = 0
    image_marks: list[int] = []

    def finalize(finished: bool, summary: str) -> AgentResult:
        rundir.write_json("session.json", _sanitize_messages_for_json(messages))
        return AgentResult(finished, summary, best_grid, best_stats, iteration, llm.total_usage)

    def report_success(grid: VoxelGrid, iteration: int, iter_dir, tc) -> dict:
        """Render/save/export a successful build and queue the critique image. Returns stats."""
        sheet, stats = views.build_contact_sheet(grid)
        rundir.save_image(f"iter_{iteration:02d}/render.png", sheet)
        (iter_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        if len(grid) > 0:
            export_schem(grid, str(iter_dir / "blueprint.schem"))
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
        return stats

    llm_turns = 0
    max_llm_turns = config.max_iters * 4 + 4  # generous cap so text-only turns can't loop forever

    def on_delta(kind: str, text: str) -> None:
        emit(f"{kind}_delta", text=text)

    while iteration < config.max_iters and llm_turns < max_llm_turns:
        llm_turns += 1
        emit("turn_start")
        result = llm.chat(
            model=config.model,
            messages=messages,
            tools=tools.ALL_TOOLS,
            reasoning=config.reasoning,
            stream=config.stream,
            on_delta=on_delta,
        )
        msg = result.message
        messages.append(_message_to_dict(msg))

        if config.cost_ceiling is not None and llm.total_usage.cost_usd >= config.cost_ceiling:
            emit("abort", reason=f"cost ceiling of ${config.cost_ceiling:.2f} reached")
            return finalize(False, f"Aborted: cost ceiling of ${config.cost_ceiling:.2f} reached.")

        reasoning_text = _extract_reasoning_text(msg)
        if reasoning_text:
            emit("reasoning", text=reasoning_text)
        content_text = getattr(msg, "content", "") or ""
        if content_text:
            emit("assistant_text", text=content_text)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
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
                if args.get("completed_interior_check") is not True:
                    messages.append(
                        _tool_result(
                            tc.id,
                            "Cannot finish: you must verify the interior with at least one query slice and one inspect cutaway.",
                        )
                    )
                    continue

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
                cumulative_source = code  # a full submit RESETS the construction history
                best_grid, best_stats = grid, report_success(grid, iteration, iter_dir, tc)
                continue

            if name in ("patch_blueprint", "edit_region"):
                if best_grid is None:
                    messages.append(
                        _tool_result(tc.id, f"No build exists yet; call submit_blueprint before {name}.")
                    )
                    continue

                iteration += 1
                code = args.get("code", "")
                design_notes = args.get("design_notes", "")
                region = args.get("region") if name == "edit_region" else None
                iter_dir = rundir.iter_dir(iteration)
                (iter_dir / "patch.py").write_text(code, encoding="utf-8")
                emit(name, iteration=iteration, design_notes=design_notes, code=code, region=region)

                # Run against a CLONE so a failing patch never corrupts the current build.
                candidate = best_grid.clone()
                if name == "edit_region" and region is not None:
                    _clear_region(candidate, region)
                try:
                    sandbox.run_blueprint(code, candidate, seed=config.seed)
                except BlueprintError as e:
                    consecutive_failures += 1
                    emit("blueprint_error", iteration=iteration, error=str(e))
                    messages.append(_tool_result(tc.id, f"{name} failed:\n{e}"))
                    if consecutive_failures >= config.max_consecutive_failures:
                        emit("abort", reason="too many consecutive blueprint failures")
                        return finalize(False, "Aborted after repeated blueprint failures.")
                    continue

                consecutive_failures = 0
                header = f"\n\n# --- {name}" + (f" region={region}" if region else "") + " ---\n"
                cumulative_source = cumulative_source + header + code
                (iter_dir / "blueprint.py").write_text(cumulative_source, encoding="utf-8")
                best_grid, best_stats = candidate, report_success(candidate, iteration, iter_dir, tc)
                continue

            if name == "inspect":
                if best_grid is None:
                    messages.append(_tool_result(tc.id, "Nothing has been built yet; call submit_blueprint first."))
                    continue
                yaw = int(args.get("yaw", 0) or 0)
                cutaway = args.get("cutaway", "none")
                slice_axis = args.get("slice_axis")
                slice_at = args.get("slice_at")
                camera_pos = args.get("camera_pos")
                look_at = args.get("look_at")
                if camera_pos is not None and look_at is not None:
                    try:
                        cam = Camera(position=tuple(camera_pos), look_at=tuple(look_at))
                        img = render_from_camera(best_grid, cam)
                    except CameraRenderError as e:
                        messages.append(_tool_result(tc.id, f"Free-camera inspect failed: {e}"))
                        continue
                    label = f"Free-camera inspection (camera_pos={camera_pos}, look_at={look_at}):"
                    emit("inspect", mode="camera", camera_pos=camera_pos, look_at=look_at, image=img)
                elif slice_axis is not None and slice_at is not None:
                    img = render_iso(best_grid, yaw=yaw, slice_spec=(slice_axis, int(slice_at)))
                    label = f"Inspection view (yaw={yaw}, slice {slice_axis}={int(slice_at)}):"
                    emit("inspect", yaw=yaw, slice_axis=slice_axis, slice_at=int(slice_at), image=img)
                else:
                    clip = None if cutaway in (None, "none") else cutaway
                    img = render_iso(best_grid, yaw=yaw, clip=clip)
                    label = f"Inspection view (yaw={yaw}, cutaway={cutaway}):"
                    emit("inspect", yaw=yaw, cutaway=cutaway, image=img)
                messages.append(_tool_result(tc.id, "Inspection render attached in the next message."))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": label},
                            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
                        ],
                    }
                )
                image_marks.append(len(messages) - 1)
                _prune_old_images(messages, image_marks)
                continue

            if name == "query":
                if best_grid is None:
                    messages.append(_tool_result(tc.id, "Nothing has been built yet; call submit_blueprint first."))
                    continue
                mode = args.get("mode")
                try:
                    if mode == "slice":
                        text = query.ascii_slice(best_grid, args.get("slice_axis", "y"), int(args.get("slice_at", 0)))
                    elif mode == "point":
                        text = query.point_query(best_grid, args.get("x", 0), args.get("y", 0), args.get("z", 0))
                    elif mode == "histogram":
                        text = query.material_histogram(best_grid, args.get("region"))
                    else:
                        text = f"Unknown query mode '{mode}'. Use slice, point, or histogram."
                except (ValueError, TypeError) as e:
                    text = f"query failed: {e}"
                emit("query", mode=mode, text=text)
                messages.append(_tool_result(tc.id, text))
                continue

            messages.append(_tool_result(tc.id, f"Unknown tool '{name}'."))

        rundir.write_json("session.json", _sanitize_messages_for_json(messages))
        if finished:
            return finalize(True, finish_summary)

    return finalize(False, "Reached max iterations without finishing.")
