"""Agent orchestrator: tool dispatch, iteration state, prompt-cache breakpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

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
        # Preserve verbatim for Anthropic extended-thinking + tool use, but normalize to
        # plain dicts (the non-streaming path hands back pydantic objects) so the message
        # round-trips as JSON when sent back to the API.
        d["reasoning_details"] = [_to_plain(rd) for rd in reasoning_details]
    return d


def _to_plain(obj: Any):
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return obj


def _tool_result(tool_call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _strip_stale_reasoning(messages: list[dict]) -> None:
    """Keep thinking blocks only on the most recent assistant message.

    Anthropic (via OpenRouter) requires the thinking block with its `signature` on the
    latest assistant turn to continue tool use, but rejects the request if any *stale*
    thinking block in history has a signature that no longer validates. Dropping older
    ones avoids that failure mode and saves tokens; the current turn's is untouched.
    """
    last_assistant = None
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_assistant = i
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and i != last_assistant and "reasoning_details" in m:
            m.pop("reasoning_details", None)


def _format_stats_for_tool(stats: dict) -> str:
    dims = stats["dims"]
    dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}" if dims else "empty"
    mats = ", ".join(f"{n} x{c}" for n, c in stats["top_materials"]) or "(none)"
    bounds = stats.get("bounds")
    if bounds:
        (minx, miny, minz), (maxx, maxy, maxz) = bounds
        bounds_str = f" bounds=[x {minx}..{maxx}, y {miny}..{maxy}, z {minz}..{maxz}]"
    else:
        bounds_str = ""
    return (
        f"Build succeeded. dimensions={dims_str}{bounds_str} "
        f"blocks={stats['block_count']} top_materials=[{mats}]"
    )


def _grid_delta(before: VoxelGrid, after: VoxelGrid) -> str:
    """Describe how `after` differs from `before` — added/removed/changed + changed bbox."""
    b = dict(before.items())
    a = dict(after.items())
    added = removed = changed = 0
    diff_coords = []
    for coord in set(a) | set(b):
        bv, av = b.get(coord), a.get(coord)
        if bv == av:
            continue
        diff_coords.append(coord)
        if bv is None:
            added += 1
        elif av is None:
            removed += 1
        else:
            changed += 1
    if not diff_coords:
        return "delta: NO CHANGE — this edit placed/removed nothing (check your coordinates/axis)."
    xs = [c[0] for c in diff_coords]
    ys = [c[1] for c in diff_coords]
    zs = [c[2] for c in diff_coords]
    region = f"x {min(xs)}..{max(xs)}, y {min(ys)}..{max(ys)}, z {min(zs)}..{max(zs)}"
    return f"delta: +{added} added, -{removed} removed, ~{changed} changed; affected region [{region}]."


def _cache_block(part: dict) -> dict:
    return {**part, "cache_control": {"type": "ephemeral"}}


def _with_cache_breakpoint(content):
    """Return `content` with a cache_control breakpoint on its trailing block.

    Anthropic (via OpenRouter) only caches a prefix when a content block carries
    cache_control; plain string content has no block to attach it to, so it's wrapped.
    """
    if isinstance(content, str):
        return [_cache_block({"type": "text", "text": content})]
    if isinstance(content, list) and content:
        return [*content[:-1], _cache_block(content[-1])]
    return content


def _with_prompt_caching(messages: list[dict]) -> list[dict]:
    """Build a request-only copy of `messages` with cache_control breakpoints on the system
    prompt and the latest user message (Anthropic allows up to 4; two is enough here).

    The system prompt (the DSL reference manual, 2k+ tokens) is byte-identical every turn
    and every run, and the latest-user breakpoint caches the whole growing prefix before
    it — cached reads are ~10x cheaper than a fresh read, and without any breakpoints every
    turn re-bills the entire context at full price. This never mutates `messages` itself, so
    a stale breakpoint from an older user turn can't linger into the next request.
    """
    last_user_idx = None
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            last_user_idx = i

    out = []
    for i, m in enumerate(messages):
        m2 = dict(m)
        if (i == 0 and m.get("role") == "system") or i == last_user_idx:
            m2["content"] = _with_cache_breakpoint(m["content"])
        out.append(m2)
    return out


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
    ref_thumb_url: str | None = None
    if reference_image is not None:
        user_content.append({"type": "image_url", "image_url": {"url": image_to_data_url(reference_image)}})
        thumb = reference_image.copy()
        thumb.thumbnail((320, 320))
        ref_thumb_url = image_to_data_url(thumb)
    messages.append({"role": "user", "content": user_content})

    best_grid: VoxelGrid | None = None
    best_stats: dict | None = None
    cumulative_source = ""  # full replayable program reflecting best_grid's construction
    iteration = 0  # per-attempt artifact counter (includes failed attempts)
    builds_done = 0  # successful builds; the edit budget (config.max_iters) counts these only
    consecutive_failures = 0

    def finalize(finished: bool, summary: str) -> AgentResult:
        rundir.write_json("session.json", _sanitize_messages_for_json(messages))
        return AgentResult(finished, summary, best_grid, best_stats, iteration, llm.total_usage)

    def report_success(grid: VoxelGrid, iteration: int, iter_dir, tc, note: str = "") -> dict:
        """Render/save/export a successful build and queue the critique image. Returns stats."""
        sheet, stats = views.build_contact_sheet(grid)
        rundir.save_image(f"iter_{iteration:02d}/render.png", sheet)
        (iter_dir / "stats.json").write_text(json.dumps(stats, indent=2))
        if len(grid) > 0:
            export_schem(grid, str(iter_dir / "blueprint.schem"))
        emit("render", iteration=iteration, stats=stats, image=sheet)
        result_text = _format_stats_for_tool(stats)
        if note:
            result_text += "\n" + note
        messages.append(_tool_result(tc.id, result_text))

        content: list[dict] = []
        if ref_thumb_url is not None:
            content.append({"type": "text", "text": "REFERENCE (reproduce this closely):"})
            content.append({"type": "image_url", "image_url": {"url": ref_thumb_url}})
            content.append({"type": "text", "text": "YOUR CURRENT BUILD:"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(sheet)}})
            content.append({"type": "text", "text": prompts.build_reference_critique_nudge()})
        else:
            content.append({"type": "text", "text": prompts.build_critique_nudge()})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(sheet)}})
        messages.append({"role": "user", "content": content})
        return stats

    llm_turns = 0
    max_llm_turns = config.max_iters * 4 + 6  # generous cap so text-only/inspect turns can't loop forever

    def on_delta(kind: str, text: str) -> None:
        emit(f"{kind}_delta", text=text)

    def budget_line() -> str:
        remaining = config.max_iters - builds_done
        if remaining <= 0:
            return "Edit budget exhausted — that was your FINAL edit. Verify, then call finish() now."
        if remaining == 1:
            return "Edits remaining: 1 — this is your last edit, make it count, then finish()."
        return f"Edits remaining: {remaining}. (inspect and query are FREE and do not use the budget.)"

    # The build-edit budget counts successful builds only (failed attempts and free
    # inspect/query turns don't burn it); llm_turns is a hard stop against runaway loops.
    while llm_turns < max_llm_turns:
        llm_turns += 1
        emit("turn_start")
        _strip_stale_reasoning(messages)  # keep thinking only on the latest assistant turn
        result = llm.chat(
            model=config.model,
            messages=_with_prompt_caching(messages),
            tools=tools.ALL_TOOLS,
            reasoning=config.reasoning,
            stream=config.stream,
            on_delta=on_delta,
        )
        msg = result.message
        messages.append(_message_to_dict(msg))
        emit(
            "turn_usage",
            turn=llm_turns,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            cached_tokens=result.usage.cached_tokens,
            cache_rate=result.usage.cache_rate,
            cost_usd=result.usage.cost_usd,
            cumulative_cost_usd=llm.total_usage.cost_usd,
        )

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
            if best_grid is None:
                nudge = "Please proceed: call submit_blueprint with your design."
            else:
                nudge = (
                    "Please proceed: refine with str_replace or edit_region, verify with "
                    "inspect/query (free), or call finish() if the build is done."
                )
            messages.append({"role": "user", "content": nudge})
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
                            "Cannot finish: you must verify the interior with at least one query slice "
                            "and one inspect cutaway.",
                        )
                    )
                    continue

                finished = True
                finish_summary = args.get("summary", "")
                messages.append(_tool_result(tc.id, "Build finished."))
                emit("finish", summary=finish_summary)
                break

            if name == "submit_blueprint":
                if builds_done >= config.max_iters:
                    messages.append(_tool_result(
                        tc.id,
                        "Edit budget reached — no edits remaining. Call finish() to export your best "
                        "build (further build calls are ignored).",
                    ))
                    continue
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
                    messages.append(_tool_result(tc.id, f"Blueprint failed (this did NOT use an edit):\n{e}"))
                    if consecutive_failures >= config.max_consecutive_failures:
                        emit("abort", reason="too many consecutive blueprint failures")
                        return finalize(False, "Aborted after repeated blueprint failures.")
                    continue

                consecutive_failures = 0
                builds_done += 1
                cumulative_source = code  # a full submit RESETS the construction history
                best_grid, best_stats = grid, report_success(grid, iteration, iter_dir, tc, note=budget_line())
                continue

            if name == "edit_region":
                if best_grid is None:
                    messages.append(
                        _tool_result(tc.id, f"No build exists yet; call submit_blueprint before {name}.")
                    )
                    continue
                if builds_done >= config.max_iters:
                    messages.append(_tool_result(
                        tc.id,
                        "Edit budget reached — no edits remaining. Call finish() to export your best "
                        "build (further build calls are ignored).",
                    ))
                    continue

                iteration += 1
                code = args.get("code", "")
                design_notes = args.get("design_notes", "")
                region = args.get("region")
                iter_dir = rundir.iter_dir(iteration)
                (iter_dir / "patch.py").write_text(code, encoding="utf-8")
                emit(name, iteration=iteration, design_notes=design_notes, code=code, region=region)

                # Run against a CLONE so a failing patch never corrupts the current build.
                candidate = best_grid.clone()
                if region is not None:
                    _clear_region(candidate, region)
                try:
                    sandbox.run_blueprint(code, candidate, seed=config.seed)
                except BlueprintError as e:
                    consecutive_failures += 1
                    emit("blueprint_error", iteration=iteration, error=str(e))
                    messages.append(_tool_result(tc.id, f"{name} failed (this did NOT use an edit):\n{e}"))
                    if consecutive_failures >= config.max_consecutive_failures:
                        emit("abort", reason="too many consecutive blueprint failures")
                        return finalize(False, "Aborted after repeated blueprint failures.")
                    continue

                consecutive_failures = 0
                builds_done += 1
                delta = _grid_delta(best_grid, candidate)
                header = f"\n\n# --- {name}" + (f" region={region}" if region else "") + " ---\n"
                cumulative_source = cumulative_source + header + code
                (iter_dir / "blueprint.py").write_text(cumulative_source, encoding="utf-8")
                best_grid, best_stats = candidate, report_success(
                    candidate, iteration, iter_dir, tc, note=delta + "\n" + budget_line()
                )
                continue

            if name == "str_replace":
                if best_grid is None:
                    messages.append(
                        _tool_result(tc.id, "No build exists yet; call submit_blueprint before str_replace.")
                    )
                    continue

                old_str = args.get("old_str", "")
                new_str = args.get("new_str", "")
                design_notes = args.get("design_notes", "")
                submit = args.get("submit", True)

                occurrences = cumulative_source.count(old_str) if old_str else 0
                if occurrences == 0:
                    messages.append(_tool_result(
                        tc.id,
                        "str_replace failed (this did NOT use an edit): old_str not found in the "
                        "current blueprint source. It must match exactly, whitespace included.",
                    ))
                    continue
                if occurrences > 1:
                    messages.append(_tool_result(
                        tc.id,
                        f"str_replace failed (this did NOT use an edit): old_str matches {occurrences} "
                        "locations in the current source. Include more surrounding context to make it unique.",
                    ))
                    continue

                new_source = cumulative_source.replace(old_str, new_str, 1)

                if not submit:
                    cumulative_source = new_source
                    messages.append(_tool_result(
                        tc.id,
                        "Edit staged (free, not built/rendered yet). Call str_replace with submit=true "
                        "(or submit_blueprint) when ready to build the accumulated edits.",
                    ))
                    continue

                if builds_done >= config.max_iters:
                    messages.append(_tool_result(
                        tc.id,
                        "Edit budget reached — no edits remaining. Call finish() to export your best "
                        "build (further build calls are ignored).",
                    ))
                    continue

                iteration += 1
                iter_dir = rundir.iter_dir(iteration)
                (iter_dir / "blueprint.py").write_text(new_source, encoding="utf-8")
                emit("str_replace", iteration=iteration, design_notes=design_notes, code=new_source)

                candidate = VoxelGrid()
                try:
                    sandbox.run_blueprint(new_source, candidate, seed=config.seed)
                except BlueprintError as e:
                    consecutive_failures += 1
                    emit("blueprint_error", iteration=iteration, error=str(e))
                    messages.append(_tool_result(tc.id, f"str_replace failed (this did NOT use an edit):\n{e}"))
                    if consecutive_failures >= config.max_consecutive_failures:
                        emit("abort", reason="too many consecutive blueprint failures")
                        return finalize(False, "Aborted after repeated blueprint failures.")
                    continue

                consecutive_failures = 0
                builds_done += 1
                delta = _grid_delta(best_grid, candidate)
                cumulative_source = new_source
                best_grid, best_stats = candidate, report_success(
                    candidate, iteration, iter_dir, tc, note=delta + "\n" + budget_line()
                )
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

    return finalize(False, "Stopped without an explicit finish (edit budget or turn limit reached).")
