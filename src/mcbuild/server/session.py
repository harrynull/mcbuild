"""Runs one agent build against a live client connection, streaming block deltas + chat."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from mcbuild.agent.loop import run_agent
from mcbuild.config import Config
from mcbuild.export.schem import export_schem
from mcbuild.llm.client import OpenRouterClient
from mcbuild.rundir import RunDir
from mcbuild.server import protocol
from mcbuild.server.delta import compute_block_delta
from mcbuild.voxel import VoxelGrid

Send = Callable[[str], Awaitable[None]]


def _format_render_chat(iteration: int, stats: dict) -> str:
    dims = stats["dims"]
    dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}" if dims else "empty"
    return f"[mcbuild] iteration {iteration}: {dims_str}, {stats['block_count']} blocks"


def _format_design_chat(event_type: str, data: dict) -> str:
    region = data.get("region")
    where = f" region={region}" if region else ""
    notes = data.get("design_notes") or "(no notes)"
    return f"[mcbuild] iteration {data['iteration']} {event_type}{where}: {notes}"


def _format_cost_chat(data: dict) -> str:
    return f"[mcbuild] turn {data['turn']}: ${data['cost_usd']:.2f} (total ${data['cumulative_cost_usd']:.2f})"


async def run_build_session(prompt: str, send: Send, config: Config | None = None) -> None:
    """Run the agent loop for `prompt`, streaming block deltas + chat lines via `send`.

    `on_event` fires from the worker thread running `run_agent` (it's fully synchronous),
    so outgoing messages are marshaled onto this coroutine's event loop via a thread-safe
    queue; a background task drains the queue and performs the actual (async) websocket
    send, keeping the worker thread free of network I/O.

    `config` defaults to `Config()` (same defaults as the CLI, e.g. anthropic/claude-sonnet-5)
    when not given; `ws_server.py` builds one from server-wide `--model`/etc. CLI flags.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def enqueue(message: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, message)

    drain_task = asyncio.create_task(_drain_forever(queue, send))

    rundir = RunDir.create(prompt, base="runs")
    enqueue(protocol.hello(str(rundir.root)))

    try:
        llm = OpenRouterClient()
    except RuntimeError as e:
        enqueue(protocol.error(str(e)))
        await queue.join()
        drain_task.cancel()
        return

    config = config or Config()
    prev_grid: VoxelGrid | None = None

    def on_event(event_type: str, data: dict) -> None:
        nonlocal prev_grid
        if event_type in ("submit_blueprint", "edit_region", "str_replace"):
            enqueue(protocol.chat(_format_design_chat(event_type, data)))
        elif event_type == "render":
            grid = data["grid"]
            changes = compute_block_delta(prev_grid, grid)
            prev_grid = grid
            if changes:
                enqueue(protocol.blocks(data["iteration"], changes))
            enqueue(protocol.chat(_format_render_chat(data["iteration"], data["stats"])))
        elif event_type == "turn_usage":
            enqueue(protocol.chat(_format_cost_chat(data)))
        elif event_type == "blueprint_error":
            enqueue(protocol.chat(f"[mcbuild] iteration {data['iteration']} failed: {data['error']}"))
        elif event_type == "abort":
            enqueue(protocol.chat(f"[mcbuild] aborted: {data['reason']}"))
        elif event_type == "finish":
            enqueue(protocol.chat(f"[mcbuild] finished: {data['summary']}"))
        # reasoning/assistant_text/content_delta/reasoning_delta are intentionally not
        # forwarded to chat -- only tool-call design notes and build/cost progress are.

    result = await asyncio.to_thread(run_agent, prompt, llm, config, rundir, on_event=on_event)

    if result.grid is not None and len(result.grid) > 0:
        export_schem(result.grid, str(rundir.root / "final.schem"))

    if result.finished:
        enqueue(protocol.finish(result.summary))
    else:
        enqueue(protocol.error(result.summary))

    await queue.join()  # let the drain task flush everything enqueued above before we return
    drain_task.cancel()


async def _drain_forever(queue: asyncio.Queue[str], send: Send) -> None:
    while True:
        message = await queue.get()
        await send(message)
        queue.task_done()
