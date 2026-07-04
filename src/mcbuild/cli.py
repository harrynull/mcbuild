"""mcbuild CLI: typer entrypoint, rich progress feed, sixel/ANSI render display."""

from __future__ import annotations

import sys
from io import BytesIO

import typer
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from mcbuild.agent.loop import run_agent
from mcbuild.agent.prompts import build_reference_image_prompt
from mcbuild.config import Config
from mcbuild.export.schem import export_schem
from mcbuild.llm.client import OpenRouterClient
from mcbuild.llm.fake import FakeLLM
from mcbuild.render.sixel import encode_sixel, supports_sixel
from mcbuild.rundir import RunDir

app = typer.Typer(add_completion=False)
console = Console()


def _fmt_tokens(n: int) -> str:
    if n >= 1000:
        v = n / 1000
        return f"{v:.0f}k" if v >= 10 else f"{v:.1f}k"
    return str(n)


def _fit_for_terminal(img: Image.Image, max_width: int = 800) -> Image.Image:
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)))


def _display_image(img: Image.Image, display: str) -> None:
    if display == "off":
        return
    mode = display
    if mode == "auto":
        mode = "sixel" if supports_sixel() else "ansi"
    if mode == "sixel":
        try:
            small = _fit_for_terminal(img)
            sys.stdout.write(encode_sixel(small))
            sys.stdout.write("\n")
            sys.stdout.flush()
            return
        except Exception:
            mode = "ansi"
    if mode == "ansi":
        try:
            from rich_pixels import Pixels

            small = _fit_for_terminal(img, max_width=100)
            console.print(Pixels.from_image(small))
        except Exception:
            pass


@app.command()
def build(
    prompt: str = typer.Argument(..., help="Natural-language description of the build."),
    model: str = typer.Option("anthropic/claude-sonnet-5", "--model", help="Vision-capable OpenRouter model id."),
    max_iters: int = typer.Option(6, "--max-iters"),
    seed: int = typer.Option(0, "--seed"),
    display: str = typer.Option("auto", "--display", help="auto|sixel|ansi|off"),
    out: str = typer.Option("runs", "--out"),
    reference: bool = typer.Option(False, "--reference/--no-reference"),
    ref_model: str = typer.Option("openai/gpt-image-2", "--ref-model"),
    reasoning: str = typer.Option("medium", "--reasoning", help="off|low|medium|high"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream reasoning/completion text live."),
    cost_ceiling: float | None = typer.Option(
        None, "--cost-ceiling", help="Abort (keeping the best build so far) once usage cost reaches this many USD."
    ),
    session_id: str | None = typer.Option(
        None,
        "--session-id",
        help=(
            "Pin OpenRouter's `user` field to this value for sticky routing (keeps every "
            "request in this run on the same upstream, which prompt caching relies on). "
            "Defaults to a random id per run; pass an existing one to keep re-using its cache."
        ),
    ),
    fake_llm: bool = typer.Option(
        False, "--fake-llm", hidden=True, help="Use a scripted offline LLM (no network) for demos/tests."
    ),
) -> None:
    """Turn a natural-language prompt into a Minecraft building (.schem)."""
    load_dotenv()

    config = Config(
        model=model,
        ref_model=ref_model,
        max_iters=max_iters,
        seed=seed,
        display=display,
        out_dir=out,
        reference=reference,
        reasoning=reasoning,
        stream=stream,
        cost_ceiling=cost_ceiling,
    )

    rundir = RunDir.create(prompt, base=out)

    if fake_llm:
        llm: object = FakeLLM()
    else:
        try:
            llm = OpenRouterClient(session_id=session_id)
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

    session_line = f"\nsession: {llm.session_id}" if hasattr(llm, "session_id") else ""
    console.print(Panel(f"[bold]{prompt}[/bold]\nrun dir: {rundir.root}{session_line}", title="mcbuild"))

    reference_image = None
    if reference and not fake_llm:
        console.print("[dim]Generating concept reference image...[/dim]")
        img_prompt = build_reference_image_prompt(prompt)
        data = llm.generate_image(config.ref_model, img_prompt)
        if data is None:
            console.print(f"[dim]{config.ref_model} unavailable, trying fallback {config.ref_model_fallback}...[/dim]")
            data = llm.generate_image(config.ref_model_fallback, img_prompt)
        if data is None:
            console.print("[yellow]Reference image generation failed; continuing without it.[/yellow]")
        else:
            reference_image = Image.open(BytesIO(data)).convert("RGB")
            rundir.save_image("reference.png", reference_image)
            console.print(f"[green]Saved concept reference:[/green] {rundir.root / 'reference.png'}")
            _display_image(reference_image, config.display)

    # Streaming state for the current turn: reasoning/content each stream into their own
    # live-updating panel (only one Live may be active on a console at a time, so reasoning
    # is always torn down before content's Live starts). Reasoning is transient — once real
    # output starts (or the turn ends), its panel is cleared and replaced by a one-line
    # marker; assistant content is finalized into a permanent panel instead.
    stream_state: dict = {
        "reasoning_live": None,
        "content_live": None,
        "reasoning_buf": "",
        "content_buf": "",
        "reasoning_hidden_announced": False,
    }

    def _hide_reasoning_live(char_count: int) -> None:
        live = stream_state["reasoning_live"]
        if live is not None:
            live.stop()
            stream_state["reasoning_live"] = None
        console.print(f"[dim]· reasoning ({char_count} chars) — hidden[/dim]")
        stream_state["reasoning_hidden_announced"] = True

    def _stop_dangling_lives() -> None:
        """Safety net: tear down any Live left open by an early return (e.g. cost-ceiling abort)."""
        if stream_state["reasoning_live"] is not None:
            stream_state["reasoning_live"].stop()
            stream_state["reasoning_live"] = None
        if stream_state["content_live"] is not None:
            stream_state["content_live"].stop()
            stream_state["content_live"] = None

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "turn_start":
            _stop_dangling_lives()
            stream_state["reasoning_buf"] = ""
            stream_state["content_buf"] = ""
            stream_state["reasoning_hidden_announced"] = False
        elif event_type == "reasoning_delta":
            if stream_state["content_live"] is not None:
                return  # content already started this turn; nothing sensible to show live
            stream_state["reasoning_buf"] += data["text"]
            if stream_state["reasoning_live"] is None:
                stream_state["reasoning_live"] = Live(console=console, transient=True, refresh_per_second=12)
                stream_state["reasoning_live"].start()
            stream_state["reasoning_live"].update(
                Panel(
                    Text(stream_state["reasoning_buf"], style="dim italic"),
                    title="reasoning",
                    border_style="grey50",
                )
            )
        elif event_type == "content_delta":
            if stream_state["reasoning_live"] is not None:
                # real output started: the thinking panel's job is done, hide it
                _hide_reasoning_live(len(stream_state["reasoning_buf"]))
            stream_state["content_buf"] += data["text"]
            if stream_state["content_live"] is None:
                stream_state["content_live"] = Live(console=console, transient=True, refresh_per_second=12)
                stream_state["content_live"].start()
            stream_state["content_live"].update(
                Panel(Text(stream_state["content_buf"]), title="assistant", border_style="cyan")
            )
        elif event_type == "reasoning":
            if stream_state["reasoning_live"] is not None:
                # streamed live but the turn ended before any content arrived to hide it
                _hide_reasoning_live(len(data["text"]))
            elif not stream_state["reasoning_hidden_announced"]:
                # never streamed (non-streaming mode): show it in full, same as before
                console.print(
                    Panel(f"[dim italic]{data['text']}[/dim italic]", title="reasoning", border_style="grey50")
                )
        elif event_type == "assistant_text":
            if stream_state["content_live"] is not None:
                stream_state["content_live"].stop()
                stream_state["content_live"] = None
            console.print(Panel(data["text"], title="assistant", border_style="cyan"))
        elif event_type == "turn_usage":
            cache_note = f" (cache {data['cache_rate']:.0%})" if data.get("cached_tokens") else ""
            console.print(
                f"[dim]turn {data['turn']}: {_fmt_tokens(data['prompt_tokens'])} in{cache_note} / "
                f"{_fmt_tokens(data['completion_tokens'])} out / "
                f"{_fmt_tokens(data['reasoning_tokens'])} reasoning / "
                f"${data['cost_usd']:.2f}, cumulative ${data['cumulative_cost_usd']:.2f}[/dim]"
            )
        elif event_type in ("submit_blueprint", "str_replace", "edit_region"):
            region = data.get("region")
            title = f"iteration {data['iteration']}: {event_type}"
            if region:
                title += f" region={region}"
            console.print(
                Panel(
                    data["design_notes"] or "(no notes)",
                    title=title,
                    border_style="blue",
                )
            )
        elif event_type == "blueprint_error":
            console.print(
                Panel(data["error"], title=f"iteration {data['iteration']}: blueprint error", border_style="red")
            )
        elif event_type == "render":
            stats = data["stats"]
            dims = stats["dims"]
            dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}" if dims else "empty"
            iter_dir = rundir.root / f"iter_{data['iteration']:02d}"
            console.print(
                f"[green]render:[/green] {iter_dir / 'render.png'}  dims={dims_str}  blocks={stats['block_count']}"
            )
            if (iter_dir / "blueprint.schem").exists():
                console.print(f"[green]schem:[/green]  {iter_dir / 'blueprint.schem'}")
            _display_image(data["image"], config.display)
        elif event_type == "inspect":
            if data.get("mode") == "camera":
                console.print(f"[cyan]inspect (free camera pos={data['camera_pos']} look_at={data['look_at']})[/cyan]")
            elif data.get("slice_axis") is not None:
                console.print(
                    f"[cyan]inspect view (yaw={data['yaw']}, slice {data['slice_axis']}={data['slice_at']})[/cyan]"
                )
            else:
                console.print(f"[cyan]inspect view (yaw={data['yaw']}, cutaway={data.get('cutaway')})[/cyan]")
            _display_image(data["image"], config.display)
        elif event_type == "query":
            console.print(Panel(data["text"], title=f"query: {data['mode']}", border_style="cyan"))
        elif event_type == "finish":
            console.print(Panel(data["summary"], title="finished", border_style="green"))
        elif event_type == "abort":
            _stop_dangling_lives()
            console.print(Panel(data["reason"], title="aborted", border_style="red"))

    result = run_agent(prompt, llm, config, rundir, reference_image=reference_image, on_event=on_event)
    _stop_dangling_lives()

    usage = llm.total_usage
    reasoning_line = f"  reasoning tokens: {usage.reasoning_tokens}" if usage.reasoning_tokens else ""
    cache_line = f"  cache rate: {usage.cache_rate:.0%}" if usage.cached_tokens else ""
    console.print(
        Panel(
            f"iterations: {result.iterations}\n"
            f"prompt tokens: {usage.prompt_tokens}  completion tokens: {usage.completion_tokens}"
            f"{reasoning_line}{cache_line}\n"
            f"cost: ${usage.cost_usd:.4f}",
            title="usage",
            border_style="magenta",
        )
    )

    if result.grid is not None and len(result.grid) > 0:
        schem_path = rundir.root / "final.schem"
        export_schem(result.grid, str(schem_path))
        console.print(f"[bold green]Exported:[/bold green] {schem_path}")

        last_blueprint = rundir.root / f"iter_{result.iterations:02d}" / "blueprint.py"
        if result.iterations > 0 and last_blueprint.exists():
            (rundir.root / "final_blueprint.py").write_text(
                last_blueprint.read_text(encoding="utf-8"), encoding="utf-8"
            )
    else:
        console.print("[yellow]No successful build to export.[/yellow]")

    if not result.finished:
        console.print(f"[yellow]{result.summary}[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
