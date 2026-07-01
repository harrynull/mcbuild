"""mcbuild CLI: typer entrypoint, rich progress feed, sixel/ANSI render display."""

from __future__ import annotations

import sys
from io import BytesIO

import typer
from dotenv import load_dotenv
from PIL import Image
from rich.console import Console
from rich.panel import Panel

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
    console.print(Panel(f"[bold]{prompt}[/bold]\nrun dir: {rundir.root}", title="mcbuild"))

    if fake_llm:
        llm: object = FakeLLM()
    else:
        try:
            llm = OpenRouterClient()
        except RuntimeError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1) from e

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

    stream_state = {"reasoning": False, "content": False}

    def on_event(event_type: str, data: dict) -> None:
        if event_type == "turn_start":
            stream_state["reasoning"] = False
            stream_state["content"] = False
        elif event_type == "reasoning_delta":
            if not stream_state["reasoning"]:
                console.print("[dim italic]reasoning:[/dim italic] ", end="")
            stream_state["reasoning"] = True
            console.print(data["text"], end="", style="dim italic")
        elif event_type == "content_delta":
            if not stream_state["content"]:
                if stream_state["reasoning"]:
                    console.print()  # close the reasoning line first
                console.print("[cyan]assistant:[/cyan] ", end="")
            stream_state["content"] = True
            console.print(data["text"], end="")
        elif event_type == "reasoning":
            if stream_state["reasoning"]:
                console.print()  # already streamed live; just close the line
            else:
                console.print(Panel(f"[dim italic]{data['text']}[/dim italic]", title="reasoning", border_style="grey50"))
        elif event_type == "assistant_text":
            if stream_state["content"]:
                console.print()  # already streamed live; just close the line
            else:
                console.print(Panel(data["text"], title="assistant", border_style="cyan"))
        elif event_type == "submit_blueprint":
            console.print(
                Panel(
                    data["design_notes"] or "(no notes)",
                    title=f"iteration {data['iteration']}: submit_blueprint",
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
            png_path = rundir.root / f"iter_{data['iteration']:02d}" / "render.png"
            console.print(f"[green]render:[/green] {png_path}  dims={dims_str}  blocks={stats['block_count']}")
            _display_image(data["image"], config.display)
        elif event_type == "inspect":
            console.print(f"[cyan]inspect view (yaw={data['yaw']}, cutaway={data['cutaway']})[/cyan]")
            _display_image(data["image"], config.display)
        elif event_type == "finish":
            console.print(Panel(data["summary"], title="finished", border_style="green"))
        elif event_type == "abort":
            console.print(Panel(data["reason"], title="aborted", border_style="red"))

    result = run_agent(prompt, llm, config, rundir, reference_image=reference_image, on_event=on_event)

    usage = llm.total_usage
    reasoning_line = f"  reasoning tokens: {usage.reasoning_tokens}" if usage.reasoning_tokens else ""
    console.print(
        Panel(
            f"iterations: {result.iterations}\n"
            f"prompt tokens: {usage.prompt_tokens}  completion tokens: {usage.completion_tokens}{reasoning_line}\n"
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
            (rundir.root / "final_blueprint.py").write_text(last_blueprint.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        console.print("[yellow]No successful build to export.[/yellow]")

    if not result.finished:
        console.print(f"[yellow]{result.summary}[/yellow]")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
