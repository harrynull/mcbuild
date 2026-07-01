# mcbuild

A from-scratch LLM agent that turns a natural-language prompt into a Minecraft
building. It writes a blueprint program in a sandboxed Python DSL, interprets
it into voxel data, renders labeled multi-view screenshots with a pure-software
isometric renderer, and lets a vision-capable LLM (via [OpenRouter](https://openrouter.ai))
critique its own renders and iterate via tool calls. Final output is a
WorldEdit-compatible `.schem` file plus a run directory with every artifact.

## Install

```bash
uv sync --extra dev
```

Set your OpenRouter key:

```bash
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY=...
```

## Usage

```bash
uv run mcbuild "a medieval watchtower with interior spiral stairs"
```

Options:

```
mcbuild PROMPT
  --model TEXT           Vision-capable OpenRouter model id [default: anthropic/claude-sonnet-5]
  --max-iters INTEGER     [default: 6]
  --seed INTEGER          [default: 0]
  --display TEXT          auto|sixel|ansi|off  [default: auto]
  --out TEXT              Run directory base  [default: runs]
  --reference/--no-reference   Generate a concept-reference image first  [default: no-reference]
  --ref-model TEXT        [default: openai/gpt-image-2]
  --reasoning TEXT        off|low|medium|high  [default: medium]
```

`--display auto` probes the terminal for sixel support (via a DA1 query) and
falls back to ANSI half-block rendering (`rich-pixels`) if unavailable.

Each run writes to `runs/<timestamp>-<slug>/`:

```
prompt.txt
reference.png          (if --reference)
iter_NN/blueprint.py
iter_NN/render.png
iter_NN/stats.json
final.schem
final_blueprint.py
session.json            (full message log, for debugging)
```

## Offline demo (no API key / no network)

```bash
uv run mcbuild "a tiny stone hut" --fake-llm --max-iters 3
```

Runs the full pipeline against a scripted stand-in LLM (broken blueprint ->
line-mapped error -> fixed blueprint -> render -> finish) so you can see the
whole loop and artifact layout without hitting the network.

## The blueprint DSL

Blueprints are sandboxed Python — no imports, no `_`-prefixed attribute access,
no dangerous builtins, and a line-count + wall-clock execution budget. See
[`src/mcbuild/dsl/REFERENCE.md`](src/mcbuild/dsl/REFERENCE.md) for the full
primitive/transform reference and worked examples.

## Development

```bash
uv run pytest
```

Tests cover: sandbox security (imports/dunders/budget), DSL shape primitives,
palette lookup + suggestions, isometric renderer output, sixel encoding,
Sponge Schematic v2 round-trip via `nbtlib`, and an offline agent-loop
integration test (scripted LLM: error -> fix -> finish) via the CLI's
`--fake-llm` path.

## Project layout

```
src/mcbuild/
  cli.py            typer CLI, rich progress feed, sixel/ANSI display
  config.py         run configuration
  voxel.py          sparse VoxelGrid
  palette.py        curated block palette + fuzzy suggestions
  rundir.py         runs/<timestamp>-<slug>/ artifact management
  dsl/               sandbox, stdlib primitives, errors, REFERENCE.md
  render/            isometric renderer, contact-sheet views, sixel encoder
  llm/               OpenRouter client, scripted offline FakeLLM
  agent/             orchestration loop, prompts, tool schemas
  export/            Sponge Schematic v2 (.schem) export
```

## Notes / v1 limitations

- Full-cube blocks only (no stairs/slabs/orientation states) — noted as a v2
  stretch goal.
- No cross-run memory / few-shot retrieval of past builds.
- No RCON live placement; output is a `.schem` file for WorldEdit's `//schem load` + `//paste`.
