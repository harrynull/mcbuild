"""System/user prompt construction: role + DSL manual + workflow + critique nudge."""

from __future__ import annotations

from pathlib import Path

_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "dsl" / "REFERENCE.md"


def _reference_manual() -> str:
    return _REFERENCE_PATH.read_text(encoding="utf-8")


def build_system_prompt() -> str:
    return f"""You are mcbuild, an expert Minecraft build designer. You turn a natural-language \
prompt into a voxel structure by writing a blueprint program in a sandboxed Python DSL. An \
interpreter executes your blueprint into voxel data, and a renderer shows you multi-view \
screenshots of the result so you can critique and refine your own work.

{_reference_manual()}

## Workflow

1. Start with a short design brief (plain text, no tool call): target dimensions, footprint, \
   palette of block types, and 2-4 key features. Keep it to a few sentences.
2. Call `submit_blueprint` with a complete, self-contained blueprint that builds the whole \
   structure from (0, 0, 0) outward. Each call rebuilds from scratch — always submit the full \
   program, not a diff.
3. If it fails, the tool result gives you a line-mapped traceback with a code excerpt. Fix the \
   bug and resubmit.
4. If it succeeds, you'll receive build stats, then a follow-up message with a labeled \
   contact-sheet image (4 isometric angles, a top-down view, and 2 interior cutaways). Critique \
   your own render against the prompt using this checklist:
   - Does it match the prompt's description and scale?
   - Are the proportions and massing believable?
   - Do the materials/palette fit the theme?
   - Is the interior (visible in the cutaways) sensible, not just a hollow shell?
   - Are there missing details (windows, doors, trim, roofline) that would make it read as
     finished rather than a blockout?
5. Use `inspect` for a close look at a specific angle or cutaway before deciding your next move.
6. Revise with another `submit_blueprint` call, or call `finish` once you're satisfied.

Keep iterating until the build genuinely looks right in the renders — don't call `finish` on a \
rough blockout just because it executed without errors.
"""


def build_user_prompt(prompt: str, seed: int, has_reference: bool) -> str:
    ref_note = (
        "\nA concept-reference image is attached below — use it as inspiration for massing, "
        "proportions, and palette, but express it through the blueprint DSL in your own way.\n"
        if has_reference
        else ""
    )
    return f"""Build this: {prompt}

Place the structure near the origin. Keep total size reasonable (roughly under 60 blocks per \
axis unless the prompt clearly calls for something larger). Use seed {seed} for any randomness \
via `rng` so the build is reproducible.
{ref_note}
Begin with your design brief."""


def build_reference_image_prompt(building_prompt: str) -> str:
    return (
        f"isometric view of {building_prompt}, blocky Minecraft voxel style, "
        "entire building visible, plain background"
    )


def build_critique_nudge() -> str:
    return (
        "Here is the current render: 4 isometric angles, a top-down view, and 2 interior "
        "cutaways, with build stats below. Compare it against the prompt (and the concept "
        "reference image, if one was provided) and critique it: does it match the prompt and "
        "scale? Are proportions and materials right? Does the interior make sense? What details "
        "are missing? Then either submit a revised blueprint or call finish if it's genuinely done."
    )
