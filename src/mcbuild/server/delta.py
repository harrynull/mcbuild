"""Structured per-iteration voxel diffing, for streaming block placements to a live client."""

from __future__ import annotations

from mcbuild import palette
from mcbuild.voxel import VoxelGrid


def compute_block_delta(before: VoxelGrid | None, after: VoxelGrid) -> list[dict]:
    """Diff two grids into a list of {"x", "y", "z", "block"} changes.

    `block` is the full mc_id string (e.g. "minecraft:oak_stairs[facing=north]") for an
    added/changed cell, or None for a removed cell (the caller places air for these).
    Cells unchanged between `before` and `after` are omitted. `before=None` is treated as
    an empty grid, so every placed cell in `after` is reported as added.
    """
    b = dict(before.items()) if before is not None else {}
    a = dict(after.items())
    changes: list[dict] = []
    for coord in set(a) | set(b):
        bv, av = b.get(coord), a.get(coord)
        if bv == av:
            continue
        x, y, z = coord
        block = palette.get_block_by_index(av).mc_id if av is not None else None
        changes.append({"x": x, "y": y, "z": z, "block": block})
    return changes
