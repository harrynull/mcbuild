"""Lossless text views of a build: ASCII slice plans, point queries, material histograms.

LLMs read character grids far more reliably than tiny per-block renders, so these give
the agent ground truth to verify interiors and layouts against.
"""

from __future__ import annotations

from mcbuild.palette import get_block_by_index
from mcbuild.voxel import VoxelGrid

_AXES = {"x": 0, "y": 1, "z": 2}
# Distinct glyphs for the legend, assigned in first-seen order.
_GLYPHS = "#@O%X=+*softceruvwxyz0123456789ABDEFGHIJKLMNPQRSTUVWZ"


def _assign_glyphs(names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for i, name in enumerate(names):
        mapping[name] = _GLYPHS[i] if i < len(_GLYPHS) else "?"
    return mapping


def ascii_slice(grid: VoxelGrid, axis: str, at: int) -> str:
    """A one-char-per-block plan of the plane `axis == at` (world coords), plus a legend.

    axis 'y' → a horizontal floor plan (rows=z, cols=x); 'x'/'z' → a vertical section.
    '.' marks empty/air; each material gets a stable glyph explained in the legend.
    """
    if axis not in _AXES:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    bounds = grid.bounds
    if bounds is None:
        return "(empty build)"
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    ai = _AXES[axis]

    # Which two axes span the plane, and their world ranges.
    if axis == "y":
        row_axis, col_axis = ("z", minz, maxz), ("x", minx, maxx)
    elif axis == "z":
        row_axis, col_axis = ("y", miny, maxy), ("x", minx, maxx)
    else:  # axis == "x"
        row_axis, col_axis = ("y", miny, maxy), ("z", minz, maxz)

    _, r_lo, r_hi = row_axis
    _, c_lo, c_hi = col_axis

    def cell(rv: int, cv: int):
        coord = [0, 0, 0]
        coord[ai] = at
        coord[_AXES[row_axis[0]]] = rv
        coord[_AXES[col_axis[0]]] = cv
        return grid.get(coord[0], coord[1], coord[2])

    # Collect materials present in this slice (first-seen order for stable glyphs).
    order: list[str] = []
    seen: set[str] = set()
    for rv in range(r_lo, r_hi + 1):
        for cv in range(c_lo, c_hi + 1):
            idx = cell(rv, cv)
            if idx is None:
                continue
            name = get_block_by_index(idx).name
            if name not in seen:
                seen.add(name)
                order.append(name)
    glyphs = _assign_glyphs(order)

    rows_out = []
    # Render top-of-image = high row value for y-up-ish readability (z increases downward).
    for rv in range(r_hi, r_lo - 1, -1):
        line = []
        for cv in range(c_lo, c_hi + 1):
            idx = cell(rv, cv)
            line.append("." if idx is None else glyphs[get_block_by_index(idx).name])
        rows_out.append("".join(line))

    legend = ", ".join(f"{glyphs[n]}={n}" for n in order) or "(no blocks in this slice)"
    header = (
        f"slice {axis}={at}  "
        f"{row_axis[0]}:{r_lo}..{r_hi} (top→bottom)  {col_axis[0]}:{c_lo}..{c_hi} (left→right)"
    )
    return header + "\n" + "\n".join(rows_out) + "\n\nlegend: " + legend


def point_query(grid: VoxelGrid, x: int, y: int, z: int) -> str:
    idx = grid.get(int(x), int(y), int(z))
    if idx is None:
        return f"({x}, {y}, {z}): air / empty"
    return f"({x}, {y}, {z}): {get_block_by_index(idx).mc_id}"


def material_histogram(grid: VoxelGrid, region=None) -> str:
    """Counts of each material, optionally within [x1,y1,z1,x2,y2,z2]."""
    if region is not None:
        x1, y1, z1, x2, y2, z2 = (int(round(v)) for v in region)
        xlo, xhi = sorted((x1, x2))
        ylo, yhi = sorted((y1, y2))
        zlo, zhi = sorted((z1, z2))

        def inside(c):
            x, y, z = c
            return xlo <= x <= xhi and ylo <= y <= yhi and zlo <= z <= zhi
    else:
        def inside(c):
            return True

    counts: dict[str, int] = {}
    for coord, idx in grid.items():
        if not inside(coord):
            continue
        name = get_block_by_index(idx).name
        counts[name] = counts.get(name, 0) + 1

    if not counts:
        return "(no blocks in region)"
    lines = [f"{n}: {c}" for n, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    total = sum(counts.values())
    scope = f" in region {list(region)}" if region is not None else ""
    return f"material histogram{scope} (total {total}):\n" + "\n".join(lines)
