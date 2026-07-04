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

# Hard ceiling per side: a slice through a huge, densely-filled build could otherwise print
# tens of thousands of characters. Windowing to the slice's own occupied extent (rather than
# the whole build's bounds) already shrinks sparse slices; this caps the dense/large-footprint
# case too.
MAX_SLICE_DIM = 128


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

    _, full_r_lo, full_r_hi = row_axis
    _, full_c_lo, full_c_hi = col_axis

    def cell(rv: int, cv: int):
        coord = [0, 0, 0]
        coord[ai] = at
        coord[_AXES[row_axis[0]]] = rv
        coord[_AXES[col_axis[0]]] = cv
        return grid.get(coord[0], coord[1], coord[2])

    # One pass over the whole build's bounds on this plane to find: (a) the slice's own
    # occupied extent (usually far tighter than the whole build — a slice through one wing
    # shouldn't print the empty rest of the footprint), and (b) the materials present
    # (first-seen order for stable glyphs), so the final print pass only touches a small window.
    order: list[str] = []
    seen: set[str] = set()
    occ: tuple[int, int, int, int] | None = None
    for rv in range(full_r_lo, full_r_hi + 1):
        for cv in range(full_c_lo, full_c_hi + 1):
            idx = cell(rv, cv)
            if idx is None:
                continue
            if occ is None:
                occ = (rv, rv, cv, cv)
            else:
                occ_r_lo, occ_r_hi, occ_c_lo, occ_c_hi = occ
                occ = (min(occ_r_lo, rv), max(occ_r_hi, rv), min(occ_c_lo, cv), max(occ_c_hi, cv))
            name = get_block_by_index(idx).name
            if name not in seen:
                seen.add(name)
                order.append(name)

    if occ is None:
        return f"slice {axis}={at}: (empty — no blocks in this plane)"

    r_lo, r_hi, c_lo, c_hi = occ
    truncated = r_hi - r_lo + 1 > MAX_SLICE_DIM or c_hi - c_lo + 1 > MAX_SLICE_DIM
    r_hi = min(r_hi, r_lo + MAX_SLICE_DIM - 1)
    c_hi = min(c_hi, c_lo + MAX_SLICE_DIM - 1)

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
    if truncated:
        header += (
            f"\n(truncated to the {MAX_SLICE_DIM}x{MAX_SLICE_DIM} window shown; use a region-"
            "scoped histogram or a narrower slice for the rest)"
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
