"""Compose model-requested renderings into one contact sheet."""

from __future__ import annotations

from PIL import Image

from mcbuild.palette import get_block_by_index
from mcbuild.render.iso import render_iso, render_topdown
from mcbuild.voxel import VoxelGrid

MAX_WIDTH = 1100
THUMB = 360
BG = (24, 24, 28)
MAX_COLS = 4


def build_stats(grid: VoxelGrid) -> dict:
    bounds = grid.bounds
    counts: dict[int, int] = {}
    for _, idx in grid.items():
        counts[idx] = counts.get(idx, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
    top_materials = [(get_block_by_index(idx).name, n) for idx, n in top]
    dims = None
    if bounds is not None:
        (minx, miny, minz), (maxx, maxy, maxz) = bounds
        dims = (maxx - minx + 1, maxy - miny + 1, maxz - minz + 1)
    return {
        "dims": dims,
        "bounds": bounds,  # ((minx,miny,minz),(maxx,maxy,maxz)) or None
        "block_count": len(grid),
        "top_materials": top_materials,
    }


def _fit(img: Image.Image, size: int) -> Image.Image:
    if img.width == 0 or img.height == 0:
        img = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    ratio = min(size / img.width, size / img.height)
    new_w = max(1, round(img.width * ratio))
    new_h = max(1, round(img.height * ratio))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2), resized)
    return canvas


def render_view(grid: VoxelGrid, spec: dict) -> tuple[str, Image.Image]:
    """Render one contact-sheet tile from a model-requested view spec.

    Spec fields: mode ("iso", default, or "top-down"), yaw (0-3, iso only), cutaway
    ("none"/"x"/"z"), or slice_axis+slice_at for an arbitrary plane (overrides cutaway).
    Specs come pre-validated by the agent loop (_normalized_view_spec); the yaw wrap here
    is only a last-resort guard so the label always matches what was rendered.
    """
    mode = spec.get("mode", "iso")
    if mode == "top-down":
        return "top-down", render_topdown(grid)

    yaw = int(spec.get("yaw", 0) or 0) % 4
    slice_axis = spec.get("slice_axis")
    slice_at = spec.get("slice_at")
    if slice_axis is not None and slice_at is not None:
        img = render_iso(grid, yaw=yaw, slice_spec=(slice_axis, int(slice_at)))
        return f"yaw {yaw * 90}deg, slice {slice_axis}={int(slice_at)}", img

    cutaway = spec.get("cutaway", "none")
    clip = None if cutaway in (None, "none") else cutaway
    img = render_iso(grid, yaw=yaw, clip=clip)
    label = f"yaw {yaw * 90}deg" + (f", cutaway {cutaway}" if clip else "")
    return label, img


def build_contact_sheet(grid: VoxelGrid, view_specs: list[dict]) -> tuple[Image.Image, list[str], dict]:
    """Render exactly the views the model requested (at least one) into one contact sheet."""
    stats = build_stats(grid)

    tiles = [render_view(grid, spec) for spec in view_specs]
    labels = [label for label, _ in tiles]

    cols = max(1, min(MAX_COLS, len(tiles)))
    rows = -(-len(tiles) // cols)  # ceil division
    cell_w, cell_h = THUMB + 20, THUMB + 20
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), BG)

    for i, (_label, tile) in enumerate(tiles):
        r, c = divmod(i, cols)
        fitted = _fit(tile, THUMB)
        x = c * cell_w + (cell_w - fitted.width) // 2
        y = r * cell_h + 10
        sheet.paste(fitted, (x, y), fitted)

    if sheet.width > MAX_WIDTH:
        ratio = MAX_WIDTH / sheet.width
        sheet = sheet.resize((MAX_WIDTH, int(sheet.height * ratio)), Image.Resampling.LANCZOS)

    return sheet, labels, stats
