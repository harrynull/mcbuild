"""Compose the 4 iso yaws + top-down + 2 cutaways into one contact sheet."""

from __future__ import annotations

from PIL import Image

from mcbuild.palette import get_block_by_index
from mcbuild.render.iso import render_iso, render_topdown
from mcbuild.voxel import VoxelGrid

MAX_WIDTH = 1100
THUMB = 360
BG = (24, 24, 28)


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
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2), resized)
    return canvas


def build_contact_sheet(grid: VoxelGrid) -> tuple[Image.Image, dict]:
    stats = build_stats(grid)

    tiles: list[tuple[str, Image.Image]] = []
    for yaw in range(4):
        tiles.append((f"iso {yaw * 90}deg", render_iso(grid, yaw=yaw)))
    tiles.append(("top-down", render_topdown(grid)))
    # clip keeps the FAR half and drops the near half; the newly-exposed cut face only faces
    # the camera when the camera sits on that same near side. yaw=0's azimuth is on the wrong
    # side for both axes, so it just showed a smaller, uncut-looking version of the "iso 0deg"
    # tile — yaw=2 (x) / yaw=1 (z) are actually on the near side and reveal the interior.
    tiles.append(("cutaway x", render_iso(grid, yaw=2, clip="x")))
    tiles.append(("cutaway z", render_iso(grid, yaw=1, clip="z")))

    cols, rows = 4, 2
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
        sheet = sheet.resize((MAX_WIDTH, int(sheet.height * ratio)), Image.LANCZOS)

    return sheet, stats
