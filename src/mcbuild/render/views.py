"""Compose the 4 iso yaws + top-down + 2 cutaways into one labeled contact sheet."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from mcbuild.palette import get_block_by_index
from mcbuild.render.iso import render_iso, render_topdown
from mcbuild.voxel import VoxelGrid

MAX_WIDTH = 1568
THUMB = 360
BG = (24, 24, 28)
FG = (230, 230, 230)


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
    return {"dims": dims, "block_count": len(grid), "top_materials": top_materials}


def _format_stats(stats: dict) -> str:
    dims = stats["dims"]
    dims_str = f"{dims[0]}x{dims[1]}x{dims[2]}" if dims else "empty"
    mats = ", ".join(f"{name} x{n}" for name, n in stats["top_materials"]) or "(none)"
    return f"dimensions: {dims_str}    blocks: {stats['block_count']:,}\ntop materials: {mats}"


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
    tiles.append(("cutaway x", render_iso(grid, yaw=0, clip="x")))
    tiles.append(("cutaway z", render_iso(grid, yaw=0, clip="z")))

    cols, rows = 4, 2
    cell_w, cell_h = THUMB + 20, THUMB + 40
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), BG)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for i, (label, tile) in enumerate(tiles):
        r, c = divmod(i, cols)
        fitted = _fit(tile, THUMB)
        x = c * cell_w + (cell_w - fitted.width) // 2
        y = r * cell_h + 10
        sheet.paste(fitted, (x, y), fitted)
        draw.text((c * cell_w + 10, y + fitted.height + 4), label, fill=FG, font=font)

    stats_text = _format_stats(stats)
    stats_h = 20 * (stats_text.count("\n") + 2)
    final = Image.new("RGB", (sheet.width, sheet.height + stats_h), (20, 20, 24))
    final.paste(sheet, (0, 0))
    draw2 = ImageDraw.Draw(final)
    draw2.text((10, sheet.height + 8), stats_text, fill=FG, font=font)

    if final.width > MAX_WIDTH:
        ratio = MAX_WIDTH / final.width
        final = final.resize((MAX_WIDTH, int(final.height * ratio)), Image.LANCZOS)

    return final, stats
