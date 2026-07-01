"""Pure-software isometric voxel renderer: cached block sprites + painter's algorithm."""

from __future__ import annotations

from functools import lru_cache

from PIL import Image, ImageChops, ImageDraw

from mcbuild.palette import get_block_by_index
from mcbuild.render import textures
from mcbuild.voxel import VoxelGrid

TW = 16  # top-face diamond width
TH = 8  # top-face diamond height
WALL = 16  # side-face height per y-unit
PAD = 16

_tw2, _th2 = TW // 2, TH // 2

# face -> (polygon vertices in sprite-local coords, origin, vecU, vecV, shade factor)
# Each face is the affine (parallelogram) image of the unit square [0,1]x[0,1],
# i.e. dest = origin + u*vecU + v*vecV, which lets a square texture be warped
# straight onto it as a shear transform (see _affine_coeffs).
_FACES: dict[str, tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], tuple[int, int], float]] = {
    "top": (
        [(_tw2, 0), (TW, _th2), (_tw2, TH), (0, _th2)],
        (_tw2, 0),
        (_tw2, _th2),
        (-_tw2, _th2),
        1.0,
    ),
    "left": (
        [(0, _th2), (_tw2, TH), (_tw2, TH + WALL), (0, _th2 + WALL)],
        (0, _th2),
        (_tw2, _th2),
        (0, WALL),
        0.8,
    ),
    "right": (
        [(_tw2, TH), (TW, _th2), (TW, _th2 + WALL), (_tw2, TH + WALL)],
        (_tw2, TH),
        (_tw2, -_th2),
        (0, WALL),
        0.6,
    ),
}


def _shade(rgb: tuple[int, int, int], factor: float) -> tuple[int, int, int]:
    return tuple(min(255, max(0, int(c * factor))) for c in rgb)  # type: ignore[return-value]


def _shade_image(img: Image.Image, factor: float) -> Image.Image:
    r, g, b, a = img.split()
    lut = [min(255, int(v * factor)) for v in range(256)]
    return Image.merge("RGBA", (r.point(lut), g.point(lut), b.point(lut), a))


def _affine_coeffs(
    origin: tuple[float, float], vec_u: tuple[float, float], vec_v: tuple[float, float], tex_w: int, tex_h: int
) -> tuple[float, float, float, float, float, float]:
    """PIL AFFINE coeffs mapping dest pixel -> source texture pixel for a parallelogram face."""
    ox, oy = origin
    ux, uy = vec_u
    vx, vy = vec_v
    det = ux * vy - vx * uy
    a = tex_w * vy / det
    b = -tex_w * vx / det
    c = -a * ox - b * oy
    d = -tex_h * uy / det
    e = tex_h * ux / det
    f = -d * ox - e * oy
    return (a, b, c, d, e, f)


def _draw_textured_face(
    canvas: Image.Image,
    poly: list[tuple[int, int]],
    origin: tuple[int, int],
    vec_u: tuple[int, int],
    vec_v: tuple[int, int],
    shade: float,
    tex: Image.Image,
    alpha: int,
) -> None:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
    bw, bh = bx1 - bx0, by1 - by0
    local_origin = (origin[0] - bx0, origin[1] - by0)
    coeffs = _affine_coeffs(local_origin, vec_u, vec_v, tex.width, tex.height)
    warped = tex.transform((bw, bh), Image.AFFINE, coeffs, resample=Image.NEAREST)
    warped = _shade_image(warped, shade)

    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).polygon([(x - bx0, y - by0) for x, y in poly], fill=255)
    r, g, b, a = warped.split()
    combined_alpha = ImageChops.multiply(mask, a)
    if alpha < 255:
        combined_alpha = combined_alpha.point(lambda v: v * alpha // 255)
    warped_final = Image.merge("RGBA", (r, g, b, combined_alpha))
    canvas.alpha_composite(warped_final, (bx0, by0))


@lru_cache(maxsize=None)
def _sprite(block_name: str, rgb: tuple[int, int, int], alpha: int) -> Image.Image:
    img = Image.new("RGBA", (TW, TH + WALL), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for face_key, (poly, origin, vec_u, vec_v, shade) in _FACES.items():
        face_type = "top" if face_key == "top" else "side"
        tex = textures.get_face_texture(block_name, face_type)
        if tex is not None:
            if textures.needs_tint(block_name, face_type):
                tex = textures.apply_tint(tex, rgb)
            _draw_textured_face(img, poly, origin, vec_u, vec_v, shade, tex, alpha)
        else:
            draw.polygon(poly, fill=_shade(rgb, shade) + (alpha,))
    return img


def _rot90_xz(x: int, z: int, X: int, Z: int) -> tuple[int, int, int, int]:
    return Z - 1 - z, x, Z, X


def _normalized_blocks(
    grid: VoxelGrid,
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int, int]]:
    bounds = grid.bounds
    if bounds is None:
        return [], (0, 0, 0)
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    dims = (maxx - minx + 1, maxy - miny + 1, maxz - minz + 1)
    # Blocks with no known appearance yet are treated as air in the preview render
    # (still placed/exported normally — see Block.renderable).
    blocks = [
        (x - minx, y - miny, z - minz, idx)
        for (x, y, z), idx in grid.items()
        if get_block_by_index(idx).renderable
    ]
    return blocks, dims


def render_iso(grid: VoxelGrid, yaw: int = 0, clip: str | None = None) -> Image.Image:
    """Render an isometric view of the grid.

    yaw: 0-3, each step a 90-degree turn around the vertical axis.
    clip: None, 'x', or 'z' — drop the near half along that axis for a cutaway view.
    """
    blocks, (X, Y, Z) = _normalized_blocks(grid)
    if not blocks:
        return Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    if clip == "x":
        blocks = [b for b in blocks if b[0] >= X // 2]
    elif clip == "z":
        blocks = [b for b in blocks if b[2] >= Z // 2]

    rotated = []
    for x, y, z, idx in blocks:
        rx, rz, RX, RZ = x, z, X, Z
        for _ in range(yaw % 4):
            rx, rz, RX, RZ = _rot90_xz(rx, rz, RX, RZ)
        rotated.append((rx, y, rz, idx))
    RX, RZ = X, Z
    for _ in range(yaw % 4):
        RX, RZ = RZ, RX

    if not rotated:
        return Image.new("RGBA", (64, 64), (0, 0, 0, 0))

    occupied = {(x, y, z) for x, y, z, _ in rotated}
    visible = []
    for x, y, z, idx in rotated:
        neighbors = (
            (x + 1, y, z),
            (x - 1, y, z),
            (x, y + 1, z),
            (x, y - 1, z),
            (x, y, z + 1),
            (x, y, z - 1),
        )
        if all(n in occupied for n in neighbors):
            continue
        visible.append((x, y, z, idx))

    tw2, th2 = TW // 2, TH // 2
    min_sx = -(RZ - 1) * tw2
    max_sx = (RX - 1) * tw2
    min_sy = -(Y - 1) * WALL
    max_sy = (RX - 1 + RZ - 1) * th2
    ox = -min_sx + PAD
    oy = -min_sy + PAD
    canvas_w = max_sx - min_sx + TW + 2 * PAD
    canvas_h = max_sy - min_sy + TH + WALL + 2 * PAD

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    visible.sort(key=lambda b: (b[1], b[0] + b[2], b[0]))

    for x, y, z, idx in visible:
        block = get_block_by_index(idx)
        alpha = 170 if block.transparent else 255
        sprite = _sprite(block.name, block.rgb, alpha)
        sx = (x - z) * tw2 + ox
        sy = (x + z) * th2 - y * WALL + oy
        px, py = sx - tw2, sy - th2
        canvas.alpha_composite(sprite, (px, py))

    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    return canvas


def render_topdown(grid: VoxelGrid, cell: int = 6) -> Image.Image:
    """Flat heightmap view: topmost block per (x, z) column, shaded by height."""
    bounds = grid.bounds
    if bounds is None:
        return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    X = maxx - minx + 1
    Z = maxz - minz + 1
    Y = maxy - miny + 1

    top_y: dict[tuple[int, int], int] = {}
    top_idx: dict[tuple[int, int], int] = {}
    for (x, y, z), idx in grid.items():
        if not get_block_by_index(idx).renderable:
            continue  # treated as air until it has a known appearance
        key = (x - minx, z - minz)
        if key not in top_y or y > top_y[key]:
            top_y[key] = y
            top_idx[key] = idx

    img = Image.new("RGBA", (X * cell, Z * cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for (nx, nz), idx in top_idx.items():
        block = get_block_by_index(idx)
        y = top_y[(nx, nz)]
        shade = 0.6 + 0.4 * ((y - miny) / max(1, Y - 1))
        rgb = _shade(block.rgb, shade)
        alpha = 170 if block.transparent else 255
        draw.rectangle(
            [nx * cell, nz * cell, nx * cell + cell - 1, nz * cell + cell - 1],
            fill=rgb + (alpha,),
        )
    return img
