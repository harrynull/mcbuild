"""Shared orthographic z-buffer rasterizer for textured cuboid faces.

Used by both the free camera (render/camera.py) and the fixed-dimetric contact-sheet
renderer. A per-pixel depth test (not a per-block sort) is required because a free camera
can produce interpenetrating face projections that no single block ordering resolves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# Face shading tiers by orientation, matching the sprite renderer's look.
_SHADE = {"top": 1.0, "side": 0.72, "bottom": 0.5}


@dataclass
class DrawFace:
    corners: tuple  # 4 world (x, y, z)
    normal: tuple
    uvs: tuple  # 4 (u, v)
    tex: np.ndarray | None  # 16x16x4 uint8, or None
    color: tuple  # (r, g, b) fallback / tint base
    kind: str  # top | side | bottom


@dataclass
class View:
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    origin: np.ndarray
    scale: float
    width: int
    height: int


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def make_view(position, look_at, up, view_size: float, width: int, height: int) -> View:
    pos = np.array(position, dtype=np.float64)
    forward = _normalize(np.array(look_at, dtype=np.float64) - pos)
    up_hint = np.array(up, dtype=np.float64)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-6:  # looking straight up/down: pick a fallback hint
        up_hint = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_hint)
    right = _normalize(right)
    true_up = _normalize(np.cross(right, forward))
    scale = min(width, height) / max(view_size, 1e-6)
    return View(right, true_up, forward, pos, scale, width, height)


def _project(view: View, pts: np.ndarray):
    rel = pts - view.origin
    vx = rel @ view.right
    vy = rel @ view.up
    vz = rel @ view.forward  # depth along view axis (larger = farther)
    sx = view.width / 2.0 + vx * view.scale
    sy = view.height / 2.0 - vy * view.scale
    return sx, sy, vz


def rasterize(faces: list[DrawFace], view: View) -> Image.Image:
    w, h = view.width, view.height
    depth = np.full((h, w), np.inf, dtype=np.float64)
    color = np.zeros((h, w, 4), dtype=np.uint8)

    for face in faces:
        # backface cull: keep faces whose outward normal points toward the camera
        if np.dot(np.array(face.normal), view.forward) >= 0:
            continue
        pts = np.array(face.corners, dtype=np.float64)
        sx, sy, vz = _project(view, pts)
        shade = _SHADE.get(face.kind, 0.72)
        # quad -> two triangles (0,1,2) and (0,2,3)
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            _raster_tri(
                depth, color,
                (sx[a], sy[a], vz[a]), (sx[b], sy[b], vz[b]), (sx[c], sy[c], vz[c]),
                (face.uvs[a], face.uvs[b], face.uvs[c]),
                face.tex, face.color, shade,
            )

    return Image.fromarray(color, mode="RGBA")


def _raster_tri(depth, color, p0, p1, p2, uvs, tex, base_color, shade):
    h, w = depth.shape
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    minx = max(0, int(math.floor(min(x0, x1, x2))))
    maxx = min(w - 1, int(math.ceil(max(x0, x1, x2))))
    miny = max(0, int(math.floor(min(y0, y1, y2))))
    maxy = min(h - 1, int(math.ceil(max(y0, y1, y2))))
    if minx > maxx or miny > maxy:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return

    ys, xs = np.mgrid[miny : maxy + 1, minx : maxx + 1]
    xs = xs.astype(np.float64) + 0.5
    ys = ys.astype(np.float64) + 0.5

    w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
    w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
    if not inside.any():
        return

    tri_depth = w0 * z0 + w1 * z1 + w2 * z2
    sub = depth[miny : maxy + 1, minx : maxx + 1]
    win = inside & (tri_depth < sub)
    if not win.any():
        return

    (u0, v0), (u1, v1), (u2, v2) = uvs
    u = w0 * u0 + w1 * u1 + w2 * u2
    v = w0 * v0 + w1 * v1 + w2 * v2

    if tex is not None:
        tx = np.clip((u * 15).astype(np.int32), 0, 15)
        ty = np.clip((v * 15).astype(np.int32), 0, 15)
        sampled = tex[ty, tx]  # (...,4)
        rgb = sampled[..., :3].astype(np.float64)
        alpha = sampled[..., 3]
    else:
        rgb = np.empty(u.shape + (3,), dtype=np.float64)
        rgb[...] = base_color
        alpha = np.full(u.shape, 255, dtype=np.uint8)

    win = win & (alpha > 8)
    if not win.any():
        return

    rgb = np.clip(rgb * shade, 0, 255).astype(np.uint8)

    sub_color = color[miny : maxy + 1, minx : maxx + 1]
    sub_color[win, 0] = rgb[win, 0]
    sub_color[win, 1] = rgb[win, 1]
    sub_color[win, 2] = rgb[win, 2]
    sub_color[win, 3] = 255
    color[miny : maxy + 1, minx : maxx + 1] = sub_color

    sub[win] = tri_depth[win]
    depth[miny : maxy + 1, minx : maxx + 1] = sub
