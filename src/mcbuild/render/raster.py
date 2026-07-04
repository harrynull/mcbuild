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

# Directional light for shadow mapping: the direction light travels (from sun to scene),
# roughly overhead and slightly from +x/+z so shadows read well in the iso views.
_LIGHT_DIR = (-0.45, -1.0, -0.30)
_SHADOW_FACTOR = 0.62  # brightness multiplier for shadowed pixels
_SHADOW_BIAS = 0.6  # world-unit depth bias to avoid self-shadow acne on blocky geometry
_SHADOW_RES = 1024  # light-space depth map resolution (square)


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


def _light_view(faces: list[DrawFace], res: int) -> View:
    """An orthographic view down the light direction, framing the whole scene."""
    corners = np.array([c for f in faces for c in f.corners], dtype=np.float64)
    lo = corners.min(axis=0)
    hi = corners.max(axis=0)
    center = (lo + hi) / 2.0
    extent = float(max(hi - lo)) + 1.0
    forward = _normalize(np.array(_LIGHT_DIR, dtype=np.float64))
    origin = center - forward * (extent * 3.0 + 20.0)
    view_size = extent * 1.25
    return make_view(tuple(origin), tuple(origin + forward), (0.0, 1.0, 0.0), view_size, res, res)


def _render_light_depth(faces: list[DrawFace], light: View) -> np.ndarray:
    """Depth buffer as seen from the light, for shadow testing."""
    depth = np.full((light.height, light.width), np.inf, dtype=np.float64)
    for face in faces:
        if np.dot(np.array(face.normal), light.forward) >= 0:
            continue  # faces turned away from the light don't define its front depth
        pts = np.array(face.corners, dtype=np.float64)
        lx, ly, lz = _project(light, pts)
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            _depth_tri(depth, (lx[a], ly[a], lz[a]), (lx[b], ly[b], lz[b]), (lx[c], ly[c], lz[c]))
    return depth


def rasterize(faces: list[DrawFace], view: View, shadows: bool = True) -> Image.Image:
    w, h = view.width, view.height
    depth = np.full((h, w), np.inf, dtype=np.float64)
    color = np.zeros((h, w, 4), dtype=np.uint8)

    light = light_depth = None
    if shadows and faces:
        light = _light_view(faces, _SHADOW_RES)
        light_depth = _render_light_depth(faces, light)

    for face in faces:
        # backface cull: keep faces whose outward normal points toward the camera
        if np.dot(np.array(face.normal), view.forward) >= 0:
            continue
        pts = np.array(face.corners, dtype=np.float64)
        sx, sy, vz = _project(view, pts)
        if light is not None:
            lx, ly, lz = _project(light, pts)
        shade = _SHADE.get(face.kind, 0.72)
        # quad -> two triangles (0,1,2) and (0,2,3)
        for a, b, c in ((0, 1, 2), (0, 2, 3)):
            light_tri = None
            if light is not None:
                light_tri = ((lx[a], ly[a], lz[a]), (lx[b], ly[b], lz[b]), (lx[c], ly[c], lz[c]))
            _raster_tri(
                depth,
                color,
                (sx[a], sy[a], vz[a]),
                (sx[b], sy[b], vz[b]),
                (sx[c], sy[c], vz[c]),
                (face.uvs[a], face.uvs[b], face.uvs[c]),
                face.tex,
                face.color,
                shade,
                light_tri,
                light_depth,
            )

    return Image.fromarray(color, mode="RGBA")


def _depth_tri(depth, p0, p1, p2):
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
    d = w0 * z0 + w1 * z1 + w2 * z2
    sub = depth[miny : maxy + 1, minx : maxx + 1]
    win = inside & (d < sub)
    sub[win] = d[win]
    depth[miny : maxy + 1, minx : maxx + 1] = sub


def _raster_tri(depth, color, p0, p1, p2, uvs, tex, base_color, shade, light_tri=None, light_depth=None):
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

    # per-pixel shade, darkened where the light is occluded (shadow map)
    shade_px = np.full(u.shape, shade, dtype=np.float64)
    if light_tri is not None and light_depth is not None:
        (lx0, ly0, ld0), (lx1, ly1, ld1), (lx2, ly2, ld2) = light_tri
        lsx = w0 * lx0 + w1 * lx1 + w2 * lx2
        lsy = w0 * ly0 + w1 * ly1 + w2 * ly2
        ldepth = w0 * ld0 + w1 * ld1 + w2 * ld2
        lh, lw = light_depth.shape
        ix = np.clip(lsx.astype(np.int32), 0, lw - 1)
        iy = np.clip(lsy.astype(np.int32), 0, lh - 1)
        in_map = (lsx >= 0) & (lsx < lw) & (lsy >= 0) & (lsy < lh)
        stored = light_depth[iy, ix]
        shadowed = in_map & np.isfinite(stored) & (ldepth > stored + _SHADOW_BIAS)
        shade_px = np.where(shadowed, shade * _SHADOW_FACTOR, shade)

    rgb = np.clip(rgb * shade_px[..., None], 0, 255).astype(np.uint8)

    sub_color = color[miny : maxy + 1, minx : maxx + 1]
    sub_color[win, 0] = rgb[win, 0]
    sub_color[win, 1] = rgb[win, 1]
    sub_color[win, 2] = rgb[win, 2]
    sub_color[win, 3] = 255
    color[miny : maxy + 1, minx : maxx + 1] = sub_color

    sub[win] = tri_depth[win]
    depth[miny : maxy + 1, minx : maxx + 1] = sub
