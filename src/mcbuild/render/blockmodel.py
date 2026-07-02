"""Block geometry: (name, state) -> list of textured cuboid faces in a 0..1 unit cell.

Full-cube blocks are a single box. Stateful architectural blocks (stairs, slabs, walls,
fences, fence gates, trapdoors, doors, panes) use hardcoded geometry templates keyed by the
model name from the bundled vanilla blockstate JSON (see blockstate.py), rotated by the
rotation that blockstate specifies. Textures are resolved from the block's base material via
the existing texture set (we don't ship the vanilla model JSONs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from mcbuild.render import blockstate, textures

# A box is (x0, y0, z0, x1, y1, z1) in 0..16 model space.
Box = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Face:
    corners: tuple[tuple[float, float, float], ...]  # 4 corners in 0..1 cell space
    normal: tuple[float, float, float]
    uvs: tuple[tuple[float, float], ...]  # per-corner UV in 0..1
    texture: str  # base texture name for textures.get_face_texture
    kind: str  # 'top' | 'bottom' | 'side' (texture + shading class, from final normal)
    tint: bool


# --- box -> faces (before rotation), in 0..16 space ---

# each face: (normal, 4 corners as (x,y,z), 4 uvs) using box min/max
def _box_faces(box: Box):
    x0, y0, z0, x1, y1, z1 = box
    faces = [
        # up (+y)
        ((0, 1, 0), [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)]),
        # down (-y)
        ((0, -1, 0), [(x0, y0, z1), (x0, y0, z0), (x1, y0, z0), (x1, y0, z1)]),
        # north (-z)
        ((0, 0, -1), [(x1, y1, z0), (x1, y0, z0), (x0, y0, z0), (x0, y1, z0)]),
        # south (+z)
        ((0, 0, 1), [(x0, y1, z1), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1)]),
        # west (-x)
        ((-1, 0, 0), [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)]),
        # east (+x)
        ((1, 0, 0), [(x1, y1, z1), (x1, y0, z1), (x1, y0, z0), (x1, y1, z0)]),
    ]
    uvs = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
    return [(n, corners, uvs) for n, corners in faces]


# --- rotation about the cell center (8,8,8), degrees ---

def _rot(px, py, pz, ax, ay):
    x, y, z = px - 8.0, py - 8.0, pz - 8.0
    if ax:
        a = math.radians(ax)
        c, s = math.cos(a), math.sin(a)
        y, z = y * c - z * s, y * s + z * c
    if ay:
        a = math.radians(ay)
        c, s = math.cos(a), math.sin(a)
        x, z = x * c + z * s, -x * s + z * c
    return x + 8.0, y + 8.0, z + 8.0


def _classify(normal) -> str:
    nx, ny, nz = normal
    if ny > 0.5:
        return "top"
    if ny < -0.5:
        return "bottom"
    return "side"


# --- shape templates: model-name predicate -> list[Box] (0..16) ---

def _template_boxes(model: str) -> list[Box] | None:
    m = model
    if m.endswith("_stairs_inner"):
        return [(0, 0, 0, 16, 8, 16), (8, 8, 0, 16, 16, 16), (0, 8, 8, 8, 16, 16)]
    if m.endswith("_stairs_outer"):
        return [(0, 0, 0, 16, 8, 16), (8, 8, 8, 16, 16, 16)]
    if m.endswith("_stairs"):
        return [(0, 0, 0, 16, 8, 16), (8, 8, 0, 16, 16, 16)]
    if m.endswith("_slab_top"):
        return [(0, 8, 0, 16, 16, 16)]
    if m.endswith("_slab"):
        return [(0, 0, 0, 16, 8, 16)]
    if m.endswith("_wall_post"):
        return [(4, 0, 4, 12, 16, 12)]
    if m.endswith("_wall_side_tall"):
        return [(5, 0, 0, 11, 16, 8)]
    if m.endswith("_wall_side"):
        return [(5, 0, 0, 11, 14, 8)]
    if m.endswith("_fence_gate"):
        return [(0, 5, 7, 2, 16, 9), (14, 5, 7, 16, 16, 9), (2, 6, 7, 14, 9, 9), (2, 12, 7, 14, 15, 9)]
    if m.endswith("_fence_post"):
        return [(6, 0, 6, 10, 16, 10)]
    if m.endswith("_fence_side"):
        return [(7, 6, 0, 9, 9, 9), (7, 12, 0, 9, 15, 9)]
    if m.endswith("_trapdoor_open"):
        return [(0, 0, 13, 16, 16, 16)]
    if m.endswith("_trapdoor_top"):
        return [(0, 13, 0, 16, 16, 16)]
    if m.endswith("_trapdoor_bottom") or m.endswith("_trapdoor"):
        return [(0, 0, 0, 16, 3, 16)]
    if "_door" in m:
        return [(0, 0, 0, 16, 16, 3)]
    if m.endswith("_bars") or m.endswith("_pane") or "_pane_" in m or "_bars_" in m:
        # thin cross (post + one arm); multipart parts add more arms via rotation
        return [(7, 0, 7, 9, 16, 9)]
    return None


# --- base texture resolution (no model JSON, so heuristic from the block name) ---

_SUFFIXES = (
    "_stairs", "_slab", "_wall", "_fence_gate", "_fence", "_trapdoor", "_door", "_bars", "_pane",
)


@lru_cache(maxsize=None)
def _base_texture(name: str) -> str | None:
    base = name
    for suf in _SUFFIXES:
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    for cand in (name, base, base + "s", base + "_planks"):
        if textures.get_face_texture(cand, "side") is not None or textures.get_face_texture(cand, "top") is not None:
            return cand
    return None


_FULL_CUBE: list[Box] = [(0, 0, 0, 16, 16, 16)]


def _faces_from_boxes(boxes, ax, ay, tex, tint) -> list[Face]:
    out: list[Face] = []
    for box in boxes:
        for normal, corners, uvs in _box_faces(box):
            rc = tuple(tuple(v / 16.0 for v in _rot(*c, ax, ay)) for c in corners)
            rn = _rot(normal[0] + 8, normal[1] + 8, normal[2] + 8, ax, ay)
            rn = (rn[0] - 8.0, rn[1] - 8.0, rn[2] - 8.0)
            out.append(
                Face(
                    corners=rc,
                    normal=rn,
                    uvs=tuple(uvs),
                    texture=tex,
                    kind=_classify(rn),
                    tint=tint,
                )
            )
    return out


@lru_cache(maxsize=4096)
def get_block_mesh(name: str, state_items: tuple = ()) -> list[Face] | None:
    """Return the block's faces in a 0..1 cell, or None if it should not render.

    `state_items` is a sorted tuple of (prop, value) pairs (hashable for caching).
    """
    tex = _base_texture(name)
    if tex is None:
        return None
    tint = textures.needs_tint(name, "side") or textures.needs_tint(name, "top")
    state = dict(state_items)

    parts = blockstate.resolve_parts(name, state)
    if not parts:
        # no blockstate / no match → full cube
        return _faces_from_boxes(_FULL_CUBE, 0, 0, tex, tint)

    faces: list[Face] = []
    matched_template = False
    for part in parts:
        boxes = _template_boxes(part.model)
        if boxes is None:
            boxes = _FULL_CUBE
        else:
            matched_template = True
        faces.extend(_faces_from_boxes(boxes, part.x, part.y, tex, tint))

    if not matched_template and len(parts) == 1:
        # a plain full-block model (e.g. oak_planks) — one clean cube
        return _faces_from_boxes(_FULL_CUBE, 0, 0, tex, tint)
    return faces
