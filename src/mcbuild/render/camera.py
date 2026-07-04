"""Free 3D camera: arbitrary position + look-at, orthographic, over the mesh rasterizer."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from mcbuild.palette import get_block_by_index
from mcbuild.render import blockmodel, culling, raster, textures
from mcbuild.voxel import VoxelGrid

Vec3 = tuple[float, float, float]

MAX_EXPOSED_BLOCKS = 20_000


class CameraRenderError(Exception):
    """Raised when a build is too large to rasterize within the safety cap."""


@dataclass
class Camera:
    position: Vec3
    look_at: Vec3
    up: Vec3 = (0.0, 1.0, 0.0)
    view_size: float = 24.0  # world units across the shorter image dimension (smaller = zoomed in)


@lru_cache(maxsize=8192)
def _texture_array(texture: str, kind: str, tint: bool, rgb: tuple) -> np.ndarray | None:
    face = "top" if kind == "top" else ("bottom" if kind == "bottom" else "side")
    img = textures.get_face_texture(texture, face)
    if img is None:
        return None
    if tint:
        img = textures.apply_tint(img, rgb)
    return np.array(img.convert("RGBA"), dtype=np.uint8)


def _build_draw_faces(
    grid: VoxelGrid, max_blocks: int, keep: Callable[[tuple], bool] | None = None
) -> list[raster.DrawFace]:
    occupied = {coord for coord, _ in grid.items()}
    # Cull against the KEPT set, not the full grid: a block cut away by `keep` must count
    # as empty here, or its surviving neighbor across the cut plane never becomes "exposed"
    # and its face is skipped entirely — showing whatever's behind it instead of the cut face.
    kept = occupied if keep is None else {c for c in occupied if keep(c)}
    exposed = culling.exposed_coords(kept)
    if len(exposed) > max_blocks:
        raise CameraRenderError(
            f"build has {len(exposed):,} exposed blocks (cap {max_blocks:,}); inspect a smaller region."
        )

    draw: list[raster.DrawFace] = []
    for cx, cy, cz in exposed:
        idx = grid.get(cx, cy, cz)
        if idx is None:
            continue
        block = get_block_by_index(idx)
        mesh = blockmodel.get_block_mesh(block.name, block.state)
        if not mesh:
            continue
        for f in mesh:
            corners = tuple((cx + c[0], cy + c[1], cz + c[2]) for c in f.corners)
            tex = _texture_array(f.texture, f.kind, f.tint, block.rgb)
            draw.append(
                raster.DrawFace(
                    corners=corners,
                    normal=f.normal,
                    uvs=f.uvs,
                    tex=tex,
                    color=block.rgb,
                    kind=f.kind,
                )
            )
    return draw


def render_from_camera(
    grid: VoxelGrid,
    camera: Camera,
    width: int = 480,
    height: int = 480,
    max_blocks: int = MAX_EXPOSED_BLOCKS,
    keep: Callable[[tuple], bool] | None = None,
    shadows: bool = True,
) -> Image.Image:
    faces = _build_draw_faces(grid, max_blocks, keep=keep)
    if not faces:
        return Image.new("RGBA", (width, height), (0, 0, 0, 0))
    view = raster.make_view(camera.position, camera.look_at, camera.up, camera.view_size, width, height)
    img = raster.rasterize(faces, view, shadows=shadows)
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img


# True-ish isometric elevation; azimuth rotates 90° per yaw step.
_ISO_ELEV_DEG = 35.264


def render_isometric(
    grid: VoxelGrid,
    yaw: int = 0,
    keep: Callable[[tuple], bool] | None = None,
    width: int = 900,
    height: int = 900,
    max_blocks: int = MAX_EXPOSED_BLOCKS,
    shadows: bool = True,
) -> Image.Image:
    """Dimetric render from a fixed isometric camera (4 azimuths via yaw), over the mesh path."""
    bounds = grid.bounds
    if bounds is None:
        return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    center = ((minx + maxx + 1) / 2.0, (miny + maxy + 1) / 2.0, (minz + maxz + 1) / 2.0)
    extent = max(maxx - minx + 1, maxy - miny + 1, maxz - minz + 1, 1)

    az = math.radians(45.0 + 90.0 * (yaw % 4))
    el = math.radians(_ISO_ELEV_DEG)
    dist = extent * 4 + 20  # orthographic: only direction matters, keep all geometry in front
    offset = (
        math.cos(el) * math.sin(az) * dist,
        math.sin(el) * dist,
        math.cos(el) * math.cos(az) * dist,
    )
    position = (center[0] + offset[0], center[1] + offset[1], center[2] + offset[2])
    view_size = extent * 2.1  # fit the build with a little margin
    cam = Camera(position=position, look_at=center, view_size=view_size)

    faces = _build_draw_faces(grid, max_blocks, keep=keep)
    if not faces:
        return Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    view = raster.make_view(cam.position, cam.look_at, cam.up, cam.view_size, width, height)
    img = raster.rasterize(faces, view, shadows=shadows)
    bbox = img.getbbox()
    return img.crop(bbox) if bbox else img
