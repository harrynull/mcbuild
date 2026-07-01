"""Real Minecraft block texture lookup + biome-tint approximation.

Textures ship at src/mcbuild/assets/textures/block/. Blocks without a matching
texture fall back to the flat palette color (handled by the sprite builder in
iso.py), so this module is purely additive.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image

TEXTURE_DIR = Path(__file__).resolve().parent.parent / "assets" / "textures" / "block"
TEXTURE_SIZE = 16

# Vanilla resource-pack textures for these blocks are grayscale tint masks,
# colorized per-biome at render time. We approximate the tint with this
# project's own curated palette RGB (palette.py) instead of a biome lookup.
TINTED_TOP_ONLY = {"grass_block"}
TINTED_ALL_FACES = {
    "oak_leaves",
    "spruce_leaves",
    "birch_leaves",
    "jungle_leaves",
    "acacia_leaves",
    "dark_oak_leaves",
    "mangrove_leaves",
    "azalea_leaves",
}


@lru_cache(maxsize=None)
def _load(path: Path) -> Image.Image | None:
    try:
        img = Image.open(path).convert("RGBA")
    except Exception:
        return None
    if img.size != (TEXTURE_SIZE, TEXTURE_SIZE):
        img = img.resize((TEXTURE_SIZE, TEXTURE_SIZE), Image.NEAREST)
    return img


def _candidates(name: str, face: str) -> list[str]:
    if face == "top":
        return [f"{name}_top", name]
    if face == "bottom":
        return [f"{name}_bottom", f"{name}_top", name]
    return [f"{name}_side", name]


@lru_cache(maxsize=None)
def get_face_texture(name: str, face: str) -> Image.Image | None:
    """face is 'top' or 'side'. Returns a 16x16 RGBA image, or None if untextured."""
    for candidate in _candidates(name, face):
        path = TEXTURE_DIR / f"{candidate}.png"
        if path.exists():
            img = _load(path)
            if img is not None:
                return img
    return None


def needs_tint(name: str, face: str) -> bool:
    if name in TINTED_ALL_FACES:
        return True
    return face == "top" and name in TINTED_TOP_ONLY


def apply_tint(img: Image.Image, rgb: tuple[int, int, int]) -> Image.Image:
    r, g, b = rgb
    tinted = Image.new("RGBA", img.size)
    src = img.load()
    dst = tinted.load()
    for y in range(img.height):
        for x in range(img.width):
            pr, pg, pb, pa = src[x, y]
            dst[x, y] = (pr * r // 255, pg * g // 255, pb * b // 255, pa)
    return tinted
