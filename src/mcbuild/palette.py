"""Block palette: names validated against the authoritative Minecraft block registry.

Colors come from a curated hand-picked table where available, otherwise are
derived lazily from the real block texture (averaging its opaque pixels), and
finally fall back to a neutral gray if neither exists.

v1's voxel model and renderer only understand full 1x1x1 cubes, so blocks with
sub-block shapes (stairs, slabs, doors, ...) are still *valid names* here (the
registry doesn't distinguish shape), but will be placed/rendered as a plain
cube rather than their true geometry — noted as a v2 improvement.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "assets" / "minecraft_block_registry.json"

# Present in the registry but not meaningfully placeable as a build voxel.
# Note: plain "air" IS allowed — it's a useful carve/erase block (renders as empty
# space, exports as minecraft:air to actively clear terrain on paste). The remaining
# entries are obscure technical variants with no build use.
_EXCLUDED_NAMES = {"cave_air", "void_air", "structure_void"}


@dataclass(frozen=True)
class Block:
    """A single palette entry.

    `name` is the bare base name (e.g. "oak_stairs"); `state` holds parsed block-state
    props as sorted (key, value) pairs (e.g. (("facing","north"),("half","top"))), and
    `mc_id` includes the `[state]` suffix for export. `renderable` is False only when no
    color/texture can be resolved even from the base material (e.g. air) — such blocks are
    still placed/exported but skipped by the preview renderer.
    """

    index: int
    name: str  # bare base name, e.g. "oak_stairs"
    mc_id: str  # e.g. "minecraft:oak_stairs[facing=north,half=top]"
    rgb: tuple[int, int, int]
    transparent: bool = False
    renderable: bool = True
    state: tuple[tuple[str, str], ...] = ()


class PaletteError(Exception):
    """Raised when a blueprint references an unknown block name."""


# name -> (rgb, transparent). Curated by hand for common/visually important blocks;
# every other registry block is resolved lazily from its texture (see _resolve).
_CURATED: dict[str, tuple[tuple[int, int, int], bool]] = {
    # --- Stone family ---
    "stone": ((125, 125, 125), False),
    "cobblestone": ((122, 122, 122), False),
    "mossy_cobblestone": ((110, 122, 100), False),
    "stone_bricks": ((122, 122, 122), False),
    "mossy_stone_bricks": ((115, 120, 105), False),
    "cracked_stone_bricks": ((117, 117, 117), False),
    "chiseled_stone_bricks": ((120, 120, 120), False),
    "smooth_stone": ((160, 160, 160), False),
    "granite": ((149, 103, 85), False),
    "polished_granite": ((152, 108, 91), False),
    "diorite": ((188, 188, 188), False),
    "polished_diorite": ((196, 196, 199), False),
    "andesite": ((132, 133, 132), False),
    "polished_andesite": ((131, 137, 133), False),
    "bedrock": ((85, 85, 85), False),
    "gravel": ((132, 128, 124), False),
    "obsidian": ((20, 18, 29), False),
    "crying_obsidian": ((32, 10, 63), False),
    # --- Deepslate family ---
    "deepslate": ((78, 78, 84), False),
    "cobbled_deepslate": ((76, 76, 79), False),
    "polished_deepslate": ((70, 70, 75), False),
    "deepslate_bricks": ((65, 65, 70), False),
    "deepslate_tiles": ((62, 62, 66), False),
    "chiseled_deepslate": ((68, 68, 72), False),
    "cracked_deepslate_bricks": ((63, 63, 67), False),
    "cracked_deepslate_tiles": ((60, 60, 64), False),
    # --- Brick / sandstone / quartz ---
    "bricks": ((150, 97, 83), False),
    "mud_bricks": ((140, 105, 78), False),
    "sandstone": ((216, 203, 155), False),
    "chiseled_sandstone": ((214, 201, 154), False),
    "cut_sandstone": ((216, 204, 158), False),
    "smooth_sandstone": ((219, 207, 163), False),
    "red_sandstone": ((181, 99, 32), False),
    "chiseled_red_sandstone": ((179, 97, 31), False),
    "cut_red_sandstone": ((181, 100, 33), False),
    "smooth_red_sandstone": ((183, 101, 34), False),
    "quartz_block": ((235, 229, 222), False),
    "smooth_quartz": ((237, 233, 226), False),
    "chiseled_quartz_block": ((233, 229, 224), False),
    "quartz_pillar": ((231, 226, 219), False),
    "quartz_bricks": ((234, 228, 221), False),
    "purpur_block": ((169, 125, 169), False),
    "purpur_pillar": ((171, 127, 171), False),
    "prismarine": ((99, 156, 151), False),
    "prismarine_bricks": ((99, 172, 153), False),
    "dark_prismarine": ((68, 99, 78), False),
    "nether_bricks": ((44, 22, 26), False),
    "red_nether_bricks": ((69, 7, 9), False),
    "blackstone": ((42, 36, 40), False),
    "polished_blackstone": ((52, 47, 53), False),
    "polished_blackstone_bricks": ((48, 43, 49), False),
    "gilded_blackstone": ((60, 45, 40), False),
    "basalt": ((69, 68, 72), False),
    "smooth_basalt": ((79, 79, 82), False),
    "end_stone": ((219, 219, 165), False),
    "end_stone_bricks": ((223, 223, 172), False),
    "terracotta": ((152, 94, 68), False),
    "white_terracotta": ((209, 178, 161), False),
    "orange_terracotta": ((161, 83, 37), False),
    "light_gray_terracotta": ((135, 107, 98), False),
    "gray_terracotta": ((57, 42, 35), False),
    "brown_terracotta": ((77, 51, 36), False),
    "black_terracotta": ((37, 22, 16), False),
    # --- Wood: planks ---
    "oak_planks": ((162, 130, 78), False),
    "spruce_planks": ((114, 84, 48), False),
    "birch_planks": ((192, 175, 121), False),
    "jungle_planks": ((160, 115, 80), False),
    "acacia_planks": ((168, 90, 50), False),
    "dark_oak_planks": ((67, 43, 21), False),
    "mangrove_planks": ((117, 54, 48), False),
    "cherry_planks": ((227, 180, 165), False),
    "crimson_planks": ((101, 48, 68), False),
    "warped_planks": ((43, 104, 99), False),
    # --- Wood: logs ---
    "oak_log": ((108, 89, 55), False),
    "spruce_log": ((65, 47, 28), False),
    "birch_log": ((216, 210, 203), False),
    "jungle_log": ((85, 68, 39), False),
    "acacia_log": ((103, 79, 56), False),
    "dark_oak_log": ((60, 47, 29), False),
    "mangrove_log": ((89, 61, 61), False),
    "cherry_log": ((54, 32, 33), False),
    "crimson_stem": ((110, 46, 82), False),
    "warped_stem": ((52, 104, 101), False),
    "stripped_oak_log": ((169, 135, 82), False),
    "stripped_dark_oak_log": ((89, 65, 42), False),
    # --- Glass ---
    "glass": ((220, 237, 237), True),
    "white_stained_glass": ((224, 224, 224), True),
    "orange_stained_glass": ((216, 127, 51), True),
    "light_blue_stained_glass": ((102, 153, 216), True),
    "blue_stained_glass": ((51, 76, 178), True),
    "green_stained_glass": ((94, 124, 22), True),
    "black_stained_glass": ((25, 25, 25), True),
    "glass_pane": ((220, 237, 237), True),
    # --- Wool / concrete (representative colors) ---
    "white_wool": ((233, 236, 236), False),
    "light_gray_wool": ((142, 142, 134), False),
    "gray_wool": ((62, 68, 71), False),
    "black_wool": ((20, 21, 25), False),
    "red_wool": ((161, 39, 34), False),
    "orange_wool": ((240, 118, 19), False),
    "yellow_wool": ((248, 197, 39), False),
    "lime_wool": ((112, 185, 25), False),
    "green_wool": ((84, 109, 27), False),
    "cyan_wool": ((21, 137, 145), False),
    "light_blue_wool": ((58, 175, 217), False),
    "blue_wool": ((53, 57, 157), False),
    "purple_wool": ((121, 42, 172), False),
    "magenta_wool": ((189, 68, 179), False),
    "pink_wool": ((238, 141, 172), False),
    "brown_wool": ((114, 71, 40), False),
    "white_concrete": ((207, 213, 214), False),
    "light_gray_concrete": ((125, 125, 115), False),
    "gray_concrete": ((54, 57, 61), False),
    "black_concrete": ((8, 10, 15), False),
    "red_concrete": ((142, 32, 32), False),
    "orange_concrete": ((224, 97, 1), False),
    "yellow_concrete": ((241, 175, 21), False),
    "lime_concrete": ((94, 168, 24), False),
    "green_concrete": ((73, 91, 36), False),
    "cyan_concrete": ((21, 119, 136), False),
    "light_blue_concrete": ((36, 137, 199), False),
    "blue_concrete": ((44, 46, 143), False),
    "purple_concrete": ((100, 32, 156), False),
    "brown_concrete": ((96, 60, 32), False),
    # --- Misc / functional ---
    "dirt": ((134, 96, 67), False),
    "coarse_dirt": ((121, 90, 64), False),
    "podzol": ((105, 74, 38), False),
    "grass_block": ((127, 178, 56), False),
    "mycelium": ((111, 98, 97), False),
    "clay": ((159, 164, 177), False),
    "packed_mud": ((152, 118, 84), False),
    "snow_block": ((249, 254, 254), False),
    "ice": ((140, 179, 237), True),
    "packed_ice": ((141, 180, 238), False),
    "iron_block": ((220, 220, 220), False),
    "gold_block": ((247, 223, 82), False),
    "diamond_block": ((98, 237, 220), False),
    "emerald_block": ((60, 178, 90), False),
    "netherite_block": ((67, 61, 63), False),
    "copper_block": ((191, 111, 80), False),
    "oxidized_copper": ((82, 162, 132), False),
    "glowstone": ((248, 202, 91), False),
    "sea_lantern": ((197, 219, 209), False),
    "shroomlight": ((245, 151, 68), False),
    "water": ((63, 118, 228), True),
    "lava": ((207, 92, 20), False),
    "hay_block": ((168, 143, 22), False),
    "bookshelf": ((109, 88, 54), False),
    "oak_leaves": ((60, 92, 30), True),
    "spruce_leaves": ((56, 79, 55), True),
    "birch_leaves": ((72, 100, 45), True),
    "jungle_leaves": ((43, 113, 21), True),
    "dark_oak_leaves": ((56, 82, 30), True),
    "azalea_leaves": ((100, 118, 54), True),
}


@lru_cache(maxsize=1)
def _load_registry_names() -> tuple[str, ...]:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return tuple(name for name in raw if name not in _EXCLUDED_NAMES)


@cache
def _texture_derived_color(name: str) -> tuple[tuple[int, int, int], bool] | None:
    from mcbuild.render import textures  # local import: avoids a palette<->render import cycle

    tex = textures.get_face_texture(name, "side") or textures.get_face_texture(name, "top")
    if tex is None:
        return None
    pixels = list(tex.convert("RGBA").tobytes())
    pixels = list(zip(pixels[0::4], pixels[1::4], pixels[2::4], pixels[3::4], strict=True))
    opaque = [p for p in pixels if p[3] > 10]
    if not opaque:
        return None
    n = len(opaque)
    rgb = (
        sum(p[0] for p in opaque) // n,
        sum(p[1] for p in opaque) // n,
        sum(p[2] for p in opaque) // n,
    )
    transparent = any(p[3] < 250 for p in pixels)
    return rgb, transparent


def _resolve(name: str) -> tuple[tuple[int, int, int], bool] | None:
    """Resolve (rgb, transparent) for a block, falling back to its base material.

    Tries: curated color → the block's own texture → the base-material texture (so
    shape variants like `oak_stairs` pick up `oak_planks`). None if nothing resolves.
    """
    curated = _CURATED.get(name)
    if curated is not None:
        return curated
    direct = _texture_derived_color(name)
    if direct is not None:
        return direct
    from mcbuild.render import blockmodel  # local import: avoids an import cycle

    base_tex = blockmodel._base_texture(name)
    if base_tex is not None and base_tex != name:
        return _texture_derived_color(base_tex)
    return None


_NAMES = _load_registry_names()
_NAME_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(_NAMES)}
_N_BASE = len(_NAMES)

# Stateful blocks ("oak_stairs[facing=north,...]") get indices allocated above the base
# registry range, on first use.
_dynamic_index: dict[str, int] = {}
_index_block: dict[int, Block] = {}


def _parse_name(name: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Split "minecraft:oak_stairs[facing=north,half=top]" -> ("oak_stairs", sorted pairs)."""
    if name.startswith("minecraft:"):
        name = name[len("minecraft:") :]
    if name.endswith("]") and "[" in name:
        base, rest = name.split("[", 1)
        pairs = []
        for part in rest[:-1].split(","):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            pairs.append((k.strip(), v.strip()))
        return base.strip(), tuple(sorted(pairs))
    return name.strip(), ()


def _mc_id(base: str, state: tuple[tuple[str, str], ...]) -> str:
    if not state:
        return f"minecraft:{base}"
    props = ",".join(f"{k}={v}" for k, v in state)
    return f"minecraft:{base}[{props}]"


def _build_block(index: int, base: str, state: tuple[tuple[str, str], ...]) -> Block:
    resolved = _resolve(base)
    if resolved is None:
        return Block(index=index, name=base, mc_id=_mc_id(base, state), rgb=(0, 0, 0), renderable=False, state=state)
    rgb, transparent = resolved
    return Block(index=index, name=base, mc_id=_mc_id(base, state), rgb=rgb, transparent=transparent, state=state)


@cache
def _base_block(index: int) -> Block:
    return _build_block(index, _NAMES[index], ())


def get_block(name: str) -> Block:
    """Look up a block by name, with optional block state, e.g. "oak_stairs[facing=north]".

    Accepts a "minecraft:" prefix. The base name is validated against the registry; block
    states are preserved for rendering and export but not individually validated.
    """
    base, state = _parse_name(name)
    index = _NAME_TO_INDEX.get(base)
    if index is None:
        suggestions = suggest(base)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise PaletteError(f"Unknown block '{base}'.{hint}")
    if not state:
        return _base_block(index)

    key = _mc_id(base, state)
    dyn = _dynamic_index.get(key)
    if dyn is None:
        dyn = _N_BASE + len(_dynamic_index)
        _dynamic_index[key] = dyn
        _index_block[dyn] = _build_block(dyn, base, state)
    return _index_block[dyn]


def get_block_by_index(index: int) -> Block:
    if index < _N_BASE:
        return _base_block(index)
    return _index_block[index]


def suggest(name: str, n: int = 3) -> list[str]:
    return difflib.get_close_matches(name, _NAMES, n=n, cutoff=0.4)


def all_block_names() -> list[str]:
    return list(_NAMES)
