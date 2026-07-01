"""Curated block palette: name -> RGB (+ transparency), with fuzzy suggestions.

v1 supports full-cube blocks only (no stairs/slabs/orientation states).
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    """A single palette entry."""

    index: int
    name: str  # bare name, e.g. "stone"
    mc_id: str  # e.g. "minecraft:stone"
    rgb: tuple[int, int, int]
    transparent: bool = False


class PaletteError(Exception):
    """Raised when a blueprint references an unknown block name."""


# (name, rgb, transparent) — order defines the stable palette index.
_RAW_BLOCKS: list[tuple[str, tuple[int, int, int], bool]] = [
    # --- Stone family ---
    ("stone", (125, 125, 125), False),
    ("cobblestone", (122, 122, 122), False),
    ("mossy_cobblestone", (110, 122, 100), False),
    ("stone_bricks", (122, 122, 122), False),
    ("mossy_stone_bricks", (115, 120, 105), False),
    ("cracked_stone_bricks", (117, 117, 117), False),
    ("chiseled_stone_bricks", (120, 120, 120), False),
    ("smooth_stone", (160, 160, 160), False),
    ("granite", (149, 103, 85), False),
    ("polished_granite", (152, 108, 91), False),
    ("diorite", (188, 188, 188), False),
    ("polished_diorite", (196, 196, 199), False),
    ("andesite", (132, 133, 132), False),
    ("polished_andesite", (131, 137, 133), False),
    ("bedrock", (85, 85, 85), False),
    ("gravel", (132, 128, 124), False),
    ("obsidian", (20, 18, 29), False),
    ("crying_obsidian", (32, 10, 63), False),
    # --- Deepslate family ---
    ("deepslate", (78, 78, 84), False),
    ("cobbled_deepslate", (76, 76, 79), False),
    ("polished_deepslate", (70, 70, 75), False),
    ("deepslate_bricks", (65, 65, 70), False),
    ("deepslate_tiles", (62, 62, 66), False),
    ("chiseled_deepslate", (68, 68, 72), False),
    ("cracked_deepslate_bricks", (63, 63, 67), False),
    ("cracked_deepslate_tiles", (60, 60, 64), False),
    # --- Brick / sandstone / quartz ---
    ("bricks", (150, 97, 83), False),
    ("mud_bricks", (140, 105, 78), False),
    ("sandstone", (216, 203, 155), False),
    ("chiseled_sandstone", (214, 201, 154), False),
    ("cut_sandstone", (216, 204, 158), False),
    ("smooth_sandstone", (219, 207, 163), False),
    ("red_sandstone", (181, 99, 32), False),
    ("chiseled_red_sandstone", (179, 97, 31), False),
    ("cut_red_sandstone", (181, 100, 33), False),
    ("smooth_red_sandstone", (183, 101, 34), False),
    ("quartz_block", (235, 229, 222), False),
    ("smooth_quartz", (237, 233, 226), False),
    ("chiseled_quartz_block", (233, 229, 224), False),
    ("quartz_pillar", (231, 226, 219), False),
    ("quartz_bricks", (234, 228, 221), False),
    ("purpur_block", (169, 125, 169), False),
    ("purpur_pillar", (171, 127, 171), False),
    ("prismarine", (99, 156, 151), False),
    ("prismarine_bricks", (99, 172, 153), False),
    ("dark_prismarine", (68, 99, 78), False),
    ("nether_bricks", (44, 22, 26), False),
    ("red_nether_bricks", (69, 7, 9), False),
    ("blackstone", (42, 36, 40), False),
    ("polished_blackstone", (52, 47, 53), False),
    ("polished_blackstone_bricks", (48, 43, 49), False),
    ("gilded_blackstone", (60, 45, 40), False),
    ("basalt", (69, 68, 72), False),
    ("smooth_basalt", (79, 79, 82), False),
    ("end_stone", (219, 219, 165), False),
    ("end_stone_bricks", (223, 223, 172), False),
    ("terracotta", (152, 94, 68), False),
    ("white_terracotta", (209, 178, 161), False),
    ("orange_terracotta", (161, 83, 37), False),
    ("light_gray_terracotta", (135, 107, 98), False),
    ("gray_terracotta", (57, 42, 35), False),
    ("brown_terracotta", (77, 51, 36), False),
    ("black_terracotta", (37, 22, 16), False),
    # --- Wood: planks ---
    ("oak_planks", (162, 130, 78), False),
    ("spruce_planks", (114, 84, 48), False),
    ("birch_planks", (192, 175, 121), False),
    ("jungle_planks", (160, 115, 80), False),
    ("acacia_planks", (168, 90, 50), False),
    ("dark_oak_planks", (67, 43, 21), False),
    ("mangrove_planks", (117, 54, 48), False),
    ("cherry_planks", (227, 180, 165), False),
    ("crimson_planks", (101, 48, 68), False),
    ("warped_planks", (43, 104, 99), False),
    # --- Wood: logs ---
    ("oak_log", (108, 89, 55), False),
    ("spruce_log", (65, 47, 28), False),
    ("birch_log", (216, 210, 203), False),
    ("jungle_log", (85, 68, 39), False),
    ("acacia_log", (103, 79, 56), False),
    ("dark_oak_log", (60, 47, 29), False),
    ("mangrove_log", (89, 61, 61), False),
    ("cherry_log", (54, 32, 33), False),
    ("crimson_stem", (110, 46, 82), False),
    ("warped_stem", (52, 104, 101), False),
    ("stripped_oak_log", (169, 135, 82), False),
    ("stripped_dark_oak_log", (89, 65, 42), False),
    # --- Glass ---
    ("glass", (220, 237, 237), True),
    ("white_stained_glass", (224, 224, 224), True),
    ("orange_stained_glass", (216, 127, 51), True),
    ("light_blue_stained_glass", (102, 153, 216), True),
    ("blue_stained_glass", (51, 76, 178), True),
    ("green_stained_glass", (94, 124, 22), True),
    ("black_stained_glass", (25, 25, 25), True),
    ("glass_pane", (220, 237, 237), True),
    # --- Wool / concrete (representative colors) ---
    ("white_wool", (233, 236, 236), False),
    ("light_gray_wool", (142, 142, 134), False),
    ("gray_wool", (62, 68, 71), False),
    ("black_wool", (20, 21, 25), False),
    ("red_wool", (161, 39, 34), False),
    ("orange_wool", (240, 118, 19), False),
    ("yellow_wool", (248, 197, 39), False),
    ("lime_wool", (112, 185, 25), False),
    ("green_wool", (84, 109, 27), False),
    ("cyan_wool", (21, 137, 145), False),
    ("light_blue_wool", (58, 175, 217), False),
    ("blue_wool", (53, 57, 157), False),
    ("purple_wool", (121, 42, 172), False),
    ("magenta_wool", (189, 68, 179), False),
    ("pink_wool", (238, 141, 172), False),
    ("brown_wool", (114, 71, 40), False),
    ("white_concrete", (207, 213, 214), False),
    ("light_gray_concrete", (125, 125, 115), False),
    ("gray_concrete", (54, 57, 61), False),
    ("black_concrete", (8, 10, 15), False),
    ("red_concrete", (142, 32, 32), False),
    ("orange_concrete", (224, 97, 1), False),
    ("yellow_concrete", (241, 175, 21), False),
    ("lime_concrete", (94, 168, 24), False),
    ("green_concrete", (73, 91, 36), False),
    ("cyan_concrete", (21, 119, 136), False),
    ("light_blue_concrete", (36, 137, 199), False),
    ("blue_concrete", (44, 46, 143), False),
    ("purple_concrete", (100, 32, 156), False),
    ("brown_concrete", (96, 60, 32), False),
    # --- Misc / functional ---
    ("dirt", (134, 96, 67), False),
    ("coarse_dirt", (121, 90, 64), False),
    ("podzol", (105, 74, 38), False),
    ("grass_block", (127, 178, 56), False),
    ("mycelium", (111, 98, 97), False),
    ("clay", (159, 164, 177), False),
    ("packed_mud", (152, 118, 84), False),
    ("snow_block", (249, 254, 254), False),
    ("ice", (140, 179, 237), True),
    ("packed_ice", (141, 180, 238), False),
    ("iron_block", (220, 220, 220), False),
    ("gold_block", (247, 223, 82), False),
    ("diamond_block", (98, 237, 220), False),
    ("emerald_block", (60, 178, 90), False),
    ("netherite_block", (67, 61, 63), False),
    ("copper_block", (191, 111, 80), False),
    ("oxidized_copper", (82, 162, 132), False),
    ("glowstone", (248, 202, 91), False),
    ("sea_lantern", (197, 219, 209), False),
    ("shroomlight", (245, 151, 68), False),
    ("water", (63, 118, 228), True),
    ("lava", (207, 92, 20), False),
    ("hay_block", (168, 143, 22), False),
    ("bookshelf", (109, 88, 54), False),
    ("oak_leaves", (60, 92, 30), True),
    ("spruce_leaves", (56, 79, 55), True),
    ("birch_leaves", (72, 100, 45), True),
    ("jungle_leaves", (43, 113, 21), True),
    ("dark_oak_leaves", (56, 82, 30), True),
    ("azalea_leaves", (100, 118, 54), True),
]


def _build_palette() -> tuple[dict[str, Block], list[Block]]:
    by_name: dict[str, Block] = {}
    by_index: list[Block] = []
    for i, (name, rgb, transparent) in enumerate(_RAW_BLOCKS):
        block = Block(index=i, name=name, mc_id=f"minecraft:{name}", rgb=rgb, transparent=transparent)
        by_name[name] = block
        by_index.append(block)
    return by_name, by_index


PALETTE_BY_NAME, PALETTE_BY_INDEX = _build_palette()


def get_block(name: str) -> Block:
    """Look up a block by bare name (e.g. "oak_planks"). Accepts a "minecraft:" prefix too."""
    if name.startswith("minecraft:"):
        name = name[len("minecraft:") :]
    block = PALETTE_BY_NAME.get(name)
    if block is not None:
        return block
    suggestions = suggest(name)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise PaletteError(f"Unknown block '{name}'.{hint}")


def get_block_by_index(index: int) -> Block:
    return PALETTE_BY_INDEX[index]


def suggest(name: str, n: int = 3) -> list[str]:
    return difflib.get_close_matches(name, PALETTE_BY_NAME.keys(), n=n, cutoff=0.4)


def all_block_names() -> list[str]:
    return [b.name for b in PALETTE_BY_INDEX]
