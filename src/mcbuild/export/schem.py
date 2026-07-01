"""Sponge Schematic v2 export (WorldEdit-compatible .schem), via nbtlib.

DataVersion 3953 corresponds to Minecraft 1.21.1.
"""

from __future__ import annotations

import nbtlib
from nbtlib.tag import ByteArray, Compound, Int, IntArray, Short

from mcbuild.palette import get_block_by_index
from mcbuild.voxel import VoxelGrid

DATA_VERSION = 3953


def _varint(value: int) -> list[int]:
    out = []
    v = value
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return out


def _to_signed_byte(b: int) -> int:
    return b - 256 if b >= 128 else b


def grid_to_schematic(grid: VoxelGrid) -> Compound:
    """Build the root NBT compound for a Sponge Schematic v2 file."""
    bounds = grid.bounds
    if bounds is None:
        raise ValueError("Cannot export an empty voxel grid.")
    (minx, miny, minz), (maxx, maxy, maxz) = bounds
    width = maxx - minx + 1
    height = maxy - miny + 1
    length = maxz - minz + 1

    used_indices = sorted({idx for _, idx in grid.items()})
    palette_compound = Compound()
    mc_id_to_palette_id: dict[int, int] = {}
    for palette_id, idx in enumerate(used_indices):
        block = get_block_by_index(idx)
        mc_id_to_palette_id[idx] = palette_id
        palette_compound[block.mc_id] = Int(palette_id)

    air_palette_id = len(palette_compound)
    palette_compound["minecraft:air"] = Int(air_palette_id)

    volume = width * height * length
    dense = [air_palette_id] * volume
    for (x, y, z), idx in grid.items():
        lx, ly, lz = x - minx, y - miny, z - minz
        pos = lx + lz * width + ly * width * length
        dense[pos] = mc_id_to_palette_id[idx]

    block_data_bytes: list[int] = []
    for v in dense:
        block_data_bytes.extend(_varint(v))

    return Compound(
        {
            "Version": Int(2),
            "DataVersion": Int(DATA_VERSION),
            "Width": Short(width),
            "Height": Short(height),
            "Length": Short(length),
            "Offset": IntArray([minx, miny, minz]),
            "PaletteMax": Int(len(palette_compound)),
            "Palette": palette_compound,
            "BlockData": ByteArray([_to_signed_byte(b) for b in block_data_bytes]),
            "Metadata": Compound({}),
        }
    )


def export_schem(grid: VoxelGrid, path: str) -> None:
    """Write the grid to a gzipped Sponge Schematic v2 (.schem) file at `path`."""
    root = grid_to_schematic(grid)
    nbt_file = nbtlib.File(root, gzipped=True, root_name="")
    nbt_file.save(path)
