import nbtlib

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.export.schem import DATA_VERSION, export_schem
from mcbuild.voxel import VoxelGrid


def _decode_varints(data: bytes, count: int) -> list[int]:
    values = []
    i = 0
    for _ in range(count):
        result = 0
        shift = 0
        while True:
            b = data[i]
            i += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        values.append(result)
    return values


def test_schem_round_trip(tmp_path):
    grid = VoxelGrid()
    run_blueprint("fill(0, 0, 0, 2, 1, 2, 'stone')\nset_block(1, 2, 1, 'glass')", grid)

    out_path = tmp_path / "test.schem"
    export_schem(grid, str(out_path))

    loaded = nbtlib.load(str(out_path))
    assert int(loaded["Version"]) == 2
    assert int(loaded["DataVersion"]) == DATA_VERSION
    assert int(loaded["Width"]) == 3
    assert int(loaded["Height"]) == 3
    assert int(loaded["Length"]) == 3

    palette = {str(k): int(v) for k, v in loaded["Palette"].items()}
    assert "minecraft:stone" in palette
    assert "minecraft:glass" in palette
    assert "minecraft:air" in palette

    raw_bytes = bytes(b & 0xFF for b in loaded["BlockData"])
    volume = int(loaded["Width"]) * int(loaded["Height"]) * int(loaded["Length"])
    values = _decode_varints(raw_bytes, volume)

    id_to_name = {v: k for k, v in palette.items()}
    stone_id = palette["minecraft:stone"]
    glass_id = palette["minecraft:glass"]

    width, length = int(loaded["Width"]), int(loaded["Length"])

    def idx_of(x, y, z):
        return x + z * width + y * width * length

    assert values[idx_of(0, 0, 0)] == stone_id
    assert values[idx_of(2, 1, 2)] == stone_id
    assert values[idx_of(1, 2, 1)] == glass_id
    assert id_to_name[values[idx_of(0, 2, 0)]] == "minecraft:air"


def test_offset_matches_grid_min_bounds(tmp_path):
    grid = VoxelGrid()
    run_blueprint("set_block(-2, -3, -1, 'stone')\nset_block(0, 0, 0, 'stone')", grid)
    out_path = tmp_path / "offset.schem"
    export_schem(grid, str(out_path))
    loaded = nbtlib.load(str(out_path))
    offset = list(loaded["Offset"])
    assert offset == [-2, -3, -1]
