from mcbuild import palette
from mcbuild.server.delta import compute_block_delta
from mcbuild.voxel import VoxelGrid


def test_delta_against_none_reports_every_cell_as_added():
    stone = palette.get_block("stone").index
    after = VoxelGrid()
    after.set(0, 0, 0, stone)
    changes = compute_block_delta(None, after)
    assert changes == [{"x": 0, "y": 0, "z": 0, "block": "minecraft:stone"}]


def test_delta_reports_added_removed_and_changed():
    stone = palette.get_block("stone").index
    dirt = palette.get_block("dirt").index

    before = VoxelGrid()
    before.set(0, 0, 0, stone)  # unchanged
    before.set(1, 0, 0, stone)  # removed in `after`

    after = VoxelGrid()
    after.set(0, 0, 0, stone)  # unchanged
    after.set(2, 0, 0, dirt)  # added

    changes = compute_block_delta(before, after)
    by_coord = {(c["x"], c["y"], c["z"]): c["block"] for c in changes}
    assert by_coord == {
        (1, 0, 0): None,
        (2, 0, 0): "minecraft:dirt",
    }


def test_delta_reports_block_state_changes():
    stairs_north = palette.get_block("oak_stairs[facing=north]").index
    stairs_south = palette.get_block("oak_stairs[facing=south]").index

    before = VoxelGrid()
    before.set(0, 0, 0, stairs_north)

    after = VoxelGrid()
    after.set(0, 0, 0, stairs_south)

    changes = compute_block_delta(before, after)
    assert changes == [{"x": 0, "y": 0, "z": 0, "block": "minecraft:oak_stairs[facing=south]"}]


def test_delta_is_empty_for_identical_grids():
    stone = palette.get_block("stone").index
    grid_a = VoxelGrid()
    grid_a.set(0, 0, 0, stone)
    grid_b = VoxelGrid()
    grid_b.set(0, 0, 0, stone)
    assert compute_block_delta(grid_a, grid_b) == []
