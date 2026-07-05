import nbtlib
import pytest

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.export.schem import export_schem
from mcbuild.palette import PaletteError, get_block
from mcbuild.render import blockmodel
from mcbuild.voxel import VoxelGrid


def test_parse_block_state():
    b = get_block("oak_stairs[facing=north,half=top]")
    assert b.name == "oak_stairs"
    assert b.mc_id == "minecraft:oak_stairs[facing=north,half=top]"
    assert b.state == (("facing", "north"), ("half", "top"))


def test_state_prop_order_is_canonical():
    a = get_block("oak_stairs[half=top,facing=north]")
    b = get_block("oak_stairs[facing=north,half=top]")
    assert a.index == b.index  # same block regardless of prop order
    assert a.mc_id == b.mc_id


def test_distinct_states_get_distinct_indices():
    a = get_block("oak_stairs[facing=north]")
    b = get_block("oak_stairs[facing=south]")
    bare = get_block("oak_stairs")
    assert len({a.index, b.index, bare.index}) == 3


def test_invalid_base_name_still_raises():
    with pytest.raises(PaletteError):
        get_block("totally_bogus_block_xyz[facing=north]")


def test_stateful_block_is_renderable_via_base_material():
    b = get_block("oak_stairs[facing=north,half=bottom,shape=straight]")
    assert b.renderable is True
    assert b.rgb != (0, 0, 0)


def test_stateful_block_placed_and_exported_with_state(tmp_path):
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'oak_stairs[facing=north,half=bottom,shape=straight]')", grid)
    assert len(grid) == 1

    out = tmp_path / "s.schem"
    export_schem(grid, str(out))
    palette = list(nbtlib.load(str(out))["Schematic"]["Blocks"]["Palette"].keys())
    assert "minecraft:oak_stairs[facing=north,half=bottom,shape=straight]" in [str(p) for p in palette]


def test_stateful_mesh_differs_from_full_cube():
    straight = tuple(sorted({"facing": "north", "half": "bottom", "shape": "straight"}.items()))
    stair_mesh = blockmodel.get_block_mesh("oak_stairs", straight)
    cube_mesh = blockmodel.get_block_mesh("oak_planks", ())
    assert stair_mesh is not None and cube_mesh is not None
    assert len(stair_mesh) != len(cube_mesh)  # stairs are two boxes, not one cube
