import pytest

from mcbuild import palette
from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.render.iso import render_iso, render_topdown
from mcbuild.voxel import VoxelGrid


def test_registry_backed_block_is_valid_but_not_renderable():
    # oak_stairs is a real registry block with no dedicated texture file / curated
    # color (stairs reuse the parent block's blockstate model, not a flat PNG match).
    block = palette.get_block("oak_stairs")
    assert block.mc_id == "minecraft:oak_stairs"
    assert block.renderable is False


def test_curated_block_is_renderable():
    block = palette.get_block("stone")
    assert block.renderable is True


def test_dsl_accepts_non_renderable_block_name():
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'oak_stairs')", grid)
    assert len(grid) == 1


def test_non_renderable_block_is_skipped_in_iso_render():
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'oak_stairs')", grid)
    img = render_iso(grid, yaw=0)
    # nothing renderable was placed, so the canvas is empty (no crop bbox)
    assert img.getbbox() is None


def test_non_renderable_block_does_not_occlude_renderable_neighbor():
    grid = VoxelGrid()
    run_blueprint(
        "set_block(0, 0, 0, 'stone')\nset_block(0, 1, 0, 'oak_stairs')",
        grid,
    )
    img = render_iso(grid, yaw=0)
    assert img.getbbox() is not None  # the stone block still renders


def test_non_renderable_block_skipped_in_topdown():
    grid = VoxelGrid()
    run_blueprint(
        "set_block(0, 0, 0, 'stone')\nset_block(0, 5, 0, 'oak_stairs')",
        grid,
    )
    img = render_topdown(grid)
    # the tall oak_stairs on top is invisible; stone underneath should still show
    assert img.getbbox() is not None
