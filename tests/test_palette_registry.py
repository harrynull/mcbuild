from mcbuild import palette
from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.render.iso import render_iso
from mcbuild.voxel import VoxelGrid


def test_registry_backed_stateful_block_is_valid():
    # oak_stairs is a real registry block; it has no curated flat color, so the palette
    # flag is False, but the mesh renderer now draws it from its base texture (see below).
    block = palette.get_block("oak_stairs")
    assert block.mc_id == "minecraft:oak_stairs"


def test_curated_block_is_renderable():
    block = palette.get_block("stone")
    assert block.renderable is True


def test_dsl_accepts_stateful_block_name():
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'oak_stairs')", grid)
    assert len(grid) == 1


def test_stateful_block_now_renders_in_iso():
    # Phase 4: stairs/slabs/etc. render via the mesh path (base texture), no longer invisible.
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'oak_stairs')", grid)
    img = render_iso(grid, yaw=0)
    assert img.getbbox() is not None


def test_truly_invisible_block_is_skipped_in_iso_render():
    # a block with no resolvable texture (e.g. air) stays invisible / non-occluding.
    grid = VoxelGrid()
    run_blueprint("set_block(0, 0, 0, 'air')", grid)
    img = render_iso(grid, yaw=0)
    assert img.getbbox() is None
