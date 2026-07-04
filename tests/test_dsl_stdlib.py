import pytest

from mcbuild.dsl.errors import BlueprintError
from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.palette import get_block
from mcbuild.voxel import VoxelGrid


def test_fill_block_count():
    grid = VoxelGrid()
    run_blueprint("fill(0, 0, 0, 4, 0, 4, 'stone')", grid)
    assert len(grid) == 25


def test_hollow_box_is_shell_only():
    grid = VoxelGrid()
    run_blueprint("hollow_box(0, 0, 0, 4, 4, 4, 'stone', thickness=1)", grid)
    # full box 5x5x5=125, minus interior 3x3x3=27 => 98
    assert len(grid) == 98
    assert grid.get(2, 2, 2) is None  # center is hollow


def test_walls_no_floor_ceiling():
    grid = VoxelGrid()
    run_blueprint("walls(0, 0, 4, 4, 0, 2, 'stone', thickness=1)", grid)
    assert grid.get(2, 0, 2) is None  # interior, not a wall
    assert grid.get(0, 0, 0) is not None  # corner is a wall


def test_floor_flat_plate():
    grid = VoxelGrid()
    run_blueprint("floor(0, 0, 3, 3, 5, 'stone')", grid)
    assert len(grid) == 16
    for (_, y, _), _ in grid.items():
        assert y == 5


def test_cylinder_solid_vs_hollow():
    grid_solid = VoxelGrid()
    run_blueprint("cylinder(0, 0, 0, height=3, r=3, block='stone', hollow=False)", grid_solid)
    grid_hollow = VoxelGrid()
    run_blueprint("cylinder(0, 0, 0, height=3, r=3, block='stone', hollow=True)", grid_hollow)
    assert len(grid_hollow) < len(grid_solid)
    assert grid_solid.get(0, 0, 0) is not None
    assert grid_hollow.get(0, 0, 0) is None  # center carved out


def test_sphere_is_roughly_symmetric():
    grid = VoxelGrid()
    run_blueprint("sphere(0, 0, 0, 3, 'stone')", grid)
    assert grid.get(0, 0, 0) is not None
    assert grid.get(3, 0, 0) is not None
    assert grid.get(4, 0, 0) is None


def test_clear_removes_blocks():
    grid = VoxelGrid()
    run_blueprint(
        "fill(0,0,0,4,4,4,'stone')\nclear(1,1,1,3,3,3)\n",
        grid,
    )
    assert grid.get(2, 2, 2) is None
    assert grid.get(0, 0, 0) is not None


def test_gable_roof_apex_higher_than_eaves():
    grid = VoxelGrid()
    run_blueprint("gable_roof(0, 0, 6, 4, 5, 'stone', ridge_axis='x', overhang=1)", grid)
    ys = [y for (_, y, _), _ in grid.items()]
    assert max(ys) > min(ys)


# --- set_blocks ---


def test_set_blocks_with_default_block():
    grid = VoxelGrid()
    run_blueprint("set_blocks([(0,0,0),(1,0,0),(2,0,0)], block='stone')", grid)
    assert len(grid) == 3
    assert grid.get(1, 0, 0) == get_block("stone").index


def test_set_blocks_per_entry_overrides_default():
    grid = VoxelGrid()
    run_blueprint("set_blocks([(0,0,0,'stone'),(1,0,0,'oak_planks')], block='glass')", grid)
    assert grid.get(0, 0, 0) == get_block("stone").index
    assert grid.get(1, 0, 0) == get_block("oak_planks").index


def test_set_blocks_mixed_3_and_4_tuples():
    grid = VoxelGrid()
    run_blueprint("set_blocks([(0,0,0),(1,0,0,'glass')], block='stone')", grid)
    assert grid.get(0, 0, 0) == get_block("stone").index
    assert grid.get(1, 0, 0) == get_block("glass").index


def test_set_blocks_no_default_raises_on_3_tuple():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError):
        run_blueprint("set_blocks([(0,0,0)])", grid)


def test_set_blocks_respects_transform_stack():
    grid = VoxelGrid()
    run_blueprint(
        "with translate(10, 0, 0):\n    set_blocks([(0,0,0),(1,0,0)], block='stone')",
        grid,
    )
    assert grid.get(10, 0, 0) is not None
    assert grid.get(11, 0, 0) is not None


# --- detail helpers ---


def test_weighted_block_only_uses_spec_materials_and_is_deterministic():
    src = "fill(0,0,0,7,0,7, weighted_block({'stone': 0.5, 'cobblestone': 0.5}))"
    g1 = VoxelGrid()
    g2 = VoxelGrid()
    run_blueprint(src, g1, seed=7)
    run_blueprint(src, g2, seed=7)
    assert dict(g1.items()) == dict(g2.items())  # seeded → deterministic
    allowed = {get_block("stone").index, get_block("cobblestone").index}
    assert {idx for _, idx in g1.items()} <= allowed
    # a 64-cell fill with a 50/50 split should use both materials
    assert len({idx for _, idx in g1.items()}) == 2


def test_scatter_density_bounds_and_determinism():
    src = "scatter(0,0,0,9,0,9, 'mossy_cobblestone', density=0.3)"
    g1 = VoxelGrid()
    g2 = VoxelGrid()
    run_blueprint(src, g1, seed=3)
    run_blueprint(src, g2, seed=3)
    assert dict(g1.items()) == dict(g2.items())
    assert 0 < len(g1) < 100  # sparse, not full, not empty


def test_frame_places_only_edges():
    grid = VoxelGrid()
    run_blueprint("frame(0,0,0,2,2,2,'stone')", grid)
    # a 3x3x3 box has 8 corners + 12 edge-midpoints = 20 edge cells; face centers/interior excluded
    assert grid.get(0, 0, 0) is not None  # corner
    assert grid.get(1, 1, 0) is None  # face center, not an edge
    assert grid.get(1, 1, 1) is None  # interior
    assert len(grid) == 20


def test_window_grid_places_in_wall_plane_only():
    grid = VoxelGrid()
    run_blueprint("window_grid(0, 0, 0, 8, 0, 4, 'glass', spacing=2, margin=1)", grid)
    assert len(grid) > 0
    # all placed cells lie on the x=0 wall plane
    assert all(x == 0 for (x, _, _), _ in grid.items())


def test_window_grid_non_axis_aligned_raises():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError):
        run_blueprint("window_grid(0, 0, 5, 8, 0, 4, 'glass')", grid)


# --- guard: reference-manual examples must execute (catch doc drift) ---


def test_reference_math_post_circle_example_runs():
    grid = VoxelGrid()
    run_blueprint(
        "n = 8\nr = 6\n"
        "posts = [(round(r*math.cos(2*math.pi*i/n)), 1, round(r*math.sin(2*math.pi*i/n)), 'glowstone') "
        "for i in range(n)]\n"
        "set_blocks(posts)",
        grid,
    )
    assert 0 < len(grid) <= 8


def test_reference_flagship_asymmetric_example_runs():
    src = """
walls(0, 0, 9, 5, 0, 4, "cobblestone")
floor(0, 0, 9, 5, 0, "oak_planks")
walls(6, 3, 10, 8, 0, 5, "cobblestone")
floor(6, 3, 10, 8, 0, "oak_planks")
set_blocks([(0, 2, 2), (0, 2, 3), (0, 2, 6)], block="glass")
clear(2, 1, 9, 3, 2, 9)
gable_roof(-1, -1, 10, 6, 5, "spruce_planks", ridge_axis="x", overhang=1)
gable_roof(5, 2, 9, 11, 6, "dark_oak_planks", ridge_axis="z", overhang=1)
scatter(0, 0, 0, 5, 4, 0, "mossy_cobblestone", density=0.15)
"""
    grid = VoxelGrid()
    run_blueprint(src, grid, seed=1)
    assert len(grid) > 0
