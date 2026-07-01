from mcbuild.dsl.sandbox import run_blueprint
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
