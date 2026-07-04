from mcbuild.agent import query
from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.voxel import VoxelGrid


def _slab_grid():
    grid = VoxelGrid()
    run_blueprint("fill(0, 0, 0, 2, 0, 2, 'stone')\nset_block(1, 0, 1, 'glass')", grid)
    return grid


def test_ascii_slice_shows_layout_and_legend():
    grid = _slab_grid()
    text = query.ascii_slice(grid, "y", 0)
    # 3x3 floor with a glass center; glyphs differ for stone vs glass
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(("slice", "legend"))]
    grid_rows = lines[:3]
    assert all(len(r) == 3 for r in grid_rows)
    center_char = grid_rows[1][1]
    corner_char = grid_rows[0][0]
    assert center_char != corner_char  # glass distinct from stone
    assert "stone" in text and "glass" in text


def test_ascii_slice_empty_level_is_all_dots():
    grid = _slab_grid()
    text = query.ascii_slice(grid, "y", 5)  # nothing at y=5
    body = [ln for ln in text.splitlines() if ln and not ln.startswith(("slice", "legend"))]
    assert all(set(row) <= {"."} for row in body) or "no blocks" in text


def test_point_query_returns_block_or_air():
    grid = _slab_grid()
    assert "minecraft:glass" in query.point_query(grid, 1, 0, 1)
    assert "air" in query.point_query(grid, 5, 5, 5)


def test_material_histogram_counts():
    grid = _slab_grid()
    text = query.material_histogram(grid)
    assert "stone: 8" in text  # 9 cells minus the 1 glass
    assert "glass: 1" in text


def test_material_histogram_region_scopes_counts():
    grid = VoxelGrid()
    run_blueprint("fill(0,0,0,4,0,4,'stone')", grid)
    text = query.material_histogram(grid, [0, 0, 0, 1, 0, 1])
    assert "stone: 4" in text  # only the 2x2 corner


def test_ascii_slice_windows_to_occupied_extent_not_whole_build():
    grid = VoxelGrid()
    # a tiny 2x2 patch inside a much larger (mostly empty) build footprint
    run_blueprint("fill(0,0,0,1,0,1,'stone')\nset_block(100, 5, 100, 'glass')", grid)
    text = query.ascii_slice(grid, "y", 0)
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(("slice", "legend", "("))]
    # windowed to the 2x2 patch actually present at y=0, not the full 0..100 footprint
    assert all(len(row) == 2 for row in lines)
    assert len(lines) == 2


def test_ascii_slice_caps_dense_large_footprint():
    grid = VoxelGrid()
    run_blueprint("fill(0,0,0,199,0,1,'stone')", grid)  # 200x1x2 dense floor
    text = query.ascii_slice(grid, "y", 0)
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith(("slice", "legend", "(")) and set(ln) <= {".", "#"}]
    assert all(len(row) <= query.MAX_SLICE_DIM for row in lines)
    assert "truncated" in text.lower()
