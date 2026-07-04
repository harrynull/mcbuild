import pytest

from mcbuild.dsl.errors import BlueprintError
from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.voxel import VoxelGrid


def test_simple_fill_executes():
    grid = VoxelGrid()
    run_blueprint("fill(0, 0, 0, 1, 1, 1, 'stone')", grid)
    assert len(grid) == 8


def test_import_is_blocked():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError) as exc_info:
        run_blueprint("import os\nfill(0,0,0,1,1,1,'stone')", grid)
    assert "not allowed" in str(exc_info.value)


def test_dunder_attribute_access_blocked():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError):
        run_blueprint("x = (1).__class__\n", grid)


def test_banned_name_blocked():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError):
        run_blueprint("eval('1+1')\n", grid)


def test_open_blocked():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError):
        run_blueprint("open('x.txt')\n", grid)


def test_syntax_error_reports_line():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError) as exc_info:
        run_blueprint("fill(0,0,0,1,1,1,'stone'\n", grid)
    assert exc_info.value.line is not None


def test_unknown_block_reports_line_and_suggestion():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError) as exc_info:
        run_blueprint("x = 1\nfill(0,0,0,1,1,1,'stoen')\n", grid)
    err = exc_info.value
    assert err.line == 2
    assert "stone" in str(err)


def test_infinite_loop_hits_budget():
    grid = VoxelGrid()
    with pytest.raises(BlueprintError) as exc_info:
        run_blueprint("while True:\n    pass\n", grid, max_lines=1000, max_seconds=2.0)
    assert "budget" in str(exc_info.value).lower() or "lines" in str(exc_info.value).lower()


def test_transform_stack_translate_and_mirror():
    grid = VoxelGrid()
    run_blueprint(
        """
def wing():
    set_block(1, 0, 0, 'stone')

wing()
with mirror('x', at=0):
    wing()
""",
        grid,
    )
    assert grid.get(1, 0, 0) == grid.get(-1, 0, 0)
    assert len(grid) == 2


def test_rotate_y_quarter_turn():
    grid = VoxelGrid()
    run_blueprint("set_block(2, 0, 0, 'stone')\nwith rotate_y(1):\n    set_block(2, 0, 0, 'stone')\n", grid)
    # rotate_y(1): (x, z) -> (-z, x) => (2, 0) -> (0, 2)
    assert grid.get(2, 0, 0) is not None
    assert grid.get(0, 0, 2) is not None


def test_seeded_rng_is_deterministic():
    grid1 = VoxelGrid()
    grid2 = VoxelGrid()
    src = "for i in range(5):\n    set_block(rng.randint(0, 10), 0, 0, 'stone')\n"
    run_blueprint(src, grid1, seed=42)
    run_blueprint(src, grid2, seed=42)
    assert dict(grid1.items()) == dict(grid2.items())


def test_get_block_returns_none_for_untouched_cell():
    grid = VoxelGrid()
    run_blueprint("x = get_block(0, 0, 0)\nset_block(9, 9, 9, 'stone' if x is None else 'glass')\n", grid)
    assert grid.get(9, 9, 9) is not None
    from mcbuild.palette import get_block as resolve

    assert grid.get(9, 9, 9) == resolve("stone").index


def test_get_block_returns_name_of_earlier_placed_block():
    grid = VoxelGrid()
    run_blueprint(
        """
set_block(0, 0, 0, 'stone')
found = get_block(0, 0, 0)
set_block(1, 0, 0, found)
""",
        grid,
    )
    assert grid.get(0, 0, 0) == grid.get(1, 0, 0)


def test_get_block_round_trips_block_state():
    grid = VoxelGrid()
    run_blueprint(
        """
set_block(0, 0, 0, "oak_stairs[facing=north,half=top]")
again = get_block(0, 0, 0)
set_block(1, 0, 0, again)
""",
        grid,
    )
    assert grid.get(0, 0, 0) == grid.get(1, 0, 0)


def test_get_block_reads_through_active_transforms():
    grid = VoxelGrid()
    run_blueprint(
        """
set_block(1, 0, 0, 'stone')
with mirror('x', at=0):
    seen = get_block(1, 0, 0)  # world coord (-1, 0, 0): nothing placed there yet
    set_block(1, 0, 0, 'glass' if seen is None else 'stone')  # writes to (-1, 0, 0)
""",
        grid,
    )
    assert grid.get(-1, 0, 0) is not None
    from mcbuild.palette import get_block as resolve

    assert grid.get(-1, 0, 0) == resolve("glass").index


def test_get_block_sees_cleared_cell_as_none_but_explicit_air_as_air():
    grid = VoxelGrid()
    run_blueprint(
        """
set_block(0, 0, 0, 'stone')
set_block(1, 0, 0, 'stone')
clear(0, 0, 0, 0, 0, 0)         # removed entirely -> None
set_block(1, 0, 0, 'air')       # explicitly placed air -> still a block
after_clear = get_block(0, 0, 0)
after_air = get_block(1, 0, 0)
set_block(5, 5, 5, 'glass' if after_clear is None else 'stone')
set_block(6, 6, 6, 'diamond_block' if after_air == 'air' else 'stone')
""",
        grid,
    )
    from mcbuild.palette import get_block as resolve

    assert grid.get(5, 5, 5) == resolve("glass").index
    assert grid.get(6, 6, 6) == resolve("diamond_block").index
