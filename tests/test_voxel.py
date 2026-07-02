import pytest

from mcbuild.voxel import MAX_BLOCKS, MAX_EXTENT, VoxelGrid, VoxelLimitError


def test_set_and_get():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 5)
    assert grid.get(0, 0, 0) == 5
    assert grid.get(1, 0, 0) is None
    assert len(grid) == 1


def test_overwrite_does_not_grow_count():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 5)
    grid.set(0, 0, 0, 7)
    assert len(grid) == 1
    assert grid.get(0, 0, 0) == 7


def test_bounds():
    grid = VoxelGrid()
    assert grid.bounds is None
    grid.set(1, 2, 3, 0)
    grid.set(-1, 5, 0, 0)
    assert grid.bounds == ((-1, 2, 0), (1, 5, 3))


def test_clear_block():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 1)
    grid.clear_block(0, 0, 0)
    assert grid.get(0, 0, 0) is None
    assert len(grid) == 0


def test_extent_guard():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 0)
    with pytest.raises(VoxelLimitError):
        grid.set(MAX_EXTENT, 0, 0, 0)


def test_to_dense_shape():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 2)
    grid.set(1, 1, 1, 3)
    arr, origin = grid.to_dense()
    assert origin == (0, 0, 0)
    assert arr.shape == (2, 2, 2)
    assert arr[0, 0, 0] == 3  # index 2 + 1
    assert arr[1, 1, 1] == 4  # index 3 + 1


def test_clone_is_independent_copy():
    grid = VoxelGrid()
    grid.set(0, 0, 0, 5)
    clone = grid.clone()
    clone.set(1, 1, 1, 7)
    assert grid.get(1, 1, 1) is None  # original untouched by clone's mutation
    assert clone.get(0, 0, 0) == 5  # clone carries original data
    assert len(grid) == 1 and len(clone) == 2


def test_clone_bounds_do_not_alias():
    grid = VoxelGrid()
    grid.set(-2, 0, 3, 1)
    clone = grid.clone()
    clone.set(10, 10, 10, 2)  # expands clone bounds; must not touch original's
    assert grid.bounds == ((-2, 0, 3), (-2, 0, 3))
    assert clone.bounds == ((-2, 0, 3), (10, 10, 10))
