"""Sparse voxel grid: y-up integer coordinates, palette-indexed blocks."""

from __future__ import annotations

from collections.abc import Iterator

MAX_BLOCKS = 4_000_000
MAX_EXTENT = 512

Coord = tuple[int, int, int]


class VoxelLimitError(Exception):
    """Raised when a build would exceed the size/extent guards."""


class VoxelGrid:
    """Sparse dict-backed voxel grid: (x, y, z) -> palette index (u16-range int)."""

    def __init__(self) -> None:
        self._blocks: dict[Coord, int] = {}
        self._min: list[int] | None = None  # [minx, miny, minz]
        self._max: list[int] | None = None  # [maxx, maxy, maxz]

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, coord: Coord) -> bool:
        return coord in self._blocks

    def get(self, x: int, y: int, z: int) -> int | None:
        return self._blocks.get((x, y, z))

    def set(self, x: int, y: int, z: int, palette_index: int) -> None:
        coord = (x, y, z)
        if coord not in self._blocks:
            self._check_extent(x, y, z)
            if len(self._blocks) >= MAX_BLOCKS:
                raise VoxelLimitError(
                    f"Build exceeds the maximum of {MAX_BLOCKS:,} placed blocks."
                )
            self._expand_bounds(x, y, z)
        self._blocks[coord] = palette_index

    def clear_block(self, x: int, y: int, z: int) -> None:
        self._blocks.pop((x, y, z), None)
        # Bounds are allowed to be conservative (not shrunk) after a clear.

    def _check_extent(self, x: int, y: int, z: int) -> None:
        if self._min is None or self._max is None:
            return
        candidate_min = (min(self._min[0], x), min(self._min[1], y), min(self._min[2], z))
        candidate_max = (max(self._max[0], x), max(self._max[1], y), max(self._max[2], z))
        for lo, hi in zip(candidate_min, candidate_max, strict=True):
            if hi - lo + 1 > MAX_EXTENT:
                raise VoxelLimitError(
                    f"Build exceeds the maximum extent of {MAX_EXTENT} blocks along an axis."
                )

    def _expand_bounds(self, x: int, y: int, z: int) -> None:
        if self._min is None or self._max is None:
            self._min = [x, y, z]
            self._max = [x, y, z]
            return
        self._min[0] = min(self._min[0], x)
        self._min[1] = min(self._min[1], y)
        self._min[2] = min(self._min[2], z)
        self._max[0] = max(self._max[0], x)
        self._max[1] = max(self._max[1], y)
        self._max[2] = max(self._max[2], z)

    @property
    def bounds(self) -> tuple[Coord, Coord] | None:
        """Return ((minx, miny, minz), (maxx, maxy, maxz)), or None if empty."""
        if self._min is None or self._max is None:
            return None
        return (tuple(self._min), tuple(self._max))  # type: ignore[return-value]

    def items(self) -> Iterator[tuple[Coord, int]]:
        return iter(self._blocks.items())

    def to_dense(self) -> tuple["object", Coord]:
        """Return (numpy array of palette indices +1 (0 = air), origin offset)."""
        import numpy as np

        bounds = self.bounds
        if bounds is None:
            return np.zeros((0, 0, 0), dtype=np.uint16), (0, 0, 0)
        (minx, miny, minz), (maxx, maxy, maxz) = bounds
        w = maxx - minx + 1
        h = maxy - miny + 1
        d = maxz - minz + 1
        arr = np.zeros((w, h, d), dtype=np.uint16)
        for (x, y, z), idx in self._blocks.items():
            arr[x - minx, y - miny, z - minz] = idx + 1  # reserve 0 for air
        return arr, (minx, miny, minz)
