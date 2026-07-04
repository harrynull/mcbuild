"""Camera-independent visibility pre-filter shared by the iso and free-camera renderers."""

from __future__ import annotations

Coord = tuple[int, int, int]

_NEIGHBORS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def exposed_coords(occupied: set[Coord]) -> set[Coord]:
    """Return the subset of `occupied` with at least one empty axis-neighbor.

    A block with all 6 neighbors filled can't be seen from any camera, so this is a
    valid coarse cull for every camera direction.
    """
    exposed = set()
    for x, y, z in occupied:
        for dx, dy, dz in _NEIGHBORS:
            if (x + dx, y + dy, z + dz) not in occupied:
                exposed.add((x, y, z))
                break
    return exposed
