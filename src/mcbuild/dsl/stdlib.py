"""Blueprint builder primitives + composable transform-stack context managers.

`make_stdlib(grid, seed)` returns a dict of names to inject into a blueprint's
global namespace: shape primitives, `translate`/`mirror`/`rotate_y` transform
contexts, a safe `math` namespace, and a seeded `rng` wrapper.
"""

from __future__ import annotations

import math as _math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from mcbuild.palette import get_block
from mcbuild.voxel import VoxelGrid


class WeightedPalette:
    """A weighted mix of block names, sampled per placed cell via the seeded rng.

    Usable anywhere a block-name string is accepted (set_block, fill, walls, ...).
    Created in blueprints via the injected `weighted_block({...})` helper.
    """

    def __init__(self, spec: dict):
        if not spec:
            raise ValueError("weighted_block: spec must be a non-empty {name: weight} mapping.")
        self.names = list(spec.keys())
        self.weights = [float(w) for w in spec.values()]
        if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
            raise ValueError("weighted_block: weights must be non-negative and sum to > 0.")

    def sample(self, rng: random.Random) -> str:
        return rng.choices(self.names, weights=self.weights, k=1)[0]


# --- transform stack -----------------------------------------------------


@dataclass
class Translate:
    dx: float
    dy: float
    dz: float

    def apply(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        return x + self.dx, y + self.dy, z + self.dz


@dataclass
class Mirror:
    axis: str
    at: float

    def apply(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        if self.axis == "x":
            return 2 * self.at - x, y, z
        if self.axis == "y":
            return x, 2 * self.at - y, z
        if self.axis == "z":
            return x, y, 2 * self.at - z
        raise ValueError("mirror axis must be 'x', 'y', or 'z'")


@dataclass
class RotateY:
    quarters: int

    def apply(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        q = self.quarters % 4
        for _ in range(q):
            x, z = -z, x
        return x, y, z


Transform = Translate | Mirror | RotateY


def _iter_box(
    x1: float, y1: float, z1: float, x2: float, y2: float, z2: float
) -> Iterator[tuple[int, int, int]]:
    xlo, xhi = sorted((int(round(x1)), int(round(x2))))
    ylo, yhi = sorted((int(round(y1)), int(round(y2))))
    zlo, zhi = sorted((int(round(z1)), int(round(z2))))
    for x in range(xlo, xhi + 1):
        for y in range(ylo, yhi + 1):
            for z in range(zlo, zhi + 1):
                yield x, y, z


def make_stdlib(grid: VoxelGrid, seed: int = 0) -> dict[str, object]:
    stack: list[Transform] = []
    _random = random.Random(seed)
    _idx_cache: dict[str, int] = {}

    def to_world(x: float, y: float, z: float) -> tuple[int, int, int]:
        for t in reversed(stack):
            x, y, z = t.apply(x, y, z)
        return int(round(x)), int(round(y)), int(round(z))

    def _resolve_idx(block) -> int:
        """Resolve a block-name string or WeightedPalette to a palette index.

        WeightedPalette is sampled fresh each call (per placed cell); plain names
        are cached so hot loops don't re-hit the registry.
        """
        name = block.sample(_random) if isinstance(block, WeightedPalette) else block
        idx = _idx_cache.get(name)
        if idx is None:
            idx = get_block(name).index
            _idx_cache[name] = idx
        return idx

    def _place(x: float, y: float, z: float, idx: int) -> None:
        wx, wy, wz = to_world(x, y, z)
        grid.set(wx, wy, wz, idx)

    def _place_block(x: float, y: float, z: float, block) -> None:
        """Place resolving the block per-cell (so WeightedPalette varies per cell)."""
        _place(x, y, z, _resolve_idx(block))

    # --- transform contexts ---

    @contextmanager
    def translate(dx: float, dy: float = 0, dz: float = 0):
        stack.append(Translate(dx, dy, dz))
        try:
            yield
        finally:
            stack.pop()

    @contextmanager
    def mirror(axis: str, at: float = 0):
        stack.append(Mirror(axis, at))
        try:
            yield
        finally:
            stack.pop()

    @contextmanager
    def rotate_y(quarters: int):
        stack.append(RotateY(quarters))
        try:
            yield
        finally:
            stack.pop()

    # --- basic primitives ---

    def set_block(x: float, y: float, z: float, block) -> None:
        _place_block(x, y, z, block)

    def set_blocks(entries, block=None) -> None:
        """Place a batch of discrete cells in one call.

        entries: iterable of (x, y, z) or (x, y, z, block_name/WeightedPalette).
        block: default for 3-tuple entries; 4-tuples override it per entry.
        """
        for entry in entries:
            entry = tuple(entry)
            if len(entry) == 4:
                x, y, z, name = entry
                _place_block(x, y, z, name)
            elif len(entry) == 3:
                if block is None:
                    raise ValueError(
                        "set_blocks: a (x, y, z) entry needs a default `block=` argument."
                    )
                x, y, z = entry
                _place_block(x, y, z, block)
            else:
                raise ValueError(
                    f"set_blocks: each entry must be a 3-tuple (x,y,z) or 4-tuple "
                    f"(x,y,z,block), got length {len(entry)}."
                )

    def fill(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, block) -> None:
        for x, y, z in _iter_box(x1, y1, z1, x2, y2, z2):
            _place_block(x, y, z, block)

    def clear(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float) -> None:
        for x, y, z in _iter_box(x1, y1, z1, x2, y2, z2):
            wx, wy, wz = to_world(x, y, z)
            grid.clear_block(wx, wy, wz)

    def hollow_box(
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
        block,
        thickness: int = 1,
    ) -> None:
        xlo, xhi = sorted((int(round(x1)), int(round(x2))))
        ylo, yhi = sorted((int(round(y1)), int(round(y2))))
        zlo, zhi = sorted((int(round(z1)), int(round(z2))))
        t = max(1, int(thickness))
        for x, y, z in _iter_box(xlo, ylo, zlo, xhi, yhi, zhi):
            near_edge = (
                x - xlo < t
                or xhi - x < t
                or y - ylo < t
                or yhi - y < t
                or z - zlo < t
                or zhi - z < t
            )
            if near_edge:
                _place_block(x, y, z, block)

    def walls(
        x1: float, z1: float, x2: float, z2: float, y1: float, y2: float, block, thickness: int = 1
    ) -> None:
        xlo, xhi = sorted((int(round(x1)), int(round(x2))))
        zlo, zhi = sorted((int(round(z1)), int(round(z2))))
        ylo, yhi = sorted((int(round(y1)), int(round(y2))))
        t = max(1, int(thickness))
        for x in range(xlo, xhi + 1):
            for z in range(zlo, zhi + 1):
                near_edge = x - xlo < t or xhi - x < t or z - zlo < t or zhi - z < t
                if not near_edge:
                    continue
                for y in range(ylo, yhi + 1):
                    _place_block(x, y, z, block)

    def floor(x1: float, z1: float, x2: float, z2: float, y: float, block) -> None:
        xlo, xhi = sorted((int(round(x1)), int(round(x2))))
        zlo, zhi = sorted((int(round(z1)), int(round(z2))))
        for x in range(xlo, xhi + 1):
            for z in range(zlo, zhi + 1):
                _place_block(x, y, z, block)

    def line(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, block) -> None:
        x1i, y1i, z1i, x2i, y2i, z2i = (int(round(v)) for v in (x1, y1, z1, x2, y2, z2))
        dx, dy, dz = x2i - x1i, y2i - y1i, z2i - z1i
        steps = max(abs(dx), abs(dy), abs(dz), 1)
        for i in range(steps + 1):
            t = i / steps
            _place_block(round(x1i + dx * t), round(y1i + dy * t), round(z1i + dz * t), block)

    # --- curved / sloped shapes ---

    def cylinder(cx: float, cz: float, y: float, height: int, r: float, block, hollow: bool = False) -> None:
        ri = _math.ceil(r)
        height = int(height)
        for dx in range(-ri, ri + 1):
            for dz in range(-ri, ri + 1):
                dist = _math.sqrt(dx * dx + dz * dz)
                if dist > r + 0.5:
                    continue
                if hollow and dist < r - 0.5:
                    continue
                for h in range(height):
                    _place_block(cx + dx, y + h, cz + dz, block)

    def sphere(cx: float, cy: float, cz: float, r: float, block, hollow: bool = False) -> None:
        ri = _math.ceil(r)
        for dx in range(-ri, ri + 1):
            for dy in range(-ri, ri + 1):
                for dz in range(-ri, ri + 1):
                    dist = _math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > r + 0.5:
                        continue
                    if hollow and dist < r - 0.5:
                        continue
                    _place_block(cx + dx, cy + dy, cz + dz, block)

    def dome(cx: float, cy: float, cz: float, r: float, block, hollow: bool = False) -> None:
        ri = _math.ceil(r)
        for dx in range(-ri, ri + 1):
            for dy in range(0, ri + 1):
                for dz in range(-ri, ri + 1):
                    dist = _math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > r + 0.5:
                        continue
                    if hollow and dist < r - 0.5:
                        continue
                    _place_block(cx + dx, cy + dy, cz + dz, block)

    def pyramid(cx: float, cz: float, y: float, base: float, height: int, block, hollow: bool = False) -> None:
        base = float(base)
        height = int(height)
        for h in range(height):
            half = (base / 2.0) * (1 - h / height)
            half_i = round(half)
            if half_i < 0:
                continue
            for dx in range(-half_i, half_i + 1):
                for dz in range(-half_i, half_i + 1):
                    if hollow and abs(dx) < half_i and abs(dz) < half_i:
                        continue
                    _place_block(cx + dx, y + h, cz + dz, block)

    def cone(cx: float, cz: float, y: float, r: float, height: int, block, hollow: bool = False) -> None:
        r = float(r)
        height = int(height)
        for h in range(height):
            cur_r = r * (1 - h / height)
            ri = _math.ceil(cur_r)
            for dx in range(-ri, ri + 1):
                for dz in range(-ri, ri + 1):
                    dist = _math.sqrt(dx * dx + dz * dz)
                    if dist > cur_r + 0.5:
                        continue
                    if hollow and dist < cur_r - 0.5:
                        continue
                    _place_block(cx + dx, y + h, cz + dz, block)

    def gable_roof(
        x1: float, z1: float, x2: float, z2: float, y: float, block, ridge_axis: str = "x", overhang: int = 1
    ) -> None:
        x1i, x2i = sorted((int(round(x1)), int(round(x2))))
        z1i, z2i = sorted((int(round(z1)), int(round(z2))))
        overhang = int(overhang)
        y = int(round(y))
        if ridge_axis not in ("x", "z"):
            raise ValueError("ridge_axis must be 'x' or 'z'")
        if ridge_axis == "x":
            span_lo, span_hi = z1i, z2i
            along_lo, along_hi = x1i - overhang, x2i + overhang
        else:
            span_lo, span_hi = x1i, x2i
            along_lo, along_hi = z1i - overhang, z2i + overhang
        center = (span_lo + span_hi) / 2.0
        max_rise = _math.ceil((span_hi - span_lo) / 2.0 + overhang)
        for p in range(span_lo - overhang, span_hi + overhang + 1):
            rise = round(max_rise - abs(p - center))
            if rise < 0:
                continue
            yy = y + rise
            for a in range(along_lo, along_hi + 1):
                if ridge_axis == "x":
                    _place_block(a, yy, p, block)
                else:
                    _place_block(p, yy, a, block)

    def hip_roof(x1: float, z1: float, x2: float, z2: float, y: float, block, overhang: int = 1) -> None:
        x1i, x2i = sorted((int(round(x1)), int(round(x2))))
        z1i, z2i = sorted((int(round(z1)), int(round(z2))))
        overhang = int(overhang)
        y = int(round(y))
        cx = (x1i + x2i) / 2.0
        cz = (z1i + z2i) / 2.0
        half_x = (x2i - x1i) / 2.0 + overhang
        half_z = (z2i - z1i) / 2.0 + overhang
        max_rise = _math.ceil(min(half_x, half_z))
        for xx in range(x1i - overhang, x2i + overhang + 1):
            for zz in range(z1i - overhang, z2i + overhang + 1):
                dx = abs(xx - cx) / half_x if half_x else 0.0
                dz = abs(zz - cz) / half_z if half_z else 0.0
                d = max(dx, dz)
                rise = round(max_rise * (1 - d))
                if rise < 0:
                    continue
                _place_block(xx, y + rise, zz, block)

    # --- detail / texture helpers ---

    def weighted_block(spec: dict) -> WeightedPalette:
        """A weighted mix of blocks, sampled per cell — e.g. for weathering.

        weighted_block({"stone_bricks": 0.7, "cracked_stone_bricks": 0.2,
                        "mossy_stone_bricks": 0.1}) can be passed anywhere a block
        name is accepted (fill, walls, set_block, scatter, ...).
        """
        return WeightedPalette(spec)

    def scatter(x1, y1, z1, x2, y2, z2, block, density: float = 0.1) -> None:
        """Randomly place `block` at `density` fraction of cells in the box (weathering)."""
        density = max(0.0, min(1.0, float(density)))
        for x, y, z in _iter_box(x1, y1, z1, x2, y2, z2):
            if _random.random() < density:
                _place_block(x, y, z, block)

    def frame(x1, y1, z1, x2, y2, z2, block) -> None:
        """Place only the 12 edges of a box (corner posts + edge trim)."""
        xlo, xhi = sorted((int(round(x1)), int(round(x2))))
        ylo, yhi = sorted((int(round(y1)), int(round(y2))))
        zlo, zhi = sorted((int(round(z1)), int(round(z2))))
        for x, y, z in _iter_box(xlo, ylo, zlo, xhi, yhi, zhi):
            on_x_edge = x == xlo or x == xhi
            on_y_edge = y == ylo or y == yhi
            on_z_edge = z == zlo or z == zhi
            # an edge cell lies on at least two of the three pairs of faces
            if (on_x_edge + on_y_edge + on_z_edge) >= 2:
                _place_block(x, y, z, block)

    def window_grid(
        x1, z1, x2, z2, y1, y2, block, spacing: int = 2, margin: int = 1
    ) -> None:
        """Place a regular grid of `block` (e.g. glass) in a vertical wall plane.

        The wall must be axis-aligned (x1==x2 for a z-facing wall, or z1==z2 for an
        x-facing wall). Openings are spaced `spacing` apart, inset by `margin`.
        """
        x1i, x2i = sorted((int(round(x1)), int(round(x2))))
        z1i, z2i = sorted((int(round(z1)), int(round(z2))))
        ylo, yhi = sorted((int(round(y1)), int(round(y2))))
        spacing = max(1, int(spacing))
        margin = int(margin)
        if x1i == x2i:
            horiz = range(z1i + margin, z2i - margin + 1)
            wall_x = x1i
            for h in horiz:
                if (h - (z1i + margin)) % spacing != 0:
                    continue
                for yy in range(ylo + margin, yhi - margin + 1, spacing):
                    _place_block(wall_x, yy, h, block)
        elif z1i == z2i:
            horiz = range(x1i + margin, x2i - margin + 1)
            wall_z = z1i
            for h in horiz:
                if (h - (x1i + margin)) % spacing != 0:
                    continue
                for yy in range(ylo + margin, yhi - margin + 1, spacing):
                    _place_block(h, yy, wall_z, block)
        else:
            raise ValueError("window_grid: wall must be axis-aligned (x1==x2 or z1==z2).")

    # --- safe math + rng namespaces ---

    math_ns = SimpleNamespace(
        sin=_math.sin,
        cos=_math.cos,
        tan=_math.tan,
        sqrt=_math.sqrt,
        floor=_math.floor,
        ceil=_math.ceil,
        pi=_math.pi,
        tau=_math.tau,
        radians=_math.radians,
        degrees=_math.degrees,
        atan2=_math.atan2,
        hypot=_math.hypot,
        pow=_math.pow,
        log=_math.log,
        e=_math.e,
    )

    rng_ns = SimpleNamespace(
        randint=_random.randint,
        choice=_random.choice,
        uniform=_random.uniform,
        random=_random.random,
    )

    return {
        "set_block": set_block,
        "set_blocks": set_blocks,
        "fill": fill,
        "clear": clear,
        "hollow_box": hollow_box,
        "walls": walls,
        "floor": floor,
        "line": line,
        "cylinder": cylinder,
        "sphere": sphere,
        "dome": dome,
        "pyramid": pyramid,
        "cone": cone,
        "gable_roof": gable_roof,
        "hip_roof": hip_roof,
        "weighted_block": weighted_block,
        "scatter": scatter,
        "frame": frame,
        "window_grid": window_grid,
        "translate": translate,
        "mirror": mirror,
        "rotate_y": rotate_y,
        "math": math_ns,
        "rng": rng_ns,
    }
