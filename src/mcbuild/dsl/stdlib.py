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

    def to_world(x: float, y: float, z: float) -> tuple[int, int, int]:
        for t in reversed(stack):
            x, y, z = t.apply(x, y, z)
        return int(round(x)), int(round(y)), int(round(z))

    def _place(x: float, y: float, z: float, idx: int) -> None:
        wx, wy, wz = to_world(x, y, z)
        grid.set(wx, wy, wz, idx)

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

    def set_block(x: float, y: float, z: float, block: str) -> None:
        idx = get_block(block).index
        _place(x, y, z, idx)

    def fill(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, block: str) -> None:
        idx = get_block(block).index
        for x, y, z in _iter_box(x1, y1, z1, x2, y2, z2):
            _place(x, y, z, idx)

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
        block: str,
        thickness: int = 1,
    ) -> None:
        idx = get_block(block).index
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
                _place(x, y, z, idx)

    def walls(
        x1: float, z1: float, x2: float, z2: float, y1: float, y2: float, block: str, thickness: int = 1
    ) -> None:
        idx = get_block(block).index
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
                    _place(x, y, z, idx)

    def floor(x1: float, z1: float, x2: float, z2: float, y: float, block: str) -> None:
        idx = get_block(block).index
        xlo, xhi = sorted((int(round(x1)), int(round(x2))))
        zlo, zhi = sorted((int(round(z1)), int(round(z2))))
        for x in range(xlo, xhi + 1):
            for z in range(zlo, zhi + 1):
                _place(x, y, z, idx)

    def line(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, block: str) -> None:
        idx = get_block(block).index
        x1i, y1i, z1i, x2i, y2i, z2i = (int(round(v)) for v in (x1, y1, z1, x2, y2, z2))
        dx, dy, dz = x2i - x1i, y2i - y1i, z2i - z1i
        steps = max(abs(dx), abs(dy), abs(dz), 1)
        for i in range(steps + 1):
            t = i / steps
            _place(round(x1i + dx * t), round(y1i + dy * t), round(z1i + dz * t), idx)

    # --- curved / sloped shapes ---

    def cylinder(cx: float, cz: float, y: float, height: int, r: float, block: str, hollow: bool = False) -> None:
        idx = get_block(block).index
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
                    _place(cx + dx, y + h, cz + dz, idx)

    def sphere(cx: float, cy: float, cz: float, r: float, block: str, hollow: bool = False) -> None:
        idx = get_block(block).index
        ri = _math.ceil(r)
        for dx in range(-ri, ri + 1):
            for dy in range(-ri, ri + 1):
                for dz in range(-ri, ri + 1):
                    dist = _math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > r + 0.5:
                        continue
                    if hollow and dist < r - 0.5:
                        continue
                    _place(cx + dx, cy + dy, cz + dz, idx)

    def dome(cx: float, cy: float, cz: float, r: float, block: str, hollow: bool = False) -> None:
        idx = get_block(block).index
        ri = _math.ceil(r)
        for dx in range(-ri, ri + 1):
            for dy in range(0, ri + 1):
                for dz in range(-ri, ri + 1):
                    dist = _math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist > r + 0.5:
                        continue
                    if hollow and dist < r - 0.5:
                        continue
                    _place(cx + dx, cy + dy, cz + dz, idx)

    def pyramid(cx: float, cz: float, y: float, base: float, height: int, block: str, hollow: bool = False) -> None:
        idx = get_block(block).index
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
                    _place(cx + dx, y + h, cz + dz, idx)

    def cone(cx: float, cz: float, y: float, r: float, height: int, block: str, hollow: bool = False) -> None:
        idx = get_block(block).index
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
                    _place(cx + dx, y + h, cz + dz, idx)

    def gable_roof(
        x1: float, z1: float, x2: float, z2: float, y: float, block: str, ridge_axis: str = "x", overhang: int = 1
    ) -> None:
        idx = get_block(block).index
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
                    _place(a, yy, p, idx)
                else:
                    _place(p, yy, a, idx)

    def hip_roof(x1: float, z1: float, x2: float, z2: float, y: float, block: str, overhang: int = 1) -> None:
        idx = get_block(block).index
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
                _place(xx, y + rise, zz, idx)

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

    _random = random.Random(seed)
    rng_ns = SimpleNamespace(
        randint=_random.randint,
        choice=_random.choice,
        uniform=_random.uniform,
        random=_random.random,
    )

    return {
        "set_block": set_block,
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
        "translate": translate,
        "mirror": mirror,
        "rotate_y": rotate_y,
        "math": math_ns,
        "rng": rng_ns,
    }
