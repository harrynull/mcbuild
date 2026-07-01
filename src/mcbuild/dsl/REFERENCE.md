# Blueprint DSL Reference

Your blueprint is a sandboxed Python program. It has no `import`, no file/network
access, and no access to `_`-prefixed names or attributes — just the building
primitives below, a safe `math` namespace, a seeded `rng`, and plain Python
control flow (`for`, `while`, `if`, `def`, list/dict comprehensions, etc).

Coordinates are integers, **y-up**: `x` and `z` are horizontal, `y` is vertical
(up is `+y`). Place your build starting near the origin `(0, 0, 0)`.

Blocks are given as bare name strings, e.g. `"oak_planks"`, `"stone_bricks"`,
`"glass"`. An unknown name raises an error with suggestions — fix the name and
resubmit.

## Primitives

```python
set_block(x, y, z, block)
fill(x1, y1, z1, x2, y2, z2, block)                       # solid box, corners in any order
clear(x1, y1, z1, x2, y2, z2)                              # remove blocks in a box
hollow_box(x1, y1, z1, x2, y2, z2, block, thickness=1)      # box shell (all 6 sides)
walls(x1, z1, x2, z2, y1, y2, block, thickness=1)           # 4 vertical walls, no floor/ceiling
floor(x1, z1, x2, z2, y, block)                             # flat rectangular plate at height y
line(x1, y1, z1, x2, y2, z2, block)                         # 3D straight line

cylinder(cx, cz, y, height, r, block, hollow=False)         # vertical cylinder, base at y
sphere(cx, cy, cz, r, block, hollow=False)
dome(cx, cy, cz, r, block, hollow=False)                    # upper hemisphere, base at cy
pyramid(cx, cz, y, base, height, block, hollow=False)       # square pyramid, base at y
cone(cx, cz, y, r, height, block, hollow=False)             # circular cone, base at y

gable_roof(x1, z1, x2, z2, y, block, ridge_axis='x', overhang=1)  # A-frame roof
hip_roof(x1, z1, x2, z2, y, block, overhang=1)                     # 4-sided sloped roof
```

## Transform contexts

Composable — nest them to build symmetric structures without recomputing coordinates:

```python
with translate(dx, dy, dz):
    ...                      # all coordinates inside are offset by (dx, dy, dz)

with mirror('x', at=5):
    ...                      # reflect the x coordinate about x=5 (also 'y', 'z')

with rotate_y(quarters):
    ...                      # rotate around the vertical axis by 90*quarters degrees
```

## Utilities

```python
math.sin/cos/tan/sqrt/floor/ceil/pi/tau/radians/degrees/atan2/hypot/pow/log/e
rng.randint(a, b) / rng.choice(seq) / rng.uniform(a, b) / rng.random()
```

## Worked examples

**Simple house shell with a gable roof:**

```python
walls(0, 0, 8, 6, 0, 4, "cobblestone", thickness=1)
floor(0, 0, 8, 6, 0, "oak_planks")
clear(3, 1, 0, 4, 3, 0)  # doorway
gable_roof(-1, -1, 9, 7, 5, "spruce_planks", ridge_axis="x", overhang=1)
```

**A tower with a domed cap:**

```python
cylinder(0, 0, 0, height=12, r=4, block="stone_bricks", hollow=True)
floor(-4, -4, 4, 4, 0, "stone_bricks")
dome(0, 12, 0, r=4, block="stone_bricks")
```

**Symmetric wings using `mirror`:**

```python
# central hall
fill(-2, 0, -2, 2, 3, 2, "quartz_block")
clear(-1, 1, -2, 1, 2, -2)

# one wing, mirrored to build the other automatically
def wing():
    fill(2, 0, -1, 6, 3, 1, "quartz_block")
    clear(2, 1, -0, 2, 2, 0)

wing()
with mirror('x', at=0):
    wing()
```

**Repeating a shape with a loop + `rotate_y`:**

```python
for i in range(4):
    with rotate_y(i):
        with translate(6, 0, 0):
            cylinder(0, 0, 0, height=6, r=1, block="cobblestone")  # 4 corner turrets
```
