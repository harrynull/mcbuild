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

Blocks can carry a **state** in `[key=value,...]` form for orientation and shape — this is
how you get real architectural detail. Stairs, slabs, walls, fences, fence gates, trapdoors,
doors and panes all render with their true geometry:

```python
set_block(3, 1, 0, "oak_stairs[facing=south,half=bottom,shape=straight]")
set_block(0, 2, 0, "oak_slab[type=top]")          # top half-slab
set_block(5, 1, 5, "cobblestone_wall")             # wall post + auto side arms
fill(0, 3, 0, 6, 3, 0, "spruce_trapdoor[facing=north,half=top,open=false]")  # eaves/awning
```

Common states: stairs — `facing=north|south|east|west, half=bottom|top, shape=straight|inner_left|inner_right|outer_left|outer_right`;
slabs — `type=bottom|top|double`; trapdoors/doors — `facing=..., half=..., open=true|false`.
Use stairs/slabs for rooflines and steps, walls/fences for railings, trapdoors for shutters
and trim — this is where believable detail comes from.

`"air"` is a valid block for carving/erasing. Unlike `clear()` (which only removes
cells from your local build and leaves whatever was already there on paste),
explicitly placing `"air"` (via `set_block`/`fill`/`set_blocks`) is written into the
exported `.schem` and actively clears/overwrites existing terrain at that position
when pasted with WorldEdit. Use `clear()` for build-local cleanup; place `"air"` when
you specifically want the build to cut into surrounding terrain (e.g. a doorway into
a hillside).

## Patching an existing build

When refining a build you already submitted (via the `patch_blueprint` or `edit_region` tools),
your code runs against the CURRENT voxel state, not a fresh world — write only the delta (e.g.
`clear(...)` a wall, `fill(...)` a new room, place `'air'` to carve). Two things do NOT carry
over between calls: transform contexts (`with translate/mirror/rotate_y` are fresh each call) and
any Python variables or functions you defined earlier (gone — talk to the grid only through
primitives). `edit_region` additionally clears its bounding box before your snippet runs, so you
can redo one region without disturbing the rest.

## Primitives

```python
set_block(x, y, z, block)
set_blocks(entries, block=None)                            # batch: entries are (x,y,z) or (x,y,z,name)
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

## Detail helpers

Use these to break up flat, monotone surfaces — the difference between a blockout and a
finished build.

```python
weighted_block({name: weight, ...})   # a block "value" sampled per cell; pass it anywhere a
                                       # block is accepted, e.g. for weathering / texture variation
scatter(x1, y1, z1, x2, y2, z2, block, density=0.1)   # random sprinkle of block in a box
frame(x1, y1, z1, x2, y2, z2, block)                  # just the 12 edges of a box (trim/posts)
window_grid(x1, z1, x2, z2, y1, y2, block, spacing=2, margin=1)  # regular openings in an
                                                                 # axis-aligned wall plane
```

Example — a weathered stone-brick wall (70% clean, 20% cracked, 10% mossy):

```python
wall = weighted_block({"stone_bricks": 0.7, "cracked_stone_bricks": 0.2, "mossy_stone_bricks": 0.1})
walls(0, 0, 10, 6, 0, 5, wall)
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

**Using `math` — items evenly around a circle (polar coordinates):**

`math` and `rng` are already available — no import. Here, glowstone posts are placed evenly
around a circular courtyard by converting an angle to (x, z) with `cos`/`sin`, collected into a
list, and placed in one `set_blocks` call:

```python
n = 8          # number of posts
r = 6          # courtyard radius
posts = [
    (round(r * math.cos(2 * math.pi * i / n)), 1, round(r * math.sin(2 * math.pi * i / n)), "glowstone")
    for i in range(n)
]
set_blocks(posts)
```

**Flagship: an asymmetric L-shaped cottage (real buildings are rarely perfectly symmetric).**

Vary massing, opening rhythm, and materials between faces instead of mirroring everything:

```python
# main wing (long, low)
walls(0, 0, 9, 5, 0, 4, "cobblestone")
floor(0, 0, 9, 5, 0, "oak_planks")
# perpendicular wing, offset — makes the footprint an L, not a box
walls(6, 3, 10, 8, 0, 5, "cobblestone")
floor(6, 3, 10, 8, 0, "oak_planks")
# irregular window rhythm on the long face (not evenly spaced)
set_blocks([(0, 2, 2), (0, 2, 3), (0, 2, 6)], block="glass")
# a single off-center door
clear(2, 1, 9, 3, 2, 9)
# weathered roofs in two materials so the wings read differently
gable_roof(-1, -1, 10, 6, 5, "spruce_planks", ridge_axis="x", overhang=1)
gable_roof(5, 2, 9, 11, 6, "dark_oak_planks", ridge_axis="z", overhang=1)
# a bit of moss on the north wall for age
scatter(0, 0, 0, 5, 4, 0, "mossy_cobblestone", density=0.15)
```

Guidance: real buildings vary massing, openings, and materials between faces. Prefer irregular
footprints (L / T / cross shapes), offset towers, and uneven window spacing over perfect mirror
symmetry. Use `weighted_block`/`scatter` to weather large surfaces so they don't read as flat.

NOTE:
1. Do NOT use import statements — `math`, `rng`, and every primitive/transform listed above are
   already injected as ready-to-use globals; nothing in this DSL ever needs an import.
2. Do NOT call rng.seed. It has been pre-seeded for you.
