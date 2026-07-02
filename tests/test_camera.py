import numpy as np
import pytest

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.palette import get_block
from mcbuild.render import blockmodel, blockstate, culling
from mcbuild.render.camera import Camera, CameraRenderError, render_from_camera
from mcbuild.voxel import VoxelGrid


def _house():
    grid = VoxelGrid()
    run_blueprint(
        "walls(0,0,4,4,0,3,'cobblestone')\nfloor(0,0,4,4,0,'oak_planks')",
        grid,
    )
    return grid


# --- culling ---


def test_exposed_coords_skips_fully_enclosed():
    grid = VoxelGrid()
    run_blueprint("fill(0,0,0,2,2,2,'stone')", grid)  # 3x3x3, center enclosed
    occ = {c for c, _ in grid.items()}
    exposed = culling.exposed_coords(occ)
    assert (1, 1, 1) not in exposed  # center hidden
    assert (0, 0, 0) in exposed  # corner visible
    assert len(exposed) == 26  # 27 minus the 1 enclosed center


# --- blockstate parsing ---


def test_blockstate_variant_match_stairs():
    parts = blockstate.resolve_parts("oak_stairs", {"facing": "north", "half": "bottom", "shape": "straight"})
    assert parts and parts[0].model == "oak_stairs"


def test_blockstate_multipart_fence_selects_matching_sides():
    parts = blockstate.resolve_parts("oak_fence", {"north": "true", "east": "false", "south": "false", "west": "false"})
    models = [p.model for p in parts]
    assert any("fence_post" in m for m in models)
    assert any("fence_side" in m for m in models)


def test_blockstate_unknown_block_returns_none():
    assert blockstate.resolve_parts("definitely_not_a_block", {}) is None


# --- mesh geometry ---


def test_full_cube_mesh_has_six_faces():
    mesh = blockmodel.get_block_mesh("stone", ())
    assert mesh is not None and len(mesh) == 6


def test_stairs_mesh_has_more_faces_than_a_cube():
    state = tuple(sorted({"facing": "north", "half": "bottom", "shape": "straight"}.items()))
    mesh = blockmodel.get_block_mesh("oak_stairs", state)
    assert mesh is not None and len(mesh) > 6  # two boxes -> more faces than a single cube


def test_base_texture_resolution():
    assert blockmodel._base_texture("oak_stairs") == "oak_planks"
    assert blockmodel._base_texture("cobblestone_slab") == "cobblestone"
    assert blockmodel._base_texture("stone_brick_stairs") == "stone_bricks"


# --- rendering: angles, determinism, depth ---


def test_render_nonempty_and_deterministic():
    grid = _house()
    cam = Camera(position=(12, 9, 12), look_at=(2, 1, 2))
    a = render_from_camera(grid, cam)
    b = render_from_camera(grid, cam)
    assert a.getbbox() is not None
    assert a.tobytes() == b.tobytes()


def test_render_oblique_37_degree_angle():
    grid = _house()
    cam = Camera(position=(15, 8, 3), look_at=(2, 1, 2))  # not a 90-degree multiple
    assert render_from_camera(grid, cam).getbbox() is not None


def test_render_straight_down_and_up_do_not_crash():
    grid = _house()
    down = render_from_camera(grid, Camera(position=(2, 30, 2), look_at=(2, 0, 2)))
    up = render_from_camera(grid, Camera(position=(2, -30, 2), look_at=(2, 0, 2)))
    assert down.getbbox() is not None
    assert isinstance(up.width, int)


def test_render_from_inside_hollow_structure():
    grid = VoxelGrid()
    run_blueprint("hollow_box(0,0,0,6,6,6,'stone',thickness=1)", grid)
    cam = Camera(position=(3, 3, 3), look_at=(6, 3, 3))
    assert render_from_camera(grid, cam).getbbox() is not None


def test_depth_ordering_near_block_occludes_far():
    grid = VoxelGrid()
    grid.set(0, 0, 0, get_block("red_wool").index)
    grid.set(0, 0, 10, get_block("blue_wool").index)
    cam = Camera(position=(0, 0, -6), look_at=(0, 0, 10), view_size=6)
    img = render_from_camera(grid, cam, 64, 64)

    red = np.array(get_block("red_wool").rgb)
    blue = np.array(get_block("blue_wool").rgb)
    arr = np.array(img.convert("RGBA"))
    opaque = arr[arr[..., 3] > 0][:, :3].astype(int)
    assert len(opaque) > 0

    def close(colors, target):
        return (np.abs(colors - target).sum(axis=1) < 60).any()

    assert close(opaque, red * 0.72)  # near red face visible (side shade)
    assert not close(opaque, blue * 0.72)  # far blue fully occluded


def test_shadows_darken_and_are_deterministic():
    # a raised roof over a floor: the roof should cast a shadow, darkening some pixels
    grid = VoxelGrid()
    run_blueprint("floor(0,0,8,8,0,'stone')\nfloor(2,2,6,6,5,'oak_planks')", grid)
    cam = Camera(position=(20, 16, 20), look_at=(4, 1, 4), view_size=16)

    lit = np.array(render_from_camera(grid, cam, shadows=False).convert("RGB")).astype(int)
    shad = np.array(render_from_camera(grid, cam, shadows=True).convert("RGB")).astype(int)
    assert lit.shape == shad.shape
    # shadows only ever darken, and must darken at least some pixels
    assert (shad <= lit + 1).all()
    assert (shad < lit - 5).any()
    # deterministic
    again = np.array(render_from_camera(grid, cam, shadows=True).convert("RGB")).astype(int)
    assert (shad == again).all()


def test_exceeds_max_blocks_raises():
    grid = VoxelGrid()
    run_blueprint("floor(0,0,20,20,0,'stone')", grid)  # 441 exposed top cells
    with pytest.raises(CameraRenderError):
        render_from_camera(grid, Camera(position=(10, 40, 10), look_at=(10, 0, 10)), max_blocks=100)
