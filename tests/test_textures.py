from PIL import Image

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.render import textures
from mcbuild.render.iso import _sprite, render_iso
from mcbuild.voxel import VoxelGrid


def test_known_block_has_texture():
    tex = textures.get_face_texture("stone", "side")
    assert tex is not None
    assert tex.size == (16, 16)
    assert tex.mode == "RGBA"


def test_unknown_texture_falls_back_gracefully():
    tex = textures.get_face_texture("crying_obsidian_that_does_not_exist", "side")
    assert tex is None


def test_directional_texture_lookup_prefers_face_specific_file():
    top = textures.get_face_texture("oak_log", "top")
    side = textures.get_face_texture("oak_log", "side")
    assert top is not None and side is not None
    # oak_log_top.png and oak_log.png are genuinely different textures
    assert top.tobytes() != side.tobytes()


def test_grass_block_top_gets_tinted_green():
    assert textures.needs_tint("grass_block", "top")
    raw = textures.get_face_texture("grass_block", "top")
    assert raw is not None
    tinted = textures.apply_tint(raw, (127, 178, 56))
    # a grayscale tint mask multiplied by a green-dominant color should end up green-dominant
    r, g, b, _ = tinted.split()

    def avg(band):
        return sum(band.tobytes()) / (band.width * band.height)

    assert avg(g) > avg(r)
    assert avg(g) > avg(b)


def test_sprite_with_texture_differs_from_flat_color_sprite():
    textured_sprite = _sprite("stone", (125, 125, 125), 255)
    flat_sprite = _sprite("nonexistent_block_xyz", (125, 125, 125), 255)
    assert textured_sprite.tobytes() != flat_sprite.tobytes()


def test_textured_render_is_nonempty_and_deterministic():
    grid = VoxelGrid()
    run_blueprint("fill(0, 0, 0, 2, 2, 2, 'cobblestone')\nfloor(-2, -2, 4, 4, 0, 'grass_block')", grid)
    img1 = render_iso(grid, yaw=0)
    img2 = render_iso(grid, yaw=0)
    assert isinstance(img1, Image.Image)
    assert img1.getbbox() is not None
    assert img1.tobytes() == img2.tobytes()
