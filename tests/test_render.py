from PIL import Image

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.render.iso import render_iso, render_topdown
from mcbuild.render.sixel import encode_sixel
from mcbuild.render.views import build_contact_sheet
from mcbuild.voxel import VoxelGrid


def _small_house_grid() -> VoxelGrid:
    grid = VoxelGrid()
    run_blueprint(
        """
walls(0, 0, 4, 4, 0, 3, 'cobblestone')
floor(0, 0, 4, 4, 0, 'oak_planks')
gable_roof(-1, -1, 5, 5, 4, 'spruce_planks', ridge_axis='x', overhang=1)
""",
        grid,
    )
    return grid


def test_render_iso_nonempty():
    grid = _small_house_grid()
    img = render_iso(grid, yaw=0)
    assert img.width > 0 and img.height > 0
    assert img.getbbox() is not None


def test_render_iso_deterministic():
    grid = _small_house_grid()
    img1 = render_iso(grid, yaw=0)
    img2 = render_iso(grid, yaw=0)
    assert img1.tobytes() == img2.tobytes()


def test_render_iso_all_yaws():
    grid = _small_house_grid()
    for yaw in range(4):
        img = render_iso(grid, yaw=yaw)
        assert img.width > 0 and img.height > 0


def test_render_topdown():
    grid = _small_house_grid()
    img = render_topdown(grid)
    assert img.width > 0 and img.height > 0


def test_empty_grid_does_not_crash():
    grid = VoxelGrid()
    img = render_iso(grid)
    assert isinstance(img, Image.Image)
    img2 = render_topdown(grid)
    assert isinstance(img2, Image.Image)


def test_contact_sheet_builds_and_has_stats():
    grid = _small_house_grid()
    sheet, stats = build_contact_sheet(grid)
    assert sheet.width <= 1568
    assert stats["block_count"] == len(grid)
    assert stats["dims"] is not None
    assert len(stats["top_materials"]) > 0


def test_sixel_encoding_smoke():
    grid = _small_house_grid()
    img = render_iso(grid, yaw=0)
    small = img.resize((32, 32))
    data = encode_sixel(small, max_colors=16)
    assert data.startswith("\x1bPq")
    assert data.endswith("\x1b\\")
