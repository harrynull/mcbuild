import numpy as np
from PIL import Image

from mcbuild.dsl.sandbox import run_blueprint
from mcbuild.render.camera import Camera, render_from_camera
from mcbuild.render.iso import _slice_keep, render_iso, render_topdown
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


_ALL_CLASSIC_VIEWS = [
    {"yaw": 0},
    {"yaw": 1},
    {"yaw": 2},
    {"yaw": 3},
    {"mode": "top-down"},
    {"yaw": 2, "cutaway": "x"},
    {"yaw": 1, "cutaway": "z"},
]


def test_contact_sheet_builds_and_has_stats():
    grid = _small_house_grid()
    sheet, labels, stats = build_contact_sheet(grid, _ALL_CLASSIC_VIEWS)
    assert sheet.width <= 1100  # MAX_WIDTH
    assert len(labels) == len(_ALL_CLASSIC_VIEWS)
    assert stats["block_count"] == len(grid)
    assert stats["dims"] is not None
    assert len(stats["top_materials"]) > 0


def test_contact_sheet_renders_a_single_requested_view():
    # the at-least-one-view requirement itself is enforced (and tested) in the agent loop
    grid = _small_house_grid()
    sheet, labels, _stats = build_contact_sheet(grid, [{"yaw": 0}])
    assert labels == ["yaw 0deg"]
    assert sheet.width > 0 and sheet.height > 0


def test_contact_sheet_cutaway_yaws_actually_face_the_cut():
    # The contact sheet's cutaway tiles must use a yaw where the camera faces the cut plane —
    # clip keeps the FAR half, so a wrong yaw just shows a smaller-looking, uncut exterior.
    grid = VoxelGrid()
    run_blueprint(
        "fill(0,0,0,7,7,7,'stone')\nfill(3,3,3,4,4,4,'diamond_block')",
        grid,
    )

    def _has_diamond(img: Image.Image) -> bool:
        rgb = np.array(img.convert("RGB"))
        blue_over_red = rgb[..., 2].astype(int) - rgb[..., 0].astype(int)
        return bool(((blue_over_red > 20) & (rgb[..., 2] > 60)).any())

    assert _has_diamond(render_iso(grid, yaw=2, clip="x"))  # matches views.py's cutaway x
    assert _has_diamond(render_iso(grid, yaw=1, clip="z"))  # matches views.py's cutaway z
    # the previous (wrong) yaw=0 for both really did hide it, confirming this isn't a no-op check
    assert not _has_diamond(render_iso(grid, yaw=0, clip="x"))
    assert not _has_diamond(render_iso(grid, yaw=0, clip="z"))


def test_contact_sheet_has_no_overlaid_text():
    from mcbuild.render.views import BG

    grid = _small_house_grid()
    sheet, _labels, _stats = build_contact_sheet(grid, _ALL_CLASSIC_VIEWS)
    # 7 requested views lay out as 4 cols x 2 rows of square cells (last cell unused) with no
    # footer/label row appended below, so the sheet's aspect ratio must stay exactly 2:1
    # regardless of any final MAX_WIDTH downscale
    assert sheet.width == 2 * sheet.height
    corner = sheet.getpixel((sheet.width - 1, sheet.height - 1))
    assert corner == BG


def test_render_iso_y_slice_shows_only_upper_storeys():
    grid = VoxelGrid()
    # two stacked floors: y=0 and y=4
    run_blueprint("floor(0,0,5,5,0,'stone')\nfloor(0,0,5,5,4,'oak_planks')", grid)
    full = render_iso(grid, yaw=0)
    sliced = render_iso(grid, yaw=0, slice_spec=("y", 4))  # keep y>=4 only
    # slicing away the ground floor changes the image and leaves it non-empty
    assert sliced.getbbox() is not None
    assert full.tobytes() != sliced.tobytes()


def test_render_iso_slice_above_everything_is_empty():
    grid = VoxelGrid()
    run_blueprint("floor(0,0,3,3,0,'stone')", grid)
    sliced = render_iso(grid, yaw=0, slice_spec=("y", 99))
    assert sliced.getbbox() is None  # nothing kept


def test_cutaway_reveals_hidden_interior_material():
    # a solid stone cube with a fully-enclosed diamond core: a correct cutaway must expose
    # the core at the cut plane, not skip straight through to the far exterior wall.
    grid = VoxelGrid()
    run_blueprint(
        "fill(0,0,0,7,7,7,'stone')\nfill(3,3,3,4,4,4,'diamond_block')",
        grid,
    )
    keep = _slice_keep(grid, "x", None)  # keeps x >= 3, removing the near half
    # camera sits in the removed void looking straight down +x into the cut face
    cam = Camera(position=(-6, 3.5, 3.5), look_at=(10, 3.5, 3.5), view_size=10)
    img = render_from_camera(grid, cam, keep=keep)
    rgb = np.array(img.convert("RGB"))
    # diamond_block's teal is nowhere close to stone's gray; presence proves the core
    # was actually rendered at the cut plane rather than culled away.
    blue_over_red = rgb[..., 2].astype(int) - rgb[..., 0].astype(int)
    assert bool(((blue_over_red > 20) & (rgb[..., 2] > 60)).any())


def test_sixel_encoding_smoke():
    grid = _small_house_grid()
    img = render_iso(grid, yaw=0)
    small = img.resize((32, 32))
    data = encode_sixel(small, max_colors=16)
    assert data.startswith("\x1bPq")
    assert data.endswith("\x1b\\")
