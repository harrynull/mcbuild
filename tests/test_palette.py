import pytest

from mcbuild.palette import PaletteError, get_block, pop_warnings, reset_warnings, suggest


def test_get_known_block():
    block = get_block("oak_planks")
    assert block.mc_id == "minecraft:oak_planks"
    assert block.rgb == (162, 130, 78)


def test_get_block_with_minecraft_prefix():
    block = get_block("minecraft:stone")
    assert block.name == "stone"


def test_unknown_block_raises_with_suggestion():
    with pytest.raises(PaletteError) as exc_info:
        get_block("totally_bogus_block_xyz")
    assert "Unknown block" in str(exc_info.value)


def test_near_miss_typo_auto_corrects_with_warning():
    reset_warnings()
    block = get_block("oak_plank")  # missing 's' — close enough to auto-correct
    assert block.mc_id == "minecraft:oak_planks"
    warnings = pop_warnings()
    assert any("oak_plank" in w and "oak_planks" in w for w in warnings)


def test_suggest_returns_close_matches():
    matches = suggest("stoen")
    assert "stone" in matches


def test_air_is_valid_and_non_renderable():
    block = get_block("air")
    assert block.mc_id == "minecraft:air"
    assert block.renderable is False


def test_cave_air_still_excluded():
    with pytest.raises(PaletteError):
        get_block("cave_air")
