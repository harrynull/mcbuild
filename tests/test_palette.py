import pytest

from mcbuild.palette import PaletteError, get_block, suggest


def test_get_known_block():
    block = get_block("oak_planks")
    assert block.mc_id == "minecraft:oak_planks"
    assert block.rgb == (162, 130, 78)


def test_get_block_with_minecraft_prefix():
    block = get_block("minecraft:stone")
    assert block.name == "stone"


def test_unknown_block_raises_with_suggestion():
    with pytest.raises(PaletteError) as exc_info:
        get_block("oak_plank")  # missing 's'
    assert "oak_planks" in str(exc_info.value)


def test_suggest_returns_close_matches():
    matches = suggest("stoen")
    assert "stone" in matches
