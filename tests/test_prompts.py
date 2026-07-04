from mcbuild.agent import prompts


def test_user_prompt_with_reference_demands_close_reproduction_and_analysis():
    p = prompts.build_user_prompt("a windmill", seed=0, has_reference=True)
    low = p.lower()
    assert "reproduce" in low
    assert "reference analysis" in low
    # asks for the structured extraction fields
    assert "storey" in low and "roof" in low and "materials" in low


def test_user_prompt_without_reference_has_no_reference_note():
    p = prompts.build_user_prompt("a windmill", seed=0, has_reference=False)
    assert "reference" not in p.lower()


def test_reference_image_prompt_is_framed_for_comparison():
    p = prompts.build_reference_image_prompt("a stone tower").lower()
    assert "isometric" in p
    assert "neutral background" in p
    assert "no people" in p


def test_system_prompt_states_inspect_and_query_are_free():
    sp = prompts.build_system_prompt().lower()
    assert "free" in sp and "budget" in sp
    assert "inspect" in sp and "query" in sp


def test_reference_critique_nudge_asks_for_discrepancies():
    n = prompts.build_reference_critique_nudge().lower()
    assert "discrepan" in n
    assert "reference" in n
