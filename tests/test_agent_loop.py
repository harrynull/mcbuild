import json

from mcbuild.agent.loop import _with_prompt_caching, run_agent
from mcbuild.config import Config
from mcbuild.llm.fake import FakeLLM, _FakeMessage, _FnCall, _ToolCall
from mcbuild.palette import get_block_by_index
from mcbuild.rundir import RunDir


def test_agent_loop_error_then_fix_then_finish(tmp_path):
    llm = FakeLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))

    events: list[tuple[str, dict]] = []

    def on_event(event_type, data):
        events.append((event_type, data))

    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=on_event)

    assert result.finished is True
    assert "hut" in result.summary.lower()
    assert result.grid is not None
    assert len(result.grid) > 0
    assert result.iterations == 2  # one failed submit + one successful submit

    event_types = [e[0] for e in events]
    assert "blueprint_error" in event_types
    assert "render" in event_types
    assert "finish" in event_types

    # error occurs before the successful render
    assert event_types.index("blueprint_error") < event_types.index("render")

    # the fixed-submit turn carries both reasoning and commentary text alongside its
    # tool call — both must surface as events, not just get swallowed into session.json
    assert "reasoning" in event_types
    assert "assistant_text" in event_types
    reasoning_event = next(d for t, d in events if t == "reasoning")
    assistant_text_event = next(d for t, d in events if t == "assistant_text")
    assert "stone" in reasoning_event["text"].lower()
    assert "doorway" in assistant_text_event["text"].lower()

    session_path = rundir.root / "session.json"
    assert session_path.exists()

    iter1 = rundir.root / "iter_01" / "blueprint.py"
    iter2 = rundir.root / "iter_02" / "blueprint.py"
    assert iter1.exists()
    assert iter2.exists()
    assert (rundir.root / "iter_02" / "render.png").exists()
    assert (rundir.root / "iter_02" / "stats.json").exists()
    assert (rundir.root / "iter_02" / "blueprint.schem").exists()
    # the failed iteration never produced a voxel grid, so no schem for it
    assert not (rundir.root / "iter_01" / "blueprint.schem").exists()


def test_agent_loop_respects_cost_ceiling(tmp_path):
    llm = FakeLLM()  # each turn adds 10 prompt + 10 completion tokens, cost_usd stays 0 by default
    llm.total_usage.cost_usd = 0.0

    class ExpensiveFakeLLM(FakeLLM):
        def chat(self, *args, **kwargs):
            result = super().chat(*args, **kwargs)
            self.total_usage.cost_usd += 1.0
            return result

    llm = ExpensiveFakeLLM()
    config = Config(max_iters=6, seed=1, cost_ceiling=0.5)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))

    events: list[tuple[str, dict]] = []
    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    assert result.finished is False
    assert "cost ceiling" in result.summary.lower()
    assert ("abort", {"reason": "cost ceiling of $0.50 reached"}) in events


def test_agent_loop_emits_turn_start_and_deltas_when_streaming(tmp_path):
    class StreamingFakeLLM(FakeLLM):
        def chat(self, model, messages, tools=None, reasoning="off", stream=False, on_delta=None, **kwargs):
            result = super().chat(model, messages, tools=tools, reasoning=reasoning)
            if stream and on_delta:
                msg = result.message
                reasoning = getattr(msg, "reasoning", None)
                if reasoning:
                    on_delta("reasoning", reasoning)
                content = getattr(msg, "content", None)
                if content:
                    on_delta("content", content)
            return result

    llm = StreamingFakeLLM()
    config = Config(max_iters=6, seed=1, stream=True)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))

    events: list[tuple[str, dict]] = []
    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    assert result.finished is True
    event_types = [e[0] for e in events]
    # design brief, broken submit, fixed submit, verify query, verify inspect, finish
    assert event_types.count("turn_start") == 6
    assert "reasoning_delta" in event_types
    assert "content_delta" in event_types

    reasoning_delta_text = "".join(d["text"] for t, d in events if t == "reasoning_delta")
    content_delta_text = "".join(d["text"] for t, d in events if t == "content_delta")
    assert "stone" in reasoning_delta_text.lower()
    assert "doorway" in content_delta_text.lower()


def test_agent_loop_respects_max_iters(tmp_path):
    class AlwaysBrokenLLM(FakeLLM):
        def _fixed_submit(self):
            return self._broken_submit()

        def _finish(self):
            return self._broken_submit()

    llm = AlwaysBrokenLLM()
    config = Config(max_iters=2, seed=1, max_consecutive_failures=10)
    rundir = RunDir.create("broken build", base=str(tmp_path))

    result = run_agent("broken build", llm, config, rundir)
    assert result.finished is False


def _tool_msg(name, args, call_id="c"):
    return _FakeMessage(tool_calls=[_ToolCall(id=call_id, function=_FnCall(name=name, arguments=json.dumps(args)))])


def test_stale_reasoning_stripped_from_all_but_latest_assistant():
    from mcbuild.agent.loop import _strip_stale_reasoning

    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a1", "reasoning_details": [{"text": "t1", "signature": "S1"}]},
        {"role": "tool", "tool_call_id": "x", "content": "r"},
        {"role": "assistant", "content": "a2", "reasoning_details": [{"text": "t2", "signature": "S2"}]},
    ]
    _strip_stale_reasoning(messages)
    assert "reasoning_details" not in messages[1]  # older turn stripped
    assert messages[3]["reasoning_details"][0]["signature"] == "S2"  # latest kept intact


def _tool_texts(rundir) -> list[str]:
    """All tool-role message contents from the saved session, for asserting result text."""
    data = json.loads((rundir.root / "session.json").read_text())
    return [m["content"] for m in data if m.get("role") == "tool" and isinstance(m.get("content"), str)]


def _finish_msg():
    return _tool_msg("finish", {"summary": "done", "completed_interior_check": True}, call_id="cf")


def test_success_result_includes_bounds_and_budget(tmp_path):
    llm = FakeLLM()  # design_brief, broken_submit, fixed_submit(success), finish
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir)
    texts = " \n ".join(_tool_texts(rundir))
    assert "bounds=[x" in texts  # min corner exposed, not just dims
    assert "Edits remaining:" in texts


def _str_replace(old_str, new_str, notes="x", submit=True):
    return _tool_msg(
        "str_replace",
        {"old_str": old_str, "new_str": new_str, "design_notes": notes, "submit": submit, "views": [{"yaw": 0}]},
    )


# the source _fixed_submit hands to submit_blueprint (used as the old_str anchor below)
_FIXED_SUBMIT_CODE = "walls(0, 0, 4, 4, 0, 2, 'stone')\nfloor(0, 0, 4, 4, 0, 'stone')\nclear(2, 1, 0, 2, 2, 0)"


def test_budget_caps_successful_builds(tmp_path):
    class SpendyLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._fixed_submit,  # build 1
                self._patch,  # build 2 (budget=2 reached)
                self._patch,  # should be REJECTED (budget exhausted)
                self._verify_query,
                self._verify_inspect,
                self._finish_ok,
            ]

        def _patch(self):
            return _str_replace("clear(2, 1, 0, 2, 2, 0)", "clear(2, 1, 0, 2, 2, 0)\nset_block(0, 4, 0, 'glass')")

        def _finish_ok(self):
            return _finish_msg()

    llm = SpendyLLM()
    config = Config(max_iters=2, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    assert any("Edit budget reached" in t for t in _tool_texts(rundir))


def test_failed_build_does_not_consume_budget(tmp_path):
    # one failure then two successful builds must all fit within a budget of 2
    class FailThenBuildLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._broken_submit,
                self._fixed_submit,
                self._patch,
                self._verify_query,
                self._verify_inspect,
                self._finish_ok,
            ]

        def _patch(self):
            return _str_replace("clear(2, 1, 0, 2, 2, 0)", "clear(2, 1, 0, 2, 2, 0)\nset_block(0, 4, 0, 'glass')")

        def _finish_ok(self):
            return _finish_msg()

    llm = FailThenBuildLLM()
    config = Config(max_iters=2, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    texts = _tool_texts(rundir)
    # the failed submit is annotated as not using an edit, and the two real builds both ran
    assert any("did NOT use an edit" in t for t in texts)
    assert not any("Edit budget reached" in t for t in texts)


def test_build_without_views_is_rejected_then_recovers(tmp_path):
    class NoViewsFirstLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._no_views_submit,
                self._fixed_submit,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _no_views_submit(self):
            return _tool_msg("submit_blueprint", {"code": _FIXED_SUBMIT_CODE, "design_notes": "no views"})

    llm = NoViewsFirstLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    assert any("at least one view" in t for t in _tool_texts(rundir))


def test_malformed_view_specs_rejected_as_tool_errors_not_crashes(tmp_path):
    # garbage view values must come back as fixable tool errors, never abort the run
    bad_views = [
        [{"yaw": "north"}],  # non-numeric yaw
        ["top-down"],  # non-dict entry
        [{"yaw": 0, "cutaway": "y"}],  # invalid cutaway axis
        [{"yaw": 0, "cutaway": "x"}],  # camera doesn't face the cut
        [{"slice_axis": "y"}],  # slice_axis without slice_at
        [{"yaw": 0}] * 9,  # over the per-build cap
    ]

    class BadViewsLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                *[self._bad_submit(v) for v in bad_views],
                self._fixed_submit,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _bad_submit(self, views):
            return lambda: _tool_msg(
                "submit_blueprint", {"code": _FIXED_SUBMIT_CODE, "design_notes": "x", "views": views}
            )

    llm = BadViewsLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    texts = _tool_texts(rundir)
    assert any("yaw must be an integer" in t for t in texts)
    assert any("each view must be an object" in t for t in texts)
    assert any("cutaway must be 'x' or 'z'" in t for t in texts)
    assert any("faces the cut" in t for t in texts)
    assert any("provided together" in t for t in texts)
    assert any("Too many views" in t for t in texts)
    # none of the rejected submits consumed an edit or ran a blueprint
    assert result.iterations == 1


def test_cutaway_view_on_build_satisfies_finish_interior_check(tmp_path):
    # _fixed_submit requests a cutaway view, so no separate inspect call should be needed
    class NoInspectLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._fixed_submit, self._verify_query, self._finish]

    llm = NoInspectLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    assert not any("Cannot finish" in t for t in _tool_texts(rundir))


def test_str_replace_result_reports_block_delta(tmp_path):
    class DeltaLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._fixed_submit, self._patch, self._finish_ok]

        def _patch(self):
            return _str_replace(
                "clear(2, 1, 0, 2, 2, 0)",
                "clear(2, 1, 0, 2, 2, 0)\nfill(0, 5, 0, 3, 5, 3, 'glass')",
                notes="roof",
            )

        def _finish_ok(self):
            return _finish_msg()

    llm = DeltaLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir)
    assert any("delta: +16 added" in t for t in _tool_texts(rundir))


def test_noop_str_replace_delta_flags_no_change(tmp_path):
    class NoopLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._fixed_submit, self._patch, self._finish_ok]

        def _patch(self):
            # clear a region with nothing in it -> no change
            return _str_replace(
                "clear(2, 1, 0, 2, 2, 0)",
                "clear(2, 1, 0, 2, 2, 0)\nclear(50, 50, 50, 52, 52, 52)",
                notes="noop",
            )

        def _finish_ok(self):
            return _finish_msg()

    llm = NoopLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir)
    assert any("NO CHANGE" in t for t in _tool_texts(rundir))


def test_str_replace_edits_cumulative_source(tmp_path):
    class ReplacingFakeLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._patch,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _patch(self):
            # cut a window into the existing wall
            return _str_replace(
                "clear(2, 1, 0, 2, 2, 0)", "clear(2, 1, 0, 2, 2, 0)\nset_block(2, 2, 4, 'air')", notes="window"
            )

    llm = ReplacingFakeLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    # iter_01 = successful submit, iter_02 = str_replace
    iter2_bp = (rundir.root / "iter_02" / "blueprint.py").read_text()
    assert "walls(" in iter2_bp  # original submit code preserved
    assert "set_block(2, 2, 4, 'air')" in iter2_bp  # replacement present
    assert not (rundir.root / "iter_02" / "patch.py").exists()  # str_replace has no patch.py, only edit_region does


def test_str_replace_submit_false_stages_without_building(tmp_path):
    class StagingLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._stage,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _stage(self):
            return _str_replace(
                "clear(2, 1, 0, 2, 2, 0)",
                "clear(2, 1, 0, 2, 2, 0)\nset_block(2, 2, 4, 'air')",
                notes="window (staged)",
                submit=False,
            )

    llm = StagingLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    texts = _tool_texts(rundir)
    assert any("staged" in t.lower() and "not built" in t.lower() for t in texts)
    # no second iteration was ever built/rendered for the staged edit
    assert not (rundir.root / "iter_02").exists()
    # the staged edit never changed the exported grid (only submit=true builds do): (2,2,4) is
    # a wall block from the original submit, still stone rather than the staged 'air'
    assert result.grid is not None
    idx = result.grid.get(2, 2, 4)
    assert idx is not None
    assert get_block_by_index(idx).name != "air"


def test_str_replace_submit_false_does_not_spend_budget(tmp_path):
    class StageThenBuildLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._fixed_submit,
                self._stage,
                self._build,
                self._verify_query,
                self._verify_inspect,
                self._finish_ok,
            ]

        def _stage(self):
            return _str_replace(
                "clear(2, 1, 0, 2, 2, 0)",
                "clear(2, 1, 0, 2, 2, 0)\nset_block(2, 2, 4, 'air')",
                submit=False,
            )

        def _build(self):
            return _str_replace(
                "set_block(2, 2, 4, 'air')",
                "set_block(2, 2, 4, 'air')\nset_block(0, 4, 0, 'glass')",
                submit=True,
            )

        def _finish_ok(self):
            return _finish_msg()

    llm = StageThenBuildLLM()
    config = Config(max_iters=2, seed=1)  # would be exhausted if staging spent budget too
    rundir = RunDir.create("hut", base=str(tmp_path))
    result = run_agent("hut", llm, config, rundir)
    assert result.finished is True
    assert not any("Edit budget reached" in t for t in _tool_texts(rundir))
    assert result.grid is not None
    # both the staged (submit=False) edit and the later submitted edit made it into the final grid
    idx_air = result.grid.get(2, 2, 4)
    idx_glass = result.grid.get(0, 4, 0)
    assert idx_air is not None and idx_glass is not None
    assert get_block_by_index(idx_air).name == "air"
    assert get_block_by_index(idx_glass).name == "glass"


def test_str_replace_old_str_not_found_returns_error(tmp_path):
    class BadAnchorLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._patch,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _patch(self):
            return _str_replace("this text does not exist in the source", "irrelevant")

    llm = BadAnchorLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    assert any("old_str not found" in t for t in _tool_texts(rundir))


def test_str_replace_old_str_not_unique_returns_error(tmp_path):
    class AmbiguousAnchorLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._patch,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _patch(self):
            # "stone'" appears twice in _FIXED_SUBMIT_CODE (walls + floor)
            return _str_replace("'stone'", "'stone_bricks'")

    llm = AmbiguousAnchorLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    assert any("matches 2 locations" in t for t in _tool_texts(rundir))


def test_str_replace_before_any_submit_returns_error(tmp_path):
    class EarlyReplaceLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._patch_first, self._finish]

        def _patch_first(self):
            return _str_replace("fill(0,0,0,1,1,1,'stone')", "fill(0,0,0,1,1,1,'stone_bricks')")

    llm = EarlyReplaceLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("x", base=str(tmp_path))
    result = run_agent("x", llm, config, rundir)
    assert result.grid is None  # nothing was ever built


def test_failing_str_replace_leaves_best_grid_untouched(tmp_path):
    class BadReplaceLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._bad_patch,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _bad_patch(self):
            return _str_replace("clear(2, 1, 0, 2, 2, 0)", "set_block(0,0,0,'not_a_real_block')")

    llm = BadReplaceLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    assert result.grid is not None and len(result.grid) > 0  # the good submit grid survives


def _session_messages(rundir):
    return json.loads((rundir.root / "session.json").read_text())


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)


def test_images_are_never_pruned(tmp_path):
    # Image pruning was removed: it saved tokens but invalidated the prompt cache from that
    # point forward on every turn (a much bigger cost). All renders, including old inspects,
    # must stay intact in history so the cached prefix stays stable.
    class InspectHeavyLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._fixed_submit,
                self._insp,
                self._insp,
                self._insp,
                self._finish_ok,
            ]

        def _insp(self):
            return _tool_msg("inspect", {"yaw": 1})

        def _finish_ok(self):
            return _tool_msg("finish", {"summary": "done", "completed_interior_check": True}, call_id="cf")

    llm = InspectHeavyLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir)

    msgs = _session_messages(rundir)
    sheet_msgs = [
        m
        for m in msgs
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "Renderings included below" in p.get("text", "") for p in m["content"])
    ]
    assert sheet_msgs, "contact-sheet critique message not found"
    assert _has_image(sheet_msgs[-1]["content"])  # latest sheet keeps its image

    inspect_msgs = [
        m
        for m in msgs
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "Inspection view" in p.get("text", "") for p in m["content"])
    ]
    assert len(inspect_msgs) == 3
    assert all(_has_image(m["content"]) for m in inspect_msgs)  # none pruned to a placeholder
    assert not any(
        isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "pruned" in p.get("text", "") for p in m["content"])
        for m in msgs
    )


def test_reference_reattached_at_critique(tmp_path):
    from PIL import Image as PILImage

    ref = PILImage.new("RGB", (128, 128), (100, 140, 90))
    llm = FakeLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir, reference_image=ref)

    msgs = _session_messages(rundir)
    critique = [
        m
        for m in msgs
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "REFERENCE" in p.get("text", "") for p in m["content"])
    ]
    assert critique, "reference-aware critique message not found"
    # both the reference thumbnail and the build render are attached together
    img_parts = [p for p in critique[-1]["content"] if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(img_parts) == 2


def test_inspect_free_camera_mode(tmp_path):
    class FreeCamLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._inspect,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _inspect(self):
            return _tool_msg("inspect", {"camera_pos": [10, 5, 10], "look_at": [2, 1, 2]})

    llm = FreeCamLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    events = []
    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    assert result.finished is True
    inspect_events = [d for t, d in events if t == "inspect"]
    assert inspect_events and inspect_events[0].get("mode") == "camera"


def test_query_tool_returns_text_ground_truth(tmp_path):
    class QueryLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._fixed_submit,
                self._query,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _query(self):
            return _tool_msg("query", {"mode": "histogram"})

    llm = QueryLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    events = []
    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    assert result.finished is True
    query_events = [d for t, d in events if t == "query"]
    assert query_events and query_events[0]["mode"] == "histogram"
    assert "stone" in query_events[0]["text"]


def test_edit_region_clears_then_rebuilds_only_the_box(tmp_path):
    class EditRegionLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [
                self._design_brief,
                self._submit_slab,
                self._edit,
                self._verify_query,
                self._verify_inspect,
                self._finish,
            ]

        def _submit_slab(self):
            # a 5x1x5 stone slab at y=0
            return _tool_msg(
                "submit_blueprint",
                {"code": "fill(0,0,0,4,0,4,'stone')", "design_notes": "slab", "views": [{"yaw": 0}]},
            )

        def _edit(self):
            # replace the middle 3x1x3 region with glass
            return _tool_msg(
                "edit_region",
                {
                    "region": [1, 0, 1, 3, 0, 3],
                    "code": "fill(1,0,1,3,0,3,'glass')",
                    "design_notes": "glass center",
                    "views": [{"yaw": 0}],
                },
            )

    from mcbuild.palette import get_block

    llm = EditRegionLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("slab", base=str(tmp_path))
    result = run_agent("slab", llm, config, rundir)

    assert result.finished is True
    grid = result.grid
    assert grid is not None
    assert grid.get(2, 0, 2) == get_block("glass").index  # center replaced
    assert grid.get(0, 0, 0) == get_block("stone").index  # corner (outside region) frozen


def test_prompt_caching_marks_system_and_latest_user_message():
    messages = [
        {"role": "system", "content": "SYSTEM PROMPT"},
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "1", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": "result"},
        {"role": "user", "content": "second (plain string)"},
    ]
    out = _with_prompt_caching(messages)

    # system prompt: wrapped into a content block with a cache breakpoint
    assert out[0]["content"] == [{"type": "text", "text": "SYSTEM PROMPT", "cache_control": {"type": "ephemeral"}}]
    # first (older) user message is untouched — only the latest user message gets a breakpoint
    assert out[1]["content"] == [{"type": "text", "text": "first"}]
    # latest user message: plain string wrapped + breakpoint
    assert out[4]["content"] == [
        {"type": "text", "text": "second (plain string)", "cache_control": {"type": "ephemeral"}}
    ]
    # original messages list is never mutated
    assert messages[0]["content"] == "SYSTEM PROMPT"
    assert messages[4]["content"] == "second (plain string)"


def test_prompt_caching_breakpoint_on_last_block_of_list_content():
    messages = [
        {"role": "system", "content": "SYS"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "data:..."}},
            ],
        },
    ]
    out = _with_prompt_caching(messages)
    assert "cache_control" not in out[1]["content"][0]
    assert out[1]["content"][1]["cache_control"] == {"type": "ephemeral"}


def test_turn_usage_event_emitted_each_llm_call(tmp_path):
    events: list[tuple[str, dict]] = []
    llm = FakeLLM()  # design_brief, broken_submit, fixed_submit, verify_query, verify_inspect, finish = 6 turns
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    turn_events = [d for t, d in events if t == "turn_usage"]
    assert len(turn_events) == 6
    assert [d["turn"] for d in turn_events] == [1, 2, 3, 4, 5, 6]
    for d in turn_events:
        assert d["prompt_tokens"] == 10
        assert d["completion_tokens"] == 10
        assert "reasoning_tokens" in d and "cost_usd" in d and "cumulative_cost_usd" in d
        # FakeLLM never reports cached tokens, so the rate should read as a clean 0.0, not an error
        assert d["cached_tokens"] == 0
        assert d["cache_rate"] == 0.0
    # cumulative cost is non-decreasing across turns
    cumulative = [d["cumulative_cost_usd"] for d in turn_events]
    assert cumulative == sorted(cumulative)
