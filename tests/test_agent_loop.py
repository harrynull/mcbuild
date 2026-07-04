import json

from mcbuild.agent.loop import run_agent
from mcbuild.config import Config
from mcbuild.llm.fake import FakeLLM, _FnCall, _FakeMessage, _ToolCall
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
                if getattr(msg, "reasoning", None):
                    on_delta("reasoning", msg.reasoning)
                if getattr(msg, "content", None):
                    on_delta("content", msg.content)
            return result

    llm = StreamingFakeLLM()
    config = Config(max_iters=6, seed=1, stream=True)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))

    events: list[tuple[str, dict]] = []
    result = run_agent("a tiny stone hut", llm, config, rundir, on_event=lambda t, d: events.append((t, d)))

    assert result.finished is True
    event_types = [e[0] for e in events]
    assert event_types.count("turn_start") == 4  # design brief, broken submit, fixed submit, finish
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
    return _FakeMessage(
        tool_calls=[_ToolCall(id=call_id, function=_FnCall(name=name, arguments=json.dumps(args)))]
    )


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


def _str_replace(old_str, new_str, notes="x"):
    return _tool_msg("str_replace", {"old_str": old_str, "new_str": new_str, "design_notes": notes})


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
            self._script = [self._broken_submit, self._fixed_submit, self._patch, self._finish_ok]

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
            self._script = [self._design_brief, self._fixed_submit, self._patch, self._finish]

        def _patch(self):
            # cut a window into the existing wall
            return _str_replace("clear(2, 1, 0, 2, 2, 0)", "clear(2, 1, 0, 2, 2, 0)\nset_block(2, 2, 4, 'air')", notes="window")

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


def test_str_replace_old_str_not_found_returns_error(tmp_path):
    class BadAnchorLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._design_brief, self._fixed_submit, self._patch, self._finish]

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
            self._script = [self._design_brief, self._fixed_submit, self._patch, self._finish]

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
            self._script = [self._design_brief, self._fixed_submit, self._bad_patch, self._finish]

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
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content
    )


def test_pruning_protects_latest_contact_sheet(tmp_path):
    # 3 inspects after a build would push the whole-building sheet out of a keep-last-2 window;
    # it must be protected so the overview survives while detail work continues.
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
        m for m in msgs
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "4 isometric angles" in p.get("text", "") for p in m["content"])
    ]
    assert sheet_msgs, "contact-sheet critique message not found"
    assert _has_image(sheet_msgs[-1]["content"])  # latest sheet keeps its image
    # at least one older inspect image was pruned to a placeholder
    pruned = sum(
        1
        for m in msgs
        if isinstance(m.get("content"), list)
        and any(isinstance(p, dict) and "pruned" in p.get("text", "") for p in m["content"])
    )
    assert pruned >= 1


def test_reference_reattached_at_critique(tmp_path):
    from PIL import Image as PILImage

    ref = PILImage.new("RGB", (128, 128), (100, 140, 90))
    llm = FakeLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("hut", base=str(tmp_path))
    run_agent("hut", llm, config, rundir, reference_image=ref)

    msgs = _session_messages(rundir)
    critique = [
        m for m in msgs
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
            self._script = [self._design_brief, self._fixed_submit, self._inspect, self._finish]

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
            self._script = [self._design_brief, self._fixed_submit, self._query, self._finish]

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
            self._script = [self._design_brief, self._submit_slab, self._edit, self._finish]

        def _submit_slab(self):
            # a 5x1x5 stone slab at y=0
            return _tool_msg(
                "submit_blueprint",
                {"code": "fill(0,0,0,4,0,4,'stone')", "design_notes": "slab"},
            )

        def _edit(self):
            # replace the middle 3x1x3 region with glass
            return _tool_msg(
                "edit_region",
                {"region": [1, 0, 1, 3, 0, 3], "code": "fill(1,0,1,3,0,3,'glass')", "design_notes": "glass center"},
            )

    from mcbuild.palette import get_block

    llm = EditRegionLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("slab", base=str(tmp_path))
    result = run_agent("slab", llm, config, rundir)

    assert result.finished is True
    grid = result.grid
    assert grid.get(2, 0, 2) == get_block("glass").index  # center replaced
    assert grid.get(0, 0, 0) == get_block("stone").index  # corner (outside region) frozen
