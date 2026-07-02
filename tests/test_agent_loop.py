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


def test_patch_blueprint_appends_to_cumulative_source(tmp_path):
    class PatchingFakeLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._design_brief, self._fixed_submit, self._patch, self._finish]

        def _patch(self):
            # cut a window into the existing wall
            return _tool_msg("patch_blueprint", {"code": "set_block(2, 2, 4, 'air')", "design_notes": "window"})

    llm = PatchingFakeLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    # iter_01 = successful submit, iter_02 = patch
    iter2_bp = (rundir.root / "iter_02" / "blueprint.py").read_text()
    assert "walls(" in iter2_bp  # original submit code preserved
    assert "set_block(2, 2, 4, 'air')" in iter2_bp  # patch appended
    assert (rundir.root / "iter_02" / "patch.py").exists()
    assert not (rundir.root / "iter_01" / "patch.py").exists()  # submit iters have no patch.py


def test_patch_before_any_submit_returns_error(tmp_path):
    class EarlyPatchLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._patch_first, self._finish]

        def _patch_first(self):
            return _tool_msg("patch_blueprint", {"code": "fill(0,0,0,1,1,1,'stone')", "design_notes": "x"})

    llm = EarlyPatchLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("x", base=str(tmp_path))
    result = run_agent("x", llm, config, rundir)
    assert result.grid is None  # nothing was ever built


def test_failing_patch_leaves_best_grid_untouched(tmp_path):
    class BadPatchLLM(FakeLLM):
        def __init__(self):
            super().__init__()
            self._script = [self._design_brief, self._fixed_submit, self._bad_patch, self._finish]

        def _bad_patch(self):
            return _tool_msg("patch_blueprint", {"code": "set_block(0,0,0,'not_a_real_block')", "design_notes": "x"})

    llm = BadPatchLLM()
    config = Config(max_iters=6, seed=1)
    rundir = RunDir.create("a tiny stone hut", base=str(tmp_path))
    result = run_agent("a tiny stone hut", llm, config, rundir)

    assert result.finished is True
    assert result.grid is not None and len(result.grid) > 0  # the good submit grid survives


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
