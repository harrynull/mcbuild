from mcbuild.agent.loop import run_agent
from mcbuild.config import Config
from mcbuild.llm.fake import FakeLLM
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
