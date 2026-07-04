from types import SimpleNamespace

from mcbuild.llm.client import OpenRouterClient, Usage, consume_stream


def _chunk(content=None, reasoning=None, reasoning_details=None, tool_calls=None, usage=None):
    delta = SimpleNamespace(
        content=content, reasoning=reasoning, reasoning_details=reasoning_details, tool_calls=tool_calls
    )
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def _tool_call_delta(index, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


def test_reasoning_details_fragments_merge_into_complete_signed_block():
    # a thinking block streams as text pieces first, signature last, all at index 0
    chunks = [
        _chunk(reasoning_details=[{"type": "reasoning.text", "text": "Let me ", "index": 0}]),
        _chunk(reasoning_details=[{"type": "reasoning.text", "text": "think.", "index": 0}]),
        _chunk(reasoning_details=[{"type": "reasoning.text", "signature": "SIG123", "index": 0}]),
    ]
    message, _ = consume_stream(chunks)
    assert message.reasoning_details is not None
    assert len(message.reasoning_details) == 1  # one merged block, not three fragments
    block = message.reasoning_details[0]
    assert block["text"] == "Let me think."
    assert block["signature"] == "SIG123"  # signature stays attached to the full text


def test_reasoning_details_separate_indices_stay_separate():
    chunks = [
        _chunk(reasoning_details=[{"type": "reasoning.text", "text": "A", "index": 0, "signature": "S0"}]),
        _chunk(reasoning_details=[{"type": "reasoning.text", "text": "B", "index": 1, "signature": "S1"}]),
    ]
    message, _ = consume_stream(chunks)
    assert message.reasoning_details is not None
    assert [b["signature"] for b in message.reasoning_details] == ["S0", "S1"]


def test_consume_stream_accumulates_content():
    chunks = [_chunk(content="Design "), _chunk(content="brief: "), _chunk(content="a hut.")]
    message, usage = consume_stream(chunks)
    assert message.content == "Design brief: a hut."
    assert message.tool_calls is None


def test_consume_stream_accumulates_reasoning_separately_from_content():
    chunks = [
        _chunk(reasoning="Thinking about "),
        _chunk(reasoning="the doorway."),
        _chunk(content="Here's my plan."),
    ]
    message, usage = consume_stream(chunks)
    assert message.reasoning == "Thinking about the doorway."
    assert message.content == "Here's my plan."


def test_consume_stream_fires_on_delta_callback_for_each_piece():
    chunks = [_chunk(reasoning="a"), _chunk(content="b"), _chunk(reasoning="c")]
    seen = []
    consume_stream(chunks, on_delta=lambda kind, text: seen.append((kind, text)))
    assert seen == [("reasoning", "a"), ("content", "b"), ("reasoning", "c")]


def test_consume_stream_accumulates_tool_call_arguments_across_chunks():
    chunks = [
        _chunk(tool_calls=[_tool_call_delta(0, id="call_1", name="submit_blueprint", arguments="")]),
        _chunk(tool_calls=[_tool_call_delta(0, arguments='{"code": "fill(')]),
        _chunk(tool_calls=[_tool_call_delta(0, arguments='0,0,0,1,1,1,\'stone\')", "design_notes": "x"}')]),
    ]
    message, usage = consume_stream(chunks)
    assert message.tool_calls is not None
    assert len(message.tool_calls) == 1
    tc = message.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.function.name == "submit_blueprint"
    assert tc.function.arguments == '{"code": "fill(0,0,0,1,1,1,\'stone\')", "design_notes": "x"}'


def test_consume_stream_handles_multiple_parallel_tool_calls_by_index():
    chunks = [
        _chunk(tool_calls=[_tool_call_delta(0, id="call_a", name="inspect", arguments="{}")]),
        _chunk(tool_calls=[_tool_call_delta(1, id="call_b", name="finish", arguments='{"summary":')]),
        _chunk(tool_calls=[_tool_call_delta(1, arguments='"done"}')]),
    ]
    message, usage = consume_stream(chunks)
    assert message.tool_calls is not None
    assert [tc.id for tc in message.tool_calls] == ["call_a", "call_b"]
    assert message.tool_calls[1].function.arguments == '{"summary":"done"}'


def test_consume_stream_captures_final_usage_chunk():
    usage_obj = SimpleNamespace(prompt_tokens=12, completion_tokens=34, cost=0.05)
    chunks = [_chunk(content="hi"), _chunk(usage=usage_obj)]
    message, usage = consume_stream(chunks)
    assert usage is usage_obj


def test_consume_stream_empty_yields_none_content_and_tool_calls():
    message, usage = consume_stream([])
    assert message.content is None
    assert message.reasoning is None
    assert message.tool_calls is None
    assert usage is None


def test_usage_cache_rate_is_fraction_of_cached_prompt_tokens():
    assert Usage(prompt_tokens=100, cached_tokens=25).cache_rate == 0.25
    assert Usage(prompt_tokens=0, cached_tokens=0).cache_rate == 0.0  # no div-by-zero on an empty turn


def test_usage_add_accumulates_cached_tokens():
    total = Usage()
    total.add(Usage(prompt_tokens=100, cached_tokens=80, cost_usd=0.01))
    total.add(Usage(prompt_tokens=50, cached_tokens=50, cost_usd=0.001))
    assert total.prompt_tokens == 150
    assert total.cached_tokens == 130
    assert total.cache_rate == 130 / 150


def test_client_session_id_defaults_to_a_generated_value_and_can_be_pinned():
    a = OpenRouterClient(api_key="test-key")
    b = OpenRouterClient(api_key="test-key")
    assert a.session_id and isinstance(a.session_id, str)
    assert a.session_id != b.session_id  # auto-generated ids don't collide

    pinned = OpenRouterClient(api_key="test-key", session_id="resume-me")
    assert pinned.session_id == "resume-me"


def test_usage_from_obj_parses_cached_tokens_for_cache_rate_reporting():
    client = OpenRouterClient(api_key="test-key")
    usage_obj = SimpleNamespace(
        prompt_tokens=1000,
        completion_tokens=50,
        cost=0.02,
        prompt_tokens_details=SimpleNamespace(cached_tokens=900),
    )
    usage = client._usage_from_obj(usage_obj)
    assert usage.cached_tokens == 900
    assert usage.cache_rate == 0.9


def test_usage_from_obj_defaults_cached_tokens_to_zero_when_absent():
    client = OpenRouterClient(api_key="test-key")
    usage_obj = SimpleNamespace(prompt_tokens=100, completion_tokens=10, cost=0.0)
    usage = client._usage_from_obj(usage_obj)
    assert usage.cached_tokens == 0
    assert usage.cache_rate == 0.0


def test_chat_blocking_sends_session_id_as_user_for_sticky_routing():
    client = OpenRouterClient(api_key="test-key", session_id="sticky-123")
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        message = SimpleNamespace(content="hi", tool_calls=None, reasoning=None, reasoning_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

    client._client.chat.completions.create = fake_create  # ty: ignore[invalid-assignment]
    client.chat(model="anthropic/claude-sonnet-5", messages=[{"role": "user", "content": "hi"}])

    assert captured["user"] == "sticky-123"
