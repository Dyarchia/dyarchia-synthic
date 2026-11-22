"""Offline tests of the Anthropic provider: hand-built SSE streams served through ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import httpx
import pytest

from synthic.provider import API_VERSION, AnthropicProvider, cost
from synthic.types import ModelRequest, ModelSpec, ProviderError, Usage

SPEC = ModelSpec(
    alias="opus",
    id="claude-opus-5-5",
    context_window=1_000_000,
    max_output_tokens=128_000,
    price_input=4.0,
    price_output=20.0,
    price_cache_write=5.0,
    price_cache_read=0.2,
    price_cache_write_1h=8.0,
)


def sse(*events: dict[str, Any], eol: str = "\n") -> bytes:
    return "".join(f"event: {e['type']}{eol}data: {json.dumps(e)}{eol}{eol}" for e in events).encode()


def start(usage: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": SPEC.id,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage or {"input_tokens": 10, "output_tokens": 1},
        },
    }


def block_start(index: int, block: dict[str, Any]) -> dict[str, Any]:
    return {"type": "content_block_start", "index": index, "content_block": block}


def json_delta(index: int, part: str) -> dict[str, Any]:
    return {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": part}}


def tool_use(tool_id: str, name: str, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input or {}}


def text_events(text: str = "Hello world", stop: str = "end_turn") -> list[dict[str, Any]]:
    half = len(text) // 2
    return [
        start(),
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:half]}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[half:]}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ]


def ok(body: bytes | None = None) -> httpx.Response:
    content = body if body is not None else sse(*text_events())
    return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})


def error_response(
    status: int, kind: str, headers: dict[str, str] | None = None, code: str | None = None
) -> httpx.Response:
    error: dict[str, Any] = {"type": kind, "message": "nope"}
    if code:
        error["details"] = {"error_code": code}
    return httpx.Response(status, json={"type": "error", "error": error, "request_id": "req_1"}, headers=headers)


def streamed(chunks: list[bytes], tail: Callable[[], Any] | None = None) -> httpx.Response:
    async def body() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk
        if tail is not None:
            await tail()

    return httpx.Response(200, content=body(), headers={"content-type": "text/event-stream"})


class Server:
    """Serves the queued responses in order and records every request it receives."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, n: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[n].content)


def make(server: Server, **kwargs: Any) -> tuple[AnthropicProvider, list[float]]:
    sleeps: list[float] = []
    now = [0.0]

    async def sleep(delay: float) -> None:
        sleeps.append(delay)
        now[0] += delay

    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    kwargs.setdefault("api_key", "test-key")
    return AnthropicProvider(client=client, sleep=sleep, clock=lambda: now[0], **kwargs), sleeps


def request(**kwargs: Any) -> ModelRequest:
    kwargs.setdefault("system", [{"type": "text", "text": "You are a test."}])
    kwargs.setdefault("messages", [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}])
    kwargs.setdefault("tools", [])
    return ModelRequest(model=SPEC, **kwargs)


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def test_text_reply_headers_and_on_text() -> None:
    server = Server(ok())
    provider, _ = make(server)
    deltas: list[str] = []
    reply = run(provider.complete(request(), on_text=deltas.append))
    assert reply.content == [{"type": "text", "text": "Hello world"}]
    assert reply.stop_reason == "end_turn"
    assert reply.model_id == SPEC.id
    assert deltas == ["Hello", " world"]
    sent = server.requests[0]
    assert str(sent.url) == "https://api.anthropic.com/v1/messages"
    assert sent.headers["x-api-key"] == "test-key"
    assert sent.headers["anthropic-version"] == API_VERSION
    assert "anthropic-beta" not in sent.headers
    body = server.body()
    assert body["stream"] is True
    assert body["model"] == SPEC.id
    assert body["max_tokens"] == SPEC.max_output_tokens
    assert "tools" not in body
    assert "output_config" not in body


def thinking_and_tools_events() -> list[dict[str, Any]]:
    return [
        start(),
        block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "check."}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBsig=="}},
        {"type": "content_block_stop", "index": 0},
        block_start(1, {"type": "redacted_thinking", "data": "opaque"}),
        {"type": "content_block_stop", "index": 1},
        block_start(2, tool_use("toolu_a", "read")),
        block_start(3, tool_use("toolu_b", "ls")),
        json_delta(2, ""),
        json_delta(2, "{\"path\": \"src/"),
        json_delta(2, "a.cls\", \"lines\": [1, "),
        json_delta(2, "2]}"),
        {"type": "content_block_stop", "index": 2},
        json_delta(3, ""),
        {"type": "content_block_stop", "index": 3},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 89}},
        {"type": "message_stop"},
    ]


EXPECTED_TOOL_TURN = [
    {"type": "thinking", "thinking": "Let me check.", "signature": "EqQBsig=="},
    {"type": "redacted_thinking", "data": "opaque"},
    tool_use("toolu_a", "read", {"path": "src/a.cls", "lines": [1, 2]}),
    tool_use("toolu_b", "ls"),
]


def test_thinking_and_parallel_tool_calls() -> None:
    server = Server(ok(sse(*thinking_and_tools_events())))
    provider, _ = make(server)
    deltas: list[str] = []
    reply = run(provider.complete(request(), on_text=deltas.append))
    assert reply.content == EXPECTED_TOOL_TURN
    assert reply.stop_reason == "tool_use"
    assert deltas == []


def test_sse_parsing_one_byte_at_a_time_with_crlf_pings_comments_and_multiline_data() -> None:
    events = thinking_and_tools_events()
    raw = sse(*events[:3], eol="\r\n")
    raw += b": keep-alive comment\r\n\r\nevent: ping\r\ndata: {\"type\": \"ping\"}\r\n\r\n"
    raw += b"event: future_event\ndata: {\"type\": \"future_event\"}\n\n"
    raw += sse(*events[3:-1], eol="\r")
    raw += b"event: message_stop\ndata: {\"type\":\ndata:  \"message_stop\"}\n\n"
    accented = sse(*text_events("café ñandú"))
    server = Server(streamed([bytes([b]) for b in raw]), streamed([bytes([b]) for b in accented]))
    provider, _ = make(server)
    assert run(provider.complete(request())).content == EXPECTED_TOOL_TURN
    assert run(provider.complete(request())).content == [{"type": "text", "text": "café ñandú"}]


def test_usage_is_cumulative_and_costed() -> None:
    events = text_events()
    events[0] = start(
        {"input_tokens": 100, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 5000, "output_tokens": 1}
    )
    events[-2] = {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}}
    first_delta = {"type": "message_delta", "delta": {"stop_reason": None}, "usage": {"output_tokens": 10, "input_tokens": 100}}
    events.insert(-2, first_delta)
    server = Server(ok(sse(*events)))
    provider, _ = make(server)
    usage = run(provider.complete(request())).usage
    counts = (usage.input_tokens, usage.output_tokens, usage.cache_write_tokens, usage.cache_read_tokens)
    assert counts == (100, 20, 2000, 5000)
    expected = (100 * 4.0 + 20 * 20.0 + 2000 * 5.0 + 5000 * 0.2) / 1_000_000
    assert usage.cost_usd == pytest.approx(expected)


def test_cost_arithmetic() -> None:
    million = 1_000_000
    usage = Usage(input_tokens=million, output_tokens=million, cache_write_tokens=million, cache_read_tokens=million)
    assert cost(SPEC, usage, "5m") == pytest.approx(4 + 20 + 5 + 0.2)
    assert cost(SPEC, usage, "1h") == pytest.approx(4 + 20 + 8 + 0.2)
    no_1h_price = ModelSpec(**{**SPEC.__dict__, "price_cache_write_1h": 0.0})
    assert cost(no_1h_price, usage, "1h") == pytest.approx(4 + 20 + 8 + 0.2)
    haiku = ModelSpec(
        alias="haiku", id="claude-haiku-5-5", context_window=1_000_000, max_output_tokens=128_000,
        price_input=0.10, price_output=0.50, price_cache_write=0.125, price_cache_read=0.01,
        price_cache_write_1h=0.20, long_prompt_threshold=100_000, long_prompt_factor=5.0,
    )
    at_threshold = Usage(input_tokens=40_000, cache_read_tokens=60_000, output_tokens=1_000)
    assert cost(haiku, at_threshold, "5m") == pytest.approx((40_000 * 0.10 + 60_000 * 0.01 + 1_000 * 0.50) / 1e6)
    above = Usage(input_tokens=40_001, cache_read_tokens=60_000, output_tokens=1_000)
    assert cost(haiku, above, "5m") == pytest.approx(5 * (40_001 * 0.10 + 60_000 * 0.01 + 1_000 * 0.50) / 1e6)


def test_cache_marks_on_a_copy() -> None:
    tools = [
        {"name": "read", "description": "r", "input_schema": {"type": "object"}},
        {"name": "ls", "description": "l", "input_schema": {"type": "object"}},
    ]
    system = [{"type": "text", "text": "stable"}, {"type": "text", "text": "role"}]
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "go"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "s"}, tool_use("t1", "ls")]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.cls"}, {"type": "text", "text": "and?"}],
        },
    ]
    extra = {"anthropic_beta": "x-2026"}
    req = request(tools=tools, system=system, messages=messages, extra=extra)
    snapshot = copy.deepcopy((tools, system, messages, extra))
    server = Server(ok())
    provider, _ = make(server, cache_ttl="1h")
    run(provider.complete(req))
    assert (tools, system, messages, extra) == snapshot
    body = server.body()
    mark = {"type": "ephemeral", "ttl": "1h"}
    assert body["tools"][-1]["cache_control"] == mark
    assert "cache_control" not in body["tools"][0]
    assert body["system"][-1]["cache_control"] == mark
    assert "cache_control" not in body["system"][0]
    assert body["messages"][-1]["content"][-1]["cache_control"] == mark
    assert json.dumps(body).count("cache_control") == 3


def test_cache_marks_walk_back_over_uncacheable_blocks() -> None:
    tools = [
        {"name": "read", "description": "r", "input_schema": {"type": "object"}},
        {"name": "rare", "description": "d", "input_schema": {"type": "object"}, "defer_loading": True},
    ]
    system = [{"type": "text", "text": "stable"}, {"type": "text", "text": ""}]
    messages = [
        {"role": "user", "content": "plain string"},
        {"role": "system", "content": [{"type": "text", "text": "mid-conversation note"}]},
        {"role": "user", "content": [{"type": "text", "text": ""}]},
    ]
    server = Server(ok())
    provider, _ = make(server)
    run(provider.complete(request(tools=tools, system=system, messages=messages)))
    body = server.body()
    mark = {"type": "ephemeral", "ttl": "5m"}
    assert body["tools"][0]["cache_control"] == mark
    assert body["system"][0]["cache_control"] == mark
    assert body["messages"][0]["content"] == [{"type": "text", "text": "plain string", "cache_control": mark}]
    assert json.dumps(body).count("cache_control") == 3
    assert messages[0]["content"] == "plain string"


def test_effort_max_tokens_extra_merge_and_beta_header() -> None:
    extra = {
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}},
        "metadata": {"user_id": "u1"},
        "anthropic_beta": ["beta-one", "beta-two"],
    }
    server = Server(ok(), ok(), ok())
    provider, _ = make(server)
    run(provider.complete(request(effort="high", max_output_tokens=500, extra=extra)))
    body = server.body()
    assert body["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    assert body["metadata"] == {"user_id": "u1"}
    assert body["max_tokens"] == 500
    assert "anthropic_beta" not in body
    assert server.requests[0].headers["anthropic-beta"] == "beta-one,beta-two"
    assert extra["anthropic_beta"] == ["beta-one", "beta-two"]
    run(provider.complete(request(max_output_tokens=0)))
    assert server.body()["max_tokens"] == 0
    assert "output_config" not in server.body()
    run(provider.complete(request(extra={"anthropic_beta": "solo"})))
    assert server.requests[-1].headers["anthropic-beta"] == "solo"


def test_retry_on_429_with_retry_after() -> None:
    server = Server(error_response(429, "rate_limit_error", {"retry-after": "7"}), ok())
    provider, sleeps = make(server)
    assert run(provider.complete(request())).stop_reason == "end_turn"
    assert len(server.requests) == 2
    assert sleeps == [7.0]


def test_retry_on_529_with_jittered_backoff() -> None:
    server = Server(error_response(529, "overloaded_error"), error_response(529, "overloaded_error"), ok())
    provider, sleeps = make(server)
    run(provider.complete(request()))
    assert len(server.requests) == 3
    assert 0.5 <= sleeps[0] <= 1.0
    assert 1.0 <= sleeps[1] <= 2.0


def test_retry_on_connection_error() -> None:
    server = Server(httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), ok())
    provider, sleeps = make(server)
    run(provider.complete(request()))
    assert len(server.requests) == 3
    assert len(sleeps) == 2


def test_retry_on_mid_stream_break() -> None:
    raw = sse(*text_events())
    cut = raw.index(b"event: content_block_stop")

    async def broken() -> None:
        raise httpx.ReadError("connection reset")

    server = Server(streamed([raw[:cut]], broken), streamed([raw[:cut]]), ok())
    provider, sleeps = make(server)
    deltas: list[str] = []
    reply = run(provider.complete(request(), on_text=deltas.append))
    assert reply.content == [{"type": "text", "text": "Hello world"}]
    assert len(server.requests) == 3
    assert len(sleeps) == 2
    assert deltas == ["Hello", " world"] * 3


def test_no_retry_on_400_and_on_spend_cap_429() -> None:
    server = Server(error_response(400, "invalid_request_error"))
    provider, sleeps = make(server)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.status, info.value.retryable, len(server.requests), sleeps) == (400, False, 1, [])
    assert "req_1" in str(info.value)

    server = Server(error_response(429, "rate_limit_error", code="enforced_spend_limit_reached"))
    provider, sleeps = make(server)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.status, info.value.retryable, len(server.requests), sleeps) == (429, False, 1, [])


def test_non_json_error_body() -> None:
    server = Server(httpx.Response(413, text="<html>Request Entity Too Large</html>"))
    provider, _ = make(server)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.status, info.value.retryable) == (413, False)


def test_retries_end_when_the_window_is_spent() -> None:
    slow = {"retry-after": "10"}
    server = Server(*(error_response(529, "overloaded_error", slow) for _ in range(10)))
    provider, sleeps = make(server, retry_seconds=25.0)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.status, info.value.retryable, len(server.requests), sleeps) == (529, True, 3, [10.0] * 2)


def test_retry_after_longer_than_the_window_ends_at_once() -> None:
    server = Server(error_response(429, "rate_limit_error", {"retry-after": "500"}))
    provider, sleeps = make(server, retry_seconds=60.0)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.retryable, len(server.requests), sleeps) == (True, 1, [])


def test_an_outage_of_many_minutes_is_ridden_out() -> None:
    server = Server(*(error_response(529, "overloaded_error") for _ in range(40)), ok())
    provider, sleeps = make(server)
    assert run(provider.complete(request())).stop_reason == "end_turn"
    assert len(server.requests) == 41
    assert sum(sleeps) > 600
    assert max(sleeps) <= 60.0


def test_backoff_never_sleeps_past_the_window() -> None:
    server = Server(*(error_response(529, "overloaded_error") for _ in range(100)))
    provider, sleeps = make(server, retry_seconds=100.0)
    with pytest.raises(ProviderError):
        run(provider.complete(request()))
    assert sum(sleeps) <= 100.0


def test_cancel_ends_the_wait_between_retries() -> None:
    server = Server(*(error_response(529, "overloaded_error", {"retry-after": "30"}) for _ in range(5)))
    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    provider = AnthropicProvider(api_key="k", client=client)

    async def scenario() -> float:
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, cancel.set)
        began = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(provider.complete(request(), cancel=cancel), timeout=5)
        return time.monotonic() - began

    assert run(scenario()) < 1.0
    assert len(server.requests) == 1


def test_timeout_seconds_reaches_the_request() -> None:
    server = Server(ok())
    provider, _ = make(server, timeout_seconds=77.0)
    run(provider.complete(request()))
    assert server.requests[0].extensions["timeout"]["read"] == 77.0


def test_error_event_inside_stream() -> None:
    overloaded = sse(start(), {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    server = Server(ok(overloaded), ok())
    provider, sleeps = make(server)
    assert run(provider.complete(request())).content == [{"type": "text", "text": "Hello world"}]
    assert (len(server.requests), len(sleeps)) == (2, 1)

    invalid = sse(start(), {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}})
    server = Server(ok(invalid))
    provider, sleeps = make(server)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert (info.value.status, info.value.retryable, len(server.requests)) == (400, False, 1)


def test_truncated_tool_input_under_max_tokens_is_kept_empty() -> None:
    events = [
        start(),
        block_start(0, tool_use("toolu_x", "edit")),
        json_delta(0, "{\"path\": \"src/"),
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]
    server = Server(ok(sse(*events)))
    provider, _ = make(server)
    reply = run(provider.complete(request()))
    assert reply.stop_reason == "max_tokens"
    assert reply.content == [tool_use("toolu_x", "edit")]


def test_failing_on_text_is_not_retried() -> None:
    server = Server(ok(), ok())
    provider, sleeps = make(server)

    def broken_callback(text: str) -> None:
        raise TypeError("callback bug")

    with pytest.raises(TypeError, match="callback bug"):
        run(provider.complete(request(), on_text=broken_callback))
    assert (len(server.requests), sleeps) == (1, [])


def test_cancel_while_stream_stalls() -> None:
    closed: list[bool] = []

    async def stall() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    raw = sse(*text_events())
    server = Server(streamed([raw[: raw.index(b"event: content_block_stop")]], stall))
    provider, _ = make(server)

    async def scenario() -> float:
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, cancel.set)
        began = time.monotonic()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(provider.complete(request(), cancel=cancel), timeout=5)
        return time.monotonic() - began

    assert run(scenario()) < 1.0
    assert closed == [True]


def test_cancel_already_set_sends_nothing() -> None:
    server = Server(ok())
    provider, _ = make(server)

    async def scenario() -> None:
        cancel = asyncio.Event()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await provider.complete(request(), cancel=cancel)

    run(scenario())
    assert server.requests == []


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    server = Server(ok())
    provider, _ = make(server, api_key=None)
    with pytest.raises(ProviderError) as info:
        run(provider.complete(request()))
    assert info.value.retryable is False
    assert server.requests == []


def test_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    server = Server(ok())
    provider, _ = make(server, api_key=None)
    run(provider.complete(request()))
    assert server.requests[0].headers["x-api-key"] == "env-key"


def test_aclose_leaves_an_injected_client_open() -> None:
    server = Server(ok())
    provider, _ = make(server)
    run(provider.aclose())
    run(provider.complete(request()))
