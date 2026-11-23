"""Server tools ``web_search`` and ``web_fetch``, offline.

The scripted model returns the blocks the docs show for these tools
(``server_tool_use`` and ``*_tool_result`` in the assistant turn), so the loop,
the journal and replay are exercised on the real shapes without the network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from test_loop import assert_replays, assert_valid, done, go, make_policy, project, records
from test_provider import SPEC, Server, block_start, json_delta, make, ok, request, sse, start
from test_provider import run as run_async

from synthic import web
from synthic.agents import Runtime
from synthic.context import Context, ReceiptStore
from synthic.policy import Level, factory_defaults, merge_levels
from synthic.provider import cost
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic.types import ContextSettings, ModelRequest, ToolCall, ToolOutput, Usage

SECRET = "sk-" + "a" * 30
WEB_ROLE = {"role": {"lead": {"tools": ["sh", "web_search", "web_fetch"]}}}


def server_use(call_id: str, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {"type": "server_tool_use", "id": call_id, "name": name, "input": tool_input}


def search_result(call_id: str) -> dict[str, Any]:
    result = {"type": "web_search_result", "url": "https://example.com", "title": "Example", "encrypted_content": SECRET}
    return {"type": "web_search_tool_result", "tool_use_id": call_id, "content": [result]}


def fetch_result(call_id: str) -> dict[str, Any]:
    document = {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": f"token={SECRET}"}}
    content = {"type": "web_fetch_result", "url": "https://example.com/a", "content": document}
    return {"type": "web_fetch_tool_result", "tool_use_id": call_id, "content": content}


def cited(value: str) -> dict[str, Any]:
    citation = {"type": "web_search_result_location", "url": "https://example.com", "encrypted_index": SECRET}
    return {"type": "text", "text": value, "citations": [citation]}


def policy(tmp_path: Path, task: dict[str, Any]) -> Any:
    return make_policy(tmp_path, {**WEB_ROLE, **task}).policy


def test_granted_server_tools_are_sent_first_with_their_options(tmp_path: Path) -> None:
    task = {
        **WEB_ROLE,
        "web": {"search_max_uses": 3, "fetch_max_uses": 2, "fetch_max_content_tokens": 5000, "fetch_citations": True},
    }
    _, result, provider = go(tmp_path, [done()], task)
    assert result.status == "delivered"
    search, fetch, first_client = provider.requests[0].tools[:3]
    assert search == {"type": "web_search_20250305", "name": "web_search", "max_uses": 3}
    assert fetch == {
        "type": "web_fetch_20250910",
        "name": "web_fetch",
        "max_uses": 2,
        "max_content_tokens": 5000,
        "citations": {"enabled": True},
    }
    assert first_client["name"] == "done"


def test_no_default_role_has_web_tools(tmp_path: Path) -> None:
    defaults = merge_levels([Level("defaults", factory_defaults())]).policy
    assert all(not {"web_search", "web_fetch"} & set(role.tools) for role in defaults.roles.values())
    _, _, provider = go(tmp_path, [done()])
    assert not any("type" in t for t in provider.requests[0].tools)


def test_permissions_decide_whether_a_tool_is_offered_and_its_domains(tmp_path: Path) -> None:
    def tools(task: dict[str, Any]) -> list[dict[str, Any]]:
        return web.definitions(policy(tmp_path, task), ["web_search", "web_fetch"])

    assert [t["name"] for t in tools({"permissions": {"web_fetch: *": "deny"}})] == ["web_search"]
    blocked = tools({"permissions": {"web_fetch: evil.com": "deny", "web_fetch: *.bad.org": "deny"}})
    assert blocked[1]["blocked_domains"] == ["evil.com", "bad.org"] and "blocked_domains" not in blocked[0]
    narrowed = tools(
        {
            "default_permission": "deny",
            "permissions": {"web_search: docs.python.org": "allow", "web_search: python.org/old": "deny"},
        }
    )
    assert narrowed == [{"type": "web_search_20250305", "name": "web_search", "allowed_domains": ["docs.python.org"]}]
    listed = tools({"web": {"allowed_domains": ["a.com", "b.com"]}, "permissions": {"web_fetch: b.com": "deny"}})
    assert listed[0]["allowed_domains"] == ["a.com", "b.com"] and listed[1]["allowed_domains"] == ["a.com"]
    assert tools({"permissions": {"web_search: *": "ask"}})[0]["name"] == "web_fetch"
    loose = policy(tmp_path, {"permissions": {"web_fetch: ex*.com": "deny"}, "human": {"permission_fallback": "allow"}})
    assert [t["name"] for t in web.definitions(loose, ["web_search", "web_fetch"])] == ["web_search"]
    assert "web_fetch is not offered" in web.warnings(loose)[0]


def test_server_blocks_are_kept_journaled_unmasked_and_replayed(tmp_path: Path) -> None:
    content = [
        server_use("srvtoolu_1", "web_search", {"query": "synthic"}),
        search_result("srvtoolu_1"),
        cited("It is a harness."),
        tool_use("done", {"summary": "Found it."}),
    ]
    runtime, result, provider = go(tmp_path, [reply(*content)], WEB_ROLE)
    assert result.status == "delivered"
    assistant = next(r for r in records(runtime) if r["kind"] == "assistant")["data"]["content"]
    assert assistant[1]["content"][0]["encrypted_content"] == SECRET
    assert assistant[2]["citations"][0]["encrypted_index"] == SECRET
    assert_replays(runtime)


def test_pause_turn_is_resent_without_a_new_user_message(tmp_path: Path) -> None:
    box: dict[str, Runtime] = {}

    def paused(request: ModelRequest) -> Any:
        box["runtime"].add("A message that must wait.")
        return reply(server_use("srvtoolu_1", "web_search", {"query": "x"}), stop_reason="pause_turn")

    def prepare(runtime: Runtime) -> None:
        box["runtime"] = runtime

    script = [paused, reply(search_result("srvtoolu_1"), text("Searched.")), done()]
    runtime, result, provider = go(tmp_path, script, WEB_ROLE, prepare)
    assert result.status == "delivered"
    continued = provider.requests[1].messages
    assert continued[-1]["role"] == "assistant" and continued[-1]["content"][0]["type"] == "server_tool_use"
    after = provider.requests[2].messages
    assert after[-1]["role"] == "user" and after[-1]["content"][-1]["text"] == "A message that must wait."
    assert_valid(after)
    assert_replays(runtime)


def test_mixed_turn_sends_only_tool_results_and_holds_other_text(tmp_path: Path) -> None:
    box: dict[str, Runtime] = {}

    def mixed(request: ModelRequest) -> Any:
        box["runtime"].add("Also check the README.")
        return reply(
            server_use("srvtoolu_1", "web_fetch", {"url": "https://example.com/a"}),
            tool_use("receipt", {"id": "r9"}, "toolu_x"),
        )

    def prepare(runtime: Runtime) -> None:
        box["runtime"] = runtime

    script = [mixed, reply(fetch_result("srvtoolu_1"), text("Fetched.")), done()]
    runtime, result, provider = go(tmp_path, script, WEB_ROLE, prepare)
    assert result.status == "delivered"
    resumed_turn = provider.requests[1].messages[-1]
    assert resumed_turn["role"] == "user" and [b["type"] for b in resumed_turn["content"]] == ["tool_result"]
    last = provider.requests[2].messages
    assert last[-1]["content"][-1]["text"] == "Also check the README."
    assert last[-2]["content"][0]["type"] == "web_fetch_tool_result"
    assert last[-2]["content"][0]["content"]["content"]["source"]["data"] == f"token={SECRET}"
    assert_replays(runtime)

    again = ScriptedProvider([done("Again.")])

    async def main() -> None:
        resumed = Runtime(make_policy(tmp_path, WEB_ROLE), project(tmp_path), again, run_id=runtime.run_id)
        try:
            assert (await resumed.resume("One more thing.")).status == "delivered"
        finally:
            await resumed.close()

    asyncio.run(main())
    assert_valid(again.requests[0].messages)
    live = provider.requests[2].messages
    assert again.requests[0].messages[: len(live)] == live


def test_a_cut_reply_drops_its_trailing_unanswered_server_call(tmp_path: Path) -> None:
    cut = reply(text("Let me search"), server_use("srvtoolu_1", "web_search", {}), stop_reason="max_tokens")
    runtime, result, provider = go(tmp_path, [cut, done()], WEB_ROLE)
    assert result.status == "delivered"
    kept = provider.requests[1].messages[1]["content"]
    assert kept == [text("Let me search")]


def test_compaction_never_keeps_a_server_result_without_its_call(tmp_path: Path) -> None:
    settings = ContextSettings(compact_keep_turns=1, compact_keep_chars=100_000)
    ctx = Context(settings, ReceiptStore(tmp_path / "r"), ["system"])
    ctx.add_user("task")
    ctx.add_assistant([server_use("srvtoolu_1", "web_fetch", {"url": "u"}), tool_use("sh", {}, "c1")])
    assert ctx.server_pending()
    ctx.add_user("held while the server call is open")
    assert ctx.build()[1][-1]["role"] == "assistant"
    ctx.add_tool_results([ctx.cap(ToolCall("c1", "sh", {}), ToolOutput("ok"))])
    ctx.add_assistant([fetch_result("srvtoolu_1"), text("done")])
    assert not ctx.server_pending() and ctx.build()[1][-1]["content"][-1]["text"] == "held while the server call is open"
    assert ctx.compaction_keep() == 0


def test_server_tool_usage_is_read_and_searches_are_billed() -> None:
    events = [
        start({"input_tokens": 100, "output_tokens": 1}),
        block_start(0, {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {}}),
        json_delta(0, '{"query": "x"}'),
        {"type": "content_block_stop", "index": 0},
        block_start(1, search_result("srvtoolu_1")),
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 20, "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}},
        },
        {"type": "message_stop"},
    ]
    provider, _ = make(Server(ok(sse(*events))), search_price=10.0)
    answer = run_async(provider.complete(request()))
    assert answer.content[0]["input"] == {"query": "x"} and answer.content[1]["type"] == "web_search_tool_result"
    usage = answer.usage
    assert (usage.web_search_requests, usage.web_fetch_requests) == (2, 1)
    assert usage.cost_usd == cost(SPEC, Usage(100, 20), "5m") + 2 * 10.0 / 1000
    total = Usage()
    total.add(usage)
    total.add(Usage(**{**usage.__dict__}))
    assert total.web_search_requests == 4


def test_ledger_counts_search_fees_reported_by_the_provider(tmp_path: Path) -> None:
    billed = Usage(input_tokens=10, output_tokens=5, cost_usd=0.02, web_search_requests=2)
    content = [server_use("srvtoolu_1", "web_search", {"query": "x"}), search_result("srvtoolu_1"), text("ok")]
    runtime, result, _ = go(tmp_path, [reply(*content, usage=billed), done()], WEB_ROLE)
    assert result.status == "delivered"
    assert runtime.ledger.total.web_search_requests == 2 and runtime.ledger.total.cost_usd >= 0.02
    calls = [r["data"]["usage"] for r in records(runtime) if r["kind"] == "model_call"]
    assert calls[0]["web_search_requests"] == 2
