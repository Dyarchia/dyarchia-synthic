from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from synthic.context import Context, ReceiptStore, ReceiptTool, ToolResultEntry
from synthic.types import (
    ContextSettings,
    ImageData,
    ModelSpec,
    Record,
    ToolCall,
    ToolContext,
    ToolOutput,
    Usage,
)

SETTINGS = ContextSettings(
    output_cap=200,
    output_head=60,
    output_tail=60,
    clean_after_seconds=300.0,
    clean_keep_recent=1,
    clean_min_chars=40,
    compact_at=0.5,
)
MODEL = ModelSpec("m", "model-id", 1000, 100, 1.0, 1.0, 1.0, 1.0)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Driver:
    """Drives a Context the way the loop does and journals every step, as JSON, like the journal would."""

    def __init__(self, tmp_path: Path, settings: ContextSettings = SETTINGS) -> None:
        self.clock = Clock()
        self.store = ReceiptStore(tmp_path / "receipts")
        self.ctx = Context(settings, self.store, ["stable system", "run system"], clock=self.clock)
        self.records: list[Record] = []
        self.calls = 0

    def log(self, kind: str, data: dict[str, Any]) -> None:
        self.records.append(Record(len(self.records), 0.0, "a1", kind, json.loads(json.dumps(data))))

    def user(self, text: str) -> None:
        self.ctx.add_user(text)
        self.log("user", {"text": text, "source": "human"})

    def assistant(self, *calls: str, text: str = "", thinking: bool = True) -> None:
        self.calls += 1
        content: list[dict[str, Any]] = []
        if thinking:
            content.append({"type": "thinking", "thinking": "", "signature": f"sig{self.calls}"})
        if text:
            content.append({"type": "text", "text": text})
        content += [{"type": "tool_use", "id": c, "name": "sh", "input": {"cmd": c}} for c in calls]
        self.log("model_call", {"model_id": "x", "stop_reason": "tool_use", "usage": {}, "cost_usd": 0})
        self.ctx.note_model_call(Usage(input_tokens=100))
        self.ctx.add_assistant(content)
        self.log("assistant", {"content": content})

    def results(self, outputs: dict[str, str]) -> list[ToolResultEntry]:
        entries = [self.ctx.cap(ToolCall(c, "sh", {}), ToolOutput(text=t)) for c, t in outputs.items()]
        self.ctx.add_tool_results(entries)
        for e in entries:
            self.log(
                "tool_result",
                {"call_id": e.call_id, "name": "sh", "content": e.content, "is_error": e.is_error,
                 "receipt": e.receipt, "seconds": 0.1},
            )
        return entries

    def clean(self) -> list[str]:
        ids = self.ctx.clean_if_cold()
        if ids:
            self.log("clean", {"call_ids": ids})
        return ids

    def compact(self, summary: str) -> None:
        self.user(self.ctx.compaction_request())
        self.assistant(text=summary)
        self.ctx.apply_compaction(summary)
        self.log("compaction", {"summary": summary})

    def replayed(self) -> Context:
        fresh = Context(SETTINGS, ReceiptStore(self.store.dir), ["stable system", "run system"], clock=Clock())
        fresh.replay(self.records)
        return fresh


def big(tag: str, lines: int = 30) -> str:
    return "\n".join(f"{tag} line {i} " + "x" * 10 for i in range(1, lines + 1))


def dump(ctx: Context) -> str:
    return json.dumps(ctx.build(), sort_keys=True)


def flat(ctx: Context) -> list[str]:
    """History as a sequence of (role, block): the API merges same-role messages, so this is what the cache sees."""
    return [json.dumps([m["role"], b], sort_keys=True) for m in ctx.build()[1] for b in m["content"]]


def assert_valid(messages: list[dict[str, Any]]) -> None:
    """Roles alternate from user; each tool_use turn is answered right after, results first."""
    assert messages[0]["role"] == "user"
    for i, m in enumerate(messages):
        assert m["content"]
        if i:
            assert m["role"] != messages[i - 1]["role"]
        if m["role"] == "user":
            types = [b["type"] for b in m["content"]]
            assert types == sorted(types, key=lambda t: t != "tool_result")
        else:
            calls = [b["id"] for b in m["content"] if b["type"] == "tool_use"]
            if calls and i + 1 < len(messages):
                answered = [b["tool_use_id"] for b in messages[i + 1]["content"] if b["type"] == "tool_result"]
                assert sorted(answered) == sorted(calls)


def thinking_positions(messages: list[dict[str, Any]]) -> list[int]:
    return [i for i, m in enumerate(messages) for b in m["content"] if b["type"] in ("thinking", "redacted_thinking")]


def stub_positions(messages: list[dict[str, Any]]) -> list[int]:
    return [
        i for i, m in enumerate(messages) for b in m["content"]
        if b["type"] == "tool_result" and str(b["content"]).startswith("[output cleared;")
    ]


def test_receipt_store_ids_continue_and_keep_newlines(tmp_path: Path) -> None:
    store = ReceiptStore(tmp_path)
    assert store.put("a\r\nb\n") == "r1"
    assert store.put("c") == "r2"
    (tmp_path / "r9.txt").write_text("x", encoding="utf-8")
    reopened = ReceiptStore(tmp_path)
    assert reopened.put("d") == "r10"
    assert reopened.get("r1") == "a\r\nb\n"
    assert reopened.info("r1") == (5, 2)
    assert reopened.exists("r2") and not reopened.exists("r3") and not reopened.exists("../r1")
    with pytest.raises(KeyError):
        reopened.get("r3")


def test_append_only_between_builds(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    snapshots: list[list[str]] = []
    d.user("task")
    snapshots.append(flat(d.ctx))
    d.assistant("c1", "c2", text="working")
    d.ctx.add_user("human note while tools run")
    d.results({"c2": "two", "c1": big("one")})
    snapshots.append(flat(d.ctx))
    d.assistant("c3")
    d.results({"c3": "three"})
    d.ctx.add_user("hook text")
    snapshots.append(flat(d.ctx))
    d.assistant(text="done", thinking=False)
    d.user("more")
    snapshots.append(flat(d.ctx))
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert later[: len(earlier)] == earlier
    assert_valid(d.ctx.build()[1])
    system, _ = d.ctx.build()
    assert system == [{"type": "text", "text": "stable system"}, {"type": "text", "text": "run system"}]


def test_roles_merge_and_results_precede_engine_text(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("a")
    d.user("b")
    d.assistant("c1", "c2")
    d.ctx.add_user("background job finished")
    d.results({"c1": "r-one"})
    d.ctx.add_user("hook says hi")
    d.results({"c2": "r-two"})
    messages = d.ctx.build()[1]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert [b["type"] for b in messages[0]["content"]] == ["text", "text"]
    last = messages[2]["content"]
    assert [b.get("tool_use_id") or b["text"] for b in last] == ["c1", "c2", "background job finished", "hook says hi"]
    assert last[0] == {"type": "tool_result", "tool_use_id": "c1", "content": "r-one", "is_error": False}
    assert_valid(messages)


def test_contract_errors(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("go")
    d.assistant("c1")
    with pytest.raises(ValueError):
        d.ctx.add_tool_results([ToolResultEntry("nope", "x")])
    with pytest.raises(ValueError):
        d.ctx.add_assistant([{"type": "text", "text": "skip results"}])
    d.results({"c1": "ok"})
    with pytest.raises(ValueError):
        d.ctx.add_tool_results([ToolResultEntry("c1", "again")])


def test_build_returns_copies(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    before = dump(d.ctx)
    system, messages = d.ctx.build()
    system[0]["text"] = "evil"
    messages[0]["content"][0]["text"] = "evil"
    messages.append({"role": "assistant", "content": []})
    assert dump(d.ctx) == before


def test_cap_arithmetic_line_cuts_and_receipt(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    lines = [f"L{i:03d}xxxxxx" for i in range(1, 41)]
    text = "\n".join(lines)
    entry = d.ctx.cap(ToolCall("c1", "sh", {}), ToolOutput(text=text))
    assert entry.receipt == "r1"
    assert d.store.get("r1") == text
    assert entry.content == (
        "\n".join(lines[:5]) + "\n"
        "[output truncated: 439 chars, 40 lines; showing lines 1-5 and 36-40. "
        "Full output: receipt r1, search it or read a line range with the receipt tool.]\n"
        + "\n".join(lines[35:])
    )
    single = "y" * 500
    cut = d.ctx.cap(ToolCall("c2", "sh", {}), ToolOutput(text=single)).content
    assert isinstance(cut, str)
    assert cut.startswith("y" * SETTINGS.output_head + "\n[output truncated: 500 chars, 1 lines")
    assert cut.endswith("\n" + "y" * SETTINGS.output_tail)


def test_receipt_threshold(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    short = d.ctx.cap(ToolCall("c1", "sh", {}), ToolOutput(text="x" * SETTINGS.clean_min_chars))
    assert short.receipt is None and short.content == "x" * SETTINGS.clean_min_chars
    mid = d.ctx.cap(ToolCall("c2", "sh", {}), ToolOutput(text="x" * (SETTINGS.clean_min_chars + 1)))
    assert mid.receipt == "r1" and mid.content == "x" * (SETTINGS.clean_min_chars + 1)
    empty = d.ctx.cap(ToolCall("c3", "sh", {}), ToolOutput(text="", is_error=True))
    assert empty.content == "(no output)" and empty.is_error


def test_images_in_results(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    img = ImageData("image/png", "iVBORw0KGgo=")
    entry = d.ctx.cap(ToolCall("c1", "shot", {}), ToolOutput(text="screen", images=[img]))
    assert entry.content == [
        {"type": "text", "text": "screen"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
    ]
    only = d.ctx.cap(ToolCall("c2", "shot", {}), ToolOutput(images=[img]))
    assert isinstance(only.content, list) and [b["type"] for b in only.content] == ["image"]


def receipt_tool(tmp_path: Path) -> tuple[ReceiptTool, str]:
    store = ReceiptStore(tmp_path)
    rid = store.put("\n".join(f"line {i}" + (" ERROR" if i in (5, 6, 20) else "") for i in range(1, 31)))
    return ReceiptTool(store, SETTINGS), rid


def test_receipt_tool_search_slice_and_overview(tmp_path: Path) -> None:
    tool, rid = receipt_tool(tmp_path)
    ctx = ToolContext("a1", tmp_path, {}, 10.0, asyncio.Event())
    assert tool.resource({"id": rid}, ctx) is None
    found = asyncio.run(tool.run({"id": rid, "search": "ERROR", "context": 1}, ctx))
    assert not found.is_error and found.meta == {"receipt": rid}
    assert "3 matching lines" in found.text
    assert "4: line 4\n5: line 5 ERROR\n6: line 6 ERROR\n7: line 7\n--\n19: line 19\n20: line 20 ERROR" in found.text
    ranged = tool.query({"id": rid, "search": "ERROR", "lines": "10-30"})
    assert "1 matching lines" in ranged.text and "5: line 5" not in ranged.text
    sliced = tool.query({"id": rid, "lines": "28-99"})
    assert sliced.text == f"Receipt {rid}, lines 28-30 of 30:\n28: line 28\n29: line 29\n30: line 30"
    overview = tool.query({"id": rid})
    assert overview.text.startswith(f"Receipt {rid}: 30 lines") and "1: line 1" in overview.text
    assert "Use `search` or `lines`" in overview.text
    none = tool.query({"id": rid, "search": "nothing"})
    assert not none.is_error and "No line matches" in none.text


def test_receipt_tool_errors(tmp_path: Path) -> None:
    tool, rid = receipt_tool(tmp_path)
    bad = tool.query({"id": rid, "search": "(unclosed"})
    assert bad.is_error and "Invalid regular expression" in bad.text
    unknown = tool.query({"id": "r77"})
    assert unknown.is_error and "Unknown receipt id" in unknown.text
    assert tool.query({"id": rid, "lines": "abc"}).is_error
    assert tool.query({"id": rid, "lines": "40-50"}).is_error
    assert tool.query({"id": rid, "context": -1, "search": "x"}).is_error


def test_receipt_tool_search_is_capped(tmp_path: Path) -> None:
    store = ReceiptStore(tmp_path)
    rid = store.put("\n".join(f"hit {i}" for i in range(1, 200)))
    out = ReceiptTool(store, SETTINGS).query({"id": rid, "search": "hit"})
    assert "more matching lines not shown" in out.text
    assert len(out.text) <= SETTINGS.output_cap + 100


def test_receipt_tool_output_is_never_stored_again(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    rid = d.store.put(big("log", 50))
    tool = ReceiptTool(d.store, SETTINGS)
    entry = d.ctx.cap(ToolCall("c1", "receipt", {}), tool.query({"id": rid, "lines": "1-50"}))
    assert entry.receipt == rid
    assert isinstance(entry.content, str) and f"view of receipt {rid}" in entry.content
    assert sorted(p.name for p in d.store.dir.iterdir()) == [f"{rid}.txt"]


def cold_run(tmp_path: Path) -> Driver:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c1")
    d.results({"c1": big("first")})
    d.assistant("c2", "c3")
    d.results({"c2": "tiny", "c3": big("third")})
    d.assistant("c4")
    d.results({"c4": big("fourth")})
    return d


def test_no_clean_before_first_call_or_ttl(tmp_path: Path) -> None:
    fresh = Driver(tmp_path / "a")
    fresh.user("x")
    fresh.clock.now += 10_000
    assert fresh.clean() == []
    d = cold_run(tmp_path / "b")
    before = dump(d.ctx)
    d.clock.now += SETTINGS.clean_after_seconds
    assert d.clean() == []
    assert dump(d.ctx) == before


def test_clean_after_ttl(tmp_path: Path) -> None:
    d = cold_run(tmp_path)
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.clean() == ["c1", "c3"]
    messages = d.ctx.build()[1]
    results = {b["tool_use_id"]: b for m in messages for b in m["content"] if b["type"] == "tool_result"}
    first = big("first")
    assert results["c1"]["content"] == f"[output cleared; receipt r1, 30 lines, {len(first):,} chars: first line 1 xxxxxxxxxx]"
    assert results["c3"]["content"].startswith("[output cleared; receipt r2, 30 lines")
    assert results["c2"]["content"] == "tiny"
    assert "fourth line 1" in results["c4"]["content"]
    assert results["c1"]["is_error"] is False
    assert_valid(messages)
    again = dump(d.ctx)
    d.clock.now += 10_000
    assert d.clean() == []
    assert dump(d.ctx) == again


def test_clean_strips_thinking_after_first_edit_only(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c0")
    d.results({"c0": "small, no receipt"})
    d.assistant("c1")
    d.results({"c1": big("first")})
    d.assistant("c2")
    d.results({"c2": big("second")})
    before = d.ctx.build()[1]
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.clean() == ["c1"]
    after = d.ctx.build()[1]
    first_stub = stub_positions(after)[0]
    assert first_stub == 4
    assert all(i < first_stub for i in thinking_positions(after))
    assert after[:first_stub] == before[:first_stub]
    assert thinking_positions(after) == [1, 3]
    for old, new in zip(before[first_stub + 1 :], after[first_stub + 1 :]):
        if old["role"] == "assistant":
            assert new["content"] == [b for b in old["content"] if b["type"] != "thinking"]
    assert_valid(after)


def test_clean_at_turn_boundary_drops_empty_assistant_and_merges(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c1")
    d.results({"c1": big("first")})
    d.ctx.note_model_call(Usage(input_tokens=100))
    d.ctx.add_assistant([{"type": "redacted_thinking", "data": "opaque"}])
    d.log("assistant", {"content": [{"type": "redacted_thinking", "data": "opaque"}]})
    d.user("next request")
    d.assistant("c2")
    d.results({"c2": big("second")})
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.clean() == ["c1"]
    messages = d.ctx.build()[1]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["type"] for b in messages[2]["content"]] == ["tool_result", "text"]
    assert thinking_positions(messages) == [1]
    assert_valid(messages)
    assert dump(d.replayed()) == dump(d.ctx)


def test_clean_shrinks_compaction_estimate_and_keeps_images(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c1")
    img = ImageData("image/png", "AAAA")
    entry = d.ctx.cap(ToolCall("c1", "shot", {}), ToolOutput(text=big("shot", 300), images=[img]))
    d.ctx.add_tool_results([entry])
    d.assistant("c2")
    d.results({"c2": "ok"})
    d.ctx.note_model_call(Usage(input_tokens=600))
    assert d.ctx.should_compact(MODEL)
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.ctx.clean_if_cold() == ["c1"]
    content = d.ctx.build()[1][2]["content"][0]["content"]
    assert content[0]["text"].startswith("[output cleared; receipt r1, 300 lines")
    assert content[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    assert not d.ctx.should_compact(MODEL)


def test_compaction_threshold_and_shape(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    assert not d.ctx.should_compact(MODEL)
    d.ctx.note_model_call(Usage(input_tokens=400, cache_read_tokens=99))
    d.ctx.add_assistant([{"type": "text", "text": "x" * 5000}])
    assert not d.ctx.should_compact(MODEL)
    d.ctx.note_model_call(Usage(input_tokens=400, cache_read_tokens=100))
    assert d.ctx.should_compact(MODEL)
    assert d.ctx.last_prompt_tokens == 500
    request = d.ctx.compaction_request()
    assert "## Receipts" in request and "## Next steps" in request
    d.ctx.apply_compaction("did things, see r1")
    messages = d.ctx.build()[1]
    assert len(messages) == 1 and messages[0]["role"] == "user"
    text = messages[0]["content"][0]["text"]
    assert text.startswith("[Summary of earlier work.") and "receipt tool" in text
    assert text.endswith("did things, see r1")
    assert not d.ctx.should_compact(MODEL)


def test_custom_compaction_template(tmp_path: Path) -> None:
    template = tmp_path / "mine.md"
    template.write_text("summarize please", encoding="utf-8")
    ctx = Context(ContextSettings(compaction_template=str(template)), ReceiptStore(tmp_path / "r"), [])
    assert ctx.compaction_request() == "summarize please"


def test_repair_dangling(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c1", "c2", "c3")
    d.results({"c2": "done"})
    d.ctx.add_user("stop")
    added = d.ctx.repair_dangling("interrupted by the human")
    assert [e.call_id for e in added] == ["c1", "c3"] and all(e.is_error for e in added)
    content = d.ctx.build()[1][-1]["content"]
    assert [b.get("tool_use_id") for b in content] == ["c2", "c1", "c3", None]
    assert "interrupted by the human" in content[1]["content"]
    assert_valid(d.ctx.build()[1])
    assert d.ctx.repair_dangling("again") == []


def test_replay_is_byte_identical_through_clean_and_compaction(tmp_path: Path) -> None:
    d = cold_run(tmp_path)
    assert dump(d.replayed()) == dump(d.ctx)
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.clean()
    d.ctx.add_user("hook text")
    d.log("user", {"text": "hook text", "source": "hook"})
    assert dump(d.replayed()) == dump(d.ctx)
    d.compact("summary keeps r2")
    assert dump(d.replayed()) == dump(d.ctx)
    d.assistant("c9")
    d.results({"c9": big("after")})
    d.ctx.add_user("human")
    d.log("user", {"text": "human", "source": "human"})
    d.records.append(Record(len(d.records), 0.0, "a1", "spawn", {"child_id": "x", "role": "r", "task": "t"}))
    assert dump(d.replayed()) == dump(d.ctx)


def test_clean_waits_for_the_longer_of_clean_after_and_the_cache_ttl(tmp_path: Path) -> None:
    d = Driver(tmp_path, replace(SETTINGS, cache_ttl="1h"))
    d.user("task")
    d.assistant("c1")
    d.results({"c1": big("first")})
    d.assistant("c2")
    d.results({"c2": big("second")})
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.ctx.clean_if_cold() == []
    d.clock.now += 3600
    assert d.ctx.clean_if_cold() == ["c1"]


def test_drop_thinking_removes_every_thinking_block(tmp_path: Path) -> None:
    d = Driver(tmp_path)
    d.user("task")
    d.assistant("c1")
    d.results({"c1": "one"})
    d.ctx.add_assistant([{"type": "redacted_thinking", "data": "opaque"}])
    d.ctx.add_user("next")
    d.assistant("c2", text="working")
    d.results({"c2": "two"})
    before = d.ctx.build()[1]
    d.ctx.drop_thinking()
    after = d.ctx.build()[1]
    assert thinking_positions(after) == []
    assert [m["role"] for m in after] == ["user", "assistant", "user", "assistant", "user"]
    assert after[2]["content"] == before[2]["content"] + before[4]["content"]
    assert_valid(after)


def test_compaction_keeps_the_newest_turns_without_thinking_and_replays(tmp_path: Path) -> None:
    d = Driver(tmp_path, replace(SETTINGS, compact_keep_turns=1, compact_keep_chars=10_000))
    d.user("task")
    d.assistant("c1")
    d.results({"c1": big("first")})
    d.assistant("c2")
    d.results({"c2": big("second")})
    keep = d.ctx.compaction_keep()
    assert keep == 2
    d.ctx.apply_compaction("summary", keep)
    d.log("compaction", {"summary": "summary", "keep": keep})
    messages = d.ctx.build()[1]
    assert_valid(messages)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert "follow it unchanged" in messages[0]["content"][0]["text"]
    assert [b["type"] for b in messages[1]["content"]] == ["tool_use"]
    assert messages[2]["content"][0]["tool_use_id"] == "c2"
    assert dump(d.replayed()) == dump(d.ctx)
    d.clock.now += SETTINGS.clean_after_seconds + 1
    d.assistant("c3")
    d.results({"c3": big("third")})
    d.clock.now += SETTINGS.clean_after_seconds + 1
    assert d.clean() == ["c2"]
    assert dump(d.replayed()) == dump(d.ctx)


def test_compaction_keeps_nothing_when_the_newest_turn_is_too_large_or_alone(tmp_path: Path) -> None:
    d = Driver(tmp_path, replace(SETTINGS, compact_keep_turns=2, compact_keep_chars=50))
    d.user("task")
    d.assistant("c1")
    d.results({"c1": big("first")})
    assert d.ctx.compaction_keep() == 0
    d.assistant("c2")
    d.results({"c2": big("second")})
    assert d.ctx.compaction_keep() == 0
    roomy = Driver(tmp_path / "b", replace(SETTINGS, compact_keep_turns=2, compact_keep_chars=10_000))
    roomy.user("task")
    roomy.assistant("c1")
    roomy.results({"c1": "ok"})
    assert roomy.ctx.compaction_keep() == 0


def test_boundary_is_due_only_above_the_token_limit(tmp_path: Path) -> None:
    d = Driver(tmp_path, replace(SETTINGS, compact_boundary_tokens=500))
    assert not d.ctx.boundary_due()
    d.ctx.restore_measure(500)
    assert d.ctx.boundary_due()
    off = Driver(tmp_path / "b", replace(SETTINGS, compact_boundary_tokens=0))
    off.ctx.restore_measure(10_000)
    assert not off.ctx.boundary_due()


def test_compaction_window_stops_at_the_price_tier(tmp_path: Path) -> None:
    tiered = replace(MODEL, long_prompt_threshold=400)
    driver = Driver(tmp_path)
    assert driver.ctx.compaction_window(MODEL) == 1000
    assert driver.ctx.compaction_window(tiered) == 400
    off = Driver(tmp_path / "off", replace(SETTINGS, compact_within_price_tier=False))
    assert off.ctx.compaction_window(tiered) == 1000
