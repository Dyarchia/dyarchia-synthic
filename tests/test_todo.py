"""The ``todo`` tool: its rules, what it costs in the context, the journal, resume and compaction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_agents import resume_run
from test_loop import assert_replays, done, dump, go, records, results

from synthic.cli import describe, main
from synthic.journal import read_records
from synthic.testing import reply, text, tool_use
from synthic.tools.todo import TodoTool, render, tally
from synthic.types import Usage

PLAN = [{"text": "Read the code"}, {"text": "Write the fix", "state": "active"}, {"text": "Run the tests"}]
WITH_TODO = {"role": {"lead": {"tools": ["sh", "edit", "todo"]}}}


def test_set_update_and_add_follow_the_rules() -> None:
    changes: list[list[dict[str, str]]] = []
    tool = TodoTool(changes.append)
    items = tool.apply({"set": PLAN})
    assert [i["state"] for i in items] == ["pending", "active", "pending"]
    tool.items = items
    items = tool.apply({"update": [{"n": 2, "state": "done"}, {"n": 3, "state": "active"}], "add": [{"text": "Ship"}]})
    assert render(items).splitlines() == [
        "[ ] 1. Read the code",
        "[x] 2. Write the fix",
        "[>] 3. Run the tests",
        "[ ] 4. Ship",
    ]
    assert tally(items) == "1/4 done; active: 3. Run the tests"
    for bad, says in (
        ({}, "Give `set`"),
        ({"update": [{"n": 9, "state": "done"}]}, "has 3 steps"),
        ({"add": [{"text": " "}]}, "non-empty"),
        ({"set": [{"text": "x", "state": "later"}]}, "must be one of"),
        ({"set": "x"}, "list of objects"),
    ):
        with pytest.raises(ValueError, match=says):
            tool.apply(bad)


def test_the_list_costs_little_and_is_journaled(tmp_path: Path) -> None:
    seen: list[tuple[str, str]] = []
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("todo", {"set": PLAN})),
            reply(tool_use("todo", {"update": [{"n": 2, "state": "done"}, {"n": 3, "state": "active"}]})),
            reply(tool_use("todo", {"update": [{"n": 3, "state": "active"}]})),
            reply(tool_use("todo", {"update": [{"n": 7}]})),
            done(),
        ],
        WITH_TODO,
        prepare=lambda rt: rt.on_output(lambda agent, kind, body: seen.append((kind, body)) if kind == "todo" else None),
    )
    assert result.status == "delivered"
    assert "todo" in [t["name"] for t in provider.requests[0].tools]
    first, second, same, bad = (results(provider.requests[n])[0] for n in (1, 2, 3, 4))
    assert first["content"] == "Saved. 0/3 done; active: 2. Write the fix."
    assert second["content"] == "Saved. 1/3 done; active: 3. Run the tests."
    assert "Read the code" not in second["content"]
    assert same["content"].startswith("No change.") and bad["is_error"]
    lists = [r["data"]["items"] for r in records(runtime) if r["kind"] == "todo"]
    assert len(lists) == 2 and lists[-1][1] == {"text": "Write the fix", "state": "done"}
    assert len(seen) == 2 and seen[-1][1].splitlines()[2] == "[>] 3. Run the tests"
    later = provider.requests[4]
    assert dump(later.system).count("Read the code") == 0
    assert dump(later.messages).count("Read the code") == 1
    assert_replays(runtime)
    trace = [describe(r, {}, False) for r in read_records(runtime.run_dir, "root") if r.kind == "todo"]
    assert trace[-1] == "todo 1/3 done; active: 3. Run the tests"


def test_a_role_without_todo_does_not_get_it(tmp_path: Path) -> None:
    _, _, provider = go(tmp_path, [done()])
    assert "todo" not in [t["name"] for t in provider.requests[0].tools]


def test_resume_starts_from_the_journaled_list(tmp_path: Path) -> None:
    runtime, _, _ = go(tmp_path, [reply(tool_use("todo", {"set": PLAN})), done()], WITH_TODO)
    result, provider = resume_run(
        tmp_path,
        runtime.run_id,
        [reply(tool_use("todo", {"update": [{"n": 1, "state": "done"}]})), done("Again.")],
        "One more thing.",
        WITH_TODO,
    )
    assert result.status == "delivered"
    assert results(provider.requests[1])[0]["content"] == "Saved. 1/3 done; active: 2. Write the fix."


def test_compaction_keeps_the_list(tmp_path: Path) -> None:
    big = Usage(input_tokens=900, output_tokens=10, cost_usd=0.001)
    task: dict[str, Any] = {
        **WITH_TODO,
        "models": {"test": {"context_window": 1000}},
        "context": {"compact_at": 0.5},
    }
    runtime, result, provider = go(
        tmp_path,
        [reply(tool_use("todo", {"set": PLAN}), usage=big), reply(text("## Task\nKeep going.")), done()],
        task,
    )
    summary = provider.requests[2].messages[0]["content"][0]["text"]
    assert summary.endswith("Your task list (`todo`) at this point:\n" + render(TodoTool(print).apply({"set": PLAN})))
    assert result.status == "delivered"
    assert_replays(runtime)


def test_todo_needs_no_permission(tmp_path: Path) -> None:
    _, result, provider = go(
        tmp_path,
        [reply(tool_use("todo", {"set": PLAN})), done()],
        {**WITH_TODO, "default_permission": "deny"},
    )
    assert not results(provider.requests[1])[0]["is_error"] and result.status == "delivered"


def test_cli_show_prints_the_last_list(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _, _ = go(tmp_path, [reply(tool_use("todo", {"set": PLAN})), done()], WITH_TODO)
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SYNTHIC_ORG_POLICY", raising=False)
    policy = tmp_path / "policy.toml"
    policy.write_text(f'[journal]\ndir = "{(tmp_path / "runs").as_posix()}"\n', encoding="utf-8")
    assert main(["show", runtime.run_id, "--policy", str(policy), "--cwd", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "task list of root:\n  [ ] 1. Read the code\n  [>] 2. Write the fix" in out
