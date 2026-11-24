"""The agent tree and the runtime: spawn, wait, limits, escalation, interrupts, resume, copy mode."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_loop import done, go, kinds, make_policy, project, records, results
from test_proc import py_command

from synthic.agents import Ledger, Runtime
from synthic.journal import ALIVE_FILE, JOURNAL_FILE, is_alive
from synthic.loop import UNKNOWN_OUTCOME, AgentResult
from synthic.policy import PolicyError
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic.types import Budget, HookOutcome, ModelReply, ModelRequest, Usage

HANG = object()
"""A script item that never answers: the call waits for the cancel event, like a real stream."""


class RoutedProvider:
    """One script per agent, chosen by a key found in the first user message (the task).

    A shared ``ScriptedProvider`` would hand replies to whichever agent asks
    first, and agents of a tree run concurrently.
    """

    def __init__(self, routes: dict[str, list[Any]]) -> None:
        self.routes = {key: ScriptedProvider(script) for key, script in routes.items()}
        self.requests: dict[str, list[ModelRequest]] = {key: [] for key in routes}

    async def complete(
        self, request: ModelRequest, on_text: Any = None, cancel: asyncio.Event | None = None
    ) -> ModelReply:
        first = request.messages[0]["content"][0]["text"]
        key = next(k for k in self.routes if k in first)
        self.requests[key].append(request)
        provider = self.routes[key]
        if provider.script and provider.script[0] is HANG:
            provider.script.pop(0)
            assert cancel is not None
            await cancel.wait()
            raise asyncio.CancelledError
        return await provider.complete(request, on_text, cancel)


def spawn(role: str, task: str) -> ModelReply:
    return reply(tool_use("spawn", {"role": role, "task": task}))


def agent(action: str, **fields: Any) -> ModelReply:
    return reply(tool_use("agent", {"action": action, **fields}))


def tree(
    tmp_path: Path,
    routes: dict[str, list[Any]],
    task: dict[str, Any] | None = None,
    prepare: Callable[[Runtime], None] | None = None,
) -> tuple[Runtime, AgentResult, RoutedProvider]:
    provider = RoutedProvider(routes)

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider)
        if prepare is not None:
            prepare(runtime)
        try:
            return runtime, await runtime.run("PARENT")
        finally:
            await runtime.close()

    runtime, result = asyncio.run(main())
    return runtime, result, provider


def text_of(request: ModelRequest) -> str:
    return str(request.messages)


def test_spawn_wait_and_shared_ledger(tmp_path: Path) -> None:
    runtime, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD count files"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [done("There are 3 files.")],
        },
    )
    assert result.status == "delivered"
    parent = provider.requests["PARENT"]
    assert "Started a1 (worker)" in results(parent[1])[0]["content"]
    waited = results(parent[2])[0]["content"]
    assert "[child a1 (worker) ended: delivered]" in waited and "There are 3 files." in waited
    assert text_of(parent[2]).count("[child a1") == 1
    assert runtime.ledger.steps == 4
    assert runtime.ledger.per_agent["a1"].cost_usd == pytest.approx(0.0002)
    assert runtime.ledger.total.cost_usd == pytest.approx(0.0008)
    assert ["spawn", "child_done"] == [k for k in kinds(runtime) if k in ("spawn", "child_done")]
    start = records(runtime, "a1")[0]
    assert start["kind"] == "start" and start["data"]["parent"] == "root" and start["data"]["role"] == "worker"


def test_spawn_does_not_wait_and_cancel_stops_the_child(tmp_path: Path) -> None:
    runtime, result, provider = tree(
        tmp_path,
        {"PARENT": [spawn("worker", "CHILD hang"), agent("cancel", id="a1"), done()], "CHILD": [HANG]},
    )
    assert "Started a1" in results(provider.requests["PARENT"][1])[0]["content"]
    assert "ended: cancelled" in results(provider.requests["PARENT"][2])[0]["content"]
    assert runtime.handles["a1"].result.status == "cancelled"
    assert result.status == "delivered"


def test_idle_parent_wakes_when_the_child_ends(tmp_path: Path) -> None:
    _, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD slow"), reply(text("Waiting for the child.")), done()],
            "CHILD": [reply(tool_use("receipt", {"id": "r1"})), done("Child result.")],
        },
    )
    third = provider.requests["PARENT"][2]
    assert "[child a1 (worker) ended: delivered]" in third.messages[-1]["content"][-1]["text"]
    assert "You ended your turn" not in text_of(third)
    assert result.status == "delivered"


def test_child_never_gets_more_than_its_parent(tmp_path: Path) -> None:
    _, _, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD try"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [spawn("helper", "GRANDCHILD"), done()],
        },
        {
            "role": {
                "lead": {"tools": ["sh", "spawn"], "can_spawn": ["worker"]},
                "worker": {"tools": ["sh", "edit", "spawn"], "can_spawn": ["helper"]},
                "helper": {"model": "test", "tools": ["sh"]},
            }
        },
    )
    child = provider.requests["CHILD"]
    assert [t["name"] for t in child[0].tools] == ["done", "ask", "receipt", "sh", "job"]
    assert "Unknown tool 'spawn'" in results(child[1])[0]["content"]


def test_depth_and_concurrency_limits_and_done_cancels_children(tmp_path: Path) -> None:
    async def slow_hook(event: str, payload: dict[str, Any]) -> HookOutcome:
        await asyncio.sleep(0)
        return HookOutcome()

    runtime, result, provider = tree(
        tmp_path,
        {
            "PARENT": [reply(tool_use("spawn", {"role": "worker", "task": "CHILD-A"}),
                             tool_use("spawn", {"role": "worker", "task": "CHILD-B"})), done()],
            "CHILD-A": [spawn("worker", "GRANDCHILD"), HANG],
            "CHILD-B": [HANG],
        },
        {
            "agents": {"depth": 1, "concurrent": 1},
            "role": {"worker": {"tools": ["sh", "spawn"], "can_spawn": ["worker"]}, "lead": {"can_spawn": ["worker"]}},
        },
        lambda r: r.hooks.subscribe("spawn", slow_hook),
    )
    contents = sorted(b["content"] for b in results(provider.requests["PARENT"][1]))
    assert contents[0].startswith("Cannot spawn now: 1 agents are running and the limit is 1")
    assert contents[1].startswith("Started a1")
    assert list(runtime.handles) == ["a1"]
    assert "spawn" not in [t["name"] for t in provider.requests["CHILD-A"][0].tools]
    assert "Unknown tool 'spawn'" in results(provider.requests["CHILD-A"][1])[0]["content"]
    assert result.status == "delivered"
    assert runtime.handles["a1"].result.status == "cancelled"
    assert all(h.runner is not None and h.runner.done() for h in runtime.handles.values())


def test_runtime_stop_cancels_the_whole_tree(tmp_path: Path) -> None:
    provider = RoutedProvider(
        {"PARENT": [spawn("worker", "CHILD hang"), reply(text("Waiting."))], "CHILD": [HANG]}
    )

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider)
        task = asyncio.create_task(runtime.run("PARENT"))
        while len(provider.requests["PARENT"]) < 2 or not provider.requests["CHILD"]:
            await asyncio.sleep(0.01)
        runtime.stop()
        result = await task
        await runtime.close()
        return runtime, result

    runtime, result = asyncio.run(main())
    assert result.status == "stopped"
    assert runtime.handles["a1"].result.status == "cancelled"
    assert records(runtime, "a1")[-1]["data"]["status"] == "cancelled"


def test_escalation_after_failed_attempts(tmp_path: Path) -> None:
    runtime, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD fix it"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [done("no luck", "failed"), done("still no luck", "failed"), done("Fixed by senior.")],
        },
        {
            "role": {
                "worker": {"on_failure": {"escalate_to": "senior", "attempts": 2}},
                "senior": {"model": "test", "tools": ["sh", "edit"]},
            }
        },
    )
    notice = results(provider.requests["PARENT"][2])[0]["content"]
    assert "[child a1 (worker) ended: delivered]" in notice and "Fixed by senior." in notice
    assert "a1 as worker: failed; a1.2 as worker: failed; a1.3 as senior: delivered" in notice
    assert [r["data"]["role"] for a in ("a1", "a1.2", "a1.3") for r in records(runtime, a)[:1]] == [
        "worker",
        "worker",
        "senior",
    ]
    assert runtime.handles["a1"].result.steps == 3
    assert result.status == "delivered"


def test_copy_mode_child_works_apart_and_reports_changes(tmp_path: Path) -> None:
    (project(tmp_path) / "base.txt").write_text("base", encoding="utf-8")
    runtime, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD write"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [reply(tool_use("edit", {"path": "new.txt", "old": "", "new": "made in a copy"})), done("Wrote.")],
        },
        {"environment": {"mode": "copy"}},
    )
    notice = results(provider.requests["PARENT"][2])[0]["content"]
    assert "A new.txt" in notice and "Work copy:" in notice
    assert not (project(tmp_path) / "new.txt").exists()
    copy = runtime.handles["a1"].copy
    assert copy is not None and copy != project(tmp_path)
    assert (copy / "new.txt").read_text(encoding="utf-8") == "made in a copy"
    assert result.status == "delivered"


def test_resume_after_a_crash_reports_the_dangling_action(tmp_path: Path) -> None:
    echo = reply(tool_use("sh", {"command": "echo hi"}))
    script = [reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "x"})), echo, done()]
    first, _, first_provider = go(tmp_path, script)
    journal = first.run_dir / "root" / JOURNAL_FILE
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    intents = [i for i, line in enumerate(lines) if '"kind": "tool_intent"' in line]
    journal.write_text("".join(lines[: intents[1] + 1]), encoding="utf-8")

    provider = ScriptedProvider([done("Recovered.")])

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider, run_id=first.run_id)
        try:
            return runtime, await runtime.resume()
        finally:
            await runtime.close()

    runtime, result = asyncio.run(main())
    before = first_provider.requests[1].messages
    after = provider.requests[0].messages
    assert after[: len(before)] == before
    assert after[len(before)] == {"role": "assistant", "content": echo.content}
    dangling = results(provider.requests[0])[0]
    assert dangling["is_error"] and dangling["content"] == UNKNOWN_OUTCOME
    assert result.status == "delivered" and result.summary == "Recovered."
    assert runtime.ledger.steps == 3
    assert provider.requests[0].system == first_provider.requests[0].system


def test_resume_reports_children_of_the_dead_process_as_cancelled(tmp_path: Path) -> None:
    first, _, _ = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD long job"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [reply(tool_use("receipt", {"id": "r1"})), done("Child finished.")],
        },
    )
    for agent_id, kind in (("root", '"kind": "spawn"'), ("a1", '"kind": "tool_result"')):
        journal = first.run_dir / agent_id / JOURNAL_FILE
        lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
        cut = next(i for i, line in enumerate(lines) if kind in line)
        journal.write_text("".join(lines[: cut + 1]), encoding="utf-8")

    provider = ScriptedProvider([done("Recovered without the child.")])

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider, run_id=first.run_id)
        try:
            return runtime, await runtime.resume()
        finally:
            await runtime.close()

    runtime, result = asyncio.run(main())
    last = provider.requests[0].messages[-1]["content"]
    assert last[0]["type"] == "tool_result" and last[0]["content"] == UNKNOWN_OUTCOME
    assert "[child a1 (worker) ended: cancelled]" in last[-1]["text"]
    assert records(runtime, "a1")[-1]["data"]["status"] == "cancelled"
    assert any(r["kind"] == "child_done" and r["data"]["status"] == "cancelled" for r in records(runtime))
    assert result.status == "delivered"


def test_undefined_role_is_an_error(tmp_path: Path) -> None:
    async def main() -> None:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), ScriptedProvider([]))
        try:
            with pytest.raises(PolicyError):
                await runtime.run("x", role="nobody")
        finally:
            await runtime.close()

    asyncio.run(main())


def test_spawn_and_agent_work_under_a_default_deny(tmp_path: Path) -> None:
    _, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD count"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [done("Counted.")],
        },
        {"default_permission": "deny", "permissions": {"spawn: worker": "allow"}},
    )
    assert "Counted." in results(provider.requests["PARENT"][2])[0]["content"]
    assert result.status == "delivered"


def test_a_child_stopped_by_its_budget_is_not_retried(tmp_path: Path) -> None:
    free = Usage()
    runtime, _, provider = tree(
        tmp_path,
        {
            "PARENT": [
                reply(tool_use("spawn", {"role": "worker", "task": "CHILD spend"}), usage=free),
                reply(tool_use("agent", {"action": "wait", "id": "a1", "seconds": 30}), usage=free),
                reply(tool_use("done", {"summary": "ok"}), usage=free),
            ],
            "CHILD": [reply(tool_use("receipt", {"id": "r1"})), done("never reached")],
        },
        {
            "budget": {"per_agent_dollars": 0.0001},
            "role": {"worker": {"on_failure": {"escalate_to": "worker", "attempts": 1}}},
        },
    )
    assert runtime.handles["a1"].attempts == ["a1 as worker: budget"]
    assert len(provider.requests["CHILD"]) == 1


def test_attempts_share_the_step_allowance(tmp_path: Path) -> None:
    runtime, _, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD try"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [reply(tool_use("receipt", {"id": "r1"})), done("no luck", "failed"), done("fresh start")],
        },
        {"role": {"worker": {"max_steps": 2, "on_failure": {"escalate_to": "worker", "attempts": 2}}}},
    )
    assert runtime.handles["a1"].attempts == ["a1 as worker: failed", "a1.2 as worker: budget"]
    assert len(provider.requests["CHILD"]) == 2
    assert list(runtime.ledger.per_agent) == ["root", "a1"]


def test_finished_children_are_dropped_and_copies_without_changes_removed(tmp_path: Path) -> None:
    ended: dict[str, Any] = {}

    def prepare(runtime: Runtime) -> None:
        async def remember(event: str, payload: dict[str, Any]) -> HookOutcome:
            ended[payload["agent_id"]] = runtime.agents[payload["agent_id"]]
            return HookOutcome()

        runtime.hooks.subscribe("end", remember)

    runtime, result, provider = tree(
        tmp_path,
        {
            "PARENT": [
                reply(tool_use("spawn", {"role": "worker", "task": "CHILD-A"}),
                      tool_use("spawn", {"role": "worker", "task": "CHILD-B"})),
                agent("wait", id="a1", seconds=30),
                agent("wait", id="a2", seconds=30),
                done(),
            ],
            "CHILD-A": [reply(tool_use("edit", {"path": "new.txt", "old": "", "new": "x"})), done("Wrote.")],
            "CHILD-B": [done("Looked only.")],
        },
        {"environment": {"mode": "copy"}},
        prepare,
    )
    assert result.status == "delivered"
    assert list(runtime.agents) == ["root"]
    handles = runtime.handles
    writer, reader = (handles[h] for h in sorted(handles, key=lambda h: handles[h].task))
    assert writer.copy is not None and (writer.copy / "new.txt").is_file()
    assert reader.copy is not None and not reader.copy.exists()
    assert "Changes: none" in reader.notice()
    for agent_id in ("a1", "a2"):
        with pytest.raises(ValueError):
            ended[agent_id].journal.append("user", {"text": "late"})


def test_a_cancelled_supervisor_still_reports_the_child(tmp_path: Path) -> None:
    holder: list[Runtime] = []

    def cancel_supervisor(request: ModelRequest) -> ModelReply:
        runner = holder[0].handles["a1"].runner
        assert runner is not None
        runner.cancel()
        return reply(text("Waiting for the child."))

    runtime, result, provider = tree(
        tmp_path,
        {"PARENT": [spawn("worker", "CHILD hang"), cancel_supervisor, done()], "CHILD": [HANG]},
        prepare=holder.append,
    )
    assert "[child a1 (worker) ended: cancelled]" in text_of(provider.requests["PARENT"][2])
    assert [r["data"]["status"] for r in records(runtime) if r["kind"] == "child_done"] == ["cancelled"]
    assert result.status == "delivered"


def test_a_finished_background_job_wakes_an_idle_parent(tmp_path: Path) -> None:
    nap = py_command("-c", "import time; time.sleep(1.5)")
    provider = RoutedProvider(
        {
            "PARENT": [
                reply(tool_use("sh", {"command": nap, "background": True})),
                spawn("worker", "CHILD hang"),
                reply(text("Waiting.")),
                done(),
            ],
            "CHILD": [HANG],
        }
    )

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider)
        try:
            return await asyncio.wait_for(runtime.run("PARENT"), 20)
        finally:
            await runtime.close()

    assert asyncio.run(main()).status == "delivered"
    assert "[background job finished] j1" in text_of(provider.requests["PARENT"][3])


def stopped_run(tmp_path: Path) -> Runtime:
    """A run the human stopped while the model was thinking."""
    provider = RoutedProvider({"PARENT": [reply(tool_use("receipt", {"id": "r1"})), reply(text("Hmm.")), HANG]})

    async def main() -> Runtime:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider)
        task = asyncio.create_task(runtime.run("PARENT"))
        while len(provider.requests["PARENT"]) < 3:
            await asyncio.sleep(0.01)
        runtime.stop()
        assert (await task).status == "stopped"
        await runtime.close()
        return runtime

    return asyncio.run(main())


def resume_run(
    tmp_path: Path, run_id: str, script: list[Any], text: str | None = None, task: dict[str, Any] | None = None
) -> tuple[AgentResult, Any]:
    provider = ScriptedProvider(script)

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider, run_id=run_id)
        try:
            return await runtime.resume(text)
        finally:
            await runtime.close()

    return asyncio.run(main()), provider


def test_a_stopped_run_can_be_resumed_and_an_ended_one_only_with_a_message(tmp_path: Path) -> None:
    first = stopped_run(tmp_path)
    result, provider = resume_run(tmp_path, first.run_id, [done("Picked up.")])
    assert result.status == "delivered" and result.summary == "Picked up."
    messages = provider.requests[0].messages
    assert messages[-1]["role"] == "user" and "resumed this run" in messages[-1]["content"][-1]["text"]
    result, provider = resume_run(tmp_path, first.run_id, [])
    assert result.summary == "Picked up." and provider.requests == []
    result, provider = resume_run(tmp_path, first.run_id, [done("Did more.")], "Also add a test.")
    assert result.summary == "Did more."
    assert provider.requests[0].messages[-1]["content"][-1] == text("Also add a test.")
    assert [r["data"]["status"] for r in records(first) if r["kind"] == "done"] == ["stopped", "delivered", "delivered"]
    journal = first.run_dir / "root" / JOURNAL_FILE
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    journal.write_text("".join(lines[:-1]), encoding="utf-8")
    result, provider = resume_run(tmp_path, first.run_id, [done("After the crash.")])
    assert result.summary == "After the crash." and len(provider.requests) == 1


def test_the_heartbeat_lives_while_the_run_does(tmp_path: Path) -> None:
    holder: list[Runtime] = []
    seen: list[bool] = []

    def check(request: ModelRequest) -> ModelReply:
        seen.append(is_alive(holder[0].run_dir))
        return done()

    runtime, _, _ = go(tmp_path, [check], prepare=holder.append)
    assert seen == [True]
    assert not (runtime.run_dir / ALIVE_FILE).exists() and not is_alive(runtime.run_dir)


def test_resumed_ledger_does_not_count_the_time_the_run_was_down(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "manual"
    (run_dir / "root").mkdir(parents=True)
    lines = [
        (1000.0, "start", {"role": "lead", "task": "x", "cwd": str(project(tmp_path)), "parent": None}),
        (1100.0, "done", {"status": "delivered", "summary": "first"}),
        (5000.0, "resume", {}),
        (5050.0, "done", {"status": "delivered", "summary": "second"}),
    ]
    text_lines = [
        json.dumps({"seq": i, "ts": ts, "agent_id": "root", "kind": kind, "data": data})
        for i, (ts, kind, data) in enumerate(lines, 1)
    ]
    (run_dir / "root" / JOURNAL_FILE).write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    async def main() -> float:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), ScriptedProvider([]), run_id="manual")
        try:
            assert (await runtime.resume()).summary == "second"
            return runtime.ledger.hours * 3600
        finally:
            await runtime.close()

    assert abs(asyncio.run(main()) - 150.0) < 5


def test_ledger_caps_and_warnings() -> None:
    now = [0.0]
    ledger = Ledger(Budget(dollars=1.0, wall_clock_hours=1.0, per_agent_dollars=0.5, warn_at=0.5), clock=lambda: now[0])
    ledger.add("a", Usage(cost_usd=0.2))
    assert ledger.check("a") is None and ledger.warning_due("a") is None
    ledger.add("a", Usage(cost_usd=0.35))
    assert ledger.warning_due("a") == "budget.dollars is 55% used (0.55 of 1)"
    assert ledger.warning_due("a") == "budget.per_agent_dollars is 110% used (0.55 of 0.5)"
    assert ledger.warning_due("a") is None and ledger.warning_due("b") is None
    assert ledger.check("a") == "budget.per_agent_dollars" and ledger.check("b") is None
    now[0] = 3600.0
    assert ledger.check("b") == "budget.wall_clock_hours"
    assert Ledger(Budget(wall_clock_hours=1.0), elapsed=3600.0).check() == "budget.wall_clock_hours"


def test_a_new_task_on_a_delivered_run_compacts_the_finished_one_first(tmp_path: Path) -> None:
    big = Usage(input_tokens=900, output_tokens=10, cost_usd=0.001)
    rules = {"context": {"compact_boundary_tokens": 500}}
    look = tool_use("receipt", {"id": "r1"})
    script = [reply(look, usage=big), reply(look, usage=big), reply(tool_use("done", {"summary": "One."}), usage=big)]
    first, result, _ = go(tmp_path, script, rules)
    assert result.status == "delivered" and "compaction" not in kinds(first)
    summary = reply(text("## Task" + chr(10) + "First task is finished."))
    result, provider = resume_run(tmp_path, first.run_id, [summary, done("Two.")], "Second task.", rules)
    assert result.summary == "Two."
    assert "Use exactly these sections" in json.dumps(provider.requests[0].messages[-1])
    after = provider.requests[1].messages
    assert "[Summary of earlier work" in after[0]["content"][0]["text"]
    assert "follow it unchanged" in after[0]["content"][0]["text"]
    assert [m["role"] for m in after] == ["user", "assistant", "user", "assistant", "user"]
    assert after[-1]["content"][-1] == text("Second task.")
    compactions = [r["data"] for r in records(first) if r["kind"] == "compaction"]
    assert [c["keep"] for c in compactions] == [4]
    small, provider = resume_run(tmp_path, first.run_id, [done("Three.")], "Third task.", rules)
    assert small.summary == "Three." and len(provider.requests) == 1
    assert provider.requests[0].messages[0] == after[0]
