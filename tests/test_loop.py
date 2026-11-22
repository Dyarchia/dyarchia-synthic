"""The loop of one agent, offline: ``ScriptedProvider``, real tools, real journal, ``tmp_path``.

Policies are built with ``merge_levels`` from small dict levels, the way a
human writes them. The helpers here are imported by ``test_agents.py``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from test_proc import async_wait_for_pids, py_command, survivors, tree_command, write_script

import synthic.agents
from synthic.agents import Runtime
from synthic.context import Context, ReceiptStore
from synthic.journal import JOURNAL_FILE
from synthic.loop import UNKNOWN_OUTCOME, AgentResult
from synthic.policy import Level, LoadedPolicy, factory_defaults, merge_levels
from synthic.prompt import build_system
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic.types import (
    HookOutcome,
    ImageData,
    ModelReply,
    ModelRequest,
    ProviderError,
    Record,
    ToolContext,
    ToolOutput,
    Usage,
)

MODEL = {
    "id": "test-model",
    "context_window": 200_000,
    "max_output_tokens": 8_000,
    "price_input": 1.0,
    "price_output": 5.0,
    "price_cache_write": 1.25,
    "price_cache_read": 0.1,
}


def make_policy(tmp_path: Path, task: dict[str, Any] | None = None) -> LoadedPolicy:
    """Defaults, a user level with test models and roles, and the test's own task level on top."""
    user = {
        "initial_role": "lead",
        "models": {"test": MODEL, "backup": {**MODEL, "id": "backup-model"}},
        "role": {
            "lead": {"model": "test", "tools": ["sh", "edit", "spawn"], "can_spawn": ["worker"]},
            "worker": {"model": "test", "tools": ["sh", "edit"]},
        },
        "journal": {"dir": str(tmp_path / "runs")},
    }
    return merge_levels([Level("defaults", factory_defaults()), Level("user", user), Level("task", task or {})])


def project(tmp_path: Path) -> Path:
    folder = tmp_path / "proj"
    folder.mkdir(exist_ok=True)
    return folder


def done(summary: str = "All done.", status: str = "delivered") -> ModelReply:
    return reply(tool_use("done", {"summary": summary, "status": status}))


def records(runtime: Runtime, agent_id: str = "root") -> list[dict[str, Any]]:
    path = runtime.run_dir / agent_id / JOURNAL_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def kinds(runtime: Runtime, agent_id: str = "root") -> list[str]:
    return [r["kind"] for r in records(runtime, agent_id)]


def results(request: ModelRequest) -> list[dict[str, Any]]:
    """The tool_result blocks of the last message of a request."""
    return [b for b in request.messages[-1]["content"] if b["type"] == "tool_result"]


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def assert_valid(messages: list[dict[str, Any]]) -> None:
    """Roles alternate and every tool_use has its tool_result in the next message."""
    for index, message in enumerate(messages):
        if index:
            assert message["role"] != messages[index - 1]["role"]
        if message["role"] == "assistant":
            calls = {b["id"] for b in message["content"] if b["type"] == "tool_use"}
            if calls:
                answered = {b["tool_use_id"] for b in messages[index + 1]["content"] if b["type"] == "tool_result"}
                assert calls <= answered


def assert_replays(runtime: Runtime, agent_id: str = "root") -> None:
    """Replaying the journal rebuilds the live context byte for byte."""
    agent = runtime.agents[agent_id]
    replayed = Context(runtime.policy.context, ReceiptStore(agent.journal.dir / "receipts"), agent.system)
    replayed.replay(Record(**r) for r in records(runtime, agent_id))
    assert dump(replayed.build()) == dump(agent.context.build())


def go(
    tmp_path: Path,
    script: list[Any],
    task: dict[str, Any] | None = None,
    prepare: Callable[[Runtime], None] | None = None,
    goal: str = "Do the task.",
) -> tuple[Runtime, AgentResult, ScriptedProvider]:
    provider = ScriptedProvider(script)

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider)
        if prepare is not None:
            prepare(runtime)
        try:
            return runtime, await runtime.run(goal)
        finally:
            await runtime.close()

    runtime, result = asyncio.run(main())
    return runtime, result, provider


def test_edit_sh_done(tmp_path: Path) -> None:
    upper = py_command("-c", "print(open('a.txt').read().upper())")
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "hello\n"})),
            reply(tool_use("sh", {"command": upper})),
            done("Wrote and checked a.txt."),
        ],
    )
    assert result.status == "delivered" and result.summary == "Wrote and checked a.txt."
    assert (project(tmp_path) / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert "HELLO" in results(provider.requests[2])[0]["content"]
    turn = ["model_call", "assistant", "tool_intent", "tool_result"]
    assert kinds(runtime) == ["start", "user", *turn, *turn, *turn, "done"]
    assert result.steps == 3 and runtime.ledger.steps == 3
    assert abs(runtime.ledger.total.cost_usd - 0.0006) < 1e-12
    assert result.usage.input_tokens == 300
    assert all(r.system == provider.requests[0].system for r in provider.requests)
    assert all(r.tools == provider.requests[0].tools for r in provider.requests)
    names = [t["name"] for t in provider.requests[0].tools]
    assert names == ["done", "ask", "receipt", "sh", "job", "edit", "spawn", "agent"]


def test_calls_run_in_parallel_and_same_resource_in_order(tmp_path: Path) -> None:
    nap = write_script(project(tmp_path), "nap.py", "import time\ntime.sleep(1.0)\n")
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": py_command(nap)}), tool_use("sh", {"command": py_command(nap)})),
            reply(
                tool_use("edit", {"path": "f.txt", "old": "", "new": "one\n"}),
                tool_use("edit", {"path": "f.txt", "old": "one", "new": "two"}),
            ),
            done(),
        ],
    )
    assert result.status == "delivered"
    log = records(runtime)
    naps = [r for r in log if r["kind"] == "tool_result" and r["data"]["name"] == "sh"]
    assert all(r["data"]["seconds"] >= 0.9 for r in naps)
    first_intent = next(r for r in log if r["kind"] == "tool_intent")
    assert naps[-1]["ts"] - first_intent["ts"] < 1.8
    assert (project(tmp_path) / "f.txt").read_text(encoding="utf-8") == "two\n"
    assert not any(b["is_error"] for b in results(provider.requests[2]))
    assert_replays(runtime)


def test_permission_deny_names_the_rule_and_does_not_run(tmp_path: Path) -> None:
    keep = project(tmp_path) / "keep.txt"
    keep.write_text("x", encoding="utf-8")
    _, result, provider = go(
        tmp_path,
        [reply(tool_use("sh", {"command": "rm keep.txt"})), done()],
        {"permissions": {"sh: rm *": "deny"}},
    )
    block = results(provider.requests[1])[0]
    assert block["is_error"] and "'sh: rm *'" in block["content"]
    assert keep.exists() and result.status == "delivered"


def test_permission_ask_without_wait_uses_the_fallback(tmp_path: Path) -> None:
    for fallback, created in (("deny", False), ("allow", True)):
        base = tmp_path / fallback
        base.mkdir()
        _, _, provider = go(
            base,
            [reply(tool_use("edit", {"path": "n.txt", "old": "", "new": "x"})), done()],
            {"permissions": {"edit: *": "ask"}, "human": {"permission_fallback": fallback}},
        )
        assert (project(base) / "n.txt").exists() is created
        assert results(provider.requests[1])[0]["is_error"] is not created


def test_permission_ask_with_wait_is_granted_by_answer(tmp_path: Path) -> None:
    async def approve(runtime: Runtime) -> None:
        while not runtime.questions():
            await asyncio.sleep(0.01)
        question = runtime.questions()[0]
        assert question.kind == "permission" and "edit n.txt" in question.text
        runtime.answer(question.id, "yes")

    provider = ScriptedProvider([reply(tool_use("edit", {"path": "n.txt", "old": "", "new": "x"})), done()])

    async def main() -> AgentResult:
        policy = make_policy(tmp_path, {"permissions": {"edit: *": "ask"}, "human": {"wait_seconds": 10}})
        runtime = Runtime(policy, project(tmp_path), provider)
        helper = asyncio.create_task(approve(runtime))
        try:
            return await runtime.run("Create n.txt.")
        finally:
            await helper
            await runtime.close()

    started = time.monotonic()
    assert asyncio.run(main()).status == "delivered"
    assert time.monotonic() - started < 5
    assert (project(tmp_path) / "n.txt").exists()


def test_hooks_block_replace_and_add_text(tmp_path: Path) -> None:
    marker = project(tmp_path) / "ran.txt"

    async def before_tool(event: str, payload: dict[str, Any]) -> HookOutcome:
        if payload["tool"] == "sh":
            return HookOutcome("block", "no shell today")
        if payload["tool"] == "edit":
            return HookOutcome(replace={"input": {**payload["input"], "new": "good"}})
        return HookOutcome()

    async def after_tool(event: str, payload: dict[str, Any]) -> HookOutcome:
        return HookOutcome(add_text=f"note after {payload['tool']}")

    def prepare(runtime: Runtime) -> None:
        runtime.hooks.subscribe("before_tool", before_tool)
        runtime.hooks.subscribe("after_tool", after_tool)

    touch = py_command("-c", "open('ran.txt', 'w').close()")
    runtime, _, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": touch}), tool_use("edit", {"path": "e.txt", "old": "", "new": "bad"})),
            done(),
        ],
        prepare=prepare,
    )
    blocked, edited = results(provider.requests[1])
    assert blocked["is_error"] and "no shell today" in blocked["content"]
    assert not marker.exists()
    assert not edited["is_error"]
    assert (project(tmp_path) / "e.txt").read_text(encoding="utf-8") == "good"
    last = provider.requests[1].messages[-1]["content"]
    assert last[-1] == {"type": "text", "text": "note after edit"}
    assert_replays(runtime)


def test_stop_before_tools_start_runs_none_of_them(tmp_path: Path) -> None:
    holder: list[Runtime] = []

    async def stop_now(event: str, payload: dict[str, Any]) -> HookOutcome:
        holder[0].stop()
        return HookOutcome()

    def prepare(runtime: Runtime) -> None:
        holder.append(runtime)
        runtime.hooks.subscribe("before_tool", stop_now)

    runtime, result, _ = go(
        tmp_path,
        [
            reply(
                tool_use("edit", {"path": "one.txt", "old": "", "new": "1"}),
                tool_use("edit", {"path": "two.txt", "old": "", "new": "2"}),
            )
        ],
        {"human": {"permission_fallback": "allow"}},
        prepare,
    )
    assert result.status == "stopped"
    assert not (project(tmp_path) / "one.txt").exists() and not (project(tmp_path) / "two.txt").exists()
    assert_valid(runtime.agents["root"].context.build()[1])


def test_before_done_block_then_pass(tmp_path: Path) -> None:
    calls = []

    async def gate(event: str, payload: dict[str, Any]) -> HookOutcome:
        calls.append(payload["summary"])
        return HookOutcome("block", "run the tests first") if len(calls) == 1 else HookOutcome()

    _, result, provider = go(
        tmp_path, [done("first try"), done("second try")], prepare=lambda r: r.hooks.subscribe("before_done", gate)
    )
    first = results(provider.requests[1])[0]
    assert first["is_error"] and "run the tests first" in first["content"]
    assert result.status == "delivered" and result.summary == "second try"


class BoomTool:
    name = "mcp"
    description = "Always raises."
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        raise RuntimeError("kaboom")


def test_unknown_tool_and_raising_tool_become_error_results(tmp_path: Path) -> None:
    def prepare(runtime: Runtime) -> None:
        runtime.plugin_tools["mcp"] = BoomTool()

    _, result, provider = go(
        tmp_path,
        [reply(tool_use("frobnicate", {}), tool_use("mcp", {})), done()],
        {"role": {"lead": {"tools": ["sh", "mcp"]}}},
        prepare,
    )
    unknown, boom = results(provider.requests[1])
    assert unknown["is_error"] and "Unknown tool 'frobnicate'" in unknown["content"]
    assert "done, ask, receipt, sh, job, mcp" in unknown["content"]
    assert boom["is_error"] and "RuntimeError: kaboom" in boom["content"]
    assert result.status == "delivered"


def test_end_turn_without_done_reminds_once_then_fails(tmp_path: Path) -> None:
    _, result, provider = go(tmp_path, [reply(text("Thinking.")), reply(text("Still thinking."))])
    assert "call `done`" in provider.requests[1].messages[-1]["content"][-1]["text"]
    assert result.status == "failed" and "Still thinking." in result.summary
    again = tmp_path / "again"
    again.mkdir()
    _, result, _ = go(again, [reply(text("Thinking.")), done()])
    assert result.status == "delivered"


def test_repetition_guard_warns_then_fails(tmp_path: Path) -> None:
    same = [reply(tool_use("receipt", {"id": "r99"})) for _ in range(4)]
    _, result, provider = go(tmp_path, same, {"agents": {"repeat_limit": 2}})
    assert "same receipt call 2 times" in dump(provider.requests[2].messages[-1])
    assert len(provider.requests) == 4
    assert result.status == "failed" and "4 times" in result.summary


def test_max_tokens_retries_at_the_model_limit_and_continues_cut_text(tmp_path: Path) -> None:
    cut_call = reply(tool_use("edit", {}), stop_reason="max_tokens")
    cut_text = reply(text("Half a sentence"), stop_reason="max_tokens")
    _, result, provider = go(tmp_path, [cut_call, cut_text, done()], {"role": {"lead": {"max_output_tokens": 100}}})
    first, second, third = provider.requests
    assert first.max_output_tokens == 100 and second.max_output_tokens == 8_000
    assert second.messages == first.messages
    assert third.max_output_tokens == 100
    assert third.messages[-2] == {"role": "assistant", "content": [text("Half a sentence")]}
    assert "cut off" in third.messages[-1]["content"][-1]["text"]
    assert result.status == "delivered"


def test_provider_failure_with_and_without_fallback(tmp_path: Path) -> None:
    runtime, result, provider = go(
        tmp_path,
        [ProviderError("overloaded", 529, retryable=True), done()],
        {"models": {"test": {"fallback": "backup"}}},
    )
    assert result.status == "delivered"
    assert [r.model.id for r in provider.requests] == ["test-model", "backup-model"]
    assert any(r["kind"] == "fallback" and r["data"]["to"] == "backup" for r in records(runtime))
    other = tmp_path / "other"
    other.mkdir()
    _, result, _ = go(other, [ProviderError("bad request", 400)])
    assert result.status == "failed" and "bad request" in result.summary


def test_budget_cap_ends_the_run_and_warns_once(tmp_path: Path) -> None:
    warnings: list[str] = []

    def prepare(runtime: Runtime) -> None:
        runtime.on_output(lambda agent, kind, line: warnings.append(line) if kind == "warning" else None)

    script = [reply(tool_use("receipt", {"id": f"r{n}"})) for n in range(1, 6)]
    runtime, result, provider = go(tmp_path, script, {"budget": {"dollars": 0.0005, "warn_at": 0.5}}, prepare)
    assert result.status == "budget" and "budget.dollars" in result.summary
    assert len(provider.requests) == 3
    assert len([w for w in warnings if "budget.dollars" in w]) == 1
    assert "[budget]" in dump(provider.requests[2].messages[-1])


def test_large_output_is_cut_with_a_receipt_the_model_can_query(tmp_path: Path) -> None:
    lines = py_command("-c", "[print(f'line {n}') for n in range(1, 5001)]")
    _, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": lines})),
            reply(tool_use("receipt", {"id": "r1", "search": "^line 4321$"})),
            done(),
        ],
        {"context": {"output_cap": 2000, "output_head": 500, "output_tail": 500}},
    )
    cut = results(provider.requests[1])[0]["content"]
    assert "output truncated" in cut and "receipt r1" in cut and len(cut) < 1500
    assert "4321: line 4321" in results(provider.requests[2])[0]["content"]
    assert result.status == "delivered"


def test_background_handoff_notice_reaches_the_model_later(tmp_path: Path) -> None:
    slow = py_command("-c", "import time; time.sleep(1.5); print('finished')")
    _, result, provider = go(
        tmp_path,
        [
            reply(tool_use("sh", {"command": slow})),
            reply(tool_use("ask", {"question": "Which color?", "assumption": "blue"})),
            done(),
        ],
        {"environment": {"tool_timeout_seconds": 0.5}, "human": {"wait_seconds": 2.5}},
    )
    assert "still running after 0.5s as job j1" in results(provider.requests[1])[0]["content"]
    assert "[background job finished] j1 exited with code 0" in dump(provider.requests[2].messages[-1])
    assert result.status == "delivered"


def test_stop_during_a_long_command_leaves_nothing_running(tmp_path: Path) -> None:
    folder = project(tmp_path)
    provider = ScriptedProvider([reply(tool_use("sh", {"command": tree_command(folder, "stay")}))])

    async def main() -> tuple[Runtime, AgentResult, list[int]]:
        runtime = Runtime(make_policy(tmp_path), folder, provider)
        task = asyncio.create_task(runtime.run("Run the tree."))
        pids = await async_wait_for_pids(folder)
        runtime.stop()
        result = await task
        await runtime.close()
        return runtime, result, pids

    runtime, result, pids = asyncio.run(main())
    assert result.status == "stopped"
    assert survivors(pids) == []
    _, messages = runtime.agents["root"].context.build()
    assert_valid(messages)
    assert messages[-1]["content"][0]["type"] == "tool_result"
    log = kinds(runtime)
    assert log[-1] == "done" and log.index("interrupt") < log.index("done")


def test_add_text_appears_in_the_next_request(tmp_path: Path) -> None:
    holder: list[Runtime] = []

    def first(request: ModelRequest) -> ModelReply:
        holder[0].add("Also mention the weather.")
        return reply(tool_use("receipt", {"id": "r1"}))

    _, result, provider = go(tmp_path, [first, done()], prepare=holder.append)
    last = provider.requests[1].messages[-1]["content"]
    assert last[0]["type"] == "tool_result" and last[-1] == text("Also mention the weather.")
    assert result.status == "delivered"


def test_redirect_aborts_the_command_and_continues(tmp_path: Path) -> None:
    folder = project(tmp_path)
    sleeper = write_script(
        folder, "sleeper.py", "import os, sys, time\nopen(sys.argv[1], 'w').write(str(os.getpid()))\ntime.sleep(30)\n"
    )
    pid_file = folder / "sleeper.pid"
    provider = ScriptedProvider(
        [
            reply(tool_use("sh", {"command": py_command(sleeper, pid_file)})),
            reply(tool_use("sh", {"command": py_command("-c", "print('fresh start')")})),
            done(),
        ]
    )

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path), folder, provider)
        task = asyncio.create_task(runtime.run("Sleep."))
        while not pid_file.exists() or not pid_file.read_text():
            await asyncio.sleep(0.02)
        runtime.redirect("Change of plan: print something.")
        result = await task
        await runtime.close()
        return result

    result = asyncio.run(main())
    assert survivors([int(pid_file.read_text())]) == []
    second = provider.requests[1].messages[-1]["content"]
    assert second[0]["type"] == "tool_result" and second[0]["is_error"]
    assert second[-1] == text("Change of plan: print something.")
    assert "fresh start" in results(provider.requests[2])[0]["content"]
    assert result.status == "delivered"


class HangingProvider:
    """Never answers: waits for the cancel event, as the real provider does while streaming."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(
        self, request: ModelRequest, on_text: Any = None, cancel: asyncio.Event | None = None
    ) -> ModelReply:
        self.requests.append(request)
        assert cancel is not None
        await cancel.wait()
        raise asyncio.CancelledError


def test_stop_during_a_model_call(tmp_path: Path) -> None:
    provider = HangingProvider()

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider)
        task = asyncio.create_task(runtime.run("Think hard."))
        while not provider.requests:
            await asyncio.sleep(0.01)
        runtime.stop()
        result = await task
        await runtime.close()
        return runtime, result

    runtime, result = asyncio.run(main())
    assert result.status == "stopped"
    assert "assistant" not in kinds(runtime)


def test_ask_is_non_blocking_with_both_fallbacks(tmp_path: Path) -> None:
    question = reply(tool_use("ask", {"question": "Blue or red?", "assumption": "blue"}))
    _, result, provider = go(tmp_path, [question, done()])
    assert "Proceed on your assumption: blue" in results(provider.requests[1])[0]["content"]
    assert result.status == "delivered"
    other = tmp_path / "stop"
    other.mkdir()
    runtime, result, provider = go(other, [question], {"human": {"question_fallback": "stop"}})
    assert result.status == "stopped" and "Blue or red?" in result.summary
    assert runtime.questions() == []


def test_late_answer_arrives_as_a_user_message(tmp_path: Path) -> None:
    holder: list[Runtime] = []

    def second(request: ModelRequest) -> ModelReply:
        runtime = holder[0]
        assert [q.id for q in runtime.questions()] == ["root-q1"]
        runtime.answer("root-q1", "Use red.")
        return reply(tool_use("receipt", {"id": "r1"}))

    question = reply(tool_use("ask", {"question": "Blue or red?", "assumption": "blue"}))
    runtime, result, provider = go(tmp_path, [question, second, done()], prepare=holder.append)
    last = provider.requests[2].messages[-1]["content"][-1]["text"]
    assert "root-q1" in last and "Blue or red?" in last and "Use red." in last
    assert "answer" in kinds(runtime)
    assert result.status == "delivered"


def test_secrets_never_reach_requests_journal_or_receipts_and_replay_is_identical(tmp_path: Path) -> None:
    secret = "sk-" + "A1b2C3d4" * 4
    leak = py_command("-c", "print('token: sk-' + 'A1b2C3d4' * 4); print('x' * 900)")
    runtime, result, provider = go(tmp_path, [reply(tool_use("sh", {"command": leak})), done()])
    assert result.status == "delivered"
    assert secret not in dump([r.messages for r in provider.requests])
    assert "[masked]" in results(provider.requests[1])[0]["content"]
    root_dir = runtime.run_dir / "root"
    assert secret not in (root_dir / JOURNAL_FILE).read_text(encoding="utf-8")
    receipts = list((root_dir / "receipts").glob("r*.txt"))
    assert receipts and all(secret not in p.read_text(encoding="utf-8") for p in receipts)
    assert_replays(runtime)


def test_compaction_replaces_the_history_with_the_summary(tmp_path: Path) -> None:
    big = Usage(input_tokens=900, output_tokens=10, cost_usd=0.001)
    runtime, result, provider = go(
        tmp_path,
        [reply(tool_use("receipt", {"id": "r1"}), usage=big), reply(text("## Task\nKeep going.")), done()],
        {"models": {"test": {"context_window": 1000}}, "context": {"compact_at": 0.5}},
    )
    assert "Use exactly these sections" in dump(provider.requests[1].messages[-1])
    after = provider.requests[2].messages
    assert len(after) == 1 and after[0]["content"][0]["text"].startswith("[Summary of earlier work")
    assert "compaction" in kinds(runtime)
    assert result.status == "delivered" and result.steps == 2


def test_system_prompt_is_short_fixed_and_shows_limits(tmp_path: Path) -> None:
    folder = project(tmp_path)
    (folder / "AGENTS.md").write_text("Repo says hi.", encoding="utf-8")
    rules = {"permissions": {"sh: git push*": "deny"}, "budget": {"dollars": 2}}
    policy = make_policy(tmp_path, rules).policy
    role = policy.roles["lead"]
    tools = ["done", "ask", "receipt", "sh", "job", "edit", "spawn", "agent"]
    texts = build_system(policy, role, folder, tools)
    assert texts == build_system(policy, role, folder, tools)
    assert "Repo says hi." not in "".join(texts)
    assert "Denied: sh: git push*" in texts[-1] and "$2.00 in total" in texts[-1] and "may spawn: worker" in texts[-1]
    opened = make_policy(tmp_path, {**rules, "instructions": {"load_repo_files": True}}).policy
    wrapped = build_system(opened, role, folder, tools)
    assert any("comes from the repository" in t and "Repo says hi." in t for t in wrapped)
    hidden = make_policy(tmp_path, {"visible_to_agent": False}).policy
    assert not any("Limits set by the human" in t for t in build_system(hidden, role, folder, tools))
    assert "nested at most 2 levels below the root agent" in texts[-1]


def test_engine_tools_skip_permissions_but_not_hooks(tmp_path: Path) -> None:
    seen: list[str] = []

    async def watch(event: str, payload: dict[str, Any]) -> HookOutcome:
        seen.append(payload["tool"])
        return HookOutcome()

    script = [
        reply(tool_use("receipt", {"id": "r9"}), tool_use("sh", {"command": "echo hi"})),
        reply(tool_use("ask", {"question": "Blue?", "assumption": "blue"})),
        done(),
    ]
    every = ({"default_permission": "deny"}, {"default_permission": "ask"}, {"permissions": {"*: *": "deny"}})
    for index, rules in enumerate(every):
        base = tmp_path / str(index)
        base.mkdir()
        _, result, provider = go(base, script, rules, lambda r: r.hooks.subscribe("before_tool", watch))
        contents = sorted(b["content"] for b in results(provider.requests[1]))
        assert contents[0].startswith("Denied by") and contents[1].startswith("Unknown receipt id")
        assert "Proceed on your assumption" in results(provider.requests[2])[0]["content"]
        assert result.status == "delivered"
    assert seen.count("done") == 3 and "sh" not in seen


SECRET = "sk-" + "A1b2C3d4" * 4
PICTURE = "AKIAIOSFODNN7EXAMPLE"


class PictureTool:
    name = "skill"
    description = "Returns a picture."
    input_schema: dict[str, Any] = {"type": "object", "properties": {}}

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput("picture", images=[ImageData("image/png", PICTURE)])


def no_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**m, "content": [b for b in m["content"] if b["type"] != "thinking"]} for m in messages]


def test_masked_reply_replays_identically_and_resume_reuses_the_start_record(tmp_path: Path) -> None:
    thinking = {"type": "thinking", "thinking": "", "signature": SECRET}
    edit = tool_use("edit", {"path": "k.txt", "old": "", "new": SECRET})
    late = {"type": "thinking", "thinking": "", "signature": "late"}
    script = [reply(thinking, text(f"Using {SECRET}"), edit, late), reply(tool_use("skill", {})), done()]
    first_task = {"role": {"lead": {"tools": ["sh", "edit", "skill"], "instructions": "Version one."}}}

    def prepare(runtime: Runtime) -> None:
        runtime.plugin_tools["skill"] = PictureTool()

    first, result, first_provider = go(tmp_path, script, first_task, prepare)
    assert result.status == "delivered"
    assert (project(tmp_path) / "k.txt").read_text(encoding="utf-8") == SECRET
    live = first_provider.requests[2].messages
    masked_edit = {**edit, "input": {"path": "k.txt", "old": "", "new": "[masked]"}}
    assert live[1]["content"] == [thinking, text("Using [masked]"), masked_edit]
    assert live[4]["content"][0]["content"][1]["source"]["data"] == PICTURE
    assert_replays(first)
    start = records(first)[0]["data"]
    assert start["system"] == first.agents["root"].system and start["tools"] == first_provider.requests[0].tools

    journal = first.run_dir / "root" / JOURNAL_FILE
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    intents = [i for i, line in enumerate(lines) if '"kind": "tool_intent"' in line]
    journal.write_text("".join(lines[: intents[1] + 1]), encoding="utf-8")
    provider = ScriptedProvider([done("Recovered.")])
    second_task = {"role": {"lead": {"tools": ["sh", "edit"], "instructions": "Version two."}}}

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, second_task), project(tmp_path), provider, run_id=first.run_id)
        try:
            return await runtime.resume()
        finally:
            await runtime.close()

    assert asyncio.run(main()).status == "delivered"
    resumed = provider.requests[0]
    assert resumed.system == first_provider.requests[0].system and resumed.tools == first_provider.requests[0].tools
    assert resumed.messages[:4] == no_thinking(live[:4])
    assert results(resumed)[0]["content"] == UNKNOWN_OUTCOME


def test_fallback_is_used_for_one_call_then_the_role_model_returns(tmp_path: Path) -> None:
    runtime, result, provider = go(
        tmp_path,
        [ProviderError("overloaded", 529, retryable=True), reply(tool_use("receipt", {"id": "r1"})), done()],
        {"models": {"test": {"fallback": "backup"}}},
    )
    assert result.status == "delivered"
    assert [r.model.id for r in provider.requests] == ["test-model", "backup-model", "test-model"]
    assert [r["data"]["to"] for r in records(runtime) if r["kind"] == "fallback"] == ["backup"]


def test_runtime_builds_its_provider_from_the_provider_table(tmp_path: Path, monkeypatch: Any) -> None:
    built: list[dict[str, Any]] = []

    class FakeProvider:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(synthic.agents, "AnthropicProvider", FakeProvider)

    async def main() -> None:
        policy = make_policy(tmp_path, {"provider": {"retry_seconds": 7200, "timeout_seconds": 90}})
        await Runtime(policy, project(tmp_path)).close()

    asyncio.run(main())
    assert built == [{"cache_ttl": "5m", "retry_seconds": 7200.0, "timeout_seconds": 90.0}]


def compaction_requests(provider: ScriptedProvider) -> list[int]:
    return [i for i, r in enumerate(provider.requests) if "Use exactly these sections" in dump(r.messages[-1])]


def test_cut_compaction_is_rejected_and_held_back_until_the_prompt_grows(tmp_path: Path) -> None:
    def used(tokens: int) -> Usage:
        return Usage(input_tokens=tokens, output_tokens=10, cost_usd=0.001)

    rules = {
        "models": {"test": {"context_window": 1000, "extra": {"service_tier": "auto"}}},
        "role": {"lead": {"extra": {"speed": "fast"}}},
        "context": {"compact_at": 0.5},
    }
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("receipt", {"id": "r1"}), usage=used(900)),
            reply(text("## Task\nhalf"), stop_reason="max_tokens", usage=used(900)),
            reply(tool_use("receipt", {"id": "r2"}), usage=used(920)),
            reply(tool_use("receipt", {"id": "r3"}), usage=used(960)),
            reply(text("## Task\nKeep going."), usage=used(960)),
            done(),
        ],
        rules,
    )
    assert result.status == "delivered"
    assert compaction_requests(provider) == [1, 4]
    assert [k for k in kinds(runtime) if k == "compaction"] == ["compaction"]
    assert all(r.extra == {"service_tier": "auto", "speed": "fast"} for r in provider.requests)
    assert abs(runtime.ledger.total.cost_usd - 0.0052) < 1e-12


def test_prompt_too_long_compacts_then_retries_once(tmp_path: Path) -> None:
    too_long = ProviderError("400 invalid_request_error: prompt is too long: 1200 > 1000", 400)
    _, result, provider = go(tmp_path, [too_long, reply(text("## Task\nShort.")), done()])
    assert result.status == "delivered" and compaction_requests(provider) == [1]
    again = tmp_path / "again"
    again.mkdir()
    large = ProviderError("413 request_too_large: no message", 413)
    _, result, _ = go(again, [large, reply(text("## Task\nShort.")), large])
    assert result.status == "failed" and "413" in result.summary


def test_cut_text_drops_a_trailing_unsigned_thinking_block(tmp_path: Path) -> None:
    unsigned = {"type": "thinking", "thinking": "half a thought", "signature": ""}
    _, result, provider = go(tmp_path, [reply(text("Half"), unsigned, stop_reason="max_tokens"), done()])
    assert provider.requests[1].messages[-2] == {"role": "assistant", "content": [text("Half")]}
    assert result.status == "delivered"


def test_replaced_input_is_checked_again(tmp_path: Path) -> None:
    async def swap(event: str, payload: dict[str, Any]) -> HookOutcome:
        return HookOutcome(replace={"input": {**payload["input"], "path": "secret.txt"}})

    _, result, provider = go(
        tmp_path,
        [reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "x"})), done()],
        {"permissions": {"edit: secret*": "deny"}},
        lambda r: r.hooks.subscribe("before_tool", swap),
    )
    block = results(provider.requests[1])[0]
    assert block["is_error"] and "'edit: secret*'" in block["content"]
    assert not (project(tmp_path) / "secret.txt").exists() and not (project(tmp_path) / "a.txt").exists()
    assert result.status == "delivered"


def test_never_send_masks_the_system_prompt(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("Ticket internal-4321 is ours.", encoding="utf-8")
    rules = {
        "never_send": [r"internal-\d{4}"],
        "instructions": {"files": [str(notes)]},
        "role": {"lead": {"instructions": "Mind internal-1234."}},
    }
    _, _, provider = go(tmp_path, [done()], rules)
    system = dump(provider.requests[0].system)
    assert "internal-" not in system and system.count("[masked]") == 2


def test_stop_cuts_a_slow_hook_and_an_interrupted_gate_does_not_deliver(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def slow_gate(event: str, payload: dict[str, Any]) -> HookOutcome:
        started.set()
        await asyncio.sleep(30)
        return HookOutcome()

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), ScriptedProvider([done()]))
        runtime.hooks.subscribe("before_done", slow_gate)
        task = asyncio.create_task(runtime.run("Finish."))
        await started.wait()
        runtime.stop()
        result = await task
        await runtime.close()
        return runtime, result

    began = time.monotonic()
    runtime, result = asyncio.run(main())
    assert time.monotonic() - began < 5
    assert result.status == "stopped"
    assert any(r["kind"] == "hook" and r["data"]["event"] == "before_done" for r in records(runtime))
