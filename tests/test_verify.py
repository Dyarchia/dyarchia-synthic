"""Verification before delivery (``[verify]``), offline: commands, the reviewer, rounds, resume, interrupts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from test_loop import done, go, make_policy, project, records, results
from test_proc import py_command, write_script

from synthic.agents import Runtime
from synthic.loop import AgentResult
from synthic.policy import Level, PolicyError, factory_defaults, merge_levels
from synthic.testing import ScriptedProvider, reply, tool_use
from synthic.types import ModelRequest, ProviderError

CHECK = "import pathlib, sys\nsys.exit(0 if pathlib.Path('ok.txt').exists() else 3)\n"


def checked(tmp_path: Path, **verify: Any) -> dict[str, Any]:
    script = write_script(project(tmp_path), "check.py", CHECK)
    task: dict[str, Any] = {"verify": {"commands": [py_command(script)], **verify}}
    task["role"] = {"checker": {"model": "test", "tools": ["sh"]}}
    return task


def create_ok() -> Any:
    return reply(tool_use("edit", {"path": "ok.txt", "old": "", "new": "ok\n"}))


def verifies(runtime: Runtime, agent_id: str = "root") -> list[dict[str, Any]]:
    return [r["data"] for r in records(runtime, agent_id) if r["kind"] == "verify"]


def test_off_by_default(tmp_path: Path) -> None:
    runtime, result, provider = go(tmp_path, [done()])
    assert result.status == "delivered" and verifies(runtime) == []
    assert "Before `done` can deliver" not in "".join(b["text"] for b in provider.requests[0].system)


def test_a_failing_command_rejects_then_the_fixed_delivery_passes(tmp_path: Path) -> None:
    script = [done("First try."), create_ok(), done("Second try.")]
    runtime, result, provider = go(tmp_path, script, checked(tmp_path))
    assert result.status == "delivered" and result.summary == "Second try."
    rejected = results(provider.requests[1])[0]
    assert rejected["is_error"] and "rejection 1 of 2 allowed" in rejected["content"]
    assert "[exit code 3]" in rejected["content"]
    assert [(v["accepted"], v["passed"]) for v in verifies(runtime)] == [(False, False), (True, True)]
    assert "Before `done` can deliver, these commands must exit 0" in "".join(
        b["text"] for b in provider.requests[0].system
    )


def test_after_the_last_round_the_delivery_is_accepted_and_marked_not_verified(tmp_path: Path) -> None:
    runtime, result, _ = go(tmp_path, [done("Try one."), done("Try two.")], checked(tmp_path, rounds=1))
    assert result.status == "delivered"
    assert result.summary.startswith("NOT VERIFIED: still failing after 1 rejection: $ ")
    assert result.summary.endswith("Try two.")
    assert [v["accepted"] for v in verifies(runtime)] == [False, True]


def test_a_command_past_its_time_is_killed_and_fails(tmp_path: Path) -> None:
    sleeper = write_script(project(tmp_path), "nap.py", "import time\ntime.sleep(30)\n")
    task = {"verify": {"commands": [py_command(sleeper)], "timeout_seconds": 0.5, "rounds": 0}}
    runtime, result, _ = go(tmp_path, [done()], task)
    assert "killed after 0.5s" in result.summary and result.summary.startswith("NOT VERIFIED")


def reviewer_saw(provider: ScriptedProvider, index: int) -> str:
    request = provider.requests[index]
    return "\n".join(b.get("text", "") for b in request.messages[0]["content"])


def test_the_reviewer_approves_with_a_fresh_context_and_its_cost_counts(tmp_path: Path) -> None:
    box: dict[str, Runtime] = {}

    def say_more(request: ModelRequest) -> Any:
        box["runtime"].add("Also keep it short.")
        return create_ok()

    def prepare(runtime: Runtime) -> None:
        box["runtime"] = runtime

    script = [say_more, done("Made ok.txt."), done("Looks complete.")]
    runtime, result, provider = go(tmp_path, script, checked(tmp_path, review="checker"), prepare, goal="Make ok.txt.")
    assert result.status == "delivered" and result.summary == "Made ok.txt."
    brief = reviewer_saw(provider, 2)
    assert "1. Make ok.txt." in brief and "2. Also keep it short." in brief and "Made ok.txt." in brief
    assert "These checks already passed" in brief and len(provider.requests[2].messages) == 1
    assert runtime.ledger.steps == 3
    assert verifies(runtime)[0]["reviewer"] == "v1" and (runtime.run_dir / "v1").is_dir()
    assert [r["kind"] for r in records(runtime, "v1")][-1] == "done"


def test_a_rejection_by_the_reviewer_returns_its_list(tmp_path: Path) -> None:
    script = [
        create_ok(),
        done("Done."),
        done("- the README was not updated", status="failed"),
        done("Done, README too."),
        done("Approved."),
    ]
    runtime, result, provider = go(tmp_path, script, checked(tmp_path, review="checker"))
    assert result.status == "delivered" and result.summary == "Done, README too."
    rejected = results(provider.requests[3])[0]
    assert rejected["is_error"] and "- the README was not updated" in rejected["content"]
    assert [v["reviewer"] for v in verifies(runtime)] == ["v1", "v2"]


def test_a_reviewer_without_a_verdict_does_not_use_a_round(tmp_path: Path) -> None:
    task = checked(tmp_path, review="checker")
    task["role"]["checker"]["max_steps"] = 1
    script = [create_ok(), done("Done."), reply(tool_use("receipt", {"id": "r9"}))]
    runtime, result, _ = go(tmp_path, script, task)
    assert result.status == "delivered"
    assert result.summary.startswith("NOT VERIFIED: the review did not finish (budget")
    assert [v["accepted"] for v in verifies(runtime)] == [True]


def test_a_child_delivery_is_not_verified(tmp_path: Path) -> None:
    script = [
        reply(tool_use("spawn", {"role": "worker", "task": "Make ok.txt."})),
        create_ok(),
        done("Child made it."),
        reply(tool_use("agent", {"action": "wait"})),
        done("Root done."),
    ]
    runtime, result, _ = go(tmp_path, script, checked(tmp_path))
    assert result.status == "delivered"
    assert verifies(runtime, "a1") == [] and [v["passed"] for v in verifies(runtime)] == [True]


def test_a_resumed_run_keeps_the_rejections_it_had(tmp_path: Path) -> None:
    task = checked(tmp_path, rounds=1)
    script = [done("One."), ProviderError("bad request", 400)]
    first, result, _ = go(tmp_path, script, task)
    assert result.status == "failed" and [v["accepted"] for v in verifies(first)] == [False]
    provider = ScriptedProvider([done("Two.")])

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider, run_id=first.run_id)
        try:
            return await runtime.resume("Go on.")
        finally:
            await runtime.close()

    resumed = asyncio.run(main())
    assert resumed.status == "delivered" and resumed.summary.startswith("NOT VERIFIED: still failing after 1")


def test_an_interrupt_during_the_checks_never_delivers(tmp_path: Path) -> None:
    sleeper = write_script(project(tmp_path), "nap.py", "import time\ntime.sleep(30)\n")
    task = {"verify": {"commands": [py_command(sleeper)]}}

    def prepare(runtime: Runtime) -> None:
        asyncio.get_running_loop().call_later(1.0, runtime.stop)

    runtime, result, _ = go(tmp_path, [done()], task, prepare)
    assert result.status == "stopped" and verifies(runtime) == []
    outcome = next(r for r in records(runtime) if r["kind"] == "tool_result")["data"]
    assert outcome["is_error"] and "interrupted during verification" in outcome["content"]


def test_the_review_role_must_exist() -> None:
    with pytest.raises(PolicyError, match="verify.review 'nobody' is not a defined role"):
        merge_levels([Level("defaults", factory_defaults()), Level("task", {"verify": {"review": "nobody"}})])
