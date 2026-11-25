"""A person at the terminal (``human.present``): prose ends a turn as ``replied``, questions wait, waits are timed."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pytest
from test_agents import HANG, agent, spawn, tree
from test_loop import done, go, make_policy, project, records, results

from synthic.agents import Runtime
from synthic.loop import AgentResult
from synthic.policy import PolicyError, load_policy
from synthic.prompt import build_system
from synthic.testing import ScriptedProvider, reply, text, tool_use

PRESENT = {"human": {"present": True}}


def converse(tmp_path: Path, script: list[Any], task: dict[str, Any] | None = None) -> tuple[Runtime, AgentResult]:
    provider = ScriptedProvider(script)

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path, task or PRESENT), project(tmp_path), provider)
        try:
            return runtime, await runtime.run("hello")
        finally:
            await runtime.close()

    return asyncio.run(main())


def test_a_prose_turn_with_a_person_present_ends_as_replied_in_one_call(tmp_path: Path) -> None:
    provider = ScriptedProvider([reply(text("Hello. What shall we do?"))])

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, PRESENT), project(tmp_path), provider)
        try:
            return await runtime.run("hello")
        finally:
            await runtime.close()

    result = asyncio.run(main())
    assert result.status == "replied" and result.summary == "Hello. What shall we do?"
    assert len(provider.requests) == 1


def test_a_replied_run_continues_with_the_next_message(tmp_path: Path) -> None:
    runtime, first = converse(tmp_path, [reply(text("Hello."))])
    assert first.status == "replied"
    provider = ScriptedProvider([reply(text("Glad to help.")), done("Finished.")])

    async def again(message: str) -> AgentResult:
        resumed = Runtime(make_policy(tmp_path, PRESENT), project(tmp_path), provider, run_id=runtime.run_id)
        try:
            return await resumed.resume(message)
        finally:
            await resumed.close()

    second = asyncio.run(again("thanks"))
    assert second.status == "replied" and second.summary == "Glad to help."
    sent = provider.requests[0].messages
    assert sent[-1]["role"] == "user" and "thanks" in str(sent[-1]) and "Hello." in str(sent[-2])
    third = asyncio.run(again("now finish"))
    assert third.status == "delivered"
    assert [r["data"]["status"] for r in records(runtime) if r["kind"] == "done"] == ["replied", "replied", "delivered"]


def test_a_replied_run_without_a_message_returns_its_recorded_turn(tmp_path: Path) -> None:
    runtime, _ = converse(tmp_path, [reply(text("Hello."))])
    provider = ScriptedProvider([])

    async def main() -> AgentResult:
        resumed = Runtime(make_policy(tmp_path, PRESENT), project(tmp_path), provider, run_id=runtime.run_id)
        try:
            return await resumed.resume()
        finally:
            await resumed.close()

    assert asyncio.run(main()).status == "replied" and provider.requests == []


def test_the_reread_check_does_not_fire_for_a_replied_turn(tmp_path: Path) -> None:
    task = {**PRESENT, "verify": {"reread": True}}
    provider = ScriptedProvider([reply(text("Hi."))])

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider)
        try:
            return await runtime.run("hello")
        finally:
            await runtime.close()

    assert asyncio.run(main()).status == "replied" and len(provider.requests) == 1


def test_a_turn_with_neither_text_nor_tool_call_is_still_nudged(tmp_path: Path) -> None:
    _, result = converse(tmp_path, [reply(), reply()])
    assert result.status == "failed" and "twice" in result.summary


def test_without_a_person_prose_is_nudged_and_then_fails(tmp_path: Path) -> None:
    _, result, provider = go(tmp_path, [reply(text("One.")), reply(text("Two."))])
    assert result.status == "failed" and "Last words: Two." in result.summary and len(provider.requests) == 2


def test_a_child_that_answers_in_prose_is_still_nudged(tmp_path: Path) -> None:
    _, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD talk"), agent("wait", id="a1", seconds=30), done()],
            "CHILD": [reply(text("Just words.")), reply(text("More words."))],
        },
        PRESENT,
    )
    assert result.status == "delivered"
    assert "ended: failed" in results(provider.requests["PARENT"][2])[0]["content"]
    assert len(provider.requests["CHILD"]) == 2


def test_the_prompt_and_the_ask_tool_tell_the_truth_about_the_person(tmp_path: Path) -> None:
    absent, present = make_policy(tmp_path).policy, make_policy(tmp_path, PRESENT).policy
    role, folder = absent.roles["lead"], project(tmp_path)
    away = build_system(absent, role, folder, role.tools)[0]
    here = build_system(present, role, folder, role.tools)[0]
    assert "Nobody is watching" in away and "Nobody is watching" not in here
    assert "A person is at the terminal" in here and "end your turn with your message" in here
    assert build_system(present, role, folder, role.tools, root=False)[0] == away

    def ask_description(task: dict[str, Any] | None, where: Path) -> str:
        where.mkdir()
        provider = ScriptedProvider([reply(text("Hi."))])

        async def main() -> str:
            runtime = Runtime(make_policy(where, task), project(where), provider)
            try:
                await runtime.run("hello")
            finally:
                await runtime.close()
            return next(t["description"] for t in provider.requests[0].tools if t["name"] == "ask")

        return asyncio.run(main())

    assert "Usually nobody answers at once" in ask_description(None, tmp_path / "a")
    waits = ask_description(PRESENT, tmp_path / "b")
    assert "waits for the answer" in waits and "nobody" not in waits.lower()


def test_ask_blocks_until_it_is_answered_when_a_person_is_present(tmp_path: Path) -> None:
    ask = reply(tool_use("ask", {"question": "Blue or red?", "assumption": "blue"}))
    provider = ScriptedProvider([ask, done()])
    order: list[str] = []

    async def answerer(runtime: Runtime) -> None:
        while not runtime.questions():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        order.append("answering")
        assert [r for r in records(runtime) if r["kind"] == "tool_result"] == []
        runtime.answer(runtime.questions()[0].id, "red")

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, PRESENT), project(tmp_path), provider)
        helper = asyncio.create_task(answerer(runtime))
        try:
            return await runtime.run("hello")
        finally:
            await helper
            await runtime.close()

    result = asyncio.run(main())
    assert order == ["answering"] and result.status == "delivered"
    answer = results(provider.requests[1])[0]["content"]
    assert re.fullmatch(r"The human answered after \d+ s: red", answer)


def test_a_permission_that_resolves_to_ask_waits_when_a_person_is_present(tmp_path: Path) -> None:
    provider = ScriptedProvider([reply(tool_use("edit", {"path": "n.txt", "old": "", "new": "x"})), done()])
    task = {**PRESENT, "permissions": {"edit: *": "ask"}}

    async def approve(runtime: Runtime) -> None:
        while not runtime.questions():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        runtime.answer(runtime.questions()[0].id, "yes")

    async def main() -> AgentResult:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider)
        helper = asyncio.create_task(approve(runtime))
        try:
            return await runtime.run("hello")
        finally:
            await helper
            await runtime.close()

    assert asyncio.run(main()).status == "delivered"
    assert (project(tmp_path) / "n.txt").exists()


def test_an_unanswered_ask_says_how_long_it_waited(tmp_path: Path) -> None:
    ask = reply(tool_use("ask", {"question": "Blue or red?", "assumption": "blue"}))
    _, _, provider = go(tmp_path, [ask, done()], {"human": {"wait_seconds": 0.05}})
    assert re.search(r"No answer after \d+ s \(root-q1\)\. Proceed on your assumption: blue", results(provider.requests[1])[0]["content"])


def test_agent_wait_says_so_when_every_child_was_already_delivered(tmp_path: Path) -> None:
    _, result, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD quick"), agent("wait", id="a1", seconds=30), agent("wait", seconds=30), done()],
            "CHILD": [done("Quick.")],
        },
    )
    assert result.status == "delivered"
    first = results(provider.requests["PARENT"][2])[0]["content"]
    assert re.search(r"\(waited after \d+ s\)$", first)
    last = results(provider.requests["PARENT"][3])[0]["content"]
    assert last == "No child is running: every result was already delivered to you."


def test_agent_wait_on_a_running_child_says_how_long_it_waited(tmp_path: Path) -> None:
    _, _, provider = tree(
        tmp_path,
        {
            "PARENT": [spawn("worker", "CHILD hang"), agent("wait", id="a1", seconds=0.05), agent("cancel", id="a1"), done()],
            "CHILD": [HANG],
        },
    )
    still = results(provider.requests["PARENT"][2])[0]["content"]
    assert re.match(r"Still running \(time is up, after \d+ s\):", still)


def test_present_is_validated_and_a_policy_file_beats_the_interface_default(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".synthic").mkdir(parents=True)
    interface = {"human": {"present": True}}
    assert load_policy(None, home=home, env={}, defaults=interface).policy.human.present is True
    assert load_policy(None, home=home, env={}).policy.human.present is False
    (home / ".synthic" / "policy.toml").write_text("[human]\npresent = false\n", encoding="utf-8")
    assert load_policy(None, home=home, env={}, defaults=interface).policy.human.present is False
    (home / ".synthic" / "policy.toml").write_text('[human]\npresent = "yes"\n', encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy(None, home=home, env={})
