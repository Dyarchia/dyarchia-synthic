"""The terminal interface, offline: a scripted model, a real runtime and journal, a headless screen."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Input, ListView, Static

from synthic.policy import LoadedPolicy, load_policy
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic_tui.app import SynthicApp, bar

POLICY = """
initial_role = "lead"

[verify]
reread = false

[memory]
enabled = false

[budget]
dollars = 2

[models.test]
id = "test-model"
context_window = 200000
max_output_tokens = 8000
price_input = 1.0
price_output = 5.0
price_cache_write = 1.25
price_cache_read = 0.1

[role.lead]
model = "test"
tools = ["sh", "edit"]
"""


def done(summary: str) -> Any:
    return reply(tool_use("done", {"summary": summary, "status": "delivered"}))


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Any]:
    (tmp_path / "task.toml").write_text(POLICY, encoding="utf-8")
    project = tmp_path / "proj"
    project.mkdir()
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))

    def load() -> LoadedPolicy:
        return load_policy(project, tmp_path / "task.toml", {"journal": {"dir": (tmp_path / "runs").as_posix()}})

    return project, load


async def settle(app: SynthicApp, pilot: Any) -> None:
    for _ in range(400):
        await pilot.pause(0.02)
        if app.job is not None and app.job.done():
            await pilot.pause(0.05)
            return
    raise AssertionError("the run did not end")


async def send(pilot: Any, line: str) -> None:
    box = pilot.app.query_one("#box", Input)
    box.focus()
    box.value = line
    await pilot.press("enter")


def test_bar_is_clamped() -> None:
    assert bar(0, 10, 4) == "----" and bar(5, 10, 4) == "##--" and bar(50, 10, 4) == "####" and bar(1, 0, 4) == "----"


def test_a_task_runs_to_delivery_and_a_second_message_continues_it(world: tuple[Path, Any]) -> None:
    project, load = world
    script = [
        reply(text("Creating the file.\nOne moment."), tool_use("edit", {"path": "a.txt", "old": "", "new": "hello\n"})),
        done("Made a.txt."),
        done("Nothing else to do."),
    ]
    provider = ScriptedProvider(script)

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(140, 40)) as pilot:
            await send(pilot, "make a.txt")
            await settle(app, pilot)
            assert (project / "a.txt").read_text(encoding="utf-8") == "hello\n"
            assert app.state == "delivered"
            assert "you: make a.txt" in app.transcript and "Creating the file." in app.transcript
            assert any(line.startswith("> edit") for line in app.transcript)
            assert any("ended delivered" in line and "Made a.txt." in line for line in app.transcript)
            status = str(app.query_one("#status", Static).render())
            assert "Steps   2" in status and "of $2" in status and "compacts at" in status
            assert len(app.query_one("#run-list", ListView).children) == 1
            first, earlier = app.run_id, app.job
            await send(pilot, "anything else?")
            assert app.job is not earlier
            await settle(app, pilot)
            assert app.run_id == first and app.state == "delivered"
            assert any("Nothing else to do." in line for line in app.transcript)
            assert len(app.query_one("#run-list", ListView).children) == 1

    asyncio.run(scenario())
    assert not provider.script


def test_a_stored_run_can_be_opened_and_new_clears_the_screen(world: tuple[Path, Any]) -> None:
    project, load = world
    provider = ScriptedProvider([done("First run."), done("Second run.")])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider, task="one")
        async with app.run_test(size=(140, 40)) as pilot:
            await settle(app, pilot)
            first = app.run_id
            await pilot.press("f2")
            assert app.run_id is None and app.transcript == []
            await send(pilot, "two")
            await settle(app, pilot)
            assert app.run_id != first
            view = app.query_one("#run-list", ListView)
            assert len(view.children) == 2
            view.index = next(i for i, item in enumerate(view.children) if item.name == first)
            view.focus()
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert app.run_id == first
            assert "you: one" in app.transcript and any("First run." in line for line in app.transcript)

    asyncio.run(scenario())
