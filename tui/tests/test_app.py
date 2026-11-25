"""The terminal interface, offline: a scripted model, a real runtime and journal, a headless screen."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from textual.widgets import Input, OptionList, Static

from synthic.policy import LoadedPolicy, deep_merge, load_policy
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic_tui.app import COMMANDS, SynthicApp

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

[models.other]
id = "other-model"
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

    def load(extra: dict[str, Any] | None = None) -> LoadedPolicy:
        overrides = deep_merge({"journal": {"dir": (tmp_path / "runs").as_posix()}}, extra or {})
        return load_policy(project, tmp_path / "task.toml", overrides)

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
    await pilot.pause(0.02)
    await pilot.press("enter")
    await pilot.pause(0.02)


def status(app: SynthicApp) -> str:
    return str(app.query_one("#status", Static).render())


def said(app: SynthicApp, piece: str) -> bool:
    return any(piece in line for line in app.transcript)


def test_a_task_runs_to_delivery_and_a_second_message_continues_it(world: tuple[Path, Any]) -> None:
    project, load = world
    script = [
        reply(text("Creating the file.\nOne moment."), tool_use("edit", {"path": "a.txt", "old": "", "new": "hello\n"})),
        reply(text("Checking."), tool_use("done", {"summary": "Made a.txt.", "status": "delivered"})),
        done("Nothing else to do."),
    ]
    provider = ScriptedProvider(script)

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, "make a.txt")
            await settle(app, pilot)
            assert (project / "a.txt").read_text(encoding="utf-8") == "hello\n"
            assert app.state == "delivered"
            assert "> make a.txt" in app.transcript and said(app, "Creating the file.")
            assert said(app, "● edit: a.txt (new file, 1 lines)  ok")
            assert said(app, "delivered: Made a.txt.") and "✓ delivered" in app.transcript
            assert not said(app, "● done") and "Checking." in app.transcript
            line = status(app)
            assert "test" in line and "2 steps" in line and "of $2" in line and "context" in line
            first, earlier = app.run_id, app.job
            await send(pilot, "anything else?")
            assert app.job is not earlier
            await settle(app, pilot)
            assert app.run_id == first and said(app, "Nothing else to do.")

    asyncio.run(scenario())
    assert not provider.script


def test_a_slash_line_is_a_command_and_never_a_task(world: tuple[Path, Any]) -> None:
    project, load = world
    provider = ScriptedProvider([done("Unused.")])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, "/model claude-haiku-5.5")
            assert said(app, "No model 'claude-haiku-5.5'. The policy has: ") and app.job is None
            await send(pilot, "/model other")
            assert said(app, "Model is now other for role lead.") and "other" in status(app)
            await send(pilot, "/effort high")
            assert "other · high" in status(app)
            await send(pilot, "/effort enormous")
            assert said(app, "No effort 'enormous'")
            await send(pilot, "/nonsense and more")
            assert said(app, "Unknown command /nonsense")
            await send(pilot, "/redirect somewhere")
            assert said(app, "/redirect works while a run is working")
            await send(pilot, "/help")
            assert all(said(app, f"/{name}") for name in COMMANDS)
            await send(pilot, "/cost")
            assert said(app, "Nothing spent")
            await send(pilot, "/context")
            assert said(app, "No context yet")
            await send(pilot, "/export")
            assert said(app, "Nothing to export")
            assert app.job is None and app.runtime is None

    asyncio.run(scenario())
    assert len(provider.script) == 1 and provider.requests == []


def test_the_chosen_model_and_effort_reach_the_next_task(world: tuple[Path, Any]) -> None:
    project, load = world
    provider = ScriptedProvider([done("Done.")])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, "/model other")
            await send(pilot, "/effort low")
            await send(pilot, "go")
            await settle(app, pilot)

    asyncio.run(scenario())
    assert provider.requests[0].model.id == "other-model" and provider.requests[0].effort == "low"


def test_the_menu_completes_commands_and_picks_a_run_to_continue(world: tuple[Path, Any]) -> None:
    project, load = world
    provider = ScriptedProvider([done("First run."), done("Second run."), done("More on the first.")])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider, task="one")
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(app, pilot)
            first = app.run_id
            await send(pilot, "/new")
            assert app.run_id is None and not said(app, "First run.")
            await send(pilot, "two")
            await settle(app, pilot)
            assert app.run_id != first

            box, menu = app.query_one("#box", Input), app.query_one("#menu", OptionList)
            box.value = "/re"
            await pilot.pause(0.05)
            assert menu.display and [menu.get_option_at_index(i).id for i in range(menu.option_count)] == ["resume", "redirect", "clear"]
            await pilot.press("tab")
            assert box.value == "/resume "
            await pilot.press("escape")
            box.value = "/res"
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.menu_mode == "resume" and menu.option_count == 2
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert app.run_id == first and not menu.display
            assert "> one" in app.transcript and said(app, "First run.") and not said(app, "Second run.")
            await send(pilot, "more")
            await settle(app, pilot)
            assert app.run_id == first and said(app, "More on the first.")
            await send(pilot, "/context")
            assert said(app, "% of the 170,000 at which it compacts")
            await send(pilot, "/export out.md")
            page = (project / "out.md").read_text(encoding="utf-8")
            assert page.startswith(f"# Run {first}") and "More on the first." in page
            await pilot.press("up")
            assert box.value == "/export out.md"

    asyncio.run(scenario())
    assert not provider.script
