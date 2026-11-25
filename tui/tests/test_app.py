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
from synthic_tui.board import Board, Step
from synthic_tui.views import VIEWS, AgentCard, FleetColumn, Row, Stream

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


def step(app: SynthicApp, kind: str, piece: str) -> Step:
    return next(s for s in app.board.steps if s.kind == kind and piece in s.text)


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
            assert app.view_key == "column"
            await send(pilot, "make a.txt")
            await settle(app, pilot)
            assert (project / "a.txt").read_text(encoding="utf-8") == "hello\n"
            assert app.state == "delivered"
            assert step(app, "user", "make a.txt") and step(app, "text", "Creating the file.")
            edit = step(app, "tool", "edit: a.txt (new file, 1 lines)")
            assert edit.result == "ok" and edit.seconds is not None and not edit.error
            assert step(app, "end", "Made a.txt.").result == "delivered" and step(app, "text", "Checking.")
            assert not any(s.text.startswith("done") for s in app.board.steps)
            assert sum(s.cost for s in app.board.steps) == pytest.approx(app.board.total) and app.board.total > 0
            line = status(app)
            assert "test" in line and "2 steps" in line and "of 2,00 $" in line and "context" in line
            assert len(app.query_one("#column").query_one(Stream).items) == len(app.board.steps)
            first, earlier = app.run_id, app.job
            await send(pilot, "anything else?")
            assert app.job is not earlier
            await settle(app, pilot)
            assert app.run_id == first and step(app, "end", "Nothing else to do.")

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
            await send(pilot, "/view sideways")
            assert said(app, "No view 'sideways'")
            assert app.job is None and app.runtime is None and not app.board.steps

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
            assert app.run_id is None and not app.board.steps
            await send(pilot, "two")
            await settle(app, pilot)
            assert app.run_id != first

            box, menu = app.query_one("#box", Input), app.query_one("#menu", OptionList)
            box.value = "/re"
            await pilot.pause(0.05)
            assert menu.display and [menu.get_option_at_index(i).id for i in range(menu.option_count)] == ["resume", "redirect", "clear"]
            assert app.query_one("#views").has_class("veiled")
            await pilot.press("tab")
            assert box.value == "/resume "
            await pilot.press("escape")
            assert not app.query_one("#views").has_class("veiled")
            box.value = "/res"
            await pilot.pause(0.05)
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert app.menu_mode == "resume" and menu.option_count == 2
            await pilot.press("down")
            await pilot.press("enter")
            await pilot.pause(0.1)
            assert app.run_id == first and not menu.display
            texts = [s.text for s in app.board.steps]
            assert "one" in texts and "First run." in texts and "Second run." not in texts
            await send(pilot, "more")
            await settle(app, pilot)
            assert app.run_id == first and step(app, "end", "More on the first.")
            await send(pilot, "/context")
            assert said(app, "% of the 170.000 at which it compacts")
            await send(pilot, "/export out.md")
            page = (project / "out.md").read_text(encoding="utf-8")
            assert page.startswith(f"# Run {first}") and "More on the first." in page
            await pilot.press("up")
            assert box.value == "/export out.md"

    asyncio.run(scenario())
    assert not provider.script


def test_every_view_draws_the_same_run_and_the_choice_is_remembered(world: tuple[Path, Any]) -> None:
    project, load = world
    (project / "a.txt").write_text("one\ntwo\n", encoding="utf-8")
    script = [
        reply(text("Changing it."), tool_use("edit", {"path": "a.txt", "old": "two\n", "new": "2\n3\n"})),
        done("Changed a.txt."),
    ]
    provider = ScriptedProvider(script)

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(140, 40)) as pilot:
            await send(pilot, "/view ledger")
            assert app.view_key == "ledger" and app.query_one("#tab-ledger").has_class("active")
            await send(pilot, "change a.txt")
            await settle(app, pilot)
            ledger = app.query_one("#ledger")
            rows = list(ledger.query(Row))
            assert len(rows) == len(app.board.steps) == 4
            edit = step(app, "tool", "edit: a.txt (+2 -1)")
            line = next(str(row.render()) for row in rows if row.step is edit)
            assert "edit: a.txt (+2 -1)" in line and " ok " in line and "0,0s" in line
            assert app.selected is app.board.steps[-1] and app.follow
            await pilot.press("ctrl+up")
            assert app.selected is edit and not app.follow and not edit.fetched
            await pilot.pause(0.5)
            assert edit.fetched
            shown =str(ledger.query_one(".detail-text", Static).render())
            assert "-two" in shown and "+2" in shown and "+3" in shown
            await pilot.press("ctrl+t")
            assert app.view_key == "column" and app.focused is app.query_one("#box", Input)
            await pilot.press("ctrl+t")
            panes = app.query_one("#panes")
            assert app.view_key == "panes" and len(list(panes.query(AgentCard))) == 1
            assert "-two" in str(panes.query_one(".detail-text", Static).render())
            assert "BUDGET" in str(panes.query_one("Meters").render())
            await pilot.press("ctrl+t")
            assert app.view_key == "fleet" and len(list(app.query_one("#fleet").query(FleetColumn))) == 1
            await pilot.pause(0.1)
            assert "SPENT" in str(app.query_one("Totals").render())
            assert len(app.query_one("#fleet").query_one(Stream).items) == 4

        again = SynthicApp(project, load, lambda: provider)
        async with again.run_test(size=(140, 40)):
            assert again.view_key == "fleet"

    asyncio.run(scenario())
    assert set(VIEWS) == {"column", "panes", "fleet", "ledger"}


def test_several_agents_and_a_question_reach_every_view(world: tuple[Path, Any]) -> None:
    project, load = world

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=(160, 44)) as pilot:
            board = app.board
            board.said("add the check")
            board.feed("root", "agent_start", "lead: add the check")
            board.feed("root", "tool_start", "spawn: apex: write it")
            board.feed("root", "tool_end", "spawn ok in 0.4s")
            for agent_id, role in (("a1", "apex"), ("a2", "tester"), ("a3", "tester")):
                board.feed(agent_id, "agent_start", f"{role}: its part")
            board.feed("a1", "text", "Writing ")
            board.feed("a1", "text", "the check.")
            board.feed("a1", "tool_start", "sh: sf project deploy start")
            board.measure("a1", 0.5, 3, 4200)
            board.feed("a1", "tool_end", "sh exit 1 in 12.5s")
            board.feed("a2", "question", "a2-q1: Full suite or one class?")
            board.feed("a3", "agent_end", "delivered: Covered.")
            app.refresh_status()
            await pilot.pause(0.05)
            assert [v.state for v in board.agents.values()] == ["working", "working", "waiting", "delivered"]
            assert step(app, "tool", "sf project deploy").cost == pytest.approx(0.5)
            deploy = step(app, "tool", "sf project deploy")
            assert deploy.result == "exit 1" and deploy.error and deploy.seconds == 12.5
            assert "◆ 1 waiting" in status(app) and "3 of 4 agents" in status(app)

            await send(pilot, "/view fleet")
            await pilot.pause(0.1)
            lanes = list(app.query_one("#fleet").query(FleetColumn))
            assert [lane.view.id for lane in lanes] == ["root", "a1", "a2", "a3"]
            head = str(lanes[2].query_one("FleetHead").render())
            assert "tester a2" in head and "◆ waiting" in head
            await pilot.press("shift+down")
            assert app.focus_agent == "a1"
            board.feed("a4", "agent_start", "apex: a late one")
            board.feed("a4", "tool_start", "sh: echo late")
            await pilot.pause(0.1)
            late = next(lane for lane in app.query_one("#fleet").query(FleetColumn) if lane.view.id == "a4")
            assert len(late.query_one(Stream).items) == 1

            await send(pilot, "/view panes")
            panes = app.query_one("#panes")
            assert len(list(panes.query(AgentCard))) == 5
            assert len(panes.query_one(Stream).items) == len(board.agents["a1"].steps) == 2
            await send(pilot, "/view column")
            column = app.query_one("#column").query_one(Stream)
            assert any("apex" in str(entry.render()) for entry in column.items.values())
            board.answered("a2-q1")
            assert board.agents["a2"].state == "working" and not board.waiting()

    asyncio.run(scenario())


def test_a_view_that_fails_to_draw_never_ends_the_run(world: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    project, load = world
    script = [reply(text("Looking."), tool_use("sh", {"command": "echo hello"})), done("Said hello.")]
    provider = ScriptedProvider(script)

    def broken(self: Any, step: Step) -> None:
        raise RuntimeError("no lane")

    monkeypatch.setattr("synthic_tui.views.ColumnView.step_added", broken)

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider, task="say hello")
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(app, pilot)
            assert app.state == "delivered" and step(app, "end", "Said hello.")
            assert said(app, "could not draw a change (RuntimeError: no lane). The run goes on.")

    asyncio.run(scenario())
    assert not provider.script


def test_a_stored_run_is_read_back_with_its_costs_and_outputs(world: tuple[Path, Any]) -> None:
    project, load = world
    script = [reply(text("Looking."), tool_use("sh", {"command": "echo hello"})), done("Said hello.")]
    provider = ScriptedProvider(script)

    async def scenario() -> Path:
        app = SynthicApp(project, load, lambda: provider, task="say hello")
        async with app.run_test(size=(120, 40)) as pilot:
            await settle(app, pilot)
            sh = step(app, "tool", "sh: echo hello")
            assert any("hello" in line for _, line in app.board.detail(sh))
            assert app.runtime is not None
            return app.runtime.run_dir

    run_dir = asyncio.run(scenario())
    board = Board()
    board.load(run_dir)
    assert [s.kind for s in board.steps] == ["user", "text", "tool", "end"]
    root = board.agents["root"]
    assert root.state == "delivered" and root.role == "lead" and root.calls == 2
    assert sum(s.cost for s in board.steps) == pytest.approx(board.total) and board.total > 0
    sh = board.steps[2]
    assert sh.name == "sh" and sh.result == "ok" and any("hello" in line for _, line in board.detail(sh))
