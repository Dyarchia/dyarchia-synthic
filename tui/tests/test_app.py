"""The terminal interface, offline: a scripted model, a real runtime and journal, a headless screen."""

from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from textual.events import Click
from textual.widgets import Input, OptionList, Static

from synthic import __version__
from synthic.cli import TUI_DEFAULTS
from synthic.policy import LoadedPolicy, deep_merge, load_policy
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic_tui.app import COMMANDS, STATUS_DROPS, SynthicApp
from synthic_tui.board import Board, Step
from synthic_tui.theme import SPACE
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
    return str(app.query_one("#stats", Static).render())


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
            assert "edit: a.txt (+2 -1)" in line and " ok " in line and re.search(r"\d,\ds", line)
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


@pytest.fixture
def person(world: tuple[Path, Any], tmp_path: Path) -> tuple[Path, Any]:
    """The same world with the interface's own default: a person is at the terminal."""
    project, _ = world

    def load(extra: dict[str, Any] | None = None) -> LoadedPolicy:
        overrides = deep_merge({"journal": {"dir": (tmp_path / "runs").as_posix()}}, extra or {})
        return load_policy(project, tmp_path / "task.toml", overrides, defaults=TUI_DEFAULTS)

    return project, load


def test_a_greeting_is_answered_in_one_call_and_the_run_waits_for_the_person(person: tuple[Path, Any]) -> None:
    project, load = person
    provider = ScriptedProvider([reply(text("Hello. What shall we work on?")), reply(text("Done with that."))])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, "hello")
            await settle(app, pilot)
            assert app.state == "replied" and len(provider.requests) == 1
            assert step(app, "text", "What shall we work on?")
            assert not any(s.kind == "end" for s in app.board.steps)
            assert "waiting for you" in status(app) and "failed" not in status(app)
            assert app.board.agents["root"].state == "replied"
            first = app.run_id
            await send(pilot, "just checking")
            await settle(app, pilot)
            assert app.run_id == first and app.state == "replied" and len(provider.requests) == 2
            assert step(app, "text", "Done with that.")

    asyncio.run(scenario())
    assert not provider.script


def test_a_plain_message_answers_the_question_the_agent_is_blocked_on(person: tuple[Path, Any]) -> None:
    project, load = person
    ask = reply(tool_use("ask", {"question": "Blue or red?", "assumption": "blue"}))
    provider = ScriptedProvider([ask, done("Painted it red.")])

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, "paint it")
            for _ in range(200):
                await pilot.pause(0.02)
                if app.blocking_question() is not None:
                    break
            assert app.blocking_question() is not None
            for _ in range(150):
                await pilot.pause(0.02)
                if "Type your answer" in str(app.query_one("#ask", Static).render()):
                    break
            assert "Type your answer" in str(app.query_one("#ask", Static).render())
            await send(pilot, "red")
            await settle(app, pilot)
            assert app.state == "delivered" and not app.board.waiting()
            assert step(app, "user", "(answer root-q1) red")
            assert "The human answered" in str(provider.requests[1].messages[-1]) and "red" in str(provider.requests[1].messages[-1])

    asyncio.run(scenario())


def test_exit_and_quit_typed_alone_leave_the_app_only_when_nothing_is_working(world: tuple[Path, Any]) -> None:
    project, load = world
    provider = ScriptedProvider([done("Unused.")])

    async def leaves(line: str) -> bool:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            await send(pilot, line)
            await pilot.pause(0.3)
            return app._exit

    assert all(asyncio.run(leaves(line)) for line in ("exit", "quit", "/exit", "/quit"))
    assert not asyncio.run(leaves("exit now"))


@pytest.mark.parametrize(("size", "fits"), [((120, 35), True), ((80, 24), False)], ids=["120x35", "80x24"])
def test_the_help_never_clips_and_scrolls_with_the_keys(world: tuple[Path, Any], size: tuple[int, int], fits: bool) -> None:
    project, load = world

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: ScriptedProvider([]))
        async with app.run_test(size=size) as pilot:
            await send(pilot, "/help")
            await pilot.pause(0.2)
            scroll, cue = app.query_one("#notice-scroll"), app.query_one("#notice-more", Static)
            status_bar, views = app.query_one("#status"), app.query_one("#views")
            assert status_bar.region.bottom <= size[1] and views.region.height >= 3
            assert scroll.region.bottom <= size[1]
            assert (scroll.max_scroll_y == 0) is fits and cue.display is (not fits)
            if fits:
                return
            assert "more below" in str(cue.render()) and scroll.scroll_y == 0
            await pilot.press("down")
            assert scroll.scroll_y == 1
            await pilot.press("pagedown", "pagedown", "pagedown")
            assert scroll.scroll_y == scroll.max_scroll_y
            await pilot.pause(0.1)
            assert "end of the list" in str(cue.render())
            await pilot.press("pageup", "up")
            assert scroll.scroll_y < scroll.max_scroll_y
            await pilot.press("escape")
            assert not app.query_one("#notice").display and not cue.display
            await send(pilot, "first")
            await pilot.press("up")
            assert app.query_one("#box", Input).value == "first"

    asyncio.run(scenario())


def test_ctrl_c_and_ctrl_d_need_a_second_press_in_every_state(world: tuple[Path, Any]) -> None:
    project, load = world

    async def leaves(*keys: str, typed: str = "") -> tuple[bool, str]:
        app = SynthicApp(project, load, lambda: ScriptedProvider([]))
        async with app.run_test(size=(100, 30)) as pilot:
            app.query_one("#box", Input).value = typed
            app.query_one("#box", Input).cursor_position = 0
            await pilot.press(keys[0])
            first = app._exit
            for key in keys[1:]:
                await pilot.press(key)
            await pilot.pause(0.3)
            return app._exit and not first, app.query_one("#box", Input).value

    assert asyncio.run(leaves("ctrl+c", "ctrl+c")) == (True, "")
    assert asyncio.run(leaves("ctrl+c", "ctrl+c", typed="text")) == (True, "")
    assert asyncio.run(leaves("ctrl+d", "ctrl+d")) == (True, "")
    assert asyncio.run(leaves("ctrl+c", "ctrl+d"))[0] is False
    assert asyncio.run(leaves("ctrl+d", "ctrl+d", typed="text")) == (False, "xt")


def screen_cells(app: SynthicApp) -> list[list[tuple[str, str]]]:
    """Every cell of the screen as it is sent to the terminal: its character and the background it is painted on."""
    rows = []
    for strip in app.screen._compositor.render_strips():
        row = []
        for segment in strip:
            colour = segment.style.bgcolor if segment.style else None
            paint = colour.get_truecolor().hex if colour is not None else "none"
            row.extend((char, paint) for char in segment.text)
        rows.append(row)
    return rows


def screen_text(app: SynthicApp) -> str:
    return "\n".join("".join(char for char, _ in row) for row in screen_cells(app))


def test_the_start_is_a_short_identity_block_and_the_help_lives_in_help(world: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    project, load = world
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: project.parent))
    home = str(Path("~") / "proj")

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: ScriptedProvider([]))
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.transcript == [f"Synthic {__version__}\n{home}"]
            visible = screen_text(app)
            assert f"Synthic {__version__}" in visible and home in visible and str(project.parent) not in visible
            for help_line in ("Enter sends", "Tab completes", "Ctrl+", "Esc interrupts", "/ lists"):
                assert help_line not in visible
            assert "/ for commands" in app.query_one("#box", Input).placeholder
            await send(pilot, "/help")
            assert "Ctrl+T" in app.transcript[-1] and "Tab" in app.transcript[-1]
            await send(pilot, "/clear")
            assert app.transcript[-1] == f"Synthic {__version__}\n{home}"

    asyncio.run(scenario())


def test_a_long_working_folder_gives_way_at_a_separator_and_is_never_split(world: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    project, load = world
    deep = project / ("a-long-folder-name" * 3) / "inner"
    deep.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: project.parent))

    async def scenario() -> None:
        app = SynthicApp(deep, load, lambda: ScriptedProvider([]))
        async with app.run_test(size=(60, 24)) as pilot:
            await pilot.pause(0.05)
            where = str(app.query_one("#notice-text", Static).render()).splitlines()[1]
            assert where.startswith("…") and where.endswith("inner") and len(where) <= 60 - 4

    asyncio.run(scenario())


def crowd(app: SynthicApp) -> None:
    """Several agents, a question, a spend and a run, so that every item of the status line has something to say."""
    board = app.board
    board.said("go")
    board.feed("root", "agent_start", "lead: go")
    board.feed("a1", "agent_start", "apex: part")
    board.feed("a1", "question", "a1-q1: which one?")
    board.measure("root", 0.5, 3, 4200)
    app.state, app.started, app.run_id = "working", time.monotonic(), "20260101-abcdef"
    app.refresh_status()


@pytest.mark.parametrize("width", [80, 120, 280])
def test_the_status_line_holds_the_tabs_and_gives_up_items_in_a_fixed_order(world: tuple[Path, Any], width: int) -> None:
    project, load = world
    marks = {"run": "run 2026", "agents": "agents", "steps": "steps", "context": "context", "model": "test", "cost": " $", "waiting": "◆"}

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=(width, 30)) as pilot:
            crowd(app)
            await pilot.pause(0.05)
            stats = app.query_one("#stats", Static)
            assert app.query_one("#status").region.bottom == 30 and not app.query("#tabs")
            assert [str(app.query_one(f"#tab-{key}", Static).render()) for key in VIEWS] == list(VIEWS.values())
            assert all(app.query_one(f"#tab-{key}").region.right <= stats.region.x for key in VIEWS)
            line = str(stats.render())
            assert len(line) <= stats.region.width and "…" not in line and "working" in line
            assert all(part.strip() for part in line.removeprefix("│ ").split(" │ "))
            present = [name for name in STATUS_DROPS if marks[name] in line]
            assert present == list(STATUS_DROPS)[len(STATUS_DROPS) - len(present):]
            visible = screen_text(app)
            assert all(label in visible for label in VIEWS.values())
            if width == 280:
                assert present == list(STATUS_DROPS)
            if width == 80:
                assert present and "run 2026" not in line and "agents" not in line

    asyncio.run(scenario())


def test_the_active_tab_is_unmistakable_and_a_click_or_ctrl_t_changes_it(world: tuple[Path, Any]) -> None:
    project, load = world

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=(120, 30)) as pilot:
            column, panes = app.query_one("#tab-column"), app.query_one("#tab-panes")
            assert column.has_class("active") and not panes.has_class("active")
            assert column.styles.text_style.bold and not panes.styles.text_style.bold
            assert column.styles.background != panes.styles.background
            assert column.styles.color.hex.lower() == SPACE["focus"] and panes.styles.color.hex.lower() == SPACE["ink-muted"]
            fleet = app.query_one("#tab-fleet")
            x, y = fleet.region.x + 2, fleet.region.y
            assert app.screen.get_widget_at(x, y)[0] is fleet
            fleet.post_message(Click(fleet, 2, 0, 0, 0, 1, False, False, False, screen_x=x, screen_y=y))
            await pilot.pause(0.05)
            assert app.view_key == "fleet" and app.query_one("#tab-fleet").has_class("active") and not column.has_class("active")
            assert app.focused is app.query_one("#box", Input)
            await pilot.press("ctrl+t")
            assert app.view_key == "ledger" and app.query_one("#tab-ledger").has_class("active")
            await pilot.press("ctrl+t")
            assert app.view_key == "column"

    asyncio.run(scenario())


def test_every_view_has_a_quiet_empty_state_and_none_is_never_printed(world: tuple[Path, Any]) -> None:
    project, load = world
    says = {"column": "No conversation yet", "panes": "No agents yet", "fleet": "No agents yet", "ledger": "No steps yet"}

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=(120, 35)) as pilot:
            for key in VIEWS:
                await send(pilot, f"/view {key}")
                await pilot.pause(0.1)
                text = screen_text(app)
                assert says[key] in text, key
                assert "None" not in text and "none" not in text, key
                assert not app.board.agents, f"showing {key} must not invent an agent"
            await send(pilot, "/cost")
            assert said(app, "Nothing spent") and "None" not in app.transcript[-1]
            app.board.said("go")
            await pilot.pause(0.1)
            for key in VIEWS:
                await send(pilot, f"/view {key}")
                await pilot.pause(0.1)
                assert says[key] not in screen_text(app), key

    asyncio.run(scenario())


@pytest.mark.parametrize("size", [(80, 24), (133, 37), (280, 60)], ids=["80x24", "133x37", "280x60"])
def test_the_screen_paints_one_background_to_every_edge_in_every_view(world: tuple[Path, Any], size: tuple[int, int]) -> None:
    project, load = world
    ground, tab = SPACE["ground"], SPACE["ground-selected"]

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=size) as pilot:
            for key in VIEWS:
                await send(pilot, f"/view {key}")
                await pilot.pause(0.1)
                cells = screen_cells(app)
                assert len(cells) == size[1] and all(len(row) == size[0] for row in cells), key
                assert {paint for row in cells for _, paint in row} <= {ground, tab, SPACE["ink"], SPACE["tide-tile"]}, key
                assert {row[-1][1] for row in cells} == {ground} and {row[0][1] for row in cells} == {ground}, key
                assert {paint for _, paint in cells[0]} == {ground} and {paint for _, paint in cells[-1]} <= {ground, tab}, key

    asyncio.run(scenario())


def test_slash_answer_names_the_question_when_several_are_open(world: tuple[Path, Any]) -> None:
    project, load = world

    class Runtime:
        def __init__(self, *ids: str) -> None:
            self.open = [SimpleNamespace(id=i, agent_id="root", waiting=True) for i in ids]
            self.given: list[tuple[str, str]] = []

        def questions(self) -> list[Any]:
            return self.open

        def answer(self, question_id: str, text: str) -> None:
            self.given.append((question_id, text))

    async def scenario() -> None:
        app = SynthicApp(project, load, None)
        async with app.run_test(size=(120, 40)):
            several = Runtime("a-q1", "b-q1")
            app.answer(several, "plain words")
            assert said(app, "Several questions are open: /answer <id> <text>") and several.given == []
            app.answer(several, "b-q1 yes, go on")
            assert several.given == [("b-q1", "yes, go on")]
            one = Runtime("a-q1")
            app.answer(one, "no")
            assert one.given == [("a-q1", "no")]
            app.answer(Runtime(), "anything")
            assert said(app, "No question is open.")

    asyncio.run(scenario())


def test_slash_instructions_lists_the_repo_files_and_switches_them_from_the_next_run(world: tuple[Path, Any]) -> None:
    project, load = world
    (project / "AGENTS.md").write_text("Repo says: use tabs.", encoding="utf-8")
    provider = ScriptedProvider([done("One."), done("Two."), done("Three.")])

    def prompt(index: int) -> str:
        return " ".join(block["text"] for block in provider.requests[index].system)

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: provider)
        async with app.run_test(size=(120, 40)) as pilot:
            assert app.loaded.policy.instructions.load_repo_files is True
            await send(pilot, "/instructions")
            assert said(app, "Repo instruction files are on.") and said(app, str(project / "AGENTS.md"))
            assert "CLAUDE.md" not in app.transcript[-1]
            await send(pilot, "/instructions off")
            assert app.loaded.policy.instructions.load_repo_files is False
            for piece in ("now off", "next run", "/clear", "instructions.load_repo_files", "this session"):
                assert piece in app.transcript[-1], piece
            await send(pilot, "/model other")
            assert app.loaded.policy.instructions.load_repo_files is False and "other" in status(app)
            await send(pilot, "first")
            await settle(app, pilot)
            assert "Repo says: use tabs." not in prompt(0) and "comes from the repository" not in prompt(0)
            await send(pilot, "/instructions bogus")
            assert said(app, "Usage: /instructions [on|off]")
            await send(pilot, "/clear")
            await send(pilot, "/instructions on")
            assert app.loaded.policy.instructions.load_repo_files is True and app.loaded.policy.roles["lead"].model == "other"
            await send(pilot, "second")
            await settle(app, pilot)
            assert "Repo says: use tabs." in prompt(1) and "comes from the repository" in prompt(1)

    asyncio.run(scenario())


def test_slash_instructions_is_refused_while_a_run_is_working_like_slash_model(world: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    project, load = world

    async def scenario() -> None:
        app = SynthicApp(project, load, lambda: ScriptedProvider([]))
        async with app.run_test(size=(120, 40)) as pilot:
            monkeypatch.setattr(app, "running", lambda: True)
            await send(pilot, "/instructions off")
            assert said(app, "A run is working: /stop it before /instructions.")
            assert app.loaded.policy.instructions.load_repo_files is True and app.extra == {}
            await send(pilot, "/model other")
            assert said(app, "A run is working: /stop it before /model.")
            assert "instructions" in COMMANDS

    asyncio.run(scenario())
