"""The terminal interface of Synthic: four views of one run, a message box and a status line.

A row of tabs chooses how the run is drawn (``Column``, ``Panes``, ``Fleet``, ``Ledger``); below it the
message box always has the keyboard, with one status line under it. A line that starts with ``/`` is a
command and never reaches the model; anything else is a task, or a message to the agent in focus.

It sits on ``synthic.agents.Runtime`` the same way the command line does. The engine knows nothing about
it: this package reads the output callback, the ledger and the journal, and calls the four controls of the
runtime (``stop``, ``add``, ``redirect``, ``answer``). The runtime runs on the event loop of the
application, so its output callback may touch widgets directly.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.events import Click, Key
from textual.widgets import ContentSwitcher, Input, OptionList, Static
from textual.widgets.option_list import Option

from synthic import __version__
from synthic.agents import Runtime, allowance_of
from synthic.cli import clock, short, summarize
from synthic.journal import agents_in, is_alive, list_runs, read_records
from synthic.policy import LoadedPolicy
from synthic.transcript import markdown
from synthic.types import Provider

from .board import LIVE, ROOT, Board, Step
from .theme import THEMES, elapsed, money, number, register
from .views import VIEWS, AgentPicked, ColumnView, FleetView, LedgerView, PanesView, StepPicked, View, badge, name_of

Load = Callable[[dict[str, Any] | None], LoadedPolicy]

COMMANDS = {
    "help": "show the commands and the keys",
    "clear": "start a new conversation",
    "resume": "open an earlier run and continue it",
    "model": "choose the model of the next tasks",
    "effort": "choose the effort of the next tasks",
    "view": "choose how the run is drawn: /view [column|panes|fleet|ledger]",
    "stop": "stop the run that is working",
    "redirect": "stop the current step and give a new direction: /redirect <text>",
    "answer": "answer an open question: /answer [id] <text>",
    "context": "show how full the context is and when it compacts",
    "cost": "show what this run has spent",
    "export": "save this run as a Markdown file: /export [file]",
    "exit": "leave Synthic",
}
ALIASES = {"new": "clear", "reset": "clear", "continue": "resume", "runs": "resume", "quit": "exit"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
PICKERS = ("resume", "model", "effort", "view")
PLACEHOLDER = "Write a task and press Enter, or / for commands"
KEYS = (
    "Enter sends, / lists the commands, Tab completes, Up recalls a message.\n"
    "Ctrl+T changes the view, Shift+Up and Shift+Down change the agent in focus,\n"
    "Ctrl+Up and Ctrl+Down move the selected step. Esc or Ctrl+C interrupts, Ctrl+C twice leaves."
)
RUNS_SHOWN = 30
QUIT_WINDOW = 2.0


def state_file() -> Path:
    """Where the interface remembers the view last used. It is not policy: nothing here binds an agent."""
    raw = os.environ.get("SYNTHIC_HOME")
    return (Path(raw) if raw else Path.home()) / ".synthic" / "tui.json"


def remembered() -> str:
    try:
        view = json.loads(state_file().read_text(encoding="utf-8")).get("view")
    except (OSError, ValueError, AttributeError):
        return "column"
    return view if view in VIEWS else "column"


class Tab(Static):
    def __init__(self, key: str) -> None:
        super().__init__(VIEWS[key], classes="tab", id=f"tab-{key}")
        self.key = key

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.show_view(self.key)


class SynthicApp(App[None]):
    TITLE = "Synthic"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: vertical; background: $background; color: $ink; padding: 1 2 0 2; overflow: hidden; }
    Static, Rule, VerticalScroll, HorizontalScroll, Vertical, Horizontal { background: transparent; }
    VerticalScroll { scrollbar-size-vertical: 1; }
    HorizontalScroll { scrollbar-size-horizontal: 1; }

    #tabs { height: 1; margin-bottom: 1; }
    .tab { width: auto; margin-right: 3; color: $ink-faint; }
    .tab.active { color: $ink; }
    .tab:hover { color: $ink-muted; }
    #views { height: 1fr; min-height: 3; }
    #views.veiled { opacity: 45%; }
    View { height: 1fr; }

    .head { height: 1; }
    .rule { height: 1; margin: 0; color: $hairline; }
    .entry { height: auto; }
    .entry.gap { margin-top: 1; }
    .lane .entry.tool, .centre .entry.tool { height: 1; text-wrap: nowrap; text-overflow: ellipsis; }
    .entry.selected, .row.selected, .card.selected { background: $ground-selected; }
    .measure { max-width: 120; }

    .side { width: 26; margin-right: 2; }
    .card { height: 2; margin-bottom: 1; }
    .centre { width: 1fr; }
    .dock { width: 36%; min-width: 34; max-width: 60; margin-left: 2; }
    .dock.wide { width: 40%; min-width: 36; max-width: 64; }
    .detail-body { height: 1fr; }
    .detail-text { height: auto; }
    Meters { height: 6; margin-top: 1; }

    .totals { height: 3; margin-bottom: 1; }
    .fleet { height: 1fr; }
    .lane { width: 1fr; min-width: 26; margin-right: 2; }
    FleetHead { height: 3; }

    .table { width: 1fr; }
    .rows { height: 1fr; }
    .row { height: 1; }

    #ask { height: auto; max-height: 8; margin-top: 1; display: none; }
    #notice { height: auto; max-height: 35vh; margin-top: 1; display: none; }
    #notice-text { height: auto; color: $ink-muted; }
    #notice.bad > #notice-text { color: $down; }
    #menu {
        height: auto; max-height: 12; max-width: 96; margin-top: 1; padding: 0 1; display: none;
        background: $ground-raised; border: round $hairline; border-top: round $ink-faint;
    }
    #menu > .option-list--option-highlighted { background: $ground-selected; color: $ink; text-style: none; }
    #entry { height: 2; margin-top: 1; border-bottom: solid $line-control; }
    #prompt { width: auto; color: $ink-muted; }
    #box, #box:focus { width: 1fr; height: 1; padding: 0; border: none; background: transparent; }
    #box > .input--placeholder, #box > .input--suggestion { color: $ink-faint; }
    #status { height: 1; }
    """
    BINDINGS = [
        Binding("ctrl+c", "leave", show=False, priority=True),
        Binding("tab", "complete", show=False, priority=True),
        Binding("ctrl+t", "next_view", show=False, priority=True),
        Binding("shift+down", "agent(1)", show=False, priority=True),
        Binding("shift+up", "agent(-1)", show=False, priority=True),
        Binding("ctrl+down", "step(1)", show=False, priority=True),
        Binding("ctrl+up", "step(-1)", show=False, priority=True),
        Binding("escape", "interrupt", show=False),
    ]

    def __init__(self, cwd: Path, load: Load, factory: Callable[[], Provider] | None = None, task: str = "") -> None:
        super().__init__()
        register(self)
        self.theme = "synthic-space"
        self.cwd = cwd
        self.load = load
        self.factory = factory
        self.first_task = task
        self.extra: dict[str, Any] = {}
        self.loaded = load(None)
        self.board = Board()
        self.board.listen(self.board_event)
        self.runtime: Runtime | None = None
        self.job: asyncio.Task[Any] | None = None
        self.run_id: str | None = None
        self.state = "idle"
        self.started = 0.0
        self.view_key = remembered()
        self.focus_agent = ROOT
        self.selected: Step | None = None
        self.follow = True
        self.transcript: list[str] = []
        self.sent: list[str] = []
        self.recall = 0
        self.menu_mode: str | None = None
        self.options: list[tuple[str, str]] = []
        self.leave_at = 0.0

    @property
    def tokens(self) -> dict[str, str]:
        return THEMES.get(self.theme, THEMES["synthic-space"])

    def compose(self) -> ComposeResult:
        with Horizontal(id="tabs"):
            for key in VIEWS:
                yield Tab(key)
        with ContentSwitcher(initial=self.view_key, id="views"):
            yield ColumnView(self.board, id="column")
            yield PanesView(self.board, id="panes")
            yield FleetView(self.board, id="fleet")
            yield LedgerView(self.board, id="ledger")
        yield Static("", id="ask")
        with VerticalScroll(id="notice") as notice:
            notice.can_focus = False
            yield Static("", id="notice-text")
        menu = OptionList(id="menu")
        menu.can_focus = False
        yield menu
        with Horizontal(id="entry"):
            yield Static("> ", id="prompt")
            yield Input(placeholder=PLACEHOLDER, id="box")
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.show_view(self.view_key, remember=False)
        self.say(f"Synthic {__version__}  {self.cwd}\n{KEYS}")
        self.set_interval(1.0, self.tick)
        self.query_one("#box", Input).focus()
        if self.first_task:
            self.begin(self.first_task)

    """The views."""

    def current(self) -> View:
        return self.query_one(f"#{self.view_key}", View)

    def show_view(self, key: str, remember: bool = True) -> None:
        self.view_key = key
        self.query_one("#views", ContentSwitcher).current = key
        for tab in self.query(Tab):
            tab.set_class(tab.key == key, "active")
        self.current().rebuild()
        self.current().selection_changed()
        if remember:
            try:
                state_file().parent.mkdir(parents=True, exist_ok=True)
                state_file().write_text(json.dumps({"view": key}), encoding="utf-8")
            except OSError:
                pass
        self.query_one("#box", Input).focus()

    def board_event(self, event: str, subject: Any) -> None:
        """Draw a change of the board. Drawing never ends a run: an error here is shown and the run goes on."""
        if not self.is_mounted:
            return
        try:
            self.draw_event(event, subject)
        except Exception as error:
            self.drawing_failed(error)

    def drawing_failed(self, error: Exception) -> None:
        self.say(f"The {VIEWS[self.view_key]} view could not draw a change ({type(error).__name__}: {error}). The run goes on.", "bad")

    def draw_event(self, event: str, subject: Any) -> None:
        view = self.current()
        match event:
            case "reset":
                self.selected, self.follow, self.focus_agent = None, True, ROOT
                view.rebuild()
            case "step":
                view.step_added(subject)
                if self.follow and subject.kind != "user":
                    self.select(subject, follow=True)
            case "change":
                view.step_changed(subject)
            case "agent":
                view.agent_changed(subject)
                self.refresh_prompt()
                self.refresh_ask()

    def select(self, step: Step | None, follow: bool = False) -> None:
        self.selected, self.follow = step, follow
        self.current().selection_changed()

    def on_step_picked(self, event: StepPicked) -> None:
        newest = bool(self.board.steps) and event.step is self.board.steps[-1]
        self.select(event.step, follow=newest)

    def on_agent_picked(self, event: AgentPicked) -> None:
        self.focus_on(event.agent_id)

    def focus_on(self, agent_id: str) -> None:
        self.focus_agent = agent_id
        steps = self.board.agent(agent_id).steps
        self.selected, self.follow = (steps[-1] if steps else None), True
        self.current().selection_changed()
        self.refresh_prompt()
        self.refresh_status()

    def action_next_view(self) -> None:
        keys = list(VIEWS)
        self.show_view(keys[(keys.index(self.view_key) + 1) % len(keys)])

    def action_agent(self, way: int) -> None:
        ids = list(self.board.agents)
        if len(ids) > 1:
            self.focus_on(ids[(ids.index(self.focus_agent) + way) % len(ids)])

    def action_step(self, way: int) -> None:
        steps = self.current().choices()
        if not steps:
            return
        at = steps.index(self.selected) if self.selected in steps else len(steps) - 1
        at = max(0, min(len(steps) - 1, at + way))
        self.select(steps[at], follow=at == len(steps) - 1)

    def model_of(self, agent_id: str, effort: bool = False) -> str:
        policy = self.loaded.policy
        view = self.board.agents.get(agent_id)
        role = policy.roles.get(view.role if view and view.role else policy.initial_role)
        if role is None:
            return ""
        return f"{role.model} · {role.effort or 'default'}" if effort else str(role.model)

    def elapsed(self) -> float:
        if self.state == "working":
            return time.monotonic() - self.started
        steps = self.board.steps
        return steps[-1].ts - steps[0].ts if steps else 0.0

    def gauges(self) -> list[tuple[str, float, float, str]]:
        """Context of the agent in focus against the size it compacts at, and spend against the budget."""
        policy = self.loaded.policy
        view = self.board.agents.get(self.focus_agent)
        used = view.context if view else 0
        limit = self.compact_limit(self.focus_agent)
        cap = policy.budget.dollars or 0.0
        spent = self.board.total
        return [
            ("CONTEXT", used, limit, f"{number(used)} of {number(limit)}" if limit else number(used)),
            ("BUDGET", spent, cap, f"{money(spent)} of {money(cap)}" if cap else money(spent)),
        ]

    def compact_limit(self, agent_id: str) -> int:
        policy = self.loaded.policy
        view = self.board.agents.get(agent_id)
        role = policy.roles.get(view.role if view and view.role else policy.initial_role)
        spec = policy.models.get(role.model) if role else None
        if spec is None:
            return 0
        window = spec.context_window
        agent = self.live_agent(agent_id)
        if agent is not None:
            window = agent.context.compaction_window(spec)
        return round(policy.context.compact_at * window)

    def live_agent(self, agent_id: str) -> Any:
        if self.runtime is None:
            return None
        try:
            return self.runtime._agent(agent_id)
        except KeyError:
            return None

    """What the interface says: the answer to a command, shown above the message box until the next one."""

    def say(self, line: str, style: str = "") -> None:
        self.transcript.append(line)
        notice = self.query_one("#notice")
        self.query_one("#notice-text", Static).update(Text(line))
        notice.set_class(style == "bad", "bad")
        notice.display = True
        notice.scroll_home(animate=False)

    def hush(self) -> None:
        self.query_one("#notice").display = False

    def engine_output(self, agent_id: str, kind: str, text: str) -> None:
        """The output callback of the runtime."""
        self.board.feed(agent_id, kind, text)
        try:
            self.measure(agent_id)
            self.refresh_ask()
            self.refresh_status()
        except Exception as error:
            self.drawing_failed(error)

    def measure(self, agent_id: str) -> None:
        runtime = self.runtime
        if runtime is None:
            return
        usage = runtime.ledger.per_agent.get(allowance_of(agent_id))
        agent = self.live_agent(agent_id)
        view = self.board.agent(agent_id)
        handle = runtime.handles.get(agent_id)
        if handle is not None:
            view.parent, view.task = handle.parent.id, view.task or handle.task
            view.depth = self.board.agent(handle.parent.id).depth + 1
        self.board.measure(
            agent_id,
            usage.cost_usd if usage else view.cost,
            agent.steps if agent is not None else view.calls,
            agent.context.last_prompt_tokens if agent is not None else view.context,
        )

    """Runs."""

    def begin(self, text: str) -> None:
        """Start a run with ``text``, or continue the open one with it as the next message."""
        try:
            self.loaded = self.load(self.extra or None)
            self.runtime = Runtime(self.loaded, self.cwd, self.factory() if self.factory else None, run_id=self.run_id)
        except Exception as error:
            self.say(f"Cannot start: {error}", "bad")
            return
        self.runtime.on_output(self.engine_output)
        self.hush()
        if self.runtime.warnings:
            self.say("\n".join(f"! {warning}" for warning in self.runtime.warnings))
        resumed = self.run_id is not None
        self.run_id = self.runtime.run_id
        self.board.run_dir = self.runtime.run_dir
        self.state, self.started = "working", time.monotonic()
        self.focus_agent = ROOT
        self.board.said(text)
        self.transcript.append(f"> {text}")
        work = self.runtime.resume(text) if resumed else self.runtime.run(text)
        self.job = asyncio.create_task(self.follow_run(self.runtime, work))
        self.refresh_status()

    async def follow_run(self, runtime: Runtime, work: Any) -> None:
        try:
            result = await work
            self.state = result.status
            self.measure(ROOT)
            self.board.ended(result.status, result.summary)
        except Exception as error:
            self.state = "failed"
            self.board.ended("failed", f"{type(error).__name__}: {error}")
        finally:
            await runtime.close()
            if self.is_running:
                self.refresh_ask()
                self.refresh_status()

    def running(self) -> bool:
        return self.job is not None and not self.job.done()

    def open_run(self, run_id: str) -> None:
        """Show what a stored run did, and make the next message continue it."""
        run_dir = next((path for path in list_runs(self.loaded.policy.journal) if path.name == run_id), None)
        if run_dir is None:
            self.say(f"No run {run_id}. /resume lists them.", "bad")
            return
        self.clear()
        self.board.load(run_dir)
        self.current().rebuild()
        if self.board.steps:
            self.select(self.board.steps[-1], follow=True)
        if is_alive(run_dir):
            self.say(f"Run {run_id} is working in another process: it is shown read-only.")
        else:
            self.run_id = run_id
            self.say(f"Run {run_id} is open: your next message continues it.")
        self.refresh_status()

    def clear(self) -> None:
        self.run_id, self.runtime, self.state = None, None, "idle"
        self.transcript.clear()
        self.board.reset()
        self.refresh_ask()

    """The message box."""

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        mode, choice = self.menu_mode, self.highlighted()
        self.close_menu()
        if mode in PICKERS:
            if choice:
                self.pick(mode, choice)
            return
        if mode == "commands" and choice and text[1:].partition(" ")[0] not in (*COMMANDS, *ALIASES):
            text = f"/{choice}"
        if not text:
            return
        self.hush()
        self.sent.append(text)
        self.recall = len(self.sent)
        if text.startswith("/"):
            self.command(text)
        elif self.running() and self.runtime is not None:
            self.tell_agent(text)
        else:
            self.begin(text)

    def target(self) -> str:
        """The agent a message goes to: the one in focus while it works, the root otherwise."""
        return self.focus_agent if self.live_agent(self.focus_agent) is not None else ROOT

    def tell_agent(self, text: str, redirect: bool = False) -> None:
        assert self.runtime is not None
        agent_id = self.target()
        try:
            (self.runtime.redirect if redirect else self.runtime.add)(text, agent_id)
        except KeyError as error:
            self.say(str(error.args[0]), "bad")
            return
        self.board.said(text, agent_id, "redirect" if redirect else "")
        self.transcript.append(f"> {'(redirect) ' if redirect else ''}{text}")

    def command(self, line: str) -> None:
        """Run a ``/`` command. Nothing typed after a slash ever reaches the model as a task."""
        word, _, rest = line[1:].partition(" ")
        word, rest = ALIASES.get(word, word), rest.strip()
        runtime = self.runtime if self.running() else None
        match word:
            case "help":
                rows = "\n".join(f"  /{name:<9} {about}" for name, about in COMMANDS.items())
                self.say(f"Commands\n{rows}\n\n{KEYS}")
            case "exit":
                self.run_action("quit")
            case "clear" | "resume" | "model" | "effort" if self.running():
                self.say(f"A run is working: /stop it before /{word}.")
            case "clear":
                self.clear()
                self.say(f"Synthic {__version__}  {self.cwd}\n{KEYS}")
                self.refresh_status()
            case "resume" | "model" | "effort" | "view" if rest:
                self.pick(word, rest)
            case "resume" | "model" | "effort" | "view":
                self.open_picker(word)
            case "stop":
                self.stop_run()
            case "redirect" if runtime is not None and rest:
                self.tell_agent(rest, redirect=True)
            case "answer" if runtime is not None and rest:
                self.answer(runtime, rest)
            case "redirect" | "answer":
                self.say(f"/{word} works while a run is working: {COMMANDS[word]}")
            case "cost":
                self.say(self.cost())
            case "context":
                self.say(self.context())
            case "export":
                self.export(rest)
            case _:
                self.say(f"Unknown command /{word}. /help lists the commands.", "bad")

    def answer(self, runtime: Runtime, rest: str) -> None:
        """``/answer id text``, or ``/answer text`` when one question is open."""
        open_ = {question.id: question for question in runtime.questions()}
        first, _, text = rest.partition(" ")
        if first in open_ and text.strip():
            question_id, text = first, text.strip()
        elif len(open_) == 1:
            question_id, text = next(iter(open_)), rest
        elif not open_:
            self.say("No question is open.")
            return
        else:
            self.say(f"Several questions are open: /answer <id> <text>, with id one of {', '.join(open_)}.")
            return
        try:
            runtime.answer(question_id, text)
        except KeyError as error:
            self.say(str(error.args[0]), "bad")
            return
        self.board.answered(question_id)
        self.board.said(text, open_[question_id].agent_id, f"answer {question_id}")
        self.transcript.append(f"> (answer {question_id}) {text}")
        self.refresh_ask()

    def context(self) -> str:
        view = self.board.agents.get(self.focus_agent)
        limit = self.compact_limit(self.focus_agent)
        if view is None or not view.context or not limit:
            return "No context yet: no run has started in this conversation."
        policy = self.loaded.policy
        role = policy.roles.get(view.role or policy.initial_role)
        spec = policy.models[role.model]
        return (
            f"The last prompt of {name_of(self.board, view.id)} was {number(view.context)} tokens, "
            f"{view.context * 100 // limit} % of the {number(limit)} at which it compacts.\n"
            f"  The window of {spec.alias} is {number(spec.context_window)} tokens."
        )

    def export(self, name: str) -> None:
        run_dir = next((path for path in list_runs(self.loaded.policy.journal) if path.name == self.run_id), None)
        if run_dir is None:
            self.say("Nothing to export: no run is open.")
            return
        target = self.cwd / (name or f"synthic-{self.run_id}.md")
        by_agent = {agent_id: list(read_records(run_dir, agent_id)) for agent_id in sorted(agents_in(run_dir))}
        try:
            target.write_text(markdown(run_dir.name, by_agent), encoding="utf-8")
        except OSError as error:
            self.say(f"Cannot write {target}: {error}", "bad")
            return
        self.say(f"Saved {target}")

    def cost(self) -> str:
        if self.runtime is None and not self.board.agents:
            return "Nothing spent: no run has started in this conversation."
        lines = [f"Run {self.run_id}: {number(sum(v.calls for v in self.board.agents.values()))} steps, {money(self.board.total)}"]
        if self.runtime is not None:
            total = self.runtime.ledger.total
            lines.append(
                f"  {number(total.input_tokens)} in, {number(total.output_tokens)} out, "
                f"{number(total.cache_read_tokens)} cache read, {number(total.cache_write_tokens)} cache write"
            )
        if len(self.board.agents) > 1:
            lines += [f"  {name_of(self.board, v.id):<16} {v.calls:>4} steps  {money(v.cost):>10}" for v in self.board.agents.values()]
        return "\n".join(lines)

    def stop_run(self) -> None:
        if self.running() and self.runtime is not None:
            self.runtime.stop()
            self.say("Interrupted: the run is stopping.")
        else:
            self.say("No run is working.")

    def pick(self, mode: str, choice: str) -> None:
        """Apply a choice of ``/resume``, ``/view``, ``/model`` or ``/effort``.

        Model and effort are policy, not a wish of the agent: the choice goes over the role as a
        ``--set`` would, and the next task (or the next message to an open run) starts under it.
        """
        policy = self.loaded.policy
        if mode == "resume":
            self.open_run(choice)
            return
        if mode == "view":
            key = choice.lower()
            if key in VIEWS:
                self.show_view(key)
            else:
                self.say(f"No view {choice!r}. Use one of: {', '.join(VIEWS)}.", "bad")
            return
        if mode == "model" and choice not in policy.models:
            self.say(f"No model {choice!r}. The policy has: {', '.join(policy.models)}.", "bad")
            return
        if mode == "effort" and choice not in EFFORTS:
            self.say(f"No effort {choice!r}. Use one of: {', '.join(EFFORTS)}.", "bad")
            return
        before = self.extra
        role = {**self.extra.get("role", {}).get(policy.initial_role, {}), mode: choice}
        self.extra = {"role": {policy.initial_role: role}}
        try:
            self.loaded = self.load(self.extra)
        except Exception as error:
            self.extra = before
            self.say(f"Cannot use {mode} {choice}: {error}", "bad")
            return
        self.say(f"{mode.capitalize()} is now {choice} for role {policy.initial_role}.")
        self.refresh_status()

    """The menu above the message box: command completion and the pickers."""

    def open_picker(self, mode: str) -> None:
        policy = self.loaded.policy
        match mode:
            case "resume":
                runs = [summarize(path) for path in list_runs(policy.journal)[:RUNS_SHOWN]]
                options = [(r.run_id, f"{clock(r.started, '%d/%m %H:%M')}  {r.status:<10} {short(r.task, 80)}") for r in runs]
            case "model":
                options = [(alias, f"{alias:<10} {spec.id}") for alias, spec in policy.models.items()]
            case "view":
                options = [(key, label) for key, label in VIEWS.items()]
            case _:
                options = [(effort, effort) for effort in EFFORTS]
        if not options:
            self.say("No runs yet.")
            return
        self.query_one("#box", Input).placeholder = "Choose with Up and Down, Enter picks, Esc cancels; type to filter"
        self.open_menu(mode, options)

    def open_menu(self, mode: str, options: list[tuple[str, str]]) -> None:
        self.menu_mode, self.options = mode, options
        self.fill_menu(options)

    def fill_menu(self, options: list[tuple[str, str]]) -> None:
        menu = self.query_one("#menu", OptionList)
        menu.clear_options()
        menu.add_options([Option(label, id=key) for key, label in options])
        menu.display = bool(options)
        menu.highlighted = 0 if options else None
        self.query_one("#views").set_class(bool(options), "veiled")

    def close_menu(self) -> None:
        self.menu_mode, self.options = None, []
        self.query_one("#menu", OptionList).display = False
        self.query_one("#views").remove_class("veiled")
        self.query_one("#box", Input).placeholder = PLACEHOLDER

    def highlighted(self) -> str | None:
        menu = self.query_one("#menu", OptionList)
        if self.menu_mode is None or not menu.display or menu.highlighted is None:
            return None
        return menu.get_option_at_index(menu.highlighted).id

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        if self.menu_mode in PICKERS:
            needle = value.strip().lower()
            self.fill_menu([(key, label) for key, label in self.options if needle in label.lower()])
        elif value.startswith("/") and " " not in value:
            typed = value[1:]
            names = [name for name in COMMANDS if name.startswith(typed)]
            names += [name for alias, name in ALIASES.items() if alias.startswith(typed) and name not in names]
            found = [(name, f"/{name:<9} {COMMANDS[name]}") for name in names]
            self.menu_mode, self.options = "commands", found
            self.fill_menu(found)
        elif self.menu_mode == "commands":
            self.close_menu()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        mode, choice = self.menu_mode, event.option.id
        box = self.query_one("#box", Input)
        box.value = ""
        self.close_menu()
        if mode in PICKERS and choice:
            self.pick(mode, choice)
        elif choice:
            self.command(f"/{choice}")
        box.focus()

    def on_key(self, event: Key) -> None:
        """Up and Down move in the menu when it is open, and through the messages already sent otherwise."""
        if event.key not in ("up", "down"):
            return
        event.stop()
        menu = self.query_one("#menu", OptionList)
        if self.menu_mode is not None and menu.display:
            menu.action_cursor_up() if event.key == "up" else menu.action_cursor_down()
            return
        if not self.sent:
            return
        self.recall = max(0, min(len(self.sent), self.recall + (-1 if event.key == "up" else 1)))
        box = self.query_one("#box", Input)
        box.value = self.sent[self.recall] if self.recall < len(self.sent) else ""
        box.cursor_position = len(box.value)

    def action_complete(self) -> None:
        choice = self.highlighted()
        if self.menu_mode == "commands" and choice:
            box = self.query_one("#box", Input)
            box.value = f"/{choice} "
            box.cursor_position = len(box.value)

    def action_interrupt(self) -> None:
        box = self.query_one("#box", Input)
        if self.menu_mode is not None:
            box.value = ""
            self.close_menu()
        elif self.query_one("#notice").display:
            self.hush()
        elif self.running():
            self.stop_run()

    def action_leave(self) -> None:
        """Ctrl+C clears the box when it holds text, else interrupts a working run, else pressed twice it leaves."""
        box = self.query_one("#box", Input)
        now = time.monotonic()
        if box.value:
            box.value = ""
        elif self.running():
            self.stop_run()
        elif now - self.leave_at < QUIT_WINDOW:
            self.run_action("quit")
        else:
            self.leave_at = now
            self.say("Press Ctrl+C again to leave.")

    async def action_quit(self) -> None:
        if self.running() and self.runtime is not None and self.job is not None:
            self.runtime.stop()
            await asyncio.wait([self.job], timeout=10)
        self.exit()

    """The question bar, the prompt and the status line."""

    def refresh_ask(self) -> None:
        """An agent that waits for an answer cannot be missed: its question sits above the message box."""
        bar = self.query_one("#ask", Static)
        if self.runtime is not None:
            open_ = {question.id for question in self.runtime.questions()} if self.running() else set()
            for step in self.board.waiting():
                if step.ref not in open_:
                    self.board.answered(step.ref)
        questions = self.board.waiting()
        if not questions:
            bar.display = False
            return
        tokens = self.tokens
        text = Text()
        for step in questions[:3]:
            text.append(badge("waiting", tokens))
            text.append(f"  {name_of(self.board, step.agent)}  {elapsed(time.time() - step.ts)}  {step.ref}\n", style=tokens["ink-muted"])
            text.append(step.text.strip() + "\n", style=tokens["ink"])
        hint = "/answer <text>" if len(questions) == 1 else "/answer <id> <text>"
        text.append(f"Answer with {hint}", style=tokens["ink-faint"])
        bar.update(text)
        bar.display = True

    def refresh_prompt(self) -> None:
        several = len(self.board.agents) > 1
        self.query_one("#prompt", Static).update(f"{name_of(self.board, self.target())} > " if several else "> ")

    def tick(self) -> None:
        for agent_id in list(self.board.agents) if self.running() else []:
            if self.board.agents[agent_id].state in LIVE:
                self.measure(agent_id)
        try:
            self.current().tick()
            self.refresh_ask()
            self.refresh_status()
        except Exception as error:
            self.drawing_failed(error)

    def refresh_status(self) -> None:
        tokens, policy = self.tokens, self.loaded.policy
        muted = tokens["ink-muted"]
        parts: list[Text] = [Text(self.model_of(self.focus_agent, effort=True), style=muted)]
        if self.state == "working":
            parts.append(Text.assemble(("working ", tokens["tide"]), (elapsed(self.elapsed()), muted)))
        else:
            colour = {"idle": muted, "delivered": tokens["moss"]}.get(self.state, tokens["down"])
            parts.append(Text(self.state, style=colour))
        if self.board.agents:
            cap = policy.budget.dollars
            spent = self.board.total
            warn = cap and spent >= 0.8 * cap
            parts.append(Text(money(spent) + (f" of {money(cap)}" if cap else ""), style=tokens["sun"] if warn else muted))
            view = self.board.agents.get(self.focus_agent)
            limit = self.compact_limit(self.focus_agent)
            if view is not None and limit:
                parts.append(Text(f"context {number(view.context)}/{number(limit)}", style=muted))
            parts.append(Text(f"{number(sum(v.calls for v in self.board.agents.values()))} steps", style=muted))
            live = [v for v in self.board.agents.values() if v.state in LIVE]
            if len(self.board.agents) > 1:
                parts.append(Text(f"{len(live)} of {len(self.board.agents)} agents", style=muted))
            if waiting := len(self.board.waiting()):
                parts.append(Text(f"◆ {waiting} waiting", style=tokens["clay"]))
        if self.run_id:
            parts.append(Text(f"run {self.run_id}", style=muted))
        line = Text(" │ ", style=tokens["ink-faint"]).join(parts)
        line.no_wrap = True
        line.overflow = "ellipsis"
        self.query_one("#status", Static).update(line)


def run(cwd: Path, load: Load, factory: Callable[[], Provider] | None = None, task: str = "") -> int:
    SynthicApp(cwd, load, factory, task).run()
    return 0
