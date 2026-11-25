"""The terminal interface of Synthic: one column that starts, follows and steers a run.

The conversation scrolls above, the message box sits below it and one status line below that, as in the
command lines of other coding agents. A line that starts with ``/`` is a command and never reaches the
model; anything else is a task, or a message to the agent that is working.

It sits on ``synthic.agents.Runtime`` the same way the command line does. The engine knows nothing about
it: this package reads the output callback, the ledger and the journal, and calls the four controls of the
runtime (``stop``, ``add``, ``redirect``, ``answer``). The runtime runs on the event loop of the
application, so its output callback may touch widgets directly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option

from synthic import __version__
from synthic.agents import Runtime
from synthic.cli import clock, short, summarize
from synthic.journal import is_alive, list_runs, read_records
from synthic.policy import LoadedPolicy
from synthic.transcript import entries, result_summary
from synthic.types import Provider

Load = Callable[[dict[str, Any] | None], LoadedPolicy]

COMMANDS = {
    "help": "show the commands and the keys",
    "new": "start a new conversation",
    "resume": "open an earlier run and continue it",
    "model": "choose the model of the next tasks",
    "effort": "choose the effort of the next tasks",
    "stop": "stop the run that is working",
    "redirect": "stop the current step and give a new direction: /redirect <text>",
    "answer": "answer an open question: /answer <id> <text>",
    "cost": "show what this run has spent",
    "quit": "leave Synthic",
}
ALIASES = {"exit": "quit", "clear": "new", "runs": "resume"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
PICKERS = ("resume", "model", "effort")
PLACEHOLDER = "Write a task and press Enter, or / for commands"
KEYS = "Enter sends, / lists the commands, Up recalls a message, Esc interrupts, Ctrl+C twice leaves."
RUNS_SHOWN = 30
QUIT_WINDOW = 2.0


class SynthicApp(App[None]):
    """Conversation above, message box below, one status line under it."""

    TITLE = "Synthic"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: vertical; }
    #chat { height: 1fr; padding: 1 2 0 2; scrollbar-size-vertical: 1; }
    #chat > Static { max-width: 120; height: auto; margin-bottom: 1; }
    #chat > .tool { margin-bottom: 0; color: $text-muted; }
    #chat > .you { background: $boost; padding: 0 1; text-style: bold; }
    #chat > .note { color: $warning; }
    #chat > .good { color: $success; }
    #chat > .bad { color: $error; }
    #chat > .quiet { color: $text-muted; }
    #menu { max-width: 124; max-height: 12; height: auto; margin: 0 2; border: round $panel; display: none; }
    #box { max-width: 124; margin: 0 2; border: round $accent; }
    #status { height: 1; padding: 0 3; color: $text-muted; }
    """
    BINDINGS = [
        Binding("ctrl+c", "leave", show=False, priority=True),
        Binding("tab", "complete", show=False, priority=True),
        Binding("escape", "interrupt", show=False),
    ]

    def __init__(self, cwd: Path, load: Load, factory: Callable[[], Provider] | None = None, task: str = "") -> None:
        super().__init__()
        self.cwd = cwd
        self.load = load
        self.factory = factory
        self.first_task = task
        self.extra: dict[str, Any] = {}
        self.loaded = load(None)
        self.runtime: Runtime | None = None
        self.job: asyncio.Task[Any] | None = None
        self.run_id: str | None = None
        self.state = "idle"
        self.started = 0.0
        self.alive: set[str] = set()
        self.transcript: list[str] = []
        self.sent: list[str] = []
        self.recall = 0
        self.speaking: Static | None = None
        self.spoken = ""
        self.calls: dict[tuple[str, str], list[tuple[Static, str]]] = {}
        self.menu_mode: str | None = None
        self.options: list[tuple[str, str]] = []
        self.leave_at = 0.0

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="chat")
        menu = OptionList(id="menu")
        menu.can_focus = False
        yield menu
        yield Input(placeholder=PLACEHOLDER, id="box")
        yield Static("", id="status")

    def on_mount(self) -> None:
        self.welcome()
        self.refresh_status()
        self.set_interval(1.0, self.refresh_status)
        self.query_one("#box", Input).focus()
        if self.first_task:
            self.begin(self.first_task)

    """The conversation."""

    def show(self, body: RenderableType, plain: str, style: str = "") -> Static:
        """Add one entry to the conversation; ``plain`` is its text, kept for whoever reads ``transcript``."""
        self.close_speech()
        return self.place(body, plain, style)

    def place(self, body: RenderableType, plain: str, style: str = "") -> Static:
        entry = Static(body, classes=style)
        chat = self.query_one("#chat", VerticalScroll)
        chat.mount(entry)
        chat.scroll_end(animate=False)
        self.transcript.append(plain)
        return entry

    def say(self, line: str, style: str = "") -> Static:
        return self.show(Text(line), line, style)

    def welcome(self) -> None:
        self.say(f"Synthic {__version__}  {self.cwd}\n{KEYS}", "quiet")

    def speak(self, chunk: str) -> None:
        """Model text arrives in pieces: one entry grows as they come and is drawn as Markdown."""
        self.spoken += chunk
        if self.speaking is None:
            self.speaking = self.place(Markdown(self.spoken), "")
            self.transcript.pop()
        else:
            self.speaking.update(Markdown(self.spoken))
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    def close_speech(self) -> None:
        if self.speaking is not None and self.spoken.strip():
            self.transcript.append(self.spoken.strip())
        elif self.speaking is not None:
            self.speaking.remove()
        self.speaking, self.spoken = None, ""

    def engine_output(self, agent_id: str, kind: str, text: str) -> None:
        """The output callback of the runtime."""
        who = f"[{agent_id}] " if len(self.alive) > 1 else ""
        name = text.partition(":")[0].partition(" ")[0]
        match kind:
            case "text":
                self.speak(text)
            case "tool_start" if name != "done":
                line = f"{who}● {text}"
                self.calls.setdefault((agent_id, name), []).append((self.say(line, "tool"), line))
            case "tool_end" if name != "done":
                waiting = self.calls.get((agent_id, name))
                outcome = text.partition(" ")[2]
                if waiting:
                    entry, line = waiting.pop(0)
                    failed = not outcome.startswith("ok")
                    entry.update(Text.assemble(line, "  ", (outcome, "red" if failed else "green")))
                    self.transcript.append(f"{line}  {outcome}")
            case "agent_start":
                self.alive.add(agent_id)
                if agent_id != "root":
                    self.say(f"{who}started {text}", "quiet")
            case "agent_end":
                self.alive.discard(agent_id)
                if agent_id != "root":
                    self.say(f"[{agent_id}] ended {text}", "quiet")
            case "question":
                question = text.partition(":")[0]
                self.say(f"{who}? {text}\n  Answer with /answer {question} <text>", "note")
            case "warning":
                self.say(f"! {text}", "note")
            case "todo":
                self.say(f"{who}Task list\n{text}", "quiet")
            case "tool_start" | "tool_end":
                """The ``done`` call is not shown, but it still ends what the model was saying."""
                self.close_speech()
            case _:
                self.say(f"{who}{kind}: {text}", "quiet")
        self.refresh_status()

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
        for warning in self.runtime.warnings:
            self.say(f"! {warning}", "note")
        resumed = self.run_id is not None
        self.run_id = self.runtime.run_id
        self.state, self.started = "working", time.monotonic()
        self.alive.clear()
        self.calls.clear()
        self.say(f"> {text}", "you")
        work = self.runtime.resume(text) if resumed else self.runtime.run(text)
        self.job = asyncio.create_task(self.follow(self.runtime, work))
        self.refresh_status()

    async def follow(self, runtime: Runtime, work: Any) -> None:
        try:
            result = await work
            self.state = result.status
            good = result.status == "delivered"
            self.show(Markdown(result.summary), f"{result.status}: {result.summary}")
            self.say(("✓ " if good else "✗ ") + result.status, "good" if good else "bad")
        except Exception as error:
            self.state = "failed"
            self.say(f"✗ {type(error).__name__}: {error}", "bad")
        finally:
            await runtime.close()
            self.alive.clear()
            self.close_speech()
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
        for entry in entries(read_records(run_dir, "root")):
            match entry.kind:
                case "user":
                    self.say(f"> {entry.text}", "you")
                case "text":
                    self.show(Markdown(entry.text), entry.text)
                case "tool":
                    line = f"● {entry.text}  {result_summary(entry)}"
                    self.show(Text.assemble(f"● {entry.text}  ", (result_summary(entry), "red" if entry.error else "green")), line, "tool")
                case "note":
                    self.say(f"! {entry.text}", "note")
                case "end":
                    self.show(Markdown(entry.detail), f"{entry.text}: {entry.detail}")
                    self.say(("✗ " if entry.error else "✓ ") + entry.text, "bad" if entry.error else "good")
        if is_alive(run_dir):
            self.say(f"Run {run_id} is working in another process: it is shown read-only.", "note")
        else:
            self.run_id = run_id
            self.say(f"Run {run_id} is open: your next message continues it.", "quiet")
        self.refresh_status()

    def clear(self) -> None:
        self.close_speech()
        self.run_id, self.runtime, self.state = None, None, "idle"
        self.transcript.clear()
        self.calls.clear()
        self.query_one("#chat", VerticalScroll).remove_children()

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
        self.sent.append(text)
        self.recall = len(self.sent)
        if text.startswith("/"):
            self.command(text)
        elif self.running() and self.runtime is not None:
            self.runtime.add(text)
            self.say(f"> {text}", "you")
        else:
            self.begin(text)

    def command(self, line: str) -> None:
        """Run a ``/`` command. Nothing typed after a slash ever reaches the model as a task."""
        word, _, rest = line[1:].partition(" ")
        word, rest = ALIASES.get(word, word), rest.strip()
        runtime = self.runtime if self.running() else None
        match word:
            case "help":
                rows = "\n".join(f"  /{name:<9} {about}" for name, about in COMMANDS.items())
                self.say(f"Commands\n{rows}\n\n{KEYS}", "quiet")
            case "quit":
                self.run_action("quit")
            case "new" | "resume" | "model" | "effort" if self.running():
                self.say(f"A run is working: /stop it before /{word}.", "note")
            case "new":
                self.clear()
                self.welcome()
                self.refresh_status()
            case "resume" | "model" | "effort" if rest:
                self.pick(word, rest)
            case "resume" | "model" | "effort":
                self.open_picker(word)
            case "stop":
                self.stop_run()
            case "redirect" if runtime is not None and rest:
                runtime.redirect(rest)
                self.say(f"> (redirect) {rest}", "you")
            case "answer" if runtime is not None and " " in rest:
                question, _, answer = rest.partition(" ")
                try:
                    runtime.answer(question, answer.strip())
                    self.say(f"> (answer {question}) {answer.strip()}", "you")
                except KeyError as error:
                    self.say(str(error.args[0]), "bad")
            case "redirect" | "answer":
                self.say(f"/{word} works while a run is working: {COMMANDS[word]}", "note")
            case "cost":
                self.say(self.cost(), "quiet")
            case _:
                self.say(f"Unknown command /{word}. /help lists the commands.", "bad")

    def cost(self) -> str:
        if self.runtime is None:
            return "Nothing spent: no run has started in this conversation."
        total = self.runtime.ledger.total
        return (
            f"Run {self.run_id}: {self.runtime.ledger.steps} steps, ${total.cost_usd:.4f}\n"
            f"  {total.input_tokens:,} in, {total.output_tokens:,} out, "
            f"{total.cache_read_tokens:,} cache read, {total.cache_write_tokens:,} cache write"
        )

    def stop_run(self) -> None:
        if self.running() and self.runtime is not None:
            self.runtime.stop()
            self.say("Interrupted: the run is stopping.", "note")
        else:
            self.say("No run is working.", "quiet")

    def pick(self, mode: str, choice: str) -> None:
        """Apply a choice of ``/resume``, ``/model`` or ``/effort``.

        Model and effort are policy, not a wish of the agent: the choice goes over the role as a
        ``--set`` would, and the next task (or the next message to an open run) starts under it.
        """
        policy = self.loaded.policy
        if mode == "resume":
            self.open_run(choice)
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
        self.say(f"{mode.capitalize()} is now {choice} for role {policy.initial_role}.", "quiet")
        self.refresh_status()

    """The menu above the message box: command completion and the three pickers."""

    def open_picker(self, mode: str) -> None:
        policy = self.loaded.policy
        match mode:
            case "resume":
                runs = [summarize(path) for path in list_runs(policy.journal)[:RUNS_SHOWN]]
                options = [(r.run_id, f"{clock(r.started, '%m-%d %H:%M')}  {r.status:<10} {short(r.task, 80)}") for r in runs]
            case "model":
                options = [(alias, f"{alias:<10} {spec.id}") for alias, spec in policy.models.items()]
            case _:
                options = [(effort, effort) for effort in EFFORTS]
        if not options:
            self.say("No runs yet.", "quiet")
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

    def close_menu(self) -> None:
        self.menu_mode, self.options = None, []
        self.query_one("#menu", OptionList).display = False
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
            found = [(name, f"/{name:<9} {about}") for name, about in COMMANDS.items() if name.startswith(value[1:])]
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
        elif self.running():
            self.stop_run()

    def action_leave(self) -> None:
        """Ctrl+C clears the box when it holds text; pressed twice on an empty box it leaves."""
        box = self.query_one("#box", Input)
        now = time.monotonic()
        if box.value:
            box.value = ""
        elif now - self.leave_at < QUIT_WINDOW:
            self.run_action("quit")
        else:
            self.leave_at = now
            self.notify("Press Ctrl+C again to leave.", timeout=QUIT_WINDOW)

    async def action_quit(self) -> None:
        if self.running() and self.runtime is not None and self.job is not None:
            self.runtime.stop()
            await asyncio.wait([self.job], timeout=10)
        self.exit()

    """The status line."""

    def refresh_status(self) -> None:
        policy = self.loaded.policy
        role = policy.roles.get(policy.initial_role)
        parts = [f"{role.model if role else '?'} · {(role.effort if role else None) or 'default effort'}"]
        if self.state == "working":
            spent = int(time.monotonic() - self.started)
            parts.append(f"working {spent // 60}:{spent % 60:02d}")
        else:
            parts.append(self.state)
        runtime = self.runtime
        if runtime is not None:
            total = runtime.ledger.total
            cap = policy.budget.dollars
            parts.append(f"${total.cost_usd:.4f}" + (f" of ${cap:g}" if cap else ""))
            agent = runtime.root_agent
            spec = policy.models.get(agent._model) if agent is not None else None
            if agent is not None and spec is not None:
                used = agent.context.last_prompt_tokens
                limit = round(policy.context.compact_at * agent.context.compaction_window(spec))
                parts.append(f"context {used:,}/{limit:,}")
            parts.append(f"{runtime.ledger.steps} steps")
            if len(self.alive) > 1:
                parts.append(f"{len(self.alive)} agents")
            if questions := runtime.questions():
                parts.append(f"{len(questions)} open question{'s' if len(questions) > 1 else ''}")
        if self.run_id:
            parts.append(f"run {self.run_id}")
        self.query_one("#status", Static).update(" │ ".join(parts))


def run(cwd: Path, load: Load, factory: Callable[[], Provider] | None = None, task: str = "") -> int:
    SynthicApp(cwd, load, factory, task).run()
    return 0
