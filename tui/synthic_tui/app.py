"""The terminal interface of Synthic: one screen that starts, follows and steers a run.

It sits on ``synthic.agents.Runtime`` the same way the command line does. The engine knows nothing
about it: this package only reads the output callback, the ledger and the journal, and calls the
four controls of the runtime (``stop``, ``add``, ``redirect``, ``answer``).

The runtime runs on the event loop of the application, so its output callback may touch widgets
directly.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, Label, ListItem, ListView, RichLog, Static

from synthic.agents import Runtime
from synthic.cli import clock, short, summarize
from synthic.journal import is_alive, list_runs, read_records
from synthic.policy import LoadedPolicy
from synthic.types import Provider

HELP = "Enter sends.  /stop  /redirect  /answer  /new  /quit"
STYLES = {
    "you": "bold cyan",
    "tool": "dim",
    "note": "yellow",
    "end": "bold green",
    "fail": "bold red",
}
RUNS_SHOWN = 40


def bar(part: float, whole: float, width: int = 18) -> str:
    """A text gauge: ``part`` of ``whole`` as filled and empty cells."""
    filled = 0 if whole <= 0 else max(0, min(width, round(width * part / whole)))
    return "#" * filled + "-" * (width - filled)


def history(run_dir: Path) -> list[tuple[str, str]]:
    """What the root agent of a stored run said and did, as ``(style, line)`` pairs."""
    lines: list[tuple[str, str]] = []
    for record in read_records(run_dir, "root"):
        data = record.data
        match record.kind:
            case "user":
                lines.append(("you", "you: " + short(str(data.get("text", "")), 600)))
            case "assistant":
                for block in data.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        lines.append(("", str(block["text"])))
            case "tool_intent":
                lines.append(("tool", f"> {data.get('name')} {short(json.dumps(data.get('input'), ensure_ascii=False), 200)}"))
            case "compaction":
                lines.append(("note", "~ compaction: the conversation was replaced by a summary"))
            case "done":
                style = "end" if data.get("status") == "delivered" else "fail"
                lines.append((style, f"* ended {data.get('status')}: {data.get('summary', '')}"))
    return lines


class SynthicApp(App[None]):
    """Runs list on the left, conversation in the middle, state on the right, message box below."""

    TITLE = "Synthic"
    CSS = """
    #runs { width: 34; border-right: solid $panel; }
    #side { width: 40; border-left: solid $panel; padding: 0 1; }
    #main { width: 1fr; }
    #log { height: 1fr; padding: 0 1; }
    #live { height: auto; max-height: 6; padding: 0 1; color: $text; }
    #help { height: 1; padding: 0 1; color: $text-muted; }
    .title { text-style: bold; padding: 0 1; color: $accent; }
    ListItem { padding: 0 1; }
    """
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("f8", "stop_run", "Stop the run"),
        Binding("f2", "new_run", "New run"),
    ]

    def __init__(
        self,
        cwd: Path,
        load: Callable[[], LoadedPolicy],
        factory: Callable[[], Provider] | None = None,
        task: str = "",
    ) -> None:
        super().__init__()
        self.cwd = cwd
        self.load = load
        self.factory = factory
        self.first_task = task
        self.loaded = load()
        self.runtime: Runtime | None = None
        self.job: asyncio.Task[Any] | None = None
        self.run_id: str | None = None
        self.state = "idle"
        self.alive: set[str] = set()
        self.pending = ""
        self.transcript: list[str] = []

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="runs"):
                yield Label("Runs", classes="title")
                yield ListView(id="run-list")
            with Vertical(id="main"):
                yield RichLog(id="log", wrap=True, markup=False, highlight=False)
                yield Static("", id="live")
                yield Static(HELP, id="help")
                yield Input(placeholder="Write a task and press Enter", id="box")
            with Vertical(id="side"):
                yield Label("State", classes="title")
                yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = str(self.cwd)
        self.refresh_runs()
        self.refresh_status()
        self.set_interval(1.0, self.refresh_status)
        self.query_one("#box", Input).focus()
        if self.first_task:
            self.begin(self.first_task)

    def say(self, line: str, style: str = "") -> None:
        """One finished line of the conversation."""
        self.flush_text()
        self.transcript.append(line)
        self.query_one("#log", RichLog).write(Text(line, style=STYLES.get(style, style)))

    def flush_text(self) -> None:
        if self.pending:
            line, self.pending = self.pending, ""
            self.transcript.append(line)
            self.query_one("#log", RichLog).write(Text(line))
        self.query_one("#live", Static).update("")

    def stream(self, chunk: str) -> None:
        """Model text arrives in pieces: whole lines go to the log, the open line stays below it."""
        *whole, self.pending = (self.pending + chunk).split("\n")
        log = self.query_one("#log", RichLog)
        for line in whole:
            self.transcript.append(line)
            log.write(Text(line))
        self.query_one("#live", Static).update(self.pending)

    def engine_output(self, agent_id: str, kind: str, text: str) -> None:
        """The output callback of the runtime."""
        who = f"[{agent_id}] " if len(self.alive) > 1 else ""
        match kind:
            case "text":
                self.stream(text)
            case "tool_start":
                self.say(f"{who}> {text}", "tool")
            case "tool_end":
                self.say(f"{who}< {text}", "tool")
            case "agent_start":
                self.alive.add(agent_id)
                self.say(f"{who}* started {text}", "note")
            case "agent_end":
                self.say(f"{who}* ended {text}", "end" if text.startswith("delivered") else "fail")
                self.alive.discard(agent_id)
            case "question":
                self.say(f"{who}? question {text}", "note")
            case "warning":
                self.say(f"! {text}", "note")
            case _:
                self.say(f"{who}{kind}: {text}", "tool")
        self.refresh_status()

    def begin(self, text: str) -> None:
        """Start a run with ``text``, or continue the selected one with it as the next message."""
        self.loaded = self.load()
        provider = self.factory() if self.factory else None
        try:
            self.runtime = Runtime(self.loaded, self.cwd, provider, run_id=self.run_id)
        except Exception as error:
            self.say(f"cannot start: {error}", "fail")
            return
        self.runtime.on_output(self.engine_output)
        for warning in self.runtime.warnings:
            self.say(f"! {warning}", "note")
        resumed = self.run_id is not None
        self.run_id = self.runtime.run_id
        self.state = "running"
        self.alive.clear()
        self.say(f"you: {text}", "you")
        work = self.runtime.resume(text) if resumed else self.runtime.run(text)
        self.job = asyncio.create_task(self.follow(work))
        self.refresh_status()

    async def follow(self, work: Any) -> None:
        runtime = self.runtime
        assert runtime is not None
        try:
            result = await work
            self.state = result.status
        except Exception as error:
            self.say(f"error: {type(error).__name__}: {error}", "fail")
            self.state = "failed"
        finally:
            await runtime.close()
            self.alive.clear()
            self.flush_text()
            self.refresh_runs()
            self.refresh_status()

    def running(self) -> bool:
        return self.job is not None and not self.job.done()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        command, _, rest = text.partition(" ")
        rest = rest.strip()
        match command:
            case "/quit":
                self.run_action("quit")
            case "/new":
                self.action_new_run()
            case "/stop":
                self.action_stop_run()
            case "/redirect" if self.running() and rest:
                assert self.runtime is not None
                self.runtime.redirect(rest)
                self.say(f"you (redirect): {rest}", "you")
            case "/answer" if self.running() and " " in rest:
                assert self.runtime is not None
                question, _, answer = rest.partition(" ")
                try:
                    self.runtime.answer(question, answer.strip())
                    self.say(f"you (answer {question}): {answer.strip()}", "you")
                except KeyError as error:
                    self.say(str(error.args[0]), "fail")
            case "/redirect" | "/answer":
                self.say("usage: /redirect <text> or /answer <id> <text>, while a run is working", "note")
            case _ if self.running():
                assert self.runtime is not None
                self.runtime.add(text)
                self.say(f"you: {text}", "you")
            case _:
                self.begin(text)

    def action_stop_run(self) -> None:
        if self.running() and self.runtime is not None:
            self.runtime.stop()
            self.say("= stop requested", "note")
        else:
            self.say("= no run is working", "note")

    def action_new_run(self) -> None:
        if self.running():
            self.say("= a run is working: stop it first", "note")
            return
        self.run_id = None
        self.runtime = None
        self.state = "idle"
        self.transcript.clear()
        self.query_one("#log", RichLog).clear()
        self.refresh_status()

    async def action_quit(self) -> None:
        if self.running() and self.runtime is not None and self.job is not None:
            self.runtime.stop()
            await asyncio.wait([self.job], timeout=10)
        self.exit()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Open a stored run: show what it did, and make the next message continue it."""
        run_id = event.item.name
        if not run_id or self.running():
            return
        run_dir = next((path for path in list_runs(self.loaded.policy.journal) if path.name == run_id), None)
        if run_dir is None:
            return
        self.transcript.clear()
        self.query_one("#log", RichLog).clear()
        for style, line in history(run_dir):
            self.say(line, style)
        if is_alive(run_dir):
            self.run_id = None
            self.say("= this run is working in another process: it is shown read-only", "note")
        else:
            self.run_id = run_id
            self.say("= your next message continues this run", "note")
        self.runtime = None
        self.state = "idle"
        self.refresh_status()
        self.query_one("#box", Input).focus()

    def refresh_runs(self) -> None:
        view = self.query_one("#run-list", ListView)
        view.clear()
        for path in list_runs(self.loaded.policy.journal)[:RUNS_SHOWN]:
            info = summarize(path)
            label = f"{clock(info.started, '%m-%d %H:%M')} {info.status}\n{short(info.task, 28)}"
            view.append(ListItem(Label(label), name=info.run_id))

    def refresh_status(self) -> None:
        policy = self.loaded.policy
        role = policy.roles.get(policy.initial_role)
        lines = [
            f"Run     {self.run_id or 'new'}",
            f"State   {self.state}",
            f"Role    {policy.initial_role}",
            f"Model   {role.model if role else '?'}",
            f"Effort  {(role.effort if role else None) or 'default'}",
        ]
        runtime = self.runtime
        if runtime is not None:
            total = runtime.ledger.total
            cap = policy.budget.dollars
            lines += ["", f"Steps   {runtime.ledger.steps}"]
            lines += [f"Cost    ${total.cost_usd:.4f}" + (f" of ${cap:g}" if cap else "")]
            if cap:
                lines += [f"        {bar(total.cost_usd, cap)}"]
            agent = runtime.root_agent
            spec = policy.models.get(agent._model) if agent is not None else None
            if agent is not None and spec is not None:
                used = agent.context.last_prompt_tokens
                limit = round(policy.context.compact_at * agent.context.compaction_window(spec))
                lines += [f"Prompt  {used:,} tokens", f"        {bar(used, limit)}", f"        compacts at {limit:,}"]
            lines += [
                "",
                f"Out     {total.output_tokens:,} tokens",
                f"Cache   {total.cache_read_tokens:,} read",
                f"Agents  {len(self.alive)} working",
            ]
            questions = runtime.questions()
            if questions:
                lines += ["", "Open questions"] + [f"  {q.id}: {short(q.text, 60)}" for q in questions]
        self.query_one("#status", Static).update("\n".join(lines))


def run(cwd: Path, load: Callable[[], LoadedPolicy], factory: Callable[[], Provider] | None = None, task: str = "") -> int:
    SynthicApp(cwd, load, factory, task).run()
    return 0
