"""The command line of Synthic: ``run``, ``resume``, ``runs``, ``show``, ``policy``, ``questions``, ``prune``,
``rewind`` and the four remote controls ``stop``, ``say``, ``redirect`` and ``answer``.

Two product rules shape everything here. Once the rules are set, a run never prompts: ``run`` reads stdin only
when it is a terminal, and then only to let the human interrupt. And the human can always interrupt: from the
same terminal (``/stop``, ``/redirect``, ``/answer``, any other line is a message, Ctrl+C is a stop) or from
another one, through ``<run folder>/control.jsonl``. The remote commands append one JSON line to that file and
the running process polls it by size, twice a second, applying only the lines it has not seen. A file keeps the
protocol free of sockets and ports, and it works the same for a run started hours ago by a scheduler.

Journals are read with ``journal.read_records``, which never repairs a file: a reader must never touch the
journal of a run that is still being written. A run is live while its runtime keeps ``<run folder>/alive``
fresh; the remote commands refuse to write to a run that is not.
"""

from __future__ import annotations

import argparse
import asyncio
import codecs
import json
import os
import re
import shutil
import signal
import sys
import textwrap
import threading
import time
import tomllib
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

from . import __version__
from .agents import Runtime
from .checkpoints import describe as describe_checkpoint
from .checkpoints import rewind
from .journal import agents_in, ended, is_alive, list_runs, prune, read_records
from .loop import AgentResult
from .media import MediaError
from .media import load as load_media
from .policy import LoadedPolicy, PolicyError, deep_merge, load_policy
from .provider import KEY_MISSING, find_api_key
from .tools.todo import render as render_todo
from .tools.todo import tally as tally_todo
from .transcript import markdown, tool_line
from .types import Attachment, Provider, Record, Usage

CONTROL_FILE = "control.jsonl"
POLL_SECONDS = 0.5
RUN_ID_SHAPE = re.compile(r"\d{8}-\d{6}-[0-9a-f]{4}")
EXIT_CODES = {"delivered": 0, "replied": 0, "failed": 1, "budget": 3, "stopped": 130, "cancelled": 1}
ANSI = {"dim": "2", "red": "31", "green": "32", "yellow": "33"}
STATUS_COLOR = {"delivered": "green", "replied": "green", "failed": "red", "budget": "yellow", "stopped": "yellow"}

TUI_DEFAULTS = {"human": {"present": True}}
"""What the terminal interface assumes before any policy file speaks: a person is at the terminal."""

ProviderFactory = Callable[[], Provider]


class CliError(Exception):
    """A problem the human can fix: it is printed as one line and ends the command with ``code``."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


def eprint(text: str) -> None:
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def parse_value(raw: str) -> Any:
    """A ``--set`` value: a TOML scalar, array or table when it parses as one, else the raw string."""
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw
    return value if isinstance(value, str | int | float | bool | list | dict) else raw


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """Turn ``["role.build.model=sonnet", "budget.steps=40"]`` into the nested dict ``load_policy`` takes."""
    tree: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        parts = key.strip().split(".")
        if not sep or not all(parts):
            raise CliError(f"--set expects KEY=VALUE with a dotted key, got {pair!r}")
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise CliError(f"--set {pair!r} conflicts with another --set on {part!r}")
            node = child
        if isinstance(node.get(parts[-1]), dict):
            raise CliError(f"--set {pair!r} conflicts with another --set below {parts[-1]!r}")
        node[parts[-1]] = parse_value(raw)
    return tree


def synthic_home() -> Path:
    """The folder holding ``.synthic/policy.toml`` of the user level: ``SYNTHIC_HOME`` or the real home."""
    raw = os.environ.get("SYNTHIC_HOME")
    return Path(raw) if raw else Path.home()


def load(
    args: argparse.Namespace, extra: dict[str, Any] | None = None, defaults: dict[str, Any] | None = None
) -> tuple[Path, LoadedPolicy]:
    """The policy in force for ``args``; ``extra`` goes over the ``--set`` values, as an interface's own choices,
    and ``defaults`` under every policy file, as its starting values."""
    folder = Path(args.cwd or Path.cwd()).resolve()
    if not folder.is_dir():
        raise CliError(f"--cwd {folder} is not a folder")
    overrides = deep_merge(deep_merge(parse_overrides(args.set), getattr(args, "schema_override", {})), extra or {})
    loaded = load_policy(
        folder,
        Path(args.policy) if args.policy else None,
        overrides or None,
        args.trust_project,
        home=synthic_home(),
        defaults=defaults,
    )
    return folder, loaded


def agent_order(agent_id: str) -> tuple[int, int, int]:
    if agent_id == "root":
        return (0, 0, 0)
    match = re.fullmatch(r"a(\d+)(?:\.(\d+))?", agent_id)
    return (1, int(match[1]), int(match[2] or 1)) if match else (2, 0, 0)


def root_end(run_dir: Path) -> Record | None:
    return ended(read_records(run_dir, "root"))


def run_folder(loaded: LoadedPolicy, run_id: str | None) -> Path:
    """The folder of ``run_id``, or of the most recently active run when it is omitted."""
    base = Path(loaded.policy.journal.dir).expanduser()
    if run_id is None:
        runs = list_runs(loaded.policy.journal)
        if not runs:
            raise CliError(f"no runs in {base}")
        return runs[0]
    path = base / run_id
    if Path(run_id).name != run_id or run_id in (".", "..") or not path.is_dir():
        raise CliError(f"no run {run_id!r} in {base}")
    return path


def short(text: str, limit: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def addressed(text: str) -> tuple[dict[str, str], str]:
    """Split a leading ``@agent`` from ``text``: ``({"agent": "a1"}, "rest")``, or ``({}, text)`` without one."""
    if text.startswith("@") and len(text) > 1:
        name, _, body = text[1:].partition(" ")
        return {"agent": name}, body.strip()
    return {}, text


def parse_line(line: str) -> dict[str, str] | None:
    """One line typed while a run is in progress, as a control command; ``None`` for an empty line.

    ``/stop``, ``/redirect <text>``, ``/answer <question-id> <text>``, ``/questions`` and ``/add <text>`` are
    commands. Any other line is a message for the agent. ``@a1 <text>`` after ``/add`` or ``/redirect``, or at
    the start of a plain line, addresses child ``a1`` instead of the root. A command with missing arguments
    becomes an ``error`` command that carries the usage text.
    """
    parts = line.split(maxsplit=1)
    if not parts:
        return None
    verb, rest = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""
    match verb:
        case "/stop":
            return {"cmd": "stop"}
        case "/questions":
            return {"cmd": "questions"}
        case "/redirect" | "/add":
            agent, body = addressed(rest)
            if not body:
                return {"cmd": "error", "text": f"usage: {verb} [@agent] <text>"}
            return {"cmd": "redirect" if verb == "/redirect" else "say", "text": body, **agent}
        case "/answer":
            asked = rest.split(maxsplit=1)
            if len(asked) < 2:
                return {"cmd": "error", "text": "usage: /answer <question-id> <text>"}
            return {"cmd": "answer", "question": asked[0], "text": asked[1].strip()}
    agent, body = addressed(line.strip())
    return {"cmd": "say", "text": body, **agent} if body else {"cmd": "error", "text": "usage: @agent <text>"}


class Printer:
    """The output callback of a run: plain text for terminals and logs, JSON lines for programs, or nothing.

    Colour only when stdout is a terminal that understands it and ``NO_COLOR`` is unset. With more than one
    agent alive every line carries the agent id as a prefix, and a line of streamed text is never cut by
    another agent's event.
    """

    def __init__(self, mode: str = "text") -> None:
        self.mode = mode
        self.alive: set[str] = set()
        self.at_line_start = True
        self.text_agent: str | None = None
        self.hidden: list[str] = []
        isatty = sys.stdout.isatty()
        ansi_ready = sys.platform != "win32" or any(k in os.environ for k in ("WT_SESSION", "TERM", "TERM_PROGRAM"))
        self.color = isatty and ansi_ready and not os.environ.get("NO_COLOR")

    def hide_warnings(self, warnings: Iterable[str]) -> None:
        """Warnings already printed on stderr at the start: the runtime repeats them, this drops the repeat."""
        self.hidden = list(warnings)

    def paint(self, text: str, style: str) -> str:
        return f"\x1b[{ANSI[style]}m{text}\x1b[0m" if self.color else text

    def write(self, text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def prefix(self, agent_id: str) -> str:
        return f"[{agent_id}] " if len(self.alive) > 1 else ""

    def end_line(self) -> None:
        if not self.at_line_start:
            self.write("\n")
            self.at_line_start = True

    def line(self, agent_id: str, body: str, style: str | None = None) -> None:
        self.end_line()
        text = self.prefix(agent_id) + body
        self.write((self.paint(text, style) if style else text) + "\n")

    def __call__(self, agent_id: str, kind: str, text: str) -> None:
        if self.mode == "quiet":
            return
        if kind == "warning" and text in self.hidden:
            self.hidden.remove(text)
            return
        if self.mode == "json":
            self.write(json.dumps({"agent": agent_id, "kind": kind, "text": text}, ensure_ascii=False) + "\n")
            return
        match kind:
            case "text":
                self.stream(agent_id, text)
            case "tool_start":
                self.line(agent_id, f"> {text}", "dim")
            case "tool_end":
                self.line(agent_id, f"< {text}", "dim")
            case "agent_start":
                self.alive.add(agent_id)
                self.line(agent_id, f"* started {text}")
            case "agent_end":
                self.line(agent_id, f"* ended {text}")
                self.alive.discard(agent_id)
            case "question":
                self.line(agent_id, f"? question {text}", "yellow")
            case "warning":
                self.line(agent_id, f"! {text}", "yellow")
            case "todo":
                self.line(agent_id, "todo:\n" + textwrap.indent(text, "  "), "dim")
            case _:
                self.line(agent_id, f"{kind}: {text}")

    def stream(self, agent_id: str, chunk: str) -> None:
        if not self.at_line_start and self.text_agent != agent_id:
            self.end_line()
        self.text_agent = agent_id
        pieces = chunk.split("\n")
        for index, piece in enumerate(pieces):
            if piece:
                if self.at_line_start:
                    self.write(self.prefix(agent_id))
                self.write(piece)
                self.at_line_start = False
            if index < len(pieces) - 1:
                self.write("\n")
                self.at_line_start = True

    def note(self, text: str) -> None:
        """A message from the CLI itself (an interrupt was applied, the open questions)."""
        if self.mode == "quiet":
            return
        if self.mode == "json":
            self.write(json.dumps({"agent": "cli", "kind": "note", "text": text}, ensure_ascii=False) + "\n")
            return
        self.line("cli", f"= {text}")

    def finish(self, result: AgentResult, runtime: Runtime, code: int) -> None:
        total = runtime.ledger.total
        if self.mode == "json":
            payload = {
                "kind": "result",
                "status": result.status,
                "summary": result.summary,
                "steps": runtime.ledger.steps,
                "tokens": asdict(total),
                "cost_usd": total.cost_usd,
                "run_id": runtime.run_id,
                "changes": result.changes,
                "result": result.result,
                "exit_code": code,
            }
            self.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return
        self.end_line()
        if self.mode == "text":
            self.write("\n")
        status = self.paint(f"status: {result.status}", STATUS_COLOR.get(result.status, "dim"))
        lines = [
            status,
            f"summary: {result.summary}",
            f"steps: {runtime.ledger.steps}",
            f"tokens: {total.input_tokens} in, {total.output_tokens} out, "
            f"{total.cache_read_tokens} cache read, {total.cache_write_tokens} cache write",
            f"cost: ${total.cost_usd:.4f}",
            f"run: {runtime.run_id}",
        ]
        if total.web_search_requests or total.web_fetch_requests:
            lines.insert(4, f"web: {total.web_search_requests} searches, {total.web_fetch_requests} fetches")
        if result.result is not None:
            lines.insert(2, "result: " + json.dumps(result.result, ensure_ascii=False))
        if result.changes:
            lines += ["changes:", result.changes]
        self.write("\n".join(lines) + "\n")


class Control:
    """Applies control commands, whether they come from a typed line or from ``control.jsonl``."""

    def __init__(self, runtime: Runtime, printer: Printer) -> None:
        self.runtime = runtime
        self.printer = printer

    def line(self, line: str) -> None:
        if (command := parse_line(line)) is not None:
            self.apply(command)

    def apply(self, command: dict[str, Any]) -> None:
        text = str(command.get("text", ""))
        note = self.printer.note
        try:
            match command.get("cmd"):
                case "stop":
                    note("stop requested")
                    self.runtime.stop()
                case "say":
                    files = [str(f) for f in command.get("attach", []) or []]
                    extra = {"attachments": attachments(files)} if files else {}
                    self.runtime.add(text, command.get("agent"), **extra)
                    shown = f" with {len(files)} attachment{'s' if len(files) > 1 else ''}" if files else ""
                    note(f"message queued for {command.get('agent') or 'the agent'}{shown}")
                case "redirect":
                    self.runtime.redirect(text, command.get("agent"))
                    note(f"redirecting {command.get('agent') or 'the agent'}")
                case "answer":
                    question = str(command.get("question", ""))
                    self.runtime.answer(question, text)
                    note(f"answered {question}")
                case "questions":
                    open_ones = self.runtime.questions()
                    note("\n".join(f"{q.id}: {q.text}" for q in open_ones) if open_ones else "no open questions")
                case "error":
                    note(text)
                case other:
                    note(f"unknown control command {other!r}")
        except KeyError as error:
            note(str(error.args[0]))
        except CliError as error:
            note(f"not sent: {error}")

    async def watch(self, path: Path, offset: int) -> None:
        """Apply the complete lines appended to ``path`` after ``offset``, polling by file size."""
        while True:
            await asyncio.sleep(POLL_SECONDS)
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            if size <= offset:
                continue
            with path.open("rb") as handle:
                handle.seek(offset)
                data = handle.read(size - offset)
            end = data.rfind(b"\n") + 1
            offset += end
            for raw in data[:end].splitlines():
                try:
                    command = json.loads(raw)
                except ValueError:
                    command = None
                if isinstance(command, dict):
                    self.apply(command)
                else:
                    self.printer.note("ignored a malformed line of control.jsonl")


def attachments(paths: Iterable[str]) -> list[Attachment]:
    """Load and check each file given with ``--attach``; the first one the API would refuse is a usage error."""
    try:
        return [load_media(path) for path in paths]
    except MediaError as error:
        raise CliError(f"cannot attach {error}") from None


def should_listen(stdin: TextIO | None, task_from_stdin: bool) -> bool:
    """Typed interrupts are read only from a terminal, and never when the task already consumed stdin."""
    return not task_from_stdin and stdin is not None and stdin.isatty()


def pump_stdin(loop: asyncio.AbstractEventLoop, control: Control) -> None:
    """Hand each typed line to the event loop. Runs in a daemon thread: asyncio cannot read a console.

    It reads the raw stream, not the buffered one, so a thread still blocked at interpreter exit holds no
    lock that finalization would wait for.
    """
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    raw = sys.stdin.buffer.raw
    while True:
        try:
            chunk = raw.read(4096)
        except (OSError, ValueError):
            return
        if not chunk:
            return
        pending += decoder.decode(chunk)
        *lines, pending = pending.split("\n")
        for line in lines:
            try:
                loop.call_soon_threadsafe(control.line, line.rstrip("\r"))
            except RuntimeError:
                return


def install_sigint(loop: asyncio.AbstractEventLoop, control: Control) -> Callable[[], None]:
    """Ctrl+C once is a clean stop; a second one leaves at once. Only the main thread can own the signal."""
    if threading.current_thread() is not threading.main_thread():
        return lambda: None
    presses = 0

    def handler(signum: int, frame: Any) -> None:
        nonlocal presses
        presses += 1
        if presses > 1:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(130)
        loop.call_soon_threadsafe(control.apply, {"cmd": "stop"})

    previous = signal.signal(signal.SIGINT, handler)
    return lambda: signal.signal(signal.SIGINT, previous if previous is not None else signal.SIG_DFL)


async def drive(
    loaded: LoadedPolicy,
    cwd: Path,
    factory: ProviderFactory | None,
    printer: Printer,
    *,
    task: str = "",
    role: str | None = None,
    run_id: str | None = None,
    text: str | None = None,
    listen: bool = False,
    files: Sequence[Attachment] = (),
) -> int:
    runtime = Runtime(loaded, cwd, factory() if factory else None, run_id=run_id)
    runtime.on_output(printer)
    printer.hide_warnings(runtime.warnings)
    for warning in runtime.warnings:
        eprint(f"warning: {warning}")
    control = Control(runtime, printer)
    loop = asyncio.get_running_loop()
    path = runtime.run_dir / CONTROL_FILE
    offset = path.stat().st_size if path.exists() else 0
    job = asyncio.ensure_future(runtime.resume(text, files) if run_id else runtime.run(task, role, files))
    watcher = loop.create_task(control.watch(path, offset))
    restore = install_sigint(loop, control)
    if listen:
        printer.note("type /stop, /redirect <text>, /answer <id> <text>, /questions, or any line to message the agent")
        threading.Thread(target=pump_stdin, args=(loop, control), daemon=True, name="synthic-stdin").start()
    try:
        result = await job
    except Exception as error:
        printer.end_line()
        eprint(f"error: {type(error).__name__}: {error}")
        return 1
    finally:
        watcher.cancel()
        with suppress(asyncio.CancelledError):
            await watcher
        restore()
        await runtime.close()
    if run_id and runtime.root_agent is None:
        printer.note("the run had already ended: this is its recorded result; give a text to continue it")
    code = EXIT_CODES.get(result.status, 1)
    printer.finish(result, runtime, code)
    return code


def read_task(args: argparse.Namespace) -> tuple[str, bool]:
    from_stdin = False
    if args.task_file:
        if args.task:
            raise CliError("give the task as an argument or with --task-file, not both")
        try:
            text = Path(args.task_file).read_text(encoding="utf-8")
        except OSError as error:
            raise CliError(f"cannot read the task file: {error}") from None
    elif args.task == "-":
        text, from_stdin = sys.stdin.read(), True
    elif args.task:
        text = args.task
    else:
        raise CliError('give a task: synthic run "<task>", --task-file FILE, or - to read it from stdin')
    if not text.strip():
        raise CliError("the task is empty")
    return text, from_stdin


def require_key(factory: ProviderFactory | None) -> None:
    if factory is None and not find_api_key():
        raise CliError(KEY_MISSING)


def printer_for(args: argparse.Namespace) -> Printer:
    return Printer("json" if args.json else "quiet" if args.quiet else "text")


def result_schema_override(args: argparse.Namespace, role: str) -> None:
    """``--result-schema FILE``: the JSON Schema becomes ``role.<role>.result_schema`` at the task level."""
    try:
        schema = json.loads(Path(args.result_schema).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CliError(f"cannot read the result schema: {error}") from None
    args.schema_override = {"role": {role: {"result_schema": schema}}}


def cmd_run(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    cwd, loaded = load(args)
    task, from_stdin = read_task(args)
    role = args.role or loaded.policy.initial_role
    if role not in loaded.policy.roles:
        raise CliError(f"role '{role}' is not defined")
    if args.result_schema:
        result_schema_override(args, role)
        cwd, loaded = load(args)
    files = attachments(args.attach)
    require_key(factory)
    listen = should_listen(sys.stdin, from_stdin)
    return asyncio.run(
        drive(loaded, cwd, factory, printer_for(args), task=task, role=args.role, listen=listen, files=files)
    )


NO_CONSOLE = (
    "the interface needs a terminal and standard input or output is not one (a pipe, or a Git Bash window "
    "without a Windows console). In Git Bash start it as `winpty synthic`, or use PowerShell, cmd or Windows "
    "Terminal. To run a task from a script, use `synthic run \"task\"`."
)


def has_terminal() -> bool:
    """Whether standard input and output are a real terminal. On Windows ``isatty`` is also true for NUL, so the
    console mode of both handles is asked as well: the interface reads keys through that console."""
    streams = (sys.stdin, sys.stdout)
    if not all(stream is not None and stream.isatty() for stream in streams):
        return False
    if os.name != "nt":
        return True
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetStdHandle.argtypes = [wintypes.DWORD]
    kernel.GetStdHandle.restype = wintypes.HANDLE
    kernel.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetConsoleMode.restype = wintypes.BOOL
    mode = wintypes.DWORD()
    return all(kernel.GetConsoleMode(kernel.GetStdHandle(handle), ctypes.byref(mode)) for handle in (-10, -11))


def cmd_tui(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    """Open the terminal interface. It is a separate package, so the engine carries no interface code.

    Without a real terminal the interface would draw into a pipe and ignore every key, then never end: it
    refuses to start instead.
    """
    if not has_terminal():
        raise CliError(NO_CONSOLE)
    try:
        from synthic_tui.app import run
    except ImportError:
        raise CliError("the terminal interface is not installed: install the synthic-tui package") from None
    cwd, _ = load(args)
    require_key(factory)
    return run(cwd, lambda extra=None: load(args, extra, TUI_DEFAULTS)[1], factory, " ".join(args.task))


def cmd_resume(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_id, words = split_run(loaded, args.words, 0)
    run_dir = run_folder(loaded, run_id)
    start = next((r for r in read_records(run_dir, "root") if r.kind == "start"), None)
    if start is None:
        raise CliError(f"run {run_dir.name} has no root journal to resume")
    if is_alive(run_dir):
        raise CliError(f"run {run_dir.name} is running in another process: stop it first, or wait for it to end")
    cwd = Path(args.cwd or start.data.get("cwd") or Path.cwd()).resolve()
    text = " ".join(words) or None
    if args.attach and not text:
        raise CliError("--attach goes with a message: synthic resume [run-id] \"<text>\" --attach FILE")
    files = attachments(args.attach)
    require_key(factory)
    listen = should_listen(sys.stdin, False)
    return asyncio.run(
        drive(loaded, cwd, factory, printer_for(args), run_id=run_dir.name, text=text, listen=listen, files=files)
    )


@dataclass
class RunInfo:
    run_id: str
    started: float
    status: str
    steps: int
    cost: float
    task: str


def summarize(run_dir: Path) -> RunInfo:
    steps, cost = 0, 0.0
    for agent_id in agents_in(run_dir):
        for record in read_records(run_dir, agent_id):
            if record.kind == "model_call":
                cost += float(record.data.get("cost_usd", 0.0))
                steps += "purpose" not in record.data
    start = next((r for r in read_records(run_dir, "root") if r.kind == "start"), None)
    end = root_end(run_dir)
    return RunInfo(
        run_dir.name,
        start.ts if start else run_dir.stat().st_mtime,
        "running" if is_alive(run_dir) else end.data["status"] if end else "crashed",
        steps,
        cost,
        short(str(start.data.get("task", "")), 400) if start else "",
    )


def clock(ts: float, pattern: str = "%Y-%m-%d %H:%M") -> str:
    return time.strftime(pattern, time.localtime(ts))


def cmd_runs(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    infos = [summarize(path) for path in list_runs(loaded.policy.journal)[: args.limit]]
    if not infos:
        print("No runs.")
        return 0
    width = min(shutil.get_terminal_size((110, 24)).columns, 128)
    id_width = max(len(i.run_id) for i in infos)
    rows = [("RUN".ljust(id_width), "WHEN".ljust(16), "STATUS".ljust(10), "STEPS".rjust(5), "COST".rjust(8), "TASK")]
    rows += [
        (i.run_id.ljust(id_width), clock(i.started), i.status.ljust(10), str(i.steps).rjust(5), f"${i.cost:.4f}".rjust(8), i.task)
        for i in infos
    ]
    room = max(20, width - (id_width + 16 + 10 + 5 + 8 + 10))
    for row in rows:
        print("  ".join(row[:-1]) + "  " + short(row[-1], room))
    return 0


def text_length(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    return sum(len(block.get("text", "")) for block in content if isinstance(block, dict))


def describe(record: Record, results: dict[tuple[str, Any], Record], full: bool) -> str | None:
    """One trace line for a journal record, or ``None`` for the kinds that do not matter to a reader."""
    data = record.data

    def fit(text: Any) -> str:
        return str(text).strip() if full else short(str(text))

    match record.kind:
        case "start":
            return f"start {data.get('role')} in {data.get('cwd')}: {fit(data.get('task', ''))}"
        case "user":
            files = [str(a.get("name", "")) for a in data.get("attachments", [])]
            shown = f" (attached: {', '.join(files)})" if files else ""
            return f"user [{data.get('source')}]: {fit(data.get('text', ''))}{shown}"
        case "assistant":
            content = data.get("content", [])
            says = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
            lines = [f"says: {fit(says)}"] if says.strip() else []
            lines += [
                f"server tool {b.get('name')} {fit(json.dumps(b.get('input', {}), ensure_ascii=False))}"
                for b in content
                if b.get("type") == "server_tool_use"
            ]
            return "\n".join(lines) or None
        case "tool_intent":
            name = str(data.get("name"))
            if name in ("done", "ask"):
                return None
            line = f"tool {tool_line(name, data.get('input', {}), None if full else 160)}"
            result = results.get((record.agent_id, data.get("call_id")))
            if result is None:
                return line + "  -> no result"
            size = text_length(result.data.get("content", ""))
            state = "error" if result.data.get("is_error") else "ok"
            line += f"  -> {state}, {size} chars, {result.data.get('seconds', 0)}s"
            if full:
                content = result.data.get("content", "")
                body = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                line += "\n" + textwrap.indent(body, "    ")
            return line
        case "question":
            return f"question {data.get('question_id')}: {fit(data.get('text', ''))} (assumes: {fit(data.get('assumption', ''))})"
        case "answer":
            return f"answer {data.get('question_id')}: {fit(data.get('text', ''))}"
        case "spawn":
            return f"spawn {data.get('child_id')} ({data.get('role')}): {fit(data.get('task', ''))}"
        case "child_done":
            return f"child {data.get('child_id')} {data.get('status')}: {fit(data.get('summary', ''))}"
        case "interrupt":
            return f"interrupt: {data.get('mode')}"
        case "fallback":
            return f"model fallback {data.get('from')} -> {data.get('to')}: {fit(data.get('reason', ''))}"
        case "compaction":
            return f"compaction: {fit(data.get('summary', ''))}"
        case "hook":
            return f"hook {data.get('event')} {data.get('action')}: {fit(data.get('reason', ''))}"
        case "checkpoint":
            return describe_checkpoint(record)
        case "todo":
            items = data.get("items", [])
            return f"todo {tally_todo(items)}" + ("\n" + render_todo(items) if full else "")
        case "verify":
            verdict = "passed" if data.get("passed") else "accepted, not verified" if data.get("accepted") else "rejected"
            by = f" by {data['reviewer']}" if data.get("reviewer") else ""
            report = f": {fit(data['report'])}" if data.get("report") else ""
            return f"verify round {data.get('round')}{by} {verdict}{report}"
        case "done":
            return f"done {data.get('status')}: {fit(data.get('summary', ''))}"
    return None


def open_questions(by_agent: dict[str, list[Record]]) -> list[Record]:
    records = [r for recs in by_agent.values() for r in recs]
    answered = {r.data.get("question_id") for r in records if r.kind == "answer"}
    asked = [r for r in records if r.kind == "question" and r.data.get("question_id") not in answered]
    return sorted(asked, key=lambda r: r.ts)


def cmd_show(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_dir = run_folder(loaded, args.run_id)
    agents = sorted(agents_in(run_dir), key=agent_order)
    if args.agent and args.agent not in agents:
        raise CliError(f"run {run_dir.name} has no agent {args.agent!r}; it has: {', '.join(agents) or 'none'}")
    by_agent = {agent_id: list(read_records(run_dir, agent_id)) for agent_id in agents}
    results = {
        (r.agent_id, r.data.get("call_id")): r for recs in by_agent.values() for r in recs if r.kind == "tool_result"
    }
    chosen = [args.agent] if args.agent else agents
    if args.md:
        sys.stdout.write(markdown(run_dir.name, {agent_id: by_agent[agent_id] for agent_id in chosen}, args.full))
        return 0
    events = sorted(
        (r for agent_id in chosen for r in by_agent[agent_id]),
        key=lambda r: (r.ts, agent_order(r.agent_id), r.seq),
    )
    label = max((len(a) for a in chosen), default=4)
    print(f"run {run_dir.name}")
    for record in events:
        body = describe(record, results, args.full)
        if body is None:
            continue
        prefix = f"{clock(record.ts, '%H:%M:%S')}  {record.agent_id.ljust(label)}  "
        first, *rest = body.split("\n")
        print(prefix + first)
        for extra in rest:
            print(" " * len(prefix) + extra)
    print("\ncost:")
    grand = Usage()
    for agent_id in chosen:
        usage, steps = Usage(), 0
        for record in by_agent[agent_id]:
            if record.kind == "model_call":
                usage.add(Usage(**record.data["usage"]))
                steps += "purpose" not in record.data
        grand.add(usage)
        print(f"  {agent_id.ljust(label)}  ${usage.cost_usd:.4f}  {steps} steps  {usage.input_tokens} in, {usage.output_tokens} out")
    if len(chosen) > 1:
        print(f"  {'total'.ljust(label)}  ${grand.cost_usd:.4f}")
    for agent_id in chosen:
        lists = [r for r in by_agent[agent_id] if r.kind == "todo"]
        if lists:
            print(f"\ntask list of {agent_id}:")
            print(textwrap.indent(render_todo(lists[-1].data.get("items", [])), "  "))
    pending = open_questions(by_agent)
    print("\nopen questions:" + ("" if pending else " none"))
    for record in pending:
        print(f"  {record.data.get('question_id')}: {short(str(record.data.get('text', '')))}")
    return 0


def cmd_policy(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    print(loaded.explain())
    if loaded.warnings:
        print("\nwarnings:")
        for warning in loaded.warnings:
            print(f"  {warning}")
    return 0


def cmd_questions(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_dir = run_folder(loaded, args.run_id)
    by_agent = {agent_id: list(read_records(run_dir, agent_id)) for agent_id in agents_in(run_dir)}
    pending = open_questions(by_agent)
    if not pending:
        print("No open questions.")
    for record in pending:
        print(f"{record.data.get('question_id')}  ({record.agent_id}): {short(str(record.data.get('text', '')), 300)}")
        print(f"    assumes: {short(str(record.data.get('assumption', '')), 300)}")
    return 0


def cmd_prune(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    removed = prune(loaded.policy.journal)
    noun = "run" if removed == 1 else "runs"
    note = "" if loaded.policy.journal.retention_days is not None else " (journal.retention_days is not set)"
    print(f"Removed {removed} {noun}{note}.")
    return 0


def cmd_rewind(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_dir = run_folder(loaded, args.run_id)
    if is_alive(run_dir):
        raise CliError(f"run {run_dir.name} is running: stop it before rewinding its files")
    if args.agent and args.agent not in agents_in(run_dir):
        raise CliError(f"run {run_dir.name} has no agent {args.agent!r}")
    if args.to < 0:
        raise CliError("--to must be 0 or more")
    try:
        lines = rewind(run_dir, args.to, args.agent, apply=not args.dry_run)
    except (OSError, RuntimeError, KeyError) as error:
        raise CliError(f"rewind stopped: {error}", 1) from None
    if not lines:
        print(f"Nothing to rewind after step {args.to} (checkpoints are recorded only when the policy turns them on).")
        return 0
    print(("Would do:" if args.dry_run else "Done:") + "\n" + "\n".join(f"  {line}" for line in lines))
    if not args.dry_run:
        print(f'A resumed agent does not know: say it, as in synthic resume {run_dir.name} "I rewound the files".')
    return 0


def send_control(loaded: LoadedPolicy, run_id: str | None, command: dict[str, Any]) -> int:
    run_dir = run_folder(loaded, run_id)
    if not is_alive(run_dir):
        hint = f' Use: synthic resume {run_dir.name} "<text>"' if command["cmd"] == "say" else ""
        raise CliError(f"run {run_dir.name} is not running, so nothing would read the command.{hint}")
    line = json.dumps({**command, "ts": time.time()}, ensure_ascii=False) + "\n"
    with (run_dir / CONTROL_FILE).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
    print(f"Sent {command['cmd']} to run {run_dir.name}.")
    return 0


def split_run(loaded: LoadedPolicy, parts: list[str], needed: int) -> tuple[str | None, list[str]]:
    """Separate an optional leading run id from the words that follow.

    The first word is a run id when such a folder exists. A word shaped like a generated run id that names no
    folder is a typo, not text, and is an error: text must never be sent to the wrong run by accident.
    """
    base = Path(loaded.policy.journal.dir).expanduser()
    first = parts[0] if parts else ""
    if RUN_ID_SHAPE.fullmatch(first) and not (base / first).is_dir():
        raise CliError(f"no run {first!r} in {base}")
    if first and len(parts) > needed and Path(first).name == first and (base / first).is_dir():
        return first, parts[1:]
    return None, parts


def cmd_stop(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    return send_control(loaded, args.run_id, {"cmd": "stop"})


def cmd_say(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_id, words = split_run(loaded, args.words, 1)
    agent = {"agent": args.agent} if args.agent else {}
    files = [a.name for a in attachments(getattr(args, "attach", []))]
    extra: dict[str, Any] = {"attach": files} if files else {}
    return send_control(loaded, run_id, {"cmd": args.command, "text": " ".join(words), **agent, **extra})


def cmd_answer(args: argparse.Namespace, factory: ProviderFactory | None) -> int:
    _, loaded = load(args)
    run_id, words = split_run(loaded, args.words, 2)
    if len(words) < 2:
        raise CliError("usage: synthic answer [<run-id>] <question-id> <text>")
    return send_control(loaded, run_id, {"cmd": "answer", "question": words[0], "text": " ".join(words[1:])})


def build_parser() -> argparse.ArgumentParser:
    policy = argparse.ArgumentParser(add_help=False)
    policy.add_argument("--cwd", metavar="DIR", help="the project folder (default: the current folder)")
    policy.add_argument("--policy", metavar="FILE", help="a task-level policy file (TOML)")
    policy.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="a task-level override")
    policy.add_argument("--trust-project", action="store_true", help="trust the repo's .synthic/policy.toml")
    output = argparse.ArgumentParser(add_help=False)
    mode = output.add_mutually_exclusive_group()
    mode.add_argument("--quiet", action="store_true", help="print only the final summary")
    mode.add_argument("--json", action="store_true", help="print one JSON object per event")
    attach = argparse.ArgumentParser(add_help=False)
    attach.add_argument(
        "--attach", action="append", default=[], metavar="FILE", help="send an image or a PDF with the message"
    )

    parser = argparse.ArgumentParser(prog="synthic", description="An agent harness: the engine and nothing else.")
    parser.add_argument("--version", action="version", version=f"synthic {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, handler: Callable[..., int], help_text: str, *parents: argparse.ArgumentParser) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, parents=[policy, *parents], help=help_text, description=help_text)
        sub.set_defaults(handler=handler)
        return sub

    run = add("run", cmd_run, "run a task to the end without prompting", output, attach)
    run.add_argument("task", nargs="?", help='the task, or "-" to read it from stdin')
    run.add_argument("--task-file", metavar="FILE", help="read the task from a file")
    run.add_argument("--role", help="the role of the root agent (default: initial_role)")
    run.add_argument(
        "--result-schema", metavar="FILE", help="a JSON Schema file: the delivery must carry a result matching it"
    )
    tui = add("tui", cmd_tui, "open the terminal interface: start, follow and steer runs on one screen")
    tui.add_argument("task", nargs="*", help="a task to start with (optional)")
    resume = add(
        "resume", cmd_resume, "continue a crashed or stopped run, or an ended one when given a text", output, attach
    )
    resume.add_argument("words", nargs="*", metavar="[RUN_ID] [TEXT]", help="run (default: the most recent) and a message")
    runs = add("runs", cmd_runs, "list the recent runs, newest first")
    runs.add_argument("--limit", type=int, default=20, help="how many runs to list (default 20)")
    show = add("show", cmd_show, "print a readable trace of a run")
    show.add_argument("run_id", nargs="?", help="default: the most recent run")
    show.add_argument("--agent", metavar="ID", help="only this agent")
    show.add_argument("--full", action="store_true", help="print texts and tool results in full")
    show.add_argument("--md", action="store_true", help="print the run as a Markdown document (redirect it to a file to keep it)")
    add("policy", cmd_policy, "print the policy in force and where each value came from")
    questions = add("questions", cmd_questions, "list the open questions of a run")
    questions.add_argument("run_id", nargs="?", help="default: the most recent run")
    add("prune", cmd_prune, "delete the runs older than journal.retention_days")
    back = add("rewind", cmd_rewind, "undo the file changes recorded by checkpoints after a step")
    back.add_argument("run_id", nargs="?", help="default: the most recent run")
    back.add_argument("--to", type=int, default=0, metavar="STEP", help="keep the changes up to this step (default 0)")
    back.add_argument("--agent", metavar="ID", help="only this agent's changes, with its own step numbers")
    back.add_argument("--dry-run", action="store_true", help="print what would be done and change nothing")
    stop = add("stop", cmd_stop, "stop a running run from another terminal")
    stop.add_argument("run_id", nargs="?", help="default: the most recent run")
    for name, help_text in (("say", "add a message for the agent of a running run"), ("redirect", "redirect a running run")):
        sub = add(name, cmd_say, help_text, *([attach] if name == "say" else []))
        sub.add_argument("words", nargs="+", metavar="[RUN_ID] TEXT")
        sub.add_argument("--agent", metavar="ID", help="address a child agent instead of the root")
    answer = add("answer", cmd_answer, "answer an open question of a running run")
    answer.add_argument("words", nargs="+", metavar="[RUN_ID] QUESTION_ID TEXT")
    return parser


def interactive(argv: Sequence[str]) -> list[str]:
    """``synthic`` alone opens the terminal interface, and ``synthic "a task"`` opens it on that task.

    A first word that is neither a command nor a sentence stays as it is, so a mistyped command is an
    error and never a task that reaches the model.
    """
    words = list(argv)
    first = words[0] if words else ""
    if first in ("-h", "--help", "--version"):
        return words
    if not first or first.startswith("-") or any(mark.isspace() for mark in first.strip()):
        return ["tui", *words]
    return words


def main(argv: Sequence[str] | None = None, *, provider_factory: ProviderFactory | None = None) -> int:
    """Run one command and return its exit code. ``provider_factory`` replaces the Anthropic provider (tests)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    try:
        args = build_parser().parse_args(interactive(sys.argv[1:] if argv is None else argv))
    except SystemExit as exit_:
        return exit_.code if isinstance(exit_.code, int) else 2
    try:
        return args.handler(args, provider_factory)
    except PolicyError as error:
        for problem in error.problems:
            eprint(f"error: {problem}")
        return 2
    except CliError as error:
        eprint(f"error: {error}")
        return error.code
