"""The command line, offline: ``ScriptedProvider`` injected through ``provider_factory``, real journals in ``tmp_path``.

Remote controls are tested the way a human uses them: the run executes in the test thread and a second thread
calls ``main`` again, exactly like a second terminal would.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from test_proc import py_command, survivors, tree_command, wait_for_pids, write_script

from synthic import cli
from synthic.cli import CliError, Control, Printer, main, parse_line, parse_overrides, parse_value, pump_stdin, should_listen
from synthic.journal import JOURNAL_FILE
from synthic.testing import ScriptedProvider, reply, text, tool_use

POLICY = """
initial_role = "lead"

[verify]
reread = false

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


@dataclass
class Setup:
    folder: Path
    runs: Path
    project: Path

    @property
    def opts(self) -> list[str]:
        return ["--policy", str(self.folder / "task.toml"), "--set", f"journal.dir={self.runs.as_posix()}", "--cwd", str(self.project)]


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    (tmp_path / "task.toml").write_text(POLICY, encoding="utf-8")
    (tmp_path / "proj").mkdir()
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SYNTHIC_ORG_POLICY", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli, "POLL_SECONDS", 0.05)
    return Setup(tmp_path, tmp_path / "runs", tmp_path / "proj")


def done(summary: str = "All done.", status: str = "delivered") -> Any:
    return reply(tool_use("done", {"summary": summary, "status": status}))


def create_file() -> Any:
    return reply(text("Creating the file."), tool_use("edit", {"path": "a.txt", "old": "", "new": "hello\n"}))


def run_cli(setup: Setup, script: list[Any], *extra: str, task: str = "make a.txt") -> tuple[int, ScriptedProvider]:
    provider = ScriptedProvider(script)
    return main(["run", task, *setup.opts, *extra], provider_factory=lambda: provider), provider


def run_id_of(output: str) -> str:
    match = re.search(r"^run: (\S+)$", output, re.MULTILINE)
    assert match, output
    return match[1]


def wait_for(condition: Callable[[], Any], timeout: float = 20.0) -> Any:
    deadline = time.monotonic() + timeout
    while not (value := condition()):
        assert time.monotonic() < deadline, "timed out waiting for the run"
        time.sleep(0.02)
    return value


def newest_run(setup: Setup) -> str:
    return wait_for(lambda: next((p.name for p in setup.runs.glob("*") if p.is_dir()), None))


def root_text(setup: Setup, run_id: str) -> str:
    path = setup.runs / run_id / "root" / JOURNAL_FILE
    return path.read_text(encoding="utf-8") if path.exists() else ""


def with_side(side: Callable[[], Any], argv: list[str], provider: ScriptedProvider) -> tuple[int, Any]:
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = side()
        except BaseException as error:
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    code = main(argv, provider_factory=lambda: provider)
    thread.join(30)
    if "error" in box:
        raise box["error"]
    return code, box.get("value")


def test_run_edits_a_file_delivers_and_is_listed_and_shown(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli(setup, [create_file(), done("Created a.txt.")])
    out = capsys.readouterr().out
    assert code == 0
    assert (setup.project / "a.txt").read_text(encoding="utf-8") == "hello\n"
    assert "Creating the file." in out
    assert "> edit" in out and "< edit ok" in out
    assert "status: delivered" in out and "summary: Created a.txt." in out
    assert "steps: 2" in out and "cost: $0.0004" in out and "tokens: 200 in, 40 out" in out
    run_id = run_id_of(out)

    assert main(["runs", *setup.opts]) == 0
    listing = capsys.readouterr().out
    row = next(line for line in listing.splitlines() if run_id in line)
    assert "delivered" in row and "make a.txt" in row and "$0.0004" in row

    assert main(["show", run_id, *setup.opts]) == 0
    trace = capsys.readouterr().out
    assert f"run {run_id}" in trace
    assert "start lead in" in trace and "user [human]: make a.txt" in trace
    assert re.search(r"tool edit: a\.txt \(new file, 1 lines\)  -> ok, \d+ chars", trace)
    assert main(["show", run_id, "--md", *setup.opts]) == 0
    page = capsys.readouterr().out
    assert page.startswith(f"# Run {run_id}") and "### You (" in page and "make a.txt" in page
    assert "- **edit: a.txt (new file, 1 lines)** (ok" in page and "**Ended delivered.** Created a.txt." in page
    assert "- root: 2 steps, $0.0004" in page
    assert "done delivered: Created a.txt." in trace
    assert "cost:" in trace and "root" in trace and "open questions: none" in trace


def test_show_without_run_id_agent_filter_and_full(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    long = "word " * 80
    run_cli(setup, [reply(tool_use("sh", {"command": "echo marker-output"})), done(long)])
    capsys.readouterr()
    assert main(["show", *setup.opts]) == 0
    short_trace = capsys.readouterr().out
    assert "tool sh: echo marker-output  -> ok" in short_trace and "marker-output\n" not in short_trace.split("tool sh")[1]
    assert "..." in short_trace
    assert main(["show", "--full", "--agent", "root", *setup.opts]) == 0
    full = capsys.readouterr().out
    assert long.strip() in full and "    marker-output" in full
    assert main(["show", "--agent", "zzz", *setup.opts]) == 2
    assert "no agent 'zzz'" in capsys.readouterr().err


def test_exit_codes(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(setup, [done("Could not.", "failed")])[0] == 1
    assert "status: failed" in capsys.readouterr().out
    assert run_cli(setup, [reply(tool_use("sh", {"command": "echo hi"}))], "--set", "budget.steps=1")[0] == 3
    assert "status: budget" in capsys.readouterr().out
    assert run_cli(setup, [done()], "--set", "role.lead.model=ghost")[0] == 2
    assert "ghost" in capsys.readouterr().err
    assert run_cli(setup, [done()], "--role", "nobody")[0] == 2
    assert "role 'nobody' is not defined" in capsys.readouterr().err


def test_every_policy_problem_is_printed_one_per_line(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(setup, [done()], "--set", "nonsense=1", "--set", "budget.bogus=2")[0] == 2
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("error:")]
    assert len(lines) == 2 and "nonsense" in lines[0] and "budget.bogus" in lines[1]


def test_usage_errors_exit_2(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", *setup.opts]) == 2
    assert "give a task" in capsys.readouterr().err
    assert main(["run", "x", "--task-file", "y.txt", *setup.opts]) == 2
    assert main(["nonsense"]) == 2
    assert main([]) == 2
    assert main(["run", "x", "--set", "novalue", *setup.opts]) == 2
    assert main(["--help"]) == 0


def test_set_parsing_nested_numbers_booleans_arrays_and_strings() -> None:
    assert parse_overrides(
        [
            "role.build.model=sonnet",
            "role.build.tools=[\"sh\", \"edit\"]",
            "budget.steps=40",
            "budget.dollars=2.5",
            "visible_to_agent=false",
            "journal.dir=C:/Users/x/runs",
            "plugins.path=[]",
            "instructions.files=\"a b.md\"",
        ]
    ) == {
        "role": {"build": {"model": "sonnet", "tools": ["sh", "edit"]}},
        "budget": {"steps": 40, "dollars": 2.5},
        "visible_to_agent": False,
        "journal": {"dir": "C:/Users/x/runs"},
        "plugins": {"path": []},
        "instructions": {"files": "a b.md"},
    }
    assert parse_value("5m") == "5m" and parse_value("2026-01-01") == "2026-01-01" and parse_value("true") is True
    with pytest.raises(CliError):
        parse_overrides(["budget=1", "budget.steps=2"])
    with pytest.raises(CliError):
        parse_overrides(["budget.steps=2", "budget=1"])
    with pytest.raises(CliError):
        parse_overrides(["=1"])


def test_set_reaches_the_policy(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["policy", *setup.opts, "--set", "role.lead.max_steps=7", "--set", "agents.concurrent=2"]) == 0
    out = capsys.readouterr().out
    assert "role.lead.max_steps = 7  (task" in out and "agents.concurrent = 2  (task" in out


def test_json_mode_prints_one_object_per_line(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli(setup, [create_file(), done("Created a.txt.")], "--json")
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line) for line in lines]
    assert code == 0 and all(isinstance(e, dict) for e in events)
    kinds = [e["kind"] for e in events]
    assert {"agent_start", "text", "tool_start", "tool_end", "agent_end", "result"} <= set(kinds)
    final = events[-1]
    assert final["kind"] == "result" and final["status"] == "delivered" and final["exit_code"] == 0
    assert final["summary"] == "Created a.txt." and final["tokens"]["input_tokens"] == 200 and final["run_id"]
    assert events[0] == {"agent": "root", "kind": "agent_start", "text": events[0]["text"]}


def test_quiet_mode_prints_only_the_summary(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run_cli(setup, [create_file(), done("Created a.txt.")], "--quiet")
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("status: delivered\nsummary: Created a.txt.\n")
    assert "> edit" not in out and "Creating the file." not in out and "run: " in out


def test_output_has_no_colour_when_stdout_is_not_a_terminal(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [create_file(), done()])
    assert "\x1b" not in capsys.readouterr().out


def test_printer_prefixes_agents_only_while_several_are_alive(capsys: pytest.CaptureFixture[str]) -> None:
    printer = Printer()
    printer("root", "agent_start", "lead: task")
    printer("root", "text", "Hello ")
    printer("root", "text", "world\nsecond")
    printer("root", "tool_start", "sh {}")
    printer("a1", "agent_start", "worker: child task")
    printer("a1", "text", "child talking")
    printer("root", "text", "parent talking\n")
    printer("a1", "agent_end", "delivered: ok")
    printer("root", "tool_end", "sh ok in 0.1s")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "* started lead: task",
        "Hello world",
        "second",
        "> sh {}",
        "[a1] * started worker: child task",
        "[a1] child talking",
        "[root] parent talking",
        "[a1] * ended delivered: ok",
        "< sh ok in 0.1s",
    ]


def test_printer_hides_the_warnings_already_printed_at_the_start(capsys: pytest.CaptureFixture[str]) -> None:
    printer = Printer()
    printer.hide_warnings(["policy says hello"])
    printer("root", "warning", "policy says hello")
    printer("root", "warning", "policy says hello")
    printer("root", "warning", "budget is 80% used")
    assert capsys.readouterr().out == "! policy says hello\n! budget is 80% used\n"


def test_policy_command_names_the_levels_and_the_warnings(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    (setup.project / ".synthic").mkdir()
    (setup.project / ".synthic" / "policy.toml").write_text('[budget]\nsteps = 9\n[permissions]\n"sh: rm*" = "deny"\n', encoding="utf-8")
    assert main(["policy", *setup.opts]) == 0
    out = capsys.readouterr().out
    assert "initial_role = 'lead'  (task" in out and "(defaults" in out and "journal.dir" in out
    assert "not trusted" in out and "warnings:" in out
    assert "permission 'sh: rm*' = deny  (project)" in out
    assert main(["policy", *setup.opts, "--trust-project"]) == 0
    assert "budget.steps = 9  (project)" in capsys.readouterr().out


def test_user_level_comes_from_synthic_home(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    (setup.folder / "home" / ".synthic").mkdir(parents=True)
    (setup.folder / "home" / ".synthic" / "policy.toml").write_text("[agents]\ndepth = 1\n", encoding="utf-8")
    assert main(["policy", *setup.opts]) == 0
    assert "agents.depth = 1  (user" in capsys.readouterr().out


def test_policy_error_prints_every_problem_and_exits_2(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["policy", *setup.opts, "--set", "role.lead.model=ghost", "--set", "zzz=1"]) == 2
    err = capsys.readouterr().err.splitlines()
    assert len(err) == 2 and all(line.startswith("error: ") for line in err)


def test_missing_api_key_is_one_clear_line_before_anything_starts(
    setup: Setup, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main(["run", "task", *setup.opts]) == 2
    err = capsys.readouterr().err.strip().splitlines()
    assert len(err) == 1 and "ANTHROPIC_API_KEY" in err[0]
    assert not setup.runs.exists()
    assert main(["resume", *setup.opts]) == 2


def test_task_from_a_file_and_from_stdin(setup: Setup, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    (setup.folder / "task.md").write_text("task from a file", encoding="utf-8")
    provider = ScriptedProvider([done(), done()])
    assert main(["run", "--task-file", str(setup.folder / "task.md"), *setup.opts], provider_factory=lambda: provider) == 0
    monkeypatch.setattr("sys.stdin", io.StringIO("task from stdin"))
    assert main(["run", "-", *setup.opts], provider_factory=lambda: provider) == 0
    first, second = (str(r.messages[0]) for r in provider.requests)
    assert "task from a file" in first and "task from stdin" in second
    monkeypatch.setattr("sys.stdin", io.StringIO("  \n"))
    assert main(["run", "-", *setup.opts], provider_factory=lambda: provider) == 2
    assert "the task is empty" in capsys.readouterr().err


class TrapStdin:
    """Fails the test if anything reads it, and is not a terminal."""

    def isatty(self) -> bool:
        return False

    def read(self, *args: Any) -> str:
        raise AssertionError("stdin was read")

    readline = read

    @property
    def buffer(self) -> Any:
        raise AssertionError("stdin was read")


def test_stdin_is_not_read_when_it_is_not_a_terminal(setup: Setup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", TrapStdin())
    started: list[Any] = []
    monkeypatch.setattr(cli.threading, "Thread", lambda *a, **k: started.append(k) or pytest.fail("thread started"))
    assert run_cli(setup, [done()])[0] == 0
    assert started == []


def test_should_listen_only_on_a_terminal_that_the_task_did_not_consume() -> None:
    class Terminal(TrapStdin):
        def isatty(self) -> bool:
            return True

    assert should_listen(Terminal(), False) is True
    assert should_listen(Terminal(), True) is False
    assert should_listen(TrapStdin(), False) is False
    assert should_listen(None, False) is False


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("", None),
        ("   \n", None),
        ("/stop\n", {"cmd": "stop"}),
        ("/STOP now please", {"cmd": "stop"}),
        ("/questions", {"cmd": "questions"}),
        ("/redirect try the other file", {"cmd": "redirect", "text": "try the other file"}),
        ("/add  also run the tests \n", {"cmd": "say", "text": "also run the tests"}),
        ("/answer root-q1 yes, use B", {"cmd": "answer", "question": "root-q1", "text": "yes, use B"}),
        ("use the second option\n", {"cmd": "say", "text": "use the second option"}),
        ("/unknown thing", {"cmd": "say", "text": "/unknown thing"}),
        ("/add @a1 check the tests", {"cmd": "say", "text": "check the tests", "agent": "a1"}),
        ("/redirect @a2 stop", {"cmd": "redirect", "text": "stop", "agent": "a2"}),
        ("@a1 hello there", {"cmd": "say", "text": "hello there", "agent": "a1"}),
        ("@a1", {"cmd": "error", "text": "usage: @agent <text>"}),
        ("/redirect", {"cmd": "error", "text": "usage: /redirect [@agent] <text>"}),
        ("/add", {"cmd": "error", "text": "usage: /add [@agent] <text>"}),
        ("/answer root-q1", {"cmd": "error", "text": "usage: /answer <question-id> <text>"}),
    ],
)
def test_parse_line(line: str, expected: dict[str, str] | None) -> None:
    assert parse_line(line) == expected


def test_pump_stdin_hands_whole_lines_to_the_loop_even_when_a_character_is_split(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks = [b"hello\r\n/sto", b"p\ncaf\xc3", b"\xa9 ready\n", b""]

    class Raw:
        def read(self, size: int) -> bytes:
            return chunks.pop(0)

    class Fake:
        buffer = type("Buffer", (), {"raw": Raw()})()

    monkeypatch.setattr("sys.stdin", Fake())
    seen: list[str] = []

    class Recorder:
        line = staticmethod(seen.append)

    async def scenario() -> None:
        await asyncio.to_thread(pump_stdin, asyncio.get_running_loop(), Recorder())
        await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert seen == ["hello", "/stop", "café ready"]


def test_remote_stop_ends_the_run_as_stopped_and_leaves_no_process(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([reply(tool_use("sh", {"command": tree_command(setup.project, "stay")}))])

    def side() -> list[int]:
        pids = wait_for_pids(setup.project)
        run_id = newest_run(setup)
        assert main(["stop", run_id, *setup.opts]) == 0
        return pids

    code, pids = with_side(side, ["run", "start the tree", *setup.opts], provider)
    out = capsys.readouterr().out
    assert code == 130 and "status: stopped" in out and "= stop requested" in out
    assert survivors(pids) == []


def test_stop_without_a_run_id_means_the_most_recent_run(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([reply(tool_use("sh", {"command": tree_command(setup.project, "stay")}))])

    def side() -> list[int]:
        pids = wait_for_pids(setup.project)
        newest_run(setup)
        assert main(["stop", *setup.opts]) == 0
        return pids

    code, pids = with_side(side, ["run", "start the tree", *setup.opts], provider)
    assert code == 130 and survivors(pids) == []
    assert main(["stop", *setup.opts]) == 2
    assert "is not running" in capsys.readouterr().err


def test_say_reaches_the_next_model_request(setup: Setup) -> None:
    marker = setup.folder / "started"
    sleeper = write_script(
        setup.folder, "sleeper.py", "import sys, time, pathlib\npathlib.Path(sys.argv[1]).write_text('1')\ntime.sleep(2)\n"
    )
    provider = ScriptedProvider([reply(tool_use("sh", {"command": py_command(sleeper, marker)})), done()])

    def side() -> None:
        wait_for(marker.exists)
        run_id = newest_run(setup)
        assert main(["say", run_id, "also mention the weather", *setup.opts]) == 0
        assert main(["say", "and the traffic", *setup.opts]) == 0

    code, _ = with_side(side, ["run", "work", *setup.opts], provider)
    sent = str(provider.requests[1].messages[-1])
    assert code == 0 and "also mention the weather" in sent and "and the traffic" in sent


def test_redirect_aborts_the_command_and_continues_with_the_text(setup: Setup) -> None:
    marker = setup.folder / "started"
    sleeper = write_script(
        setup.folder, "sleeper.py", "import sys, time, pathlib\npathlib.Path(sys.argv[1]).write_text('1')\ntime.sleep(30)\n"
    )
    provider = ScriptedProvider([reply(tool_use("sh", {"command": py_command(sleeper, marker)})), done("Redirected.")])

    def side() -> None:
        wait_for(marker.exists)
        assert main(["redirect", newest_run(setup), "stop that", "and", "do this instead", *setup.opts]) == 0

    started = time.monotonic()
    code, _ = with_side(side, ["run", "work", *setup.opts], provider)
    assert code == 0 and time.monotonic() - started < 25
    assert "stop that and do this instead" in str(provider.requests[1].messages[-1])


def test_remote_answer_resolves_a_waiting_question(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([reply(tool_use("ask", {"question": "Which file?", "assumption": "a.txt"})), done("Used B.")])

    def side() -> str:
        run_id = newest_run(setup)
        wait_for(lambda: '"kind": "question"' in root_text(setup, run_id))
        assert main(["questions", run_id, *setup.opts]) == 0
        assert main(["answer", run_id, "root-q1", "use", "file", "B", *setup.opts]) == 0
        return run_id

    code, run_id = with_side(side, ["run", "work", "--set", "human.wait_seconds=30", *setup.opts], provider)
    out = capsys.readouterr().out
    assert code == 0 and "? question root-q1: Which file?" in out and "= answered root-q1" in out
    assert "root-q1  (root): Which file?" in out
    assert "The human answered: use file B" in str(provider.requests[1].messages[-1])
    assert main(["questions", run_id, *setup.opts]) == 0
    assert "No open questions." in capsys.readouterr().out


def test_questions_lists_the_open_ones_from_the_journal(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [reply(tool_use("ask", {"question": "Which file?", "assumption": "a.txt"})), done()])
    run_id = run_id_of(capsys.readouterr().out)
    assert main(["questions", *setup.opts]) == 0
    out = capsys.readouterr().out
    assert "root-q1  (root): Which file?" in out and "assumes: a.txt" in out
    assert main(["show", run_id, *setup.opts]) == 0
    assert "open questions:\n  root-q1: Which file?" in capsys.readouterr().out
    assert main(["questions", "no-such-run", *setup.opts]) == 2


def test_control_commands_check_the_run(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["stop", *setup.opts]) == 2
    assert "no runs" in capsys.readouterr().err
    run_cli(setup, [done()])
    run_id = run_id_of(capsys.readouterr().out)
    assert main(["say", run_id, "late", *setup.opts]) == 2
    assert f'Use: synthic resume {run_id} "<text>"' in capsys.readouterr().err
    assert main(["redirect", run_id, "late", *setup.opts]) == 2
    assert "Use: synthic resume" not in capsys.readouterr().err
    assert not (setup.runs / run_id / "control.jsonl").exists()
    assert main(["answer", "root-q1", *setup.opts]) == 2
    assert main(["stop", "../x", *setup.opts]) == 2


def test_resume_after_a_crash_continues_the_run(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [create_file(), reply(tool_use("sh", {"command": "echo hi"})), done()])
    run_id = run_id_of(capsys.readouterr().out)
    journal = setup.runs / run_id / "root" / JOURNAL_FILE
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    intents = [i for i, line in enumerate(lines) if '"kind": "tool_intent"' in line]
    journal.write_text("".join(lines[: intents[1] + 1]), encoding="utf-8")

    provider = ScriptedProvider([done("Recovered.")])
    assert main(["resume", *setup.opts[:4]], provider_factory=lambda: provider) == 0
    out = capsys.readouterr().out
    assert "status: delivered" in out and "summary: Recovered." in out and f"run: {run_id}" in out
    assert "steps: 3" in out
    assert str(provider.requests[0].messages[-1]).count("The process died") == 1
    assert main(["show", run_id, *setup.opts]) == 0
    assert "done delivered: Recovered." in capsys.readouterr().out


def test_resume_of_a_finished_run_shows_its_recorded_result(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [done("First time.")])
    run_id = run_id_of(capsys.readouterr().out)
    provider = ScriptedProvider([])
    assert main(["resume", run_id, *setup.opts], provider_factory=lambda: provider) == 0
    out = capsys.readouterr().out
    assert "the run had already ended" in out and "summary: First time." in out and provider.requests == []
    assert main(["resume", "20260101-000000-abcd", *setup.opts], provider_factory=lambda: provider) == 2
    assert "no run" in capsys.readouterr().err


def test_prune_applies_the_retention(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [done()])
    old = run_id_of(capsys.readouterr().out)
    time.sleep(1.1)
    run_cli(setup, [done()])
    recent = run_id_of(capsys.readouterr().out)
    assert main(["prune", *setup.opts]) == 0
    assert "Removed 0 runs (journal.retention_days is not set)." in capsys.readouterr().out
    stale = time.time() - 3 * 86400
    for path in [setup.runs / old, *(setup.runs / old).rglob("*")]:
        os.utime(path, (stale, stale))
    assert main(["prune", *setup.opts, "--set", "journal.retention_days=1"]) == 0
    assert "Removed 1 run." in capsys.readouterr().out
    assert not (setup.runs / old).exists() and (setup.runs / recent).exists()
    assert main(["runs", *setup.opts]) == 0
    listing = capsys.readouterr().out
    assert recent in listing and old not in listing


def test_runs_with_nothing_recorded(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["runs", *setup.opts]) == 0
    assert capsys.readouterr().out.strip() == "No runs."


@pytest.mark.skipif(not hasattr(signal, "raise_signal"), reason="needs signal.raise_signal")
def test_ctrl_c_once_is_a_clean_stop(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([reply(tool_use("sh", {"command": tree_command(setup.project, "stay")}))])

    def side() -> list[int]:
        pids = wait_for_pids(setup.project)
        newest_run(setup)
        time.sleep(0.2)
        signal.raise_signal(signal.SIGINT)
        return pids

    previous = signal.getsignal(signal.SIGINT)
    code, pids = with_side(side, ["run", "start the tree", *setup.opts], provider)
    assert signal.getsignal(signal.SIGINT) is previous
    assert code == 130 and "status: stopped" in capsys.readouterr().out and survivors(pids) == []


def test_control_applies_only_lines_it_has_not_seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    applied: list[dict[str, Any]] = []

    class Spy(Control):
        def __init__(self) -> None:
            self.printer = Printer("quiet")

        def apply(self, command: dict[str, Any]) -> None:
            applied.append(command)

    path = tmp_path / "control.jsonl"
    path.write_bytes(b'{"cmd": "say", "text": "old"}\n')
    offset = path.stat().st_size

    async def scenario() -> None:
        watcher = asyncio.ensure_future(Spy().watch(path, offset))
        await asyncio.sleep(0.1)
        with path.open("ab") as handle:
            handle.write(b'{"cmd": "say", "text": "new"}\n{"cmd": "sta')
        await asyncio.sleep(0.2)
        with path.open("ab") as handle:
            handle.write(b'rt"}\nnot json\n')
        await asyncio.sleep(0.2)
        watcher.cancel()

    monkeypatch.setattr(cli, "POLL_SECONDS", 0.02)
    asyncio.run(scenario())
    assert applied == [{"cmd": "say", "text": "new"}, {"cmd": "start"}]


def test_resume_with_a_text_continues_an_ended_run(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [done("First time.")])
    run_id = run_id_of(capsys.readouterr().out)
    provider = ScriptedProvider([done("Second time.")])
    assert main(["resume", run_id, "now", "also", "this", *setup.opts], provider_factory=lambda: provider) == 0
    out = capsys.readouterr().out
    assert "summary: Second time." in out and "already ended" not in out
    assert "now also this" in str(provider.requests[0].messages[-1])
    provider = ScriptedProvider([done("Third time.")])
    assert main(["resume", "one more thing", *setup.opts], provider_factory=lambda: provider) == 0
    assert "one more thing" in str(provider.requests[0].messages[-1])


def test_resume_continues_a_stopped_run(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    provider = ScriptedProvider([reply(tool_use("sh", {"command": tree_command(setup.project, "stay")}))])

    def side() -> list[int]:
        pids = wait_for_pids(setup.project)
        assert main(["stop", newest_run(setup), *setup.opts]) == 0
        return pids

    code, pids = with_side(side, ["run", "start the tree", *setup.opts], provider)
    run_id = run_id_of(capsys.readouterr().out)
    assert code == 130 and survivors(pids) == []
    resumed = ScriptedProvider([done("Picked up.")])
    assert main(["resume", run_id, *setup.opts], provider_factory=lambda: resumed) == 0
    assert "summary: Picked up." in capsys.readouterr().out


def test_runs_shows_running_crashed_and_ended(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    run_cli(setup, [create_file(), reply(tool_use("sh", {"command": "echo hi"})), done()])
    crashed = run_id_of(capsys.readouterr().out)
    journal = setup.runs / crashed / "root" / JOURNAL_FILE
    lines = journal.read_text(encoding="utf-8").splitlines(keepends=True)
    intents = [i for i, line in enumerate(lines) if '"kind": "tool_intent"' in line]
    journal.write_text("".join(lines[: intents[1] + 1]), encoding="utf-8")
    time.sleep(1.1)
    marker = setup.folder / "started"
    sleeper = write_script(
        setup.folder, "sleeper.py", "import sys, time, pathlib\npathlib.Path(sys.argv[1]).write_text('1')\ntime.sleep(2)\n"
    )
    provider = ScriptedProvider([reply(tool_use("sh", {"command": py_command(sleeper, marker)})), done()])
    listing: list[str] = []

    def side() -> None:
        wait_for(marker.exists)
        assert main(["runs", *setup.opts]) == 0
        listing.append(capsys.readouterr().out)

    code, _ = with_side(side, ["run", "work", *setup.opts], provider)
    assert code == 0
    during = listing[0]
    rows = [line for line in during.splitlines() if line[:8].isdigit()]
    assert "crashed" in next(line for line in rows if crashed in line)
    assert "running" in next(line for line in rows if crashed not in line)
    assert main(["runs", *setup.opts]) == 0
    after = capsys.readouterr().out
    assert "running" not in after and "delivered" in after


def test_say_and_redirect_can_address_a_child(setup: Setup, capsys: pytest.CaptureFixture[str]) -> None:
    marker = setup.folder / "started"
    sleeper = write_script(
        setup.folder, "sleeper.py", "import sys, time, pathlib\npathlib.Path(sys.argv[1]).write_text('1')\ntime.sleep(2)\n"
    )
    provider = ScriptedProvider([reply(tool_use("sh", {"command": py_command(sleeper, marker)})), done()])

    def side() -> None:
        wait_for(marker.exists)
        assert main(["say", "--agent", "a1", newest_run(setup), "to the child", *setup.opts]) == 0

    code, _ = with_side(side, ["run", "work", *setup.opts], provider)
    assert code == 0 and "to the child" not in str(provider.requests[1].messages)
    assert "= no agent 'a1'" in capsys.readouterr().out


def test_control_gives_one_clear_line_for_an_unknown_agent(capsys: pytest.CaptureFixture[str]) -> None:
    class Runtime:
        def add(self, text: str, agent_id: str | None = None) -> None:
            raise KeyError(f"no agent {agent_id!r}")

        def redirect(self, text: str, agent_id: str | None = None) -> None:
            raise KeyError(f"agent {agent_id!r} has ended")

    control = Control(Runtime(), Printer())
    control.line("/add @a9 hello")
    control.line("/redirect @a1 go")
    assert capsys.readouterr().out.splitlines() == ["= no agent 'a9'", "= agent 'a1' has ended"]


def test_synthic_alone_opens_the_interface_and_a_mistyped_command_is_never_a_task() -> None:
    from synthic.cli import interactive

    assert interactive([]) == ["tui"]
    assert interactive(["fix the failing test"]) == ["tui", "fix the failing test"]
    assert interactive(["--set", "budget.dollars=1"]) == ["tui", "--set", "budget.dollars=1"]
    assert interactive(["run", "a task"]) == ["run", "a task"]
    assert interactive(["shwo"]) == ["shwo"] and main(["shwo"]) == 2
    assert interactive(["--version"]) == ["--version"] and interactive(["-h"]) == ["-h"]
