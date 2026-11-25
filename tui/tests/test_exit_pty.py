"""Every way out of the interface, driven as a person drives it: a real process under a Windows pseudo console.

The headless ``Pilot`` of the other tests cannot see a key that never reaches the program, or an exit that
is requested but never carried out (v0.4.0 shipped an un-awaited ``run_action("quit")``). Here the program is
started as ``python -m synthic`` in a real console, real key bytes are written, and the process must end,
with no process of its tree left behind. A run that is working is a scripted model asking for a shell
command that sleeps (``pty_app.py``), so no model is called.
Skipped where ``pywinpty`` is not installed or the platform is not Windows.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

winpty = pytest.importorskip("winpty")
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the pseudo console is a Windows facility")

START_WAIT = 30
EXIT_WAIT = 12
KEY_PAUSE = 0.4
UP, DOWN, PAGE_UP, PAGE_DOWN = "\x1b[A", "\x1b[B", "\x1b[5~", "\x1b[6~"
CTRL_C, CTRL_D = "\x03", "\x04"
POLICY = """
initial_role = "lead"

[verify]
reread = false

[memory]
enabled = false

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
tools = ["sh"]
"""
LISTING = (
    "$ErrorActionPreference = 'Stop'; "
    "Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId, CommandLine | ConvertTo-Json -Compress"
)


def processes() -> list[dict]:
    """Every process of the machine, as ``ProcessId``, ``ParentProcessId`` and ``CommandLine``."""
    done = subprocess.run(["powershell", "-NoProfile", "-Command", LISTING], capture_output=True, text=True, timeout=60)
    found = json.loads(done.stdout or "[]")
    return found if isinstance(found, list) else [found]


def family(root: int) -> set[int]:
    """``root`` and everything below it."""
    table = processes()
    tree, grew = {root}, True
    while grew:
        grew = False
        for row in table:
            if row["ParentProcessId"] in tree and row["ProcessId"] not in tree:
                tree.add(row["ProcessId"])
                grew = True
    return tree


class Console:
    """A program in a pseudo console: it keeps what the screen received and writes keys to it."""

    def __init__(self, home: Path, cwd: Path, argv: list[str], size: tuple[int, int]) -> None:
        (home / ".synthic").mkdir(parents=True)
        (home / ".synthic" / "policy.toml").write_text(POLICY, encoding="utf-8")
        env = {**os.environ, "SYNTHIC_HOME": str(home), "ANTHROPIC_API_KEY": "sk-test-not-used"}
        self.process = winpty.PtyProcess.spawn(argv, cwd=str(cwd), env=env, dimensions=size)
        self.seen = ""
        threading.Thread(target=self.drain, daemon=True).start()

    def drain(self) -> None:
        while True:
            try:
                chunk = self.process.read(65536)
            except Exception:
                return
            if chunk:
                self.seen += chunk
            elif not self.process.isalive():
                return
            else:
                time.sleep(0.02)

    def wait_for(self, piece: str, seconds: float, after: int = 0) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if piece in self.seen[after:]:
                return True
            time.sleep(0.05)
        return False

    def ends_within(self, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if not self.process.isalive():
                return True
            time.sleep(0.05)
        return False

    def press(self, *keys: str) -> None:
        for key in keys:
            self.process.write(key)
            time.sleep(KEY_PAUSE)

    def close(self) -> None:
        if self.process.isalive():
            self.process.terminate(force=True)


@pytest.fixture
def launch(tmp_path: Path):
    """``launch(argv=None, size=(30, 100))`` starts the interface, ``python -m synthic`` unless told otherwise."""
    opened: list[Console] = []
    project = tmp_path / "proj"
    project.mkdir()

    def start(argv: list[str] | None = None, size: tuple[int, int] = (30, 100)) -> Console:
        console = Console(tmp_path / f"home{len(opened)}", project, argv or [sys.executable, "-m", "synthic"], size)
        opened.append(console)
        return console

    yield start
    for console in opened:
        console.close()


def ready(console: Console) -> None:
    assert console.wait_for("Write a task", START_WAIT), "the interface did not start"
    time.sleep(0.5)


def assert_left(console: Console, keys: str) -> None:
    """The process ended and nothing of its tree is still alive."""
    below = family(console.process.pid)
    assert console.ends_within(EXIT_WAIT), "the process is still alive after the keys: " + keys
    time.sleep(0.5)
    left = below & {row["ProcessId"] for row in processes()}
    assert not left, f"processes of the interface survived it: {sorted(left)}"


@pytest.mark.parametrize(
    "keys",
    [["exit\r"], ["quit\r"], ["/exit\r"], ["/quit\r"], [CTRL_C, CTRL_C], [CTRL_D, CTRL_D]],
    ids=["exit", "quit", "slash-exit", "slash-quit", "ctrl-c-twice", "ctrl-d-twice"],
)
def test_each_way_out_ends_the_process(launch, keys: list[str]) -> None:
    console = launch()
    ready(console)
    console.press(*keys)
    assert_left(console, repr(keys))


@pytest.mark.parametrize(
    "keys",
    [["some text", CTRL_C, CTRL_C], ["some text", CTRL_C, "/", CTRL_C], ["/", CTRL_C, CTRL_C], ["/model\r", CTRL_C, CTRL_C]],
    ids=["text", "text-cleared-then-menu", "menu", "picker"],
)
def test_a_double_ctrl_c_leaves_with_text_or_a_menu_open(launch, keys: list[str]) -> None:
    console = launch()
    ready(console)
    console.press(*keys)
    assert_left(console, repr(keys))


def test_the_first_ctrl_c_clears_the_box_and_closes_a_menu_and_warns(launch) -> None:
    console = launch()
    ready(console)
    console.press("/model\r")
    assert console.wait_for("Choose with Up and Down", 5)
    mark = len(console.seen)
    console.press(CTRL_C)
    assert console.wait_for("Press Ctrl+C again to leave", 5, mark)
    assert console.process.isalive()
    time.sleep(2.5)
    console.press(CTRL_C)
    assert console.process.isalive(), "a press after the window must only warn again"


def test_a_single_ctrl_c_only_warns(launch) -> None:
    console = launch()
    ready(console)
    console.press(CTRL_C)
    assert console.wait_for("Press Ctrl+C again to leave", 5)
    assert console.process.isalive()


def test_ctrl_d_with_text_does_not_leave_and_without_text_warns_first(launch) -> None:
    console = launch()
    ready(console)
    console.press("ab", CTRL_D, CTRL_D, CTRL_D)
    assert console.process.isalive(), "Ctrl+D must not leave while the box holds text"
    assert "Press Ctrl+D again to leave" not in console.seen
    console.press("\b\b", CTRL_D)
    assert console.wait_for("Press Ctrl+D again to leave", 5)
    assert console.process.isalive()


@pytest.mark.parametrize("key", [CTRL_C, CTRL_D], ids=["ctrl-c", "ctrl-d"])
def test_a_double_press_leaves_while_a_run_is_working_and_kills_its_shell(launch, key: str) -> None:
    marker = "synthic-pty-" + uuid.uuid4().hex
    console = launch([sys.executable, str(Path(__file__).with_name("pty_app.py")), marker])
    ready(console)
    deadline = time.monotonic() + START_WAIT
    while time.monotonic() < deadline and not any(marker in (row["CommandLine"] or "") for row in processes()):
        time.sleep(0.5)
    sleepers = {row["ProcessId"] for row in processes() if marker in (row["CommandLine"] or "")}
    assert sleepers, "the long command never started"
    console.press(key, key)
    assert_left(console, repr(key + key))
    alive = {row["ProcessId"] for row in processes() if marker in (row["CommandLine"] or "")}
    assert not alive, f"the command of the run survived the interface: {sorted(alive)}"


def test_a_letter_typed_between_two_ctrl_c_does_not_disarm_the_leave(launch) -> None:
    marker = "synthic-pty-" + uuid.uuid4().hex
    console = launch([sys.executable, str(Path(__file__).with_name("pty_app.py")), marker])
    ready(console)
    deadline = time.monotonic() + START_WAIT
    while time.monotonic() < deadline and not any(marker in (row["CommandLine"] or "") for row in processes()):
        time.sleep(0.5)
    console.press(CTRL_C, "x", CTRL_C)
    assert_left(console, "Ctrl+C, a letter, Ctrl+C")
    assert not any(marker in (row["CommandLine"] or "") for row in processes())


def test_the_help_can_be_scrolled_to_its_last_line_at_80_by_24(launch) -> None:
    console = launch(size=(24, 80))
    ready(console)
    console.press("/help")
    mark = len(console.seen)
    console.press("\r")
    assert console.wait_for("more below", 5, mark), "no cue that the help goes on"
    assert "leave Synthic" not in console.seen[mark:], "the help already shows its end, so nothing is tested"
    mark = len(console.seen)
    console.press(*([PAGE_DOWN] * 4), *([DOWN] * 6))
    assert console.wait_for("with text it deletes a char", 5, mark), "the last line was never reached"
    assert console.wait_for("end of the list", 5, mark)
    console.press(PAGE_UP, UP)
    console.press("\x1b")
    assert console.process.isalive()
    mark = len(console.seen)
    console.press("/help", "\r")
    assert console.wait_for("more below", 5, mark), "the help did not open again after Esc"


def test_the_help_fits_whole_at_120_by_35(launch) -> None:
    console = launch(size=(35, 120))
    ready(console)
    console.press("/help")
    mark = len(console.seen)
    console.press("\r")
    assert console.wait_for("with text it deletes a char", 5, mark)
    assert "more below" not in console.seen[mark:]
    assert "leave Synthic" in console.seen[mark:]
