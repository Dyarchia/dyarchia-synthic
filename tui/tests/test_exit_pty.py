"""Every way out of the interface, driven as a person drives it: a real process under a Windows pseudo console.

The headless ``Pilot`` of the other tests cannot see a key that never reaches the program, or an exit that
is requested but never carried out (v0.4.0 shipped an un-awaited ``run_action("quit")``). Here the program is
started as ``python -m synthic`` in a real console, real key bytes are written, and the process must end.
Skipped where ``pywinpty`` is not installed or the platform is not Windows.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

winpty = pytest.importorskip("winpty")
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="the pseudo console is a Windows facility")

START_WAIT = 30
EXIT_WAIT = 8


class Console:
    """A program in a pseudo console: it keeps what the screen received and writes keys to it."""

    def __init__(self, home: Path, cwd: Path) -> None:
        env = {**os.environ, "SYNTHIC_HOME": str(home), "ANTHROPIC_API_KEY": "sk-test-not-used"}
        self.process = winpty.PtyProcess.spawn(
            [sys.executable, "-m", "synthic"], cwd=str(cwd), env=env, dimensions=(30, 100)
        )
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

    def wait_for(self, piece: str, seconds: float) -> bool:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if piece in self.seen:
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

    def close(self) -> None:
        if self.process.isalive():
            self.process.terminate(force=True)


@pytest.fixture
def console(tmp_path: Path) -> Console:
    project = tmp_path / "proj"
    project.mkdir()
    opened = Console(tmp_path / "home", project)
    yield opened
    opened.close()


@pytest.mark.parametrize(
    "keys",
    [["exit\r"], ["quit\r"], ["/exit\r"], ["/quit\r"], ["\x03", "\x03"]],
    ids=["exit", "quit", "slash-exit", "slash-quit", "ctrl-c-twice"],
)
def test_each_way_out_ends_the_process(console: Console, keys: list[str]) -> None:
    assert console.wait_for("Write a task", START_WAIT), "the interface did not start"
    time.sleep(0.5)
    for key in keys:
        console.process.write(key)
        time.sleep(0.4)
    assert console.ends_within(EXIT_WAIT), "the process is still alive after the keys: " + repr(keys)


def test_a_single_ctrl_c_only_warns(console: Console) -> None:
    assert console.wait_for("Write a task", START_WAIT)
    time.sleep(0.5)
    console.process.write("\x03")
    assert console.wait_for("Press Ctrl+C again to leave", 5)
    assert console.process.isalive()
