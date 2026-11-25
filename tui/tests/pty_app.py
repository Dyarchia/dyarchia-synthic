"""The interface with a scripted model, started as a program by ``test_exit_pty.py``.

The model asks for one shell command that sleeps for minutes, so a run is working, with a child process
under its shell, until the interface stops it. The last argument is a marker that only that command line
carries, so the test can tell whether the child survived.
"""

from __future__ import annotations

import sys
from pathlib import Path

from synthic.cli import main
from synthic.testing import ScriptedProvider, reply, text, tool_use

marker = sys.argv[1]
sleeper = f'"{Path(sys.executable).as_posix()}" -c "import time; time.sleep(300)" {marker}'
provider = ScriptedProvider([reply(tool_use("sh", {"command": sleeper})), reply(text("Stopped."))])
raise SystemExit(main(["tui", "wait for the long command"], provider_factory=lambda: provider))
