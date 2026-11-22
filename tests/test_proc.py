"""Process trees: real processes, real kills.

The helpers here are imported by the other tool test files, since the suite
has no shared conftest.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

from synthic import proc
from synthic.types import EnvironmentSettings

SETTINGS = EnvironmentSettings()
SLEEP_SECONDS = 60

GRANDCHILD = """
import os, sys, time
from pathlib import Path
target = Path(sys.argv[1])
target.with_suffix(".tmp").write_text(str(os.getpid()))
target.with_suffix(".tmp").replace(target)
time.sleep(int(sys.argv[2]))
"""

CHILD = """
import os, subprocess, sys, time
from pathlib import Path
folder = Path(sys.argv[1])
flags = 0x08000000 | 0x00000200 if sys.platform == "win32" else 0
subprocess.Popen(
    [sys.executable, str(folder / "grandchild.py"), str(folder / "grandchild.pid"), sys.argv[3]],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags,
)
(folder / "child.tmp").write_text(str(os.getpid()))
(folder / "child.tmp").replace(folder / "child.pid")
if sys.argv[2] == "stay":
    time.sleep(int(sys.argv[3]))
"""


def uses_powershell(settings: EnvironmentSettings = SETTINGS) -> bool:
    return "powershell" in Path(proc.shell_argv(settings)[0]).name.lower()


def py_command(*args: str | Path, settings: EnvironmentSettings = SETTINGS) -> str:
    """A command line that runs this Python with ``args`` in whatever shell ``settings`` selects."""
    parts = [Path(sys.executable).as_posix(), *(a.as_posix() if isinstance(a, Path) else a for a in args)]
    line = " ".join(f'"{part}"' for part in parts)
    return f"& {line}" if uses_powershell(settings) else line


def write_script(folder: Path, name: str, source: str) -> Path:
    path = folder / name
    path.write_text(source, encoding="utf-8")
    return path


def tree_command(folder: Path, mode: str) -> str:
    """Shell, then a child that starts a detached sleeping grandchild; ``mode`` is ``stay`` or ``orphan``."""
    write_script(folder, "grandchild.py", GRANDCHILD)
    child = write_script(folder, "child.py", CHILD)
    return py_command(child, folder, mode, str(SLEEP_SECONDS))


def wait_for_pids(folder: Path, timeout: float = 20.0) -> list[int]:
    deadline = time.monotonic() + timeout
    files = [folder / "child.pid", folder / "grandchild.pid"]
    while not all(f.exists() for f in files):
        if time.monotonic() > deadline:
            raise AssertionError("the process tree did not record its pids in time")
        time.sleep(0.05)
    return [int(f.read_text()) for f in files]


async def async_wait_for_pids(folder: Path) -> list[int]:
    return await asyncio.to_thread(wait_for_pids, folder)


def is_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, wintypes.LPDWORD]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        code = wintypes.DWORD()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def survivors(pids: list[int], timeout: float = 5.0) -> list[int]:
    """Pids still alive after giving the OS up to ``timeout`` seconds to finish the kill."""
    deadline = time.monotonic() + timeout
    alive = [pid for pid in pids if is_alive(pid)]
    while alive and time.monotonic() < deadline:
        time.sleep(0.05)
        alive = [pid for pid in alive if is_alive(pid)]
    return alive


def expected_kill_path() -> str:
    return "job" if sys.platform == "win32" else "killpg"


def test_kill_tree_kills_running_tree(tmp_path: Path) -> None:
    async def scenario() -> None:
        tree = await proc.start(tree_command(tmp_path, "stay"), tmp_path, os.environ, tmp_path / "out.log", SETTINGS)
        try:
            assert tree.kill_path == expected_kill_path()
            pids = await async_wait_for_pids(tmp_path)
            assert all(is_alive(pid) for pid in pids)
            await tree.kill_tree()
            assert tree.returncode is not None
            assert survivors([tree.pid, *pids]) == []
        finally:
            await tree.kill_tree()

    asyncio.run(scenario())


def test_kill_tree_reaches_orphans_after_shell_exited(tmp_path: Path) -> None:
    async def scenario() -> None:
        tree = await proc.start(tree_command(tmp_path, "orphan"), tmp_path, os.environ, tmp_path / "out.log", SETTINGS)
        try:
            pids = await async_wait_for_pids(tmp_path)
            await asyncio.wait_for(tree.wait(), 20)
            grandchild = pids[1]
            assert is_alive(grandchild)
            assert tree.alive()
            await tree.kill_tree()
            assert survivors([grandchild]) == []
            assert not tree.alive()
        finally:
            await tree.kill_tree()

    asyncio.run(scenario())


@pytest.mark.skipif(sys.platform != "win32", reason="taskkill is the Windows fallback")
def test_taskkill_fallback_kills_running_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(proc, "_create_job", lambda: None)

    async def scenario() -> None:
        tree = await proc.start(tree_command(tmp_path, "stay"), tmp_path, os.environ, tmp_path / "out.log", SETTINGS)
        try:
            assert tree.kill_path == "taskkill"
            pids = await async_wait_for_pids(tmp_path)
            await tree.kill_tree()
            assert survivors([tree.pid, *pids]) == []
        finally:
            await tree.kill_tree()

    asyncio.run(scenario())


def test_output_is_merged_to_file_and_stdin_is_closed(tmp_path: Path) -> None:
    script = write_script(
        tmp_path,
        "io.py",
        "import sys\n"
        "data = sys.stdin.read()\n"
        "print('stdin', len(data), flush=True)\n"
        "print('to stderr', file=sys.stderr, flush=True)\n"
        "sys.exit(3)\n",
    )

    async def scenario() -> None:
        output = tmp_path / "out.log"
        tree = await proc.start(py_command(script), tmp_path, os.environ, output, SETTINGS)
        try:
            code = await asyncio.wait_for(tree.wait(), 30)
        finally:
            await tree.kill_tree()
        text = output.read_text(encoding="utf-8")
        assert "stdin 0" in text
        assert "to stderr" in text
        assert code != 0
        if not uses_powershell():
            assert code == 3

    asyncio.run(scenario())


def test_env_and_cwd_are_applied(tmp_path: Path) -> None:
    script = write_script(tmp_path, "env.py", "import os\nprint(os.environ['SYNTHIC_PROBE'], os.getcwd())\n")
    workdir = tmp_path / "work"
    workdir.mkdir()

    async def scenario() -> None:
        output = tmp_path / "out.log"
        env = proc.merged_env({"SYNTHIC_PROBE": "probe-value"})
        tree = await proc.start(py_command(script), workdir, env, output, SETTINGS)
        try:
            await asyncio.wait_for(tree.wait(), 30)
        finally:
            await tree.kill_tree()
        text = output.read_text(encoding="utf-8")
        assert "probe-value" in text
        assert os.path.normcase(str(workdir)) in os.path.normcase(text)

    asyncio.run(scenario())


def test_api_key_is_withheld_unless_set_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    monkeypatch.setenv("SYNTHIC_OTHER", "kept")
    assert "ANTHROPIC_API_KEY" not in proc.base_env()
    assert proc.base_env()["SYNTHIC_OTHER"] == "kept"
    assert "ANTHROPIC_API_KEY" not in proc.merged_env({})
    assert proc.merged_env({"ANTHROPIC_API_KEY": "explicit"})["ANTHROPIC_API_KEY"] == "explicit"


def test_explicit_shell_is_used_and_described() -> None:
    settings = EnvironmentSettings(shell=("custom-shell", "-x"))
    assert proc.shell_argv(settings) == ("custom-shell", "-x")
    assert "custom-shell -x" in proc.describe_shell(settings)


def test_auto_shell_is_never_the_wsl_launcher() -> None:
    argv = proc.shell_argv(SETTINGS)
    description = proc.describe_shell(SETTINGS)
    assert argv[0] in description
    if sys.platform == "win32":
        assert not proc._is_wsl_launcher(Path(argv[0]))
    else:
        assert argv == ("sh", "-c")


def test_git_bash_is_found_without_bash_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A terminal started from PowerShell has no Git Bash on PATH; it is still the shell that gets picked."""
    if not proc.IS_WINDOWS or shutil.which("git") is None:
        pytest.skip("needs Git for Windows")
    kept = [f for f in os.environ["PATH"].split(os.pathsep) if not (Path(f) / "bash.exe").is_file()]
    monkeypatch.setenv("PATH", os.pathsep.join(kept))
    proc._auto_shell.cache_clear()
    try:
        argv, name = proc._auto_shell()
    finally:
        proc._auto_shell.cache_clear()
    assert name == "Git Bash"
    assert Path(argv[0]).is_file()

