"""Processes whose whole tree can be killed.

A command runs through a shell with stdin closed and stdout and stderr merged
into a file on disk, so output survives the process, can be read while it runs
and never fills a pipe. ``ProcessTree.kill_tree`` ends the process and every
descendant, also after the shell itself has exited.

Windows: the process is created suspended, assigned to a Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` and only then resumed, so it cannot start
a child before it is in the job and there is no start/assign race at all. Every
descendant inherits the job (breakaway is not allowed), detached ones included,
and ``TerminateJobObject`` ends them all. If the harness dies, the OS closes the
job handle and the kernel kills the tree. If the job cannot be created or the
process cannot be assigned, the process is resumed anyway and killing falls
back to ``taskkill /T /F``, which walks parent links and therefore misses
descendants whose parent already exited. What escapes even the job: work handed
to processes outside the tree (services, the task scheduler, WMI, COM servers).

POSIX: the shell starts in its own session and the whole process group gets
SIGTERM, then SIGKILL after a short grace. A descendant that calls ``setsid`` or
``setpgid`` (daemonising tools, ``setsid cmd``) leaves the group and survives.
"""

from __future__ import annotations

import asyncio
import functools
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from .types import EnvironmentSettings

IS_WINDOWS = sys.platform == "win32"
KILL_GRACE_SECONDS = 2.0

KillPath = Literal["job", "taskkill", "killpg"]


if IS_WINDOWS:
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _ntdll = ctypes.WinDLL("ntdll")

    class _BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BasicLimits),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _BasicAccounting(ctypes.Structure):
        _fields_ = [
            ("TotalUserTime", ctypes.c_int64),
            ("TotalKernelTime", ctypes.c_int64),
            ("ThisPeriodTotalUserTime", ctypes.c_int64),
            ("ThisPeriodTotalKernelTime", ctypes.c_int64),
            ("TotalPageFaultCount", wintypes.DWORD),
            ("TotalProcesses", wintypes.DWORD),
            ("ActiveProcesses", wintypes.DWORD),
            ("TotalTerminatedProcesses", wintypes.DWORD),
        ]

    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    _PROCESS_TERMINATE = 0x0001
    _PROCESS_SET_QUOTA = 0x0100
    _PROCESS_SUSPEND_RESUME = 0x0800
    _CREATE_SUSPENDED = 0x00000004

    _kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL
    _kernel32.QueryInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    ]
    _kernel32.QueryInformationJobObject.restype = wintypes.BOOL
    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL
    _ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    _ntdll.NtResumeProcess.restype = ctypes.c_long

    def _create_job() -> int | None:
        """A job that kills its members when its last handle closes, or ``None`` when Windows refuses."""
        job = _kernel32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
        ):
            _kernel32.CloseHandle(job)
            return None
        return job

    def _enter_job_and_resume(pid: int) -> int | None:
        """Put the suspended process ``pid`` in a new job, resume it, and return the job handle.

        Once the process is opened it is resumed whatever happens, so a failed
        assignment never leaves it frozen; ``None`` then means "use taskkill".
        If it cannot even be opened, ``OSError`` tells the caller to kill it.
        ``NtResumeProcess`` is used because ``subprocess`` closes the primary
        thread handle that ``ResumeThread`` would need.
        """
        process = _kernel32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA | _PROCESS_SUSPEND_RESUME, False, pid)
        if not process:
            raise ctypes.WinError(ctypes.get_last_error())
        job: int | None = None
        try:
            job = _create_job()
            if job is not None and not _kernel32.AssignProcessToJobObject(job, process):
                _kernel32.CloseHandle(job)
                job = None
        finally:
            _ntdll.NtResumeProcess(process)
            _kernel32.CloseHandle(process)
        return job

    def _job_active_processes(job: int) -> int:
        info = _BasicAccounting()
        if not _kernel32.QueryInformationJobObject(
            job, _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION, ctypes.byref(info), ctypes.sizeof(info), None
        ):
            return 1
        return int(info.ActiveProcesses)


def _is_wsl_launcher(path: Path) -> bool:
    """``System32\\bash.exe`` and the ``WindowsApps`` alias start WSL, not a Windows shell."""
    folder = os.path.normcase(str(path.parent))
    system32 = os.path.normcase(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"))
    return folder == system32 or "windowsapps" in folder


@functools.cache
def _auto_shell() -> tuple[tuple[str, ...], str]:
    """The shell argv prefix picked when the policy names none, with a human name for it."""
    if not IS_WINDOWS:
        return ("sh", "-c"), "POSIX sh"
    for candidate in _git_bash_candidates():
        if candidate.is_file() and not _is_wsl_launcher(candidate):
            return (str(candidate), "-c"), "Git Bash"
    return ("powershell", "-NoProfile", "-NonInteractive", "-Command"), "Windows PowerShell"


def _git_bash_candidates() -> list[Path]:
    """Where Git Bash may live on Windows, most specific first.

    Git Bash is the default because it behaves the same as the shell on Linux
    and macOS, which is what models write best. A terminal started from
    PowerShell rarely has ``bash.exe`` on PATH (only the WSL launcher), so the
    search also derives the location from ``git.exe`` and from the standard
    install folders.
    """
    found = [Path(folder) / "bash.exe" for folder in os.environ.get("PATH", "").split(os.pathsep) if folder]
    git = shutil.which("git")
    if git:
        found.extend(parent / "bin" / "bash.exe" for parent in Path(git).resolve().parents[:3])
    for variable in ("ProgramFiles", "ProgramW6432", "LocalAppData"):
        base = os.environ.get(variable)
        if base:
            found.append(Path(base) / "Git" / "bin" / "bash.exe")
            found.append(Path(base) / "Programs" / "Git" / "bin" / "bash.exe")
    return found


def shell_argv(settings: EnvironmentSettings) -> tuple[str, ...]:
    """The argv prefix that runs a command line: the policy's choice, else the auto-detected shell."""
    return settings.shell or _auto_shell()[0]


def describe_shell(settings: EnvironmentSettings) -> str:
    """One line for the system prompt telling the model which shell interprets its commands."""
    if settings.shell:
        return f"Commands run as: {' '.join(settings.shell)} <command>"
    argv, name = _auto_shell()
    return f"Commands run in {name} ({' '.join(argv)} <command>)"


class ProcessTree:
    """A started command and everything it spawns. Create it with ``start``."""

    def __init__(self, process: asyncio.subprocess.Process, kill_path: KillPath, job: int | None) -> None:
        self.process = process
        self.pid = process.pid
        self.kill_path: KillPath = kill_path
        self._job = job

    @property
    def returncode(self) -> int | None:
        return self.process.returncode

    async def wait(self) -> int:
        """Wait for the shell itself to exit. Descendants it left behind may still run."""
        return await self.process.wait()

    def alive(self) -> bool:
        """Whether any process of the tree may still run, the shell or anything it left behind."""
        if self.returncode is None:
            return True
        match self.kill_path:
            case "job":
                return self._job is not None and _job_active_processes(self._job) > 0
            case "killpg":
                try:
                    os.killpg(self.pid, 0)
                except (ProcessLookupError, PermissionError):
                    return False
                return True
            case _:
                return False

    async def kill_tree(self) -> None:
        """Terminate the process and all its descendants, then wait for the shell to be reaped."""
        match self.kill_path:
            case "job":
                if self._job is not None:
                    _kernel32.TerminateJobObject(self._job, 1)
                    _kernel32.CloseHandle(self._job)
                    self._job = None
            case "taskkill":
                if self.returncode is None:
                    killer = await asyncio.create_subprocess_exec(
                        "taskkill",
                        "/T",
                        "/F",
                        "/PID",
                        str(self.pid),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    await killer.wait()
            case "killpg":
                await self._killpg()
        if self.returncode is None:
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        await self.process.wait()

    async def _killpg(self) -> None:
        try:
            os.killpg(self.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        loop = asyncio.get_running_loop()
        deadline = loop.time() + KILL_GRACE_SECONDS
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            if self.returncode is not None and not self.alive():
                return
        try:
            os.killpg(self.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def release(self) -> None:
        """Drop OS resources of a tree that has nothing left running."""
        if self._job is not None:
            _kernel32.CloseHandle(self._job)
            self._job = None


async def start(
    command: str,
    cwd: Path,
    env: Mapping[str, str],
    output: Path,
    settings: EnvironmentSettings,
) -> ProcessTree:
    """Run ``command`` through the shell in ``cwd`` with exactly ``env``, writing all output to ``output``."""
    argv = [*shell_argv(settings), command]
    with open(output, "wb") as sink:
        if IS_WINDOWS:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=sink,
                stderr=subprocess.STDOUT,
                creationflags=_CREATE_SUSPENDED | subprocess.CREATE_NO_WINDOW,
            )
            try:
                job = _enter_job_and_resume(process.pid)
            except OSError:
                process.kill()
                await process.wait()
                raise
            return ProcessTree(process, "job" if job is not None else "taskkill", job)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.DEVNULL,
            stdout=sink,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return ProcessTree(process, "killpg", None)


def base_env() -> dict[str, str]:
    """``os.environ`` without the engine's own ``ANTHROPIC_API_KEY``.

    Every child the engine starts (shell commands, MCP servers, hooks) begins
    from this, because those children run what the model or a plugin decided and
    could read or leak the key the engine needs for itself. The key reaches a
    child only when the policy's ``environment.env``, or a server's or hook's
    own ``env``, sets it explicitly.
    """
    return {key: value for key, value in os.environ.items() if key.upper() != "ANTHROPIC_API_KEY"}


def merged_env(overlay: Mapping[str, str]) -> dict[str, str]:
    """``base_env()`` overlaid by ``overlay``. On Windows names are case-insensitive, so keys are folded first."""
    env = base_env()
    for key, value in overlay.items():
        env[key.upper() if IS_WINDOWS else key] = value
    return env
