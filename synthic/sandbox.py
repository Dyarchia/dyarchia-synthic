"""Where an agent's commands run and its file tools read and write.

``HostSandbox`` is the host itself, as every mode but ``docker`` has always
worked. ``DockerSandbox`` (``environment.mode = "docker"``) is one container
per run, shared by every agent of the run, in which ``sh``, background jobs and
``[verify]`` commands run and through which ``edit`` and ``view`` see files.

The container:

* Started at the beginning of the run (in the background, so an image pull
  overlaps the first model call) with ``docker run -i --rm --init``. Its main
  process reads its stdin until the end: the engine holds the other end of that
  pipe, so when the engine process dies for any reason the pipe closes, the
  main process exits and ``--rm`` removes the container. On Windows the client
  also sits in the engine's Job Object and dies with it, which closes the
  container's stdin the same way. ``close`` removes it with ``docker rm -f`` at
  the end of the run (delivered, stopped, budget or timeout, failed). Before
  starting, containers labelled ``synthic.run`` whose run is no longer alive
  are removed too: a container whose run's heartbeat stopped can only be an
  orphan, and only containers this engine created carry the label.
* Mounts: the project root read-write at ``docker.workdir``, the project's
  memory folder read-write at ``MEMORY_DIR`` (when memory is on), and each of
  ``docker.mounts`` read-only. Nothing else of the host is visible.
* Environment: only ``environment.env`` and the host variables named in
  ``docker.pass_env``, passed as ``-e NAME`` so no value appears on a command
  line. ``ANTHROPIC_API_KEY`` is refused by the policy in this mode.
* Network: on or off (``--network none``). An allow-list of domains needs a
  filtering proxy, which Docker does not provide, so there is none.

Commands: each is a ``docker exec`` whose process carries the variable
``SYNTHIC_JOB=<token>``, inherited by every descendant, detached ones and
those that call ``setsid`` included. Killing the ``docker exec`` client does
not stop what runs inside, so a kill first marks the token as cancelled (a
command that starts after that exits at once) and then kills, inside the
container, every process whose environment holds the token, in rounds until
none is left. What escapes: a process that clears its environment. When the
shell exits, one more ``docker exec`` (off the critical path) checks whether it
left processes behind, which is what ``alive`` reports from then on.
Difference from the host: output that a leftover process writes more than
about two seconds after the shell exits is not captured, since the Docker
client stops reading then.

Files: a path inside a mount is read and written on the host side of the bind
mount, which holds the same bytes and costs no ``docker exec``; the path is
resolved on the host and must stay inside the mounted folder, so a symlink
that points elsewhere is followed inside the container instead. Any other path
(``/tmp``, ``/etc``...) is read and written inside the container.
"""

from __future__ import annotations

import asyncio
import os
import posixpath
import io
import secrets
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from . import proc
from .journal import is_alive
from .prompt import memory_dir
from .types import EnvironmentSettings, Location, Policy

TOKEN_VAR = "SYNTHIC_JOB"
MEMORY_DIR = "/synthic/memory"
KILL_MARKS = "/tmp/.synthic-kill"
LABEL = "synthic.run"
LABEL_DIR = "synthic.run_dir"
KEEPER_LOG = "docker.log"
POLL_SECONDS = 0.2
MAX_TREE_BYTES = 64 * 1024 * 1024
"""The most a search reads from a folder that only exists inside the container."""

_WRAPPER = f'[ -e "{KILL_MARKS}/${TOKEN_VAR}" ] && exit 137; exec "$@"'
_SCAN = (
    'for p in /proc/[0-9]*; do tr "\\0" "\\n" < "$p/environ" 2>/dev/null | grep -qx "'
    + TOKEN_VAR
    + '=$1" && {action}; done'
)
_KILL = (
    f'mkdir -p {KILL_MARKS} && : > "{KILL_MARKS}/$1"; '
    "for round in 1 2 3 4 5; do found=; "
    + _SCAN.format(action='kill -9 "${p#/proc/}" 2>/dev/null && found=1')
    + '; [ -z "$found" ] && exit 0; done'
)
_ALIVE = _SCAN.format(action="exit 0") + "; exit 1"
_READ = '[ -e "$1" ] || exit 3; [ -d "$1" ] && exit 4; cat < "$1"'
_WRITE = 'mkdir -p "$(dirname "$1")" && cat > "$1"'


def write_atomic(target: Path, data: bytes) -> None:
    """Write through a temp file in the same folder and swap it in, keeping the old file's mode."""
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
    handle, temp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as sink:
            sink.write(data)
        if mode is not None:
            os.chmod(temp, mode)
        os.replace(temp, target)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def _git_names(base: Path) -> list[str] | None:
    """Files under ``base`` that git does not ignore (tracked and untracked), or ``None`` outside a work tree."""
    try:
        done = subprocess.run(
            ["git", "-C", str(base), "ls-files", "-z", "-co", "--exclude-standard"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if done.returncode != 0:
        return None
    return [name for name in done.stdout.decode("utf-8", "replace").split("\0") if name]


def host_files(base: Path, max_bytes: int) -> Iterator[tuple[str, bytes | None]]:
    """``(relative posix name, content)`` of every file under ``base`` (``""`` when ``base`` is a file), sorted.

    Inside a git work tree only the files git does not ignore, else every file
    but those under ``.git``. Content is ``None`` for a file over ``max_bytes``.
    Lazy, so a caller that stops early reads no more.
    """
    if base.is_file():
        yield "", _capped(base, max_bytes)
        return
    names = _git_names(base)
    if names is None:
        names = []
        for folder, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != ".git"]
            names += [(Path(folder) / f).relative_to(base).as_posix() for f in files]
    for name in sorted(set(names)):
        path = base / name
        if path.is_file():
            yield name, _capped(path, max_bytes)


def _capped(path: Path, max_bytes: int) -> bytes | None:
    try:
        return None if path.stat().st_size > max_bytes else path.read_bytes()
    except OSError:
        return None


def _tar_files(data: bytes, max_bytes: int) -> Iterator[tuple[str, bytes | None]]:
    """The regular files of a tar stream, as ``host_files`` gives them; a stream cut short ends the listing."""
    found: list[tuple[str, bytes | None]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name.removeprefix("./")
                source = archive.extractfile(member) if member.size <= max_bytes else None
                found.append((name, source.read() if source is not None else None))
    except (tarfile.TarError, OSError, EOFError):
        pass
    yield from sorted(found)


class HostSandbox:
    """The host: paths resolve against the working folder, commands run through ``proc``."""

    kind = "host"

    def locate(self, raw: str, cwd: Path) -> Location:
        path = (cwd / Path(raw).expanduser()).resolve()
        return Location(str(path), path)

    def shown(self, path: Path) -> str:
        return str(path)

    async def read(self, location: Location) -> bytes:
        assert location.host is not None
        return await asyncio.to_thread(location.host.read_bytes)

    async def write(self, location: Location, data: bytes) -> None:
        assert location.host is not None
        await asyncio.to_thread(write_atomic, location.host, data)

    async def exists(self, location: Location) -> bool:
        return location.host is not None and location.host.exists()

    async def is_dir(self, location: Location) -> bool:
        return location.host is not None and location.host.is_dir()

    async def files(self, location: Location, max_bytes: int) -> Iterator[tuple[str, bytes | None]]:
        """The files under ``location`` (see ``host_files``); iterate it in a worker thread."""
        assert location.host is not None
        return host_files(location.host, max_bytes)

    async def start(
        self, command: str, cwd: Path, env: dict[str, str], output: Path, settings: EnvironmentSettings, token: str
    ) -> proc.ProcessTree:
        return await proc.start(command, cwd, proc.merged_env(env), output, settings)

    def warm(self) -> None:
        """Nothing to prepare on the host."""

    async def close(self) -> None:
        """Nothing to remove on the host."""


HOST = HostSandbox()


def of(sandbox: Any) -> Any:
    """The sandbox of a ``ToolContext``: ``None`` is the host."""
    return HOST if sandbox is None else sandbox


def _mount_field(text: str) -> str:
    """One ``key=value`` field of ``--mount``, which Docker parses as CSV: quoted when it holds a comma or quote."""
    return f'"{text.replace(chr(34), chr(34) * 2)}"' if "," in text or '"' in text else text


class DockerTree:
    """A command running in the container: the host ``docker exec`` client plus the token of what it started."""

    kill_path = "docker"

    def __init__(self, sandbox: DockerSandbox, client: proc.ProcessTree, token: str) -> None:
        self.sandbox = sandbox
        self.client = client
        self.process = client.process
        self.pid = client.pid
        self.token = token
        self._left = True

    @property
    def returncode(self) -> int | None:
        return self.client.returncode

    async def wait(self) -> int:
        return await self.client.wait()

    def alive(self) -> bool:
        """Running, or left processes behind when it exited (as far as the last ``refresh`` saw)."""
        return self.returncode is None or self._left

    async def refresh(self) -> None:
        if self.returncode is not None:
            self._left = await self.sandbox.has_processes(self.token)

    async def kill_tree(self) -> None:
        await self.sandbox.kill(self.token)
        await self.client.kill_tree()
        self._left = False

    def release(self) -> None:
        self.client.release()


class DockerSandbox:
    """One container per run; see the module docstring. Build it, ``warm()`` it, ``await close()`` it."""

    kind = "docker"

    def __init__(
        self, settings: EnvironmentSettings, root: Path, run_dir: Path, run_id: str, memory: Path | None = None
    ) -> None:
        self.environment = settings
        self.settings = settings.docker
        self.root = Path(root).resolve()
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.name = f"synthic-{run_id}"
        self.workdir = posixpath.normpath(self.settings.workdir)
        mounts = [(self.workdir, self.root, False)]
        if memory is not None:
            mounts.append((MEMORY_DIR, Path(memory).resolve(), False))
        mounts += [
            (posixpath.normpath(inner), Path(host).expanduser().resolve(), True)
            for inner, host in self.settings.mounts.items()
        ]
        self.mounts = sorted(mounts, key=lambda m: len(m[0]), reverse=True)
        self.client_env = proc.merged_env(settings.env)
        self._keeper: proc.ProcessTree | None = None
        self._starting: asyncio.Task[None] | None = None
        self._closed = False

    def inner(self, path: Path) -> str | None:
        """The container path of a host path inside a mount, or ``None``."""
        resolved = Path(path).resolve()
        for inner, host, _ in self.mounts:
            if resolved.is_relative_to(host):
                rel = resolved.relative_to(host).as_posix()
                return inner if rel == "." else posixpath.join(inner, rel)
        return None

    def shown(self, path: Path) -> str:
        return self.inner(path) or str(path)

    def locate(self, raw: str, cwd: Path) -> Location:
        inner = posixpath.normpath(posixpath.join(self.inner(cwd) or self.workdir, raw))
        for prefix, host, readonly in self.mounts:
            if inner != prefix and not inner.startswith(prefix.rstrip("/") + "/"):
                continue
            rel = inner[len(prefix) :].lstrip("/")
            try:
                candidate = (host / rel).resolve() if rel else host
            except OSError:
                return Location(inner, None, inner, readonly)
            if candidate.is_relative_to(host):
                return Location(inner, candidate, inner, readonly)
            return Location(inner, None, inner, readonly)
        return Location(inner, None, inner, False)

    async def read(self, location: Location) -> bytes:
        if location.host is not None:
            return await asyncio.to_thread(location.host.read_bytes)
        code, out, err = await self._exec(_READ, location.shown)
        if code == 3:
            raise FileNotFoundError(2, "No such file or directory", location.shown)
        if code == 4:
            raise IsADirectoryError(21, "Is a directory", location.shown)
        if code != 0:
            raise OSError(f"cannot read {location.shown} in the container: {err or f'exit code {code}'}")
        return out

    async def write(self, location: Location, data: bytes) -> None:
        if location.readonly:
            raise PermissionError(13, "Read-only mount", location.shown)
        if location.host is not None:
            await asyncio.to_thread(write_atomic, location.host, data)
            return
        code, _, err = await self._exec(_WRITE, location.shown, data=data)
        if code != 0:
            raise OSError(f"cannot write {location.shown} in the container: {err or f'exit code {code}'}")

    async def exists(self, location: Location) -> bool:
        if location.host is not None:
            return location.host.exists()
        return (await self._exec('[ -e "$1" ]', location.shown))[0] == 0

    async def is_dir(self, location: Location) -> bool:
        if location.host is not None:
            return location.host.is_dir()
        return (await self._exec('[ -d "$1" ]', location.shown))[0] == 0

    async def files(self, location: Location, max_bytes: int) -> Iterator[tuple[str, bytes | None]]:
        """Files under a mount come from the host side (``host_files``); others from a ``tar`` of the folder,
        cut at ``MAX_TREE_BYTES``."""
        if location.host is not None:
            return host_files(location.host, max_bytes)
        if not await self.is_dir(location):
            return iter([("", await self.read(location))])
        code, out, err = await self._exec(f'cd "$1" && tar -cf - . | head -c {MAX_TREE_BYTES}', location.shown)
        if code != 0 and not out:
            raise OSError(f"cannot list {location.shown} in the container: {err or f'exit code {code}'}")
        return _tar_files(out, max_bytes)

    async def start(
        self, command: str, cwd: Path, env: dict[str, str], output: Path, settings: EnvironmentSettings, token: str
    ) -> DockerTree:
        """Run ``command`` in the container in the container view of ``cwd``; ``env`` (host values) is not used."""
        await self.ensure()
        argv = [
            *self.settings.command,
            "exec",
            "-w",
            self.inner(cwd) or self.workdir,
            "-e",
            TOKEN_VAR,
            self.name,
            "sh",
            "-c",
            _WRAPPER,
            "synthic",
            *self.settings.shell,
            command,
        ]
        client = await proc.start_argv(argv, self.root, {**self.client_env, TOKEN_VAR: token}, output)
        return DockerTree(self, client, token)

    async def kill(self, token: str) -> None:
        if self._keeper is not None and not self._closed:
            await self._exec(_KILL, token)

    async def has_processes(self, token: str) -> bool:
        if self._keeper is None or self._closed:
            return False
        return (await self._exec(_ALIVE, token))[0] == 0

    async def _exec(self, script: str, arg: str, data: bytes | None = None) -> tuple[int, bytes, str]:
        """Run ``sh -c script synthic arg`` in the container, with ``data`` on stdin: exit code, stdout, stderr."""
        await self.ensure()
        flags = ["-i"] if data is not None else []
        return await self._docker("exec", *flags, self.name, "sh", "-c", script, "synthic", arg, data=data)

    async def _docker(self, *args: str, data: bytes | None = None) -> tuple[int, bytes, str]:
        try:
            process = await asyncio.create_subprocess_exec(
                *self.settings.command,
                *args,
                stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.client_env,
            )
        except OSError as error:
            return 127, b"", str(error)
        out, err = await process.communicate(data)
        code = process.returncode if process.returncode is not None else 1
        return code, out, err.decode("utf-8", "replace").strip()

    def warm(self) -> None:
        """Start the container in the background, so the first command finds it ready."""
        if self._starting is None and not self._closed:
            self._starting = asyncio.get_running_loop().create_task(self._start())

    async def ensure(self) -> None:
        """Wait until the container runs; the first caller starts it. Raises ``OSError`` when it cannot."""
        if self._closed:
            raise OSError("the run's container is already removed")
        if self._starting is None:
            self.warm()
        assert self._starting is not None
        await asyncio.shield(self._starting)

    def _run_argv(self) -> list[str]:
        s = self.settings
        argv = [*s.command, "run", "-i", "--rm", "--init", "--name", self.name]
        argv += ["--label", f"{LABEL}={self.run_id}", "--label", f"{LABEL_DIR}={self.run_dir.resolve()}"]
        for inner, host, readonly in sorted(self.mounts, key=lambda m: len(m[0])):
            fields = ["type=bind", _mount_field(f"source={host}"), _mount_field(f"target={inner}")]
            argv += ["--mount", ",".join([*fields, "readonly"] if readonly else fields)]
        if not s.network:
            argv += ["--network", "none"]
        for name in dict.fromkeys([*self.environment.env, *s.pass_env]):
            argv += ["-e", name]
        if s.user:
            argv += ["--user", s.user]
        elif hasattr(os, "getuid"):
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        return [*argv, *s.options, "--entrypoint", "sh", s.image, "-c", "cat > /dev/null"]

    async def _start(self) -> None:
        await self._sweep()
        for _, host, readonly in self.mounts:
            if not readonly:
                host.mkdir(parents=True, exist_ok=True)
        log = self.run_dir / KEEPER_LOG
        self._keeper = await proc.start_argv(self._run_argv(), self.root, self.client_env, log, stdin=subprocess.PIPE)
        while True:
            if self._keeper.returncode is not None:
                tail = log.read_text(encoding="utf-8", errors="replace").strip()[-2000:]
                raise OSError(f"the container did not start (docker exit code {self._keeper.returncode}): {tail}")
            code, out, _ = await self._docker("inspect", "-f", "{{.State.Running}}", self.name)
            if code == 0 and out.strip() == b"true":
                return
            await asyncio.sleep(POLL_SECONDS)

    async def _sweep(self) -> None:
        """Remove containers of this engine whose run is no longer alive (see the module docstring)."""
        code, out, _ = await self._docker(
            "ps", "-a", "--filter", f"label={LABEL}", "--format", f'{{{{.Names}}}}\t{{{{.Label "{LABEL_DIR}"}}}}'
        )
        if code != 0:
            return
        for line in out.decode("utf-8", "replace").splitlines():
            name, _, folder = line.partition("\t")
            if name and name != self.name and folder and not is_alive(Path(folder)):
                await self._docker("rm", "-f", name)

    async def close(self) -> None:
        """Remove the container; whatever still runs in it dies with it."""
        self._closed = True
        if self._starting is not None and not self._starting.done():
            self._starting.cancel()
        if self._starting is not None:
            await asyncio.gather(self._starting, return_exceptions=True)
        keeper, self._keeper = self._keeper, None
        if keeper is None:
            return
        await self._docker("rm", "-f", self.name)
        stdin = keeper.process.stdin
        if stdin is not None:
            stdin.close()
        await keeper.kill_tree()
        keeper.release()


def make_sandbox(policy: Policy, root: Path, run_dir: Path, run_id: str) -> HostSandbox | DockerSandbox:
    """The sandbox ``environment.mode`` asks for."""
    if policy.environment.mode != "docker":
        return HOST
    memory = memory_dir(policy, root) if policy.memory.enabled else None
    return DockerSandbox(policy.environment, root, run_dir, run_id, memory)


def job_token() -> str:
    """A fresh prefix for the tokens of one ``JobManager``'s commands."""
    return secrets.token_hex(4)
