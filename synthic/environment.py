"""Job 9: start and close the place where an agent works. Nothing more.

Merging the work of a copy back into the root is left to add-ons, which read
``Workspace.changes`` to decide what to merge. A git work tree gets a real
worktree on its own branch so the work can be merged with ordinary git tools;
anything else gets a plain copy of the folder.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from synthic.types import EnvironmentSettings

MAX_CHANGE_LINES = 200
MTIME_TOLERANCE_NS = 2_000_000_000
START_KEY = "synthic-start"


@dataclass
class _Copy:
    """``dest`` is the folder created for the agent; ``repo`` is the git top level when ``dest`` is a worktree.

    ``branch`` and ``start`` (the commit the worktree began from) are set only for a worktree.
    """

    path: Path
    dest: Path
    repo: Path | None
    branch: str | None = None
    start: str | None = None


async def _git(*args: str, cwd: Path) -> tuple[int, str]:
    """Run git without a shell. A missing executable or folder reads as exit code 127."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return 127, ""
    out, err = await proc.communicate()
    code = proc.returncode if proc.returncode is not None else 1
    return code, (out if code == 0 else err).decode("utf-8", "replace").rstrip("\r\n")


def _force_remove(path: Path) -> None:
    """Delete a tree even when it holds read-only files, which git objects are on Windows."""

    def make_writable_and_retry(func, target, _exc) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onexc=make_writable_and_retry)


def _cap(lines: list[str]) -> str:
    if len(lines) > MAX_CHANGE_LINES:
        extra = len(lines) - MAX_CHANGE_LINES
        lines = [*lines[:MAX_CHANGE_LINES], f"... and {extra} more"]
    return "\n".join(lines)


def _walk_files(base: Path, skip: Path | None) -> set[str]:
    """Relative posix paths of every file under ``base``, never entering ``skip``."""
    found: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(base):
        here = Path(dirpath)
        if skip is not None:
            dirnames[:] = [d for d in dirnames if (here / d).resolve() != skip]
        found.update((here / name).relative_to(base).as_posix() for name in filenames)
    return found


def _plain_diff(root: Path, copy: Path, skip: Path | None) -> list[str]:
    """Files that differ between a plain copy and its root, by size or modification time."""
    in_root = _walk_files(root, skip)
    in_copy = _walk_files(copy, None)
    lines: list[str] = []
    for rel in sorted(in_root | in_copy):
        if rel not in in_root:
            lines.append(f"A {rel}")
        elif rel not in in_copy:
            lines.append(f"D {rel}")
        else:
            before, after = (root / rel).stat(), (copy / rel).stat()
            if before.st_size != after.st_size or abs(before.st_mtime_ns - after.st_mtime_ns) > MTIME_TOLERANCE_NS:
                lines.append(f"M {rel}")
    return lines


class Workspace:
    """Gives each agent the folder it works in and takes it back when the agent is done.

    A git copy lives on the branch ``synthic/<run_id>/<agent_id>``. Agent ids
    restart at ``a1`` on every run, so the run id keeps the branches of
    different runs apart. The commit a copy started from is also stored in the
    repository's config, so a new process can still tell what the agent
    committed. A branch with commits is never deleted by ``close``: it holds work.
    """

    def __init__(self, settings: EnvironmentSettings, root: Path, state_dir: Path, run_id: str = "") -> None:
        self.settings = settings
        self.root = Path(root)
        self.state_dir = Path(state_dir)
        self.run_id = run_id
        self._copies: dict[str, _Copy] = {}

    def _branch(self, agent_id: str) -> str:
        return f"synthic/{self.run_id}/{agent_id}" if self.run_id else f"synthic/{agent_id}"

    async def start(self, agent_id: str) -> Path:
        if self.settings.mode != "copy":
            return self.root
        dest = self.state_dir / "work" / agent_id
        if dest.exists():
            raise FileExistsError(f"work copy already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        top = await self._repo_top()
        if top is None:
            await asyncio.to_thread(self._copy_tree, dest)
            self._copies[agent_id] = _Copy(dest, dest, None)
            return dest
        branch = self._branch(agent_id)
        code, message = await _git("worktree", "add", "-b", branch, str(dest), "HEAD", cwd=top)
        if code != 0:
            raise RuntimeError(f"git worktree add failed: {message}")
        _, start = await _git("rev-parse", "HEAD", cwd=dest)
        await _git("config", "--local", f"branch.{branch}.{START_KEY}", start, cwd=top)
        try:
            path = dest / self.root.resolve().relative_to(top)
        except ValueError:
            path = dest
        self._copies[agent_id] = _Copy(path, dest, top, branch, start)
        return path

    async def changes(self, agent_id: str) -> str:
        """What the agent did: for a git copy the commits since it started, then the uncommitted status."""
        if self.settings.mode != "copy":
            return await self._status(self.root)
        entry = await self._entry(agent_id)
        if entry is None:
            return ""
        if entry.repo is None:
            lines = await asyncio.to_thread(_plain_diff, self.root, entry.dest, self._skip())
            return _cap(lines)
        commits = await self._commits(entry)
        status = (await self._status(entry.path)).splitlines()
        if not commits:
            return _cap(status)
        lines = [f"{len(commits)} commit(s) on branch {entry.branch}, kept when the copy is closed:", *commits]
        if status:
            lines += ["uncommitted:", *status]
        return _cap(lines)

    async def has_changes(self, agent_id: str) -> bool:
        """Whether closing this agent's copy would discard or leave behind any work."""
        return bool(await self.changes(agent_id))

    async def close(self, agent_id: str, keep: bool = False) -> None:
        """Remove the copy. Uncommitted work goes with it (ask ``has_changes`` first); commits stay on their branch."""
        if self.settings.mode != "copy" or keep:
            return
        entry = await self._entry(agent_id)
        self._copies.pop(agent_id, None)
        if entry is None:
            return
        if entry.repo is None:
            if entry.dest.exists():
                await asyncio.to_thread(_force_remove, entry.dest)
            return
        committed = bool(await self._commits(entry))
        code, _ = await _git("worktree", "remove", "--force", str(entry.dest), cwd=entry.repo)
        if code != 0 and entry.dest.exists():
            await asyncio.to_thread(_force_remove, entry.dest)
            await _git("worktree", "prune", cwd=entry.repo)
        if not committed and entry.branch:
            await _git("branch", "-D", entry.branch, cwd=entry.repo)

    async def _entry(self, agent_id: str) -> _Copy | None:
        """The copy of ``agent_id``: the one this object started, else the one found on disk."""
        if agent_id in self._copies:
            return self._copies[agent_id]
        dest = self.state_dir / "work" / agent_id
        if not dest.is_dir():
            return None
        if not (dest / ".git").is_file():
            entry = _Copy(dest, dest, None)
        else:
            entry = await self._locate_worktree(dest)
            if entry is None:
                return None
        self._copies[agent_id] = entry
        return entry

    async def _locate_worktree(self, dest: Path) -> _Copy | None:
        code, common = await _git("rev-parse", "--path-format=absolute", "--git-common-dir", cwd=dest)
        if code != 0 or not common:
            return None
        top = Path(common).resolve().parent
        code, branch = await _git("rev-parse", "--abbrev-ref", "HEAD", cwd=dest)
        if code != 0 or not branch or branch == "HEAD":
            return None
        code, start = await _git("config", "--local", "--get", f"branch.{branch}.{START_KEY}", cwd=top)
        try:
            path = dest / self.root.resolve().relative_to(top)
        except ValueError:
            path = dest
        return _Copy(path, dest, top, branch, start if code == 0 and start else None)

    async def _commits(self, entry: _Copy) -> list[str]:
        """One line per commit made on the copy's branch since it started."""
        if entry.start is None:
            return []
        code, out = await _git("log", "--oneline", f"{entry.start}..HEAD", cwd=entry.dest)
        return out.splitlines() if code == 0 and out else []

    async def _repo_top(self) -> Path | None:
        """The git top level of the root, or ``None`` when it is not a repo with a commit (or git is missing)."""
        code, top = await _git("rev-parse", "--show-toplevel", cwd=self.root)
        if code != 0 or not top:
            return None
        top_path = Path(top).resolve()
        code, _ = await _git("rev-parse", "--verify", "HEAD", cwd=top_path)
        return top_path if code == 0 else None

    async def _status(self, folder: Path) -> str:
        code, out = await _git("status", "--short", cwd=folder)
        return _cap(out.splitlines()) if code == 0 else ""

    def _skip(self) -> Path | None:
        """The state folder, when it lives inside the root and so must not be copied or compared."""
        state = self.state_dir.resolve()
        return state if state.is_relative_to(self.root.resolve()) else None

    def _copy_tree(self, dest: Path) -> None:
        skip = self._skip()

        def ignore(directory: str, names: list[str]) -> list[str]:
            if skip is None:
                return []
            return [n for n in names if (Path(directory) / n).resolve() == skip]

        shutil.copytree(self.root, dest, ignore=ignore, symlinks=True)
