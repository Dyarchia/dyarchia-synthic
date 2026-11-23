"""File checkpoints: what is recorded while an agent works, so that ``synthic rewind`` can undo its file changes.

Off unless the ``[checkpoints]`` policy turns it on. Two kinds of record, both
``checkpoint`` records of the agent's journal, each with the agent's ``step``
(the model call that asked for the action):

* ``edit`` (``checkpoints.edits``): before ``edit`` writes a file, its previous
  bytes are stored once, by hash, under ``<agent>/checkpoints/``, or the record
  says the file did not exist.
* ``git`` (``checkpoints.shell``): before a turn that runs ``sh``, when the
  working folder is in a git work tree, a tree object of the folder as it is
  (tracked and untracked files git does not ignore) is written with a private
  index, which leaves the repository's own index, branches and stash alone.
  Its blobs go into the repository's object store, where git's garbage
  collection drops them once unreferenced and old. A tree equal to the last one
  is not recorded again. The journal folder is never part of it.

``rewind`` undoes every checkpoint after the chosen step, newest first: an
edit gets its previous bytes back (or its file deleted), a git checkpoint puts
the working folder back to its tree: the files that differ are written back,
byte for byte, and files git does not ignore that the tree does not hold are
deleted.

Not covered: files changed by ``sh`` when the folder is not a git work tree,
or files git ignores; changes a background job makes after its turn, between
two snapshots; ``[verify]`` commands, hooks and MCP servers; files that only
exist inside the Docker container, which is gone when the run ends; anything
outside the machine (a Salesforce org, a pushed branch, a sent message); the
byte-exactness of files that ``.gitattributes`` converts (``text``, ``eol``,
filters such as LFS). A resumed agent is not told about a rewind: say it in the
resume message.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .journal import FileJournal, agents_in, read_records
from .proc import base_env
from .sandbox import write_atomic
from .types import Location, Record

FOLDER = "checkpoints"


async def _git(*args: str, cwd: Path, env: Mapping[str, str] | None = None) -> tuple[int, str]:
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=cwd,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return 127, ""
    out, _ = await process.communicate()
    return process.returncode or 0, out.decode("utf-8", "replace").strip()


class Recorder:
    """Writes the checkpoints of one agent. ``step`` is read when each checkpoint is taken."""

    def __init__(self, journal: FileJournal, step: Any) -> None:
        self.journal = journal
        self.folder = journal.dir / FOLDER
        self._step = step
        self._last_tree: str | None = None

    def before_write(self, location: Location, previous: bytes | None) -> None:
        """Called by ``edit`` with the bytes it is about to replace (``None``: the file is new)."""
        data: dict[str, Any] = {"kind": "edit", "step": self._step(), "path": location.shown}
        data["host"] = str(location.host) if location.host is not None else None
        if previous is not None:
            name = hashlib.sha256(previous).hexdigest()[:32] + ".bin"
            self.folder.mkdir(parents=True, exist_ok=True)
            target = self.folder / name
            if not target.exists():
                write_atomic(target, previous)
            data["file"] = name
        self.journal.append("checkpoint", data)

    async def snapshot(self, cwd: Path) -> None:
        """Record a git tree of the working folder, when it is in a work tree and changed since the last one.

        The private index lives in the agent's checkpoint folder and is kept:
        the first snapshot hashes every file, later ones only what changed
        since (git's stat cache). It is never seeded from the repository's
        index, whose blobs went through the repository's line-ending rules, so
        every blob holds the bytes on disk (``RAW``).
        """
        code, top = await _git("rev-parse", "--show-toplevel", cwd=cwd)
        if code != 0 or not top:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        env = {**base_env(), "GIT_INDEX_FILE": str(self.folder / "index")}
        prefix = Path(cwd).resolve().relative_to(Path(top).resolve()).as_posix()
        exclude = _journal_scope(self.journal.dir.parent.parent, Path(top))
        if (await _git(*RAW, "add", "-A", "--", *_spec(prefix, exclude), cwd=Path(top), env=env))[0] != 0:
            return
        code, tree = await _git("write-tree", cwd=Path(top), env=env)
        if code != 0 or not tree or tree == self._last_tree:
            return
        self._last_tree = tree
        data = {"kind": "git", "step": self._step(), "tree": tree, "top": top, "prefix": prefix, "exclude": exclude}
        self.journal.append("checkpoint", data)


RAW = ("-c", "core.autocrlf=false", "-c", "core.safecrlf=false")
"""Git settings that keep blobs and restored files byte for byte what was on disk; ``.gitattributes``
rules (``text``, ``eol``, filters such as LFS) still apply, both ways."""


def _journal_scope(base: Path, top: Path) -> str | None:
    """The journal folder (every run of ``journal.dir``) relative to the work tree, when it lies inside it:
    journals must never be captured, and a rewind must never delete them."""
    base, top = base.resolve(), top.resolve()
    return base.relative_to(top).as_posix() if base.is_relative_to(top) else None


def _spec(prefix: str, exclude: str | None) -> list[str]:
    return [prefix, f":(exclude,top){exclude}"] if exclude else [prefix]


def _run_git(args: list[str], top: str, env: Mapping[str, str] | None = None, data: bytes | None = None) -> bytes:
    done = subprocess.run(["git", *args], cwd=top, env=dict(env) if env else None, input=data, capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"git failed: {done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout


def _restore_tree(record: Record, scratch: Path, apply: bool) -> list[str]:
    """Put the folder of a git checkpoint back to its tree, touching only the files that differ.

    The folder as it is now becomes a tree too (from a copy of the agent's
    private index, so only changed files are hashed); ``diff-tree`` between
    the two names what to write back and what to delete.
    """
    top, tree, prefix = record.data["top"], record.data["tree"], record.data["prefix"]
    spec = _spec(prefix, record.data.get("exclude"))
    current_index = scratch / "rewind-now"
    if (scratch / "index").is_file():
        shutil.copyfile(scratch / "index", current_index)
    env = {**base_env(), "GIT_INDEX_FILE": str(current_index)}
    try:
        _run_git([*RAW, "add", "-A", "--", *spec], top, env)
        now = _run_git(["write-tree"], top, env).decode("utf-8").strip()
    finally:
        current_index.unlink(missing_ok=True)
    fields = _run_git(["diff-tree", "-r", "-z", "--no-renames", "--name-status", now, tree, "--", prefix], top)
    parts = [p for p in fields.decode("utf-8", "replace").split("\0") if p]
    changes = list(zip(parts[::2], parts[1::2]))
    restore = sorted(path for status, path in changes if status != "D")
    delete = sorted(path for status, path in changes if status == "D")
    lines = [f"restore {path}" for path in restore] + [f"delete {path}" for path in delete]
    if apply and restore:
        target_index = scratch / "rewind-target"
        env = {**base_env(), "GIT_INDEX_FILE": str(target_index)}
        try:
            _run_git(["read-tree", tree], top, env)
            _run_git([*RAW, "checkout-index", "-f", "-z", "--stdin"], top, env, "\0".join(restore).encode("utf-8"))
        finally:
            target_index.unlink(missing_ok=True)
    if apply:
        for path in delete:
            (Path(top) / path).unlink(missing_ok=True)
    return lines or [f"nothing differs from tree {tree[:12]}"]


def _restore_edit(record: Record, folder: Path, apply: bool) -> str:
    data = record.data
    if data.get("host") is None:
        return f"skip {data['path']}: it was only inside the container"
    target = Path(data["host"])
    if data.get("file") is None:
        if apply:
            target.unlink(missing_ok=True)
        return f"delete {data['path']} (created by edit)"
    if apply:
        write_atomic(target, (folder / data["file"]).read_bytes())
    return f"restore {data['path']}"


def checkpoints_of(run_dir: Path, agent: str | None) -> list[Record]:
    """Every checkpoint record of the run (or of one agent), oldest first."""
    agents = [agent] if agent else agents_in(run_dir)
    found = [r for a in agents for r in read_records(run_dir, a) if r.kind == "checkpoint"]
    return sorted(found, key=lambda r: (r.ts, r.seq))


def after(run_dir: Path, to: int, agent: str | None) -> list[Record]:
    """The checkpoints a rewind to step ``to`` undoes.

    With ``agent``, that agent's checkpoints of later steps. Without it the
    steps are the root's: its checkpoints of later steps, and every other
    agent's from the moment the root's step ``to + 1`` began.
    """
    records = checkpoints_of(run_dir, agent)
    if agent:
        return [r for r in records if r.data["step"] > to]
    calls = [r for r in read_records(run_dir, "root") if r.kind == "model_call" and "purpose" not in r.data]
    since = float("-inf") if to == 0 else calls[to].ts if to < len(calls) else float("inf")
    return [r for r in records if (r.data["step"] > to if r.agent_id == "root" else r.ts >= since)]


def rewind(run_dir: Path, to: int = 0, agent: str | None = None, apply: bool = True) -> list[str]:
    """Undo the checkpoints after step ``to``, newest first; returns one line per action (done, or planned)."""
    lines: list[str] = []
    for record in reversed(after(run_dir, to, agent)):
        where = f"{record.agent_id} step {record.data['step']}"
        folder = run_dir / record.agent_id / FOLDER
        if record.data["kind"] == "edit":
            lines.append(f"{where}: {_restore_edit(record, folder, apply)}")
        else:
            lines += [f"{where}: {line}" for line in _restore_tree(record, folder, apply)]
    return lines


def describe(record: Record) -> str:
    data = record.data
    if data.get("kind") == "git":
        return f"checkpoint step {data.get('step')}: git tree {str(data.get('tree'))[:12]} of {data.get('prefix')}"
    state = "existed" if data.get("file") else "new file"
    return f"checkpoint step {data.get('step')}: edit {data.get('path')} ({state})"
