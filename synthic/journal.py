"""Job 7: the ordered, on-disk record of everything that happened.

Layout: ``<dir>/<run_id>/<agent_id>/journal.jsonl``, one JSON object per line
with the fields of ``Record``. Each line is flushed as it is written, so a crash
loses at most the line in flight; a truncated tail is dropped when the file is
reopened and skipped when it is read.

While a runtime works on a run it rewrites ``<run_id>/alive`` (its process id)
every ``HEARTBEAT_SECONDS`` and removes it on a clean close, so a reader can
tell a live run from one whose process died (``is_alive``).
"""

from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from synthic.types import JournalSettings, Record

JOURNAL_FILE = "journal.jsonl"
ALIVE_FILE = "alive"
HEARTBEAT_SECONDS = 5.0
"""How often a live runtime rewrites ``<run_dir>/alive``; ``is_alive`` allows three missed beats by default."""
MASK = "[masked]"
_TAIL_BLOCK = 8192
_OPAQUE_BLOCKS = frozenset({"thinking", "redacted_thinking"})

_BUILTIN: tuple[tuple[str | None, re.Pattern[str], str], ...] = (
    (
        "-----BEGIN",
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|\Z)", re.DOTALL),
        MASK,
    ),
    ("sk-", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"), MASK),
    (None, re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), MASK),
    (None, re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})"), MASK),
    ("Bearer", re.compile(r"\bBearer[ \t]+[A-Za-z0-9._~+/=\-]{16,}"), MASK),
    ("00D", re.compile(r"\b00D[A-Za-z0-9]{12}(?:[A-Za-z0-9]{3})?![A-Za-z0-9._]{20,}"), MASK),
    ("force://", re.compile(r"force://[^\s\"'<>]+"), MASK),
    (
        "=",
        re.compile(r"(?i)\b([A-Za-z_]*(?:password|token|secret)\s*=\s*[\"']?)[^\s\"'&;,]{12,}"),
        rf"\1{MASK}",
    ),
)


class Masker:
    """Replaces anything shaped like a secret with ``[masked]``.

    The built-in patterns are anchored on distinctive prefixes and each one is
    skipped with a plain substring test when its marker is absent, so large
    strings cost little. ``extra_patterns`` are regular expressions from the
    policy and are always applied.
    """

    def __init__(self, extra_patterns: tuple[str, ...] = ()) -> None:
        self._extra = tuple(re.compile(pattern) for pattern in extra_patterns)

    def mask(self, text: str) -> str:
        for marker, pattern, replacement in _BUILTIN:
            if marker is None or marker in text:
                text = pattern.sub(replacement, text)
        for pattern in self._extra:
            text = pattern.sub(MASK, text)
        return text

    def mask_obj(self, value: Any) -> Any:
        """Mask every string inside ``value``, except opaque model data.

        ``thinking`` and ``redacted_thinking`` blocks and the ``data`` of a
        base64 source (images) are returned untouched: the API checks thinking
        blocks byte for byte, and a secret-shaped run of base64 is not a
        secret. Masking them would corrupt what must go back unchanged and make
        a replayed history differ from the live one.

        Server tool results (``web_search_tool_result``, ``web_fetch_tool_result``
        and any other ``*_tool_result`` but the client ``tool_result``) and
        ``citations`` are untouched too: the docs require their
        ``encrypted_content`` and ``encrypted_index`` back unchanged or the
        request fails, and their text came from the provider, not from here.
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
        """
        if isinstance(value, str):
            return self.mask(value)
        if isinstance(value, dict):
            kind = value.get("type")
            if kind in _OPAQUE_BLOCKS or _server_result(kind):
                return value
            return {
                key: item if (kind == "base64" and key == "data") or key == "citations" else self.mask_obj(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.mask_obj(item) for item in value]
        return value


def _server_result(kind: Any) -> bool:
    return isinstance(kind, str) and kind.endswith("_tool_result") and kind != "tool_result"


def _base(settings: JournalSettings) -> Path:
    return Path(settings.dir).expanduser()


def new_run(settings: JournalSettings, run_id: str | None = None) -> Path:
    """Create and return the folder of a run. Generated ids sort by time: ``20261122-153012-ab12``."""
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
    run_dir = _base(settings) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _parse(line: bytes) -> Record | None:
    try:
        return Record(**json.loads(line))
    except (ValueError, TypeError):
        return None


def _repair_tail(path: Path) -> int:
    """Drop a partial last line left by a crash and return the last valid ``seq`` (0 for an empty file)."""
    if not path.exists():
        return 0
    with path.open("r+b") as handle:
        size = handle.seek(0, 2)
        position, buffer = size, b""
        while position > 0 and buffer.count(b"\n") < 2:
            step = min(_TAIL_BLOCK, position)
            position -= step
            handle.seek(position)
            buffer = handle.read(step) + buffer
        end = buffer.rfind(b"\n") + 1
        keep = position + end
        if keep != size:
            handle.truncate(keep)
        lines = buffer[:end].splitlines()
    if lines and (record := _parse(lines[-1])) is not None:
        return record.seq
    return max((record.seq for record in _read(path)), default=0)


def _read(path: Path) -> Iterator[Record]:
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line in handle:
            if line.strip() and (record := _parse(line)) is not None:
                yield record


class FileJournal:
    """The journal of one agent: ``Journal`` protocol over a JSON Lines file."""

    def __init__(self, run_dir: Path, agent_id: str, masker: Masker) -> None:
        self.agent_id = agent_id
        self.dir = Path(run_dir) / agent_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / JOURNAL_FILE
        self._masker = masker
        self._seq = _repair_tail(self.path)
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")

    def append(self, kind: str, data: dict[str, Any]) -> Record:
        self._seq += 1
        record = Record(self._seq, time.time(), self.agent_id, kind, self._masker.mask_obj(data))
        line = json.dumps(record.__dict__, ensure_ascii=False, default=str)
        self._handle.write(line + "\n")
        self._handle.flush()
        return record

    def records(self) -> Iterator[Record]:
        self._handle.flush()
        return _read(self.path)

    def close(self) -> None:
        self._handle.close()


def read_records(run_dir: Path, agent_id: str) -> Iterator[Record]:
    """The records of one agent, read without ever writing: safe on a run another process is writing.

    A partial last line is skipped, not repaired.
    """
    return _read(Path(run_dir) / agent_id / JOURNAL_FILE)


def ended(records: Iterable[Record]) -> Record | None:
    """The ``done`` record in force: the last ``done`` with no ``resume`` after it, or ``None`` while the agent is
    running, or was cut by a crash."""
    last = None
    for record in records:
        if record.kind in ("done", "resume"):
            last = record
    return last if last is not None and last.kind == "done" else None


def beat(run_dir: Path) -> None:
    """Write the process id to ``<run_dir>/alive``, which also refreshes its modification time."""
    (Path(run_dir) / ALIVE_FILE).write_text(str(os.getpid()), encoding="utf-8")


def is_alive(run_dir: Path, max_age: float = 15.0) -> bool:
    """Whether a runtime has beaten for this run within the last ``max_age`` seconds."""
    try:
        return time.time() - (Path(run_dir) / ALIVE_FILE).stat().st_mtime <= max_age
    except OSError:
        return False


def pending_intents(records: Iterable[Record]) -> list[Record]:
    """The ``tool_intent`` records with no ``tool_result`` of the same ``call_id``: the process died during them."""
    open_intents: dict[str, Record] = {}
    for record in records:
        call_id = record.data.get("call_id")
        if record.kind == "tool_intent":
            open_intents[call_id] = record
        elif record.kind == "tool_result":
            open_intents.pop(call_id, None)
    return list(open_intents.values())


def _last_activity(run_dir: Path) -> float:
    times = [run_dir.stat().st_mtime]
    times.extend(path.stat().st_mtime for path in run_dir.glob(f"*/{JOURNAL_FILE}"))
    return max(times)


def list_runs(settings: JournalSettings) -> list[Path]:
    """Run folders, most recently active first."""
    base = _base(settings)
    if not base.is_dir():
        return []
    runs = [path for path in base.iterdir() if path.is_dir()]
    return sorted(runs, key=_last_activity, reverse=True)


def agents_in(run_dir: Path) -> list[str]:
    return sorted(path.parent.name for path in Path(run_dir).glob(f"*/{JOURNAL_FILE}"))


def prune(settings: JournalSettings) -> int:
    """Delete runs idle for longer than ``retention_days`` and return how many went. No-op when it is ``None``."""
    if settings.retention_days is None:
        return 0
    cutoff = time.time() - settings.retention_days * 86400
    removed = 0
    for run_dir in list_runs(settings):
        if _last_activity(run_dir) < cutoff:
            shutil.rmtree(run_dir, ignore_errors=True)
            removed += not run_dir.exists()
    return removed
