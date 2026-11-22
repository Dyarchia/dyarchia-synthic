"""The ``edit`` tool: exact text replacement that keeps the file's encoding details intact."""

from __future__ import annotations

import asyncio
import difflib
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from ..types import ToolContext, ToolOutput

BOM = b"\xef\xbb\xbf"
CONTEXT_LINES = 2


class EditError(Exception):
    """A call the model has to fix; the message tells it how."""


def resolve(path: str, cwd: Path) -> Path:
    return (cwd / Path(path).expanduser()).resolve()


def _closest_line(text: str, old: str) -> str:
    """A hint for a failed match: the file line most similar to the first non-blank line of ``old``."""
    target = next((line.strip() for line in old.splitlines() if line.strip()), "")
    lines = text.splitlines()
    if not target or not lines:
        return ""
    stripped = [line.strip() for line in lines]
    best = difflib.get_close_matches(target, stripped, n=1, cutoff=0.6)
    if not best:
        return ""
    number = stripped.index(best[0]) + 1
    return f" Closest line {number}: {lines[number - 1]!r}."


def _excerpt(text: str, start: int, length: int) -> str:
    """Lines of ``text`` around ``text[start:start + length]``, numbered, with context on both sides."""
    lines = text.split("\n")
    first = text.count("\n", 0, start)
    last = first + text.count("\n", start, start + length)
    low, high = max(0, first - CONTEXT_LINES), min(len(lines), last + CONTEXT_LINES + 1)
    width = len(str(high))
    return "\n".join(f"{n + 1:>{width}}  {lines[n]}" for n in range(low, high))


def apply_edit(target: Path, old: str, new: str, replace_all: bool) -> str:
    """Perform one edit on disk and return the success message, or raise ``EditError``."""
    if not target.exists():
        if old:
            raise EditError(f"{target} does not exist. To create it, call with an empty `old`.")
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_atomic(target, new.encode("utf-8"))
        return f"Created {target}."
    if not old:
        raise EditError(f"{target} already exists. Give the exact text to replace in `old`.")
    if old == new:
        raise EditError("`old` and `new` are identical; nothing to change.")
    raw = target.read_bytes()
    bom = raw.startswith(BOM)
    try:
        text = raw[len(BOM) if bom else 0 :].decode("utf-8")
    except UnicodeDecodeError as error:
        raise EditError(f"{target} is not UTF-8 text ({error.reason} at byte {error.start}); not edited.") from None
    eol = "\r\n" if "\r\n" in text else "\n"
    old_n, new_n = (s.replace("\r\n", "\n").replace("\n", eol) for s in (old, new))
    count = text.count(old_n)
    if count == 0:
        raise EditError(f"`old` was not found in {target}.{_closest_line(text, old_n)} Copy the text exactly.")
    if count > 1 and not replace_all:
        at = [text.count("\n", 0, i) + 1 for i in _positions(text, old_n)]
        lines = ", ".join(map(str, at[:10])) + (", ..." if count > 10 else "")
        raise EditError(
            f"`old` occurs {count} times in {target} (lines {lines}). "
            "Include more surrounding text to make it unique, or set replace_all."
        )
    start = text.index(old_n)
    result = text.replace(old_n, new_n) if replace_all else text.replace(old_n, new_n, 1)
    if text.endswith(eol) and result and not result.endswith(eol):
        result += eol
    elif not text.endswith(eol) and result.endswith(eol):
        result = result[: -len(eol)]
    _write_atomic(target, (BOM if bom else b"") + result.encode("utf-8"))
    shown = result.replace("\r\n", "\n")
    offset = len(result[:start].replace("\r\n", "\n"))
    first = " (first shown)" if count > 1 else ""
    header = f"{target}: {count} replacement{'s' if count > 1 else ''}{first}"
    return f"{header}\n{_excerpt(shown, offset, len(new.replace('\r\n', '\n')))}"


def _positions(text: str, needle: str) -> list[int]:
    found, index = [], text.find(needle)
    while index != -1:
        found.append(index)
        index = text.find(needle, index + len(needle))
    return found


def _write_atomic(target: Path, data: bytes) -> None:
    """Write through a temp file in the same folder and swap it in, keeping the old file's mode."""
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


class EditTool:
    """Replaces exact text in one file, or creates a file."""

    name = "edit"
    description = (
        "Replace exact text in a file. `old` must match exactly once (whitespace included) unless `replace_all` "
        "is true. Write `\\n` line endings: CRLF files, a UTF-8 BOM and the final newline are kept as they are. "
        "An empty `old` creates a new file (with parent folders) containing `new`; it fails if the file exists. "
        "Relative paths are from the working folder."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace; empty to create a new file."},
            "new": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old", "new"],
    }

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        path = tool_input.get("path")
        if not isinstance(path, str) or not path:
            return None
        return os.path.normcase(str(resolve(path, ctx.cwd)))

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path, old, new = (tool_input.get(key) for key in ("path", "old", "new"))
        if not (isinstance(path, str) and path and isinstance(old, str) and isinstance(new, str)):
            return ToolOutput("`path` (non-empty), `old` and `new` must be strings.", is_error=True)
        target = resolve(path, ctx.cwd)
        try:
            message = await asyncio.to_thread(apply_edit, target, old, new, tool_input.get("replace_all") is True)
        except (EditError, OSError) as error:
            return ToolOutput(str(error), is_error=True)
        return ToolOutput(message)
