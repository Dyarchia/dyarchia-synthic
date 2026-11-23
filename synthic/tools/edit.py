"""The ``edit`` tool: exact text replacement that keeps the file's encoding details intact."""

from __future__ import annotations

import asyncio
import difflib
import os
from typing import Any

from ..sandbox import of
from ..types import ToolContext, ToolOutput

BOM = b"\xef\xbb\xbf"
CONTEXT_LINES = 2


class EditError(Exception):
    """A call the model has to fix; the message tells it how."""


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


def edit_bytes(raw: bytes | None, old: str, new: str, replace_all: bool, target: str) -> tuple[bytes, str]:
    """The new content of a file whose content is ``raw`` (``None``: it does not exist) and the success message.

    Raises ``EditError``. ``target`` is the path as the agent sees it, for the messages.
    """
    if raw is None:
        if old:
            raise EditError(f"{target} does not exist. To create it, call with an empty `old`.")
        return new.encode("utf-8"), f"Created {target}."
    if not old:
        raise EditError(f"{target} already exists. Give the exact text to replace in `old`.")
    if old == new:
        raise EditError("`old` and `new` are identical; nothing to change.")
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
    shown = result.replace("\r\n", "\n")
    offset = len(result[:start].replace("\r\n", "\n"))
    first = " (first shown)" if count > 1 else ""
    header = f"{target}: {count} replacement{'s' if count > 1 else ''}{first}"
    message = f"{header}\n{_excerpt(shown, offset, len(new.replace('\r\n', '\n')))}"
    return (BOM if bom else b"") + result.encode("utf-8"), message


def _positions(text: str, needle: str) -> list[int]:
    found, index = [], text.find(needle)
    while index != -1:
        found.append(index)
        index = text.find(needle, index + len(needle))
    return found


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
        location = of(ctx.sandbox).locate(path, ctx.cwd)
        return os.path.normcase(str(location.host)) if location.host is not None else location.shown

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path, old, new = (tool_input.get(key) for key in ("path", "old", "new"))
        if not (isinstance(path, str) and path and isinstance(old, str) and isinstance(new, str)):
            return ToolOutput("`path` (non-empty), `old` and `new` must be strings.", is_error=True)
        sandbox = of(ctx.sandbox)
        location = sandbox.locate(path, ctx.cwd)
        try:
            try:
                raw: bytes | None = await sandbox.read(location)
            except FileNotFoundError:
                raw = None
            data, message = await asyncio.to_thread(
                edit_bytes, raw, old, new, tool_input.get("replace_all") is True, location.shown
            )
            if ctx.before_write is not None:
                ctx.before_write(location, raw)
            await sandbox.write(location, data)
        except (EditError, OSError) as error:
            return ToolOutput(str(error), is_error=True)
        return ToolOutput(message)
