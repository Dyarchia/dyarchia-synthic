"""The ``read`` and ``grep`` tools: exact, numbered views of files, with less noise than ``sh``.

Both go through the agent's sandbox, so under the Docker environment they see
the container's files. Their results pass through the context's output cap like
every tool result: a long one is cut to head and tail and kept whole under a
receipt, which the ``receipt`` tool can search or slice.

``grep`` lists, inside a git work tree, only the files git does not ignore
(tracked and untracked), and everywhere skips ``.git``, binary files (a NUL
byte in the first ``BINARY_SNIFF`` bytes) and files over ``MAX_FILE_BYTES``.
"""

from __future__ import annotations

import asyncio
import posixpath
import re
from collections.abc import Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ..sandbox import of
from ..types import Location, ToolContext, ToolOutput

MAX_FILE_BYTES = 2 * 1024 * 1024
BINARY_SNIFF = 8192
MAX_LINE_CHARS = 300
DEFAULT_MATCHES = 100
MAX_MATCHES = 1000
_RANGE = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?")


def _binary(data: bytes) -> bool:
    return b"\0" in data[:BINARY_SNIFF]


def _lines(text: str) -> list[str]:
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return [line.removesuffix("\r") for line in parts]


def display(location: Location, cwd: Path, sandbox: Any) -> str:
    """How a location is named in results: relative to the working folder when inside it, else as shown."""
    base = sandbox.locate(".", cwd).shown
    shown = location.shown
    if shown == base:
        return "."
    for separator in ("/", "\\"):
        if shown.startswith(base.rstrip(separator) + separator):
            return shown[len(base.rstrip(separator)) + 1 :].replace("\\", "/")
    return shown.replace("\\", "/") if location.host is not None and sandbox.kind == "host" else shown


class ReadTool:
    name = "read"
    description = (
        "Read a text file, or the line range `lines` (\"120-180\" or \"120\"), with line numbers. A long result is "
        "cut and kept under a receipt. Relative paths are from the working folder. For images and PDFs use `view`."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "lines": {"type": "string", "description": "1-based inclusive line range, such as 120-180."},
        },
        "required": ["path"],
    }

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        path, spec = tool_input.get("path"), tool_input.get("lines")
        if not isinstance(path, str) or not path:
            return ToolOutput("`path` must be a non-empty string.", is_error=True)
        sandbox = of(ctx.sandbox)
        location = sandbox.locate(path, ctx.cwd)
        name = display(location, ctx.cwd, sandbox)
        try:
            data = await sandbox.read(location)
        except IsADirectoryError:
            return ToolOutput(f"{name} is a folder: use `grep` or `sh` to look inside it.", is_error=True)
        except OSError as error:
            return ToolOutput(f"{name}: cannot be read: {error.strerror or error}", is_error=True)
        if _binary(data):
            return ToolOutput(f"{name} is a binary file ({len(data):,} bytes): use `view` or `sh`.", is_error=True)
        lines = _lines(data.decode("utf-8", "replace").removeprefix("﻿"))
        total = len(lines)
        first, last = 1, total
        if spec is not None:
            match = _RANGE.fullmatch(str(spec))
            if not match:
                return ToolOutput(f"Invalid line range {spec!r}: use \"120-180\" or \"120\".", is_error=True)
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else first
            if first < 1 or last < first:
                return ToolOutput(f"Invalid line range {spec!r}: lines start at 1, end >= start.", is_error=True)
            if first > total:
                return ToolOutput(f"{name} has only {total:,} lines.", is_error=True)
            last = min(last, total)
        if total == 0:
            return ToolOutput(f"{name}: empty file.")
        body = "\n".join(f"{n}: {lines[n - 1]}" for n in range(first, last + 1))
        return ToolOutput(f"{name}, lines {first}-{last} of {total:,}:\n{body}")


class GrepTool:
    name = "grep"
    description = (
        "Search files for a Python regular expression and list the matching lines as `path:line: text`. `path` is "
        "a folder (default: the working folder) or a file; `glob` keeps only matching file names or paths (such as "
        "`*.cls` or `force-app/**`); `ignore_case`; `limit` caps the matches (default 100, at most 1000). Files git "
        "ignores, binary files and files over 2 MB are skipped."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
            "glob": {"type": "string"},
            "ignore_case": {"type": "boolean"},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["pattern"],
    }

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        pattern, path = tool_input.get("pattern"), tool_input.get("path") or "."
        glob, limit = tool_input.get("glob"), tool_input.get("limit", DEFAULT_MATCHES)
        if not isinstance(pattern, str) or not pattern or not isinstance(path, str):
            return ToolOutput("`pattern` must be a non-empty string, and `path` a string.", is_error=True)
        if glob is not None and not isinstance(glob, str):
            return ToolOutput("`glob` must be a string.", is_error=True)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            return ToolOutput("`limit` must be a positive integer.", is_error=True)
        try:
            regex = re.compile(pattern, re.IGNORECASE if tool_input.get("ignore_case") is True else 0)
        except re.error as error:
            return ToolOutput(f"Invalid regular expression {pattern!r}: {error}.", is_error=True)
        sandbox = of(ctx.sandbox)
        location = sandbox.locate(path, ctx.cwd)
        name = display(location, ctx.cwd, sandbox)
        try:
            if not await sandbox.exists(location):
                return ToolOutput(f"{name} does not exist.", is_error=True)
            files = await sandbox.files(location, MAX_FILE_BYTES)
        except OSError as error:
            return ToolOutput(f"{name}: cannot be searched: {error}", is_error=True)
        text = await asyncio.to_thread(_search, files, regex, name, glob, min(limit, MAX_MATCHES))
        return ToolOutput(text)


def _search(
    files: Iterator[tuple[str, bytes | None]], regex: re.Pattern[str], base: str, glob: str | None, limit: int
) -> str:
    """The result text of one search; reading stops at the limit."""
    out: list[str] = []
    searched = skipped = matched_files = 0
    full = False
    for rel, data in files:
        shown = posixpath.join(base, rel) if rel and base != "." else rel or base
        if glob and not (fnmatchcase(shown, glob) or fnmatchcase(posixpath.basename(shown), glob)):
            continue
        if data is None or _binary(data):
            skipped += 1
            continue
        searched += 1
        hit = False
        for number, line in enumerate(_lines(data.decode("utf-8", "replace")), 1):
            if not regex.search(line):
                continue
            if len(out) >= limit:
                full = True
                break
            hit = True
            cut = line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + "..."
            out.append(f"{shown}:{number}: {cut}")
        matched_files += hit
        if full:
            break
    skipped_note = f", {skipped} skipped (binary or over 2 MB)" if skipped else ""
    if not out:
        return f"No match for {regex.pattern!r} in {searched} files{skipped_note}."
    if full:
        footer = f"[stopped at {limit} matches: narrow the pattern, the path or the glob, or raise `limit`]"
    else:
        footer = f"[{len(out)} matches in {matched_files} files; {searched} files searched{skipped_note}]"
    return "\n".join([*out, footer])
