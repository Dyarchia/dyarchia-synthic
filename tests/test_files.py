"""The ``read`` and ``grep`` tools, on the host, through a loop, and inside the fake Docker sandbox."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from test_loop import done, go, project, results
from test_sandbox import docker_task

from synthic.testing import reply, tool_use
from synthic.tools.files import GrepTool, ReadTool
from synthic.types import ToolContext, ToolOutput


def ctx(cwd: Path) -> ToolContext:
    return ToolContext("t", cwd, {}, 30.0, asyncio.Event())


def call(tool: ReadTool | GrepTool, cwd: Path, **tool_input: object) -> ToolOutput:
    return asyncio.run(tool.run(dict(tool_input), ctx(cwd)))


def test_read_numbers_lines_and_slices(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"\xef\xbb\xbfone\r\ntwo\r\nthree\r\n")
    whole = call(ReadTool(), tmp_path, path="a.txt")
    assert whole.text == "a.txt, lines 1-3 of 3:\n1: one\n2: two\n3: three"
    assert call(ReadTool(), tmp_path, path="a.txt", lines="2-9").text == "a.txt, lines 2-3 of 3:\n2: two\n3: three"
    assert call(ReadTool(), tmp_path, path="a.txt", lines="3").text.endswith("3: three")
    assert "only 3 lines" in call(ReadTool(), tmp_path, path="a.txt", lines="4").text
    assert call(ReadTool(), tmp_path, path="a.txt", lines="x").is_error
    (tmp_path / "bin.dat").write_bytes(b"\x00\x01")
    assert "binary" in call(ReadTool(), tmp_path, path="bin.dat").text
    assert "cannot be read" in call(ReadTool(), tmp_path, path="missing.txt").text
    assert "is a folder" in call(ReadTool(), tmp_path, path=".").text or call(ReadTool(), tmp_path, path=".").is_error


def test_grep_lists_matches_with_limits(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "A.cls").write_text("public class A {\n    // TODO fix\n}\n", encoding="utf-8")
    (tmp_path / "src" / "b.js").write_text("const todo = 1;\n" + "x" * 400 + " TODO\n", encoding="utf-8")
    (tmp_path / "blob.bin").write_bytes(b"TODO\x00")
    result = call(GrepTool(), tmp_path, pattern="TODO")
    lines = result.text.splitlines()
    assert lines[:2] == ["src/A.cls:2:     // TODO fix", f"src/b.js:2: {'x' * 300}..."]
    assert lines[-1] == "[2 matches in 2 files; 2 files searched, 1 skipped (binary or over 2 MB)]"
    assert call(GrepTool(), tmp_path, pattern="todo", ignore_case=True, glob="*.js").text.count("src/b.js") == 2
    limited = call(GrepTool(), tmp_path, pattern="TODO", limit=1, ignore_case=True).text.splitlines()
    assert len(limited) == 2 and limited[-1].startswith("[stopped at 1 matches")
    assert call(GrepTool(), tmp_path, pattern="TODO", path="src/A.cls").text.startswith("src/A.cls:2:")
    assert call(GrepTool(), tmp_path, pattern="nothing").text.startswith("No match for 'nothing' in 2 files, 1 skipped")
    assert call(GrepTool(), tmp_path, pattern="(").is_error
    assert call(GrepTool(), tmp_path, pattern="x", path="nope").text == "nope does not exist."


def test_grep_respects_gitignore(tmp_path: Path) -> None:
    if subprocess.run(["git", "init", "-q", str(tmp_path)], capture_output=True).returncode != 0:
        return
    (tmp_path / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "dep.js").write_text("needle\n", encoding="utf-8")
    (tmp_path / "mine.js").write_text("needle\n", encoding="utf-8")
    assert call(GrepTool(), tmp_path, pattern="needle").text.splitlines()[0] == "mine.js:1: needle"
    assert "node_modules" not in call(GrepTool(), tmp_path, pattern="needle").text


def test_read_and_grep_through_the_loop_and_their_permissions(tmp_path: Path) -> None:
    (project(tmp_path) / "notes.md").write_text("alpha\nbeta\n", encoding="utf-8")
    _, result, provider = go(
        tmp_path,
        [
            reply(tool_use("read", {"path": "notes.md"}), tool_use("grep", {"pattern": "beta"})),
            reply(tool_use("read", {"path": "secret.env"})),
            done(),
        ],
        {"role": {"lead": {"tools": ["read", "grep"]}}, "permissions": {"read: *.env": "deny"}},
    )
    found = sorted(block["content"] for block in results(provider.requests[1]))
    assert found[0].endswith("2: beta") and found[1].startswith("notes.md:2: beta")
    assert "Denied" in results(provider.requests[2])[0]["content"] and result.status == "delivered"


def test_read_and_grep_inside_the_fake_container(tmp_path: Path) -> None:
    state = tmp_path / "state"
    (project(tmp_path) / "src.txt").write_text("in the mount\n", encoding="utf-8")
    task = docker_task(state)
    task["role"] = {"lead": {"tools": ["read", "grep", "edit"]}}
    _, result, provider = go(
        tmp_path,
        [
            reply(tool_use("edit", {"path": "/opt/inner.txt", "old": "", "new": "only inside\n"})),
            reply(tool_use("read", {"path": "/opt/inner.txt"}), tool_use("grep", {"pattern": "mount"})),
            done(),
        ],
        task,
    )
    contents = sorted(block["content"] for block in results(provider.requests[2]))
    assert contents[0] == "/opt/inner.txt, lines 1-1 of 1:\n1: only inside"
    assert contents[1].startswith("src.txt:1: in the mount") and result.status == "delivered"
