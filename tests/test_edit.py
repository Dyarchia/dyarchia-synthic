from __future__ import annotations

import asyncio
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from synthic.tools.edit import EditTool
from synthic.types import ToolContext, ToolOutput


def make_ctx(cwd: Path) -> ToolContext:
    return ToolContext(agent_id="test", cwd=cwd, env={}, timeout_seconds=30.0, cancel=asyncio.Event())


def edit(cwd: Path, **tool_input: Any) -> ToolOutput:
    return asyncio.run(EditTool().run(tool_input, make_ctx(cwd)))


def test_unique_match_is_replaced(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_bytes(b"one\ntwo\nthree\nfour\nfive\nsix\nseven\n")
    result = edit(tmp_path, path="a.py", old="four", new="FOUR\nFOUR-B")
    assert not result.is_error
    assert target.read_bytes() == b"one\ntwo\nthree\nFOUR\nFOUR-B\nfive\nsix\nseven\n"
    assert "1 replacement" in result.text
    shown = result.text.splitlines()[1:]
    assert [line.split(None, 1)[1] for line in shown] == ["two", "three", "FOUR", "FOUR-B", "five", "six"]
    assert shown[0].split()[0] == "2"


def test_no_match_reports_closest_line(tmp_path: Path) -> None:
    target = tmp_path / "a.py"
    target.write_bytes(b"def handler(event):\n    return event\n")
    result = edit(tmp_path, path="a.py", old="def handle(event):", new="x")
    assert result.is_error
    assert "not found" in result.text
    assert "line 1" in result.text and "def handler(event):" in result.text
    assert target.read_bytes() == b"def handler(event):\n    return event\n"


def test_multiple_matches_are_an_error_with_count_and_lines(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"x = 1\ny = 2\nx = 1\n")
    result = edit(tmp_path, path="a.txt", old="x = 1", new="x = 3")
    assert result.is_error
    assert "2 times" in result.text and "lines 1, 3" in result.text
    assert target.read_bytes() == b"x = 1\ny = 2\nx = 1\n"


def test_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"x = 1\ny = 2\nx = 1\n")
    result = edit(tmp_path, path="a.txt", old="x = 1", new="x = 3", replace_all=True)
    assert not result.is_error
    assert "2 replacements" in result.text
    assert target.read_bytes() == b"x = 3\ny = 2\nx = 3\n"


def test_create_with_parents_and_refuse_existing(tmp_path: Path) -> None:
    result = edit(tmp_path, path="deep/er/new.txt", old="", new="hello\n")
    assert not result.is_error
    assert (tmp_path / "deep" / "er" / "new.txt").read_bytes() == b"hello\n"
    again = edit(tmp_path, path="deep/er/new.txt", old="", new="other\n")
    assert again.is_error
    missing = edit(tmp_path, path="nope.txt", old="a", new="b")
    assert missing.is_error and "does not exist" in missing.text


def test_crlf_is_preserved(tmp_path: Path) -> None:
    target = tmp_path / "win.cls"
    target.write_bytes(b"public class A {\r\n    Integer x;\r\n}\r\n")
    result = edit(tmp_path, path="win.cls", old="{\n    Integer x;", new="{\n    Integer x;\n    Integer y;")
    assert not result.is_error
    assert target.read_bytes() == b"public class A {\r\n    Integer x;\r\n    Integer y;\r\n}\r\n"
    assert "\r" not in result.text


def test_bom_and_missing_final_newline_are_preserved(tmp_path: Path) -> None:
    target = tmp_path / "bom.txt"
    target.write_bytes(b"\xef\xbb\xbfalpha\nbeta")
    result = edit(tmp_path, path="bom.txt", old="beta", new="gamma\n")
    assert not result.is_error
    assert target.read_bytes() == b"\xef\xbb\xbfalpha\ngamma"


def test_final_newline_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    target.write_bytes(b"alpha\nbeta\n")
    edit(tmp_path, path="a.txt", old="beta\n", new="gamma")
    assert target.read_bytes() == b"alpha\ngamma\n"


def test_non_utf8_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "latin.txt"
    target.write_bytes(b"caf\xe9\n")
    result = edit(tmp_path, path="latin.txt", old="caf", new="tea")
    assert result.is_error and "UTF-8" in result.text
    assert target.read_bytes() == b"caf\xe9\n"


def test_identical_old_and_new_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"same\n")
    assert edit(tmp_path, path="a.txt", old="same", new="same").is_error


def test_absolute_path_and_no_temp_leftovers(tmp_path: Path) -> None:
    target = tmp_path / "abs.txt"
    target.write_bytes(b"old\n")
    result = edit(Path.cwd(), path=str(target), old="old", new="new")
    assert not result.is_error
    assert target.read_bytes() == b"new\n"
    assert [p.name for p in tmp_path.iterdir()] == ["abs.txt"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_file_mode_is_kept(tmp_path: Path) -> None:
    target = tmp_path / "run.sh"
    target.write_bytes(b"echo old\n")
    os.chmod(target, 0o755)
    edit(tmp_path, path="run.sh", old="old", new="new")
    assert stat.S_IMODE(target.stat().st_mode) == 0o755


def test_resource_is_equal_for_two_spellings(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.txt").write_bytes(b"x\n")
    tool = EditTool()
    ctx = make_ctx(tmp_path)
    relative = tool.resource({"path": "sub/../sub/f.txt"}, ctx)
    absolute = tool.resource({"path": str(tmp_path / "sub" / "f.txt")}, ctx)
    assert relative is not None and relative == absolute
    if sys.platform == "win32":
        assert tool.resource({"path": str(tmp_path / "SUB" / "F.TXT")}, ctx) == relative
    assert tool.resource({"path": "other.txt"}, ctx) != relative
