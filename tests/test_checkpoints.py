"""File checkpoints and ``synthic rewind``: edits, git snapshots around ``sh``, and what stays untouched."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from test_loop import done, go, kinds, project, records

from synthic.checkpoints import rewind
from synthic.cli import main
from synthic.journal import beat
from synthic.testing import reply, tool_use

EDITS = {"checkpoints": {"edits": True}}


def git(folder: Path, *args: str) -> str:
    done_ = subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=folder, capture_output=True)
    assert done_.returncode == 0, done_.stderr
    return done_.stdout.decode("utf-8")


def test_off_by_default(tmp_path: Path) -> None:
    runtime, _, _ = go(tmp_path, [reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "x"})), done()])
    assert "checkpoint" not in kinds(runtime)
    assert rewind(runtime.run_dir) == []


def test_edits_are_undone_back_to_a_step(tmp_path: Path) -> None:
    folder = project(tmp_path)
    (folder / "b.txt").write_bytes(b"original\r\n")
    runtime, result, _ = go(
        tmp_path,
        [
            reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "one\n"})),
            reply(tool_use("edit", {"path": "a.txt", "old": "one", "new": "two"})),
            reply(tool_use("edit", {"path": "b.txt", "old": "original", "new": "changed"})),
            done(),
        ],
        EDITS,
    )
    assert result.status == "delivered"
    marks = [r["data"] for r in records(runtime) if r["kind"] == "checkpoint"]
    assert [(m["step"], m["path"], "file" in m) for m in marks] == [
        (1, str((folder / "a.txt").resolve()), False),
        (2, str((folder / "a.txt").resolve()), True),
        (3, str((folder / "b.txt").resolve()), True),
    ]
    planned = rewind(runtime.run_dir, 1, apply=False)
    assert len(planned) == 2 and (folder / "a.txt").read_text(encoding="utf-8") == "two\n"
    rewind(runtime.run_dir, 1)
    assert (folder / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert (folder / "b.txt").read_bytes() == b"original\r\n"
    rewind(runtime.run_dir, 0)
    assert not (folder / "a.txt").exists()


def test_git_snapshots_undo_what_sh_changed_without_touching_the_index(tmp_path: Path) -> None:
    folder = project(tmp_path)
    if subprocess.run(["git", "init", "-q"], cwd=folder, capture_output=True).returncode != 0:
        pytest.skip("git is not available")
    (folder / "t.txt").write_bytes(b"tracked\n")
    (folder / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    (folder / "mixed.txt").write_bytes(b"a\nb\r\n")
    (folder / ".gitignore").write_text("ignored.log\n", encoding="utf-8")
    git(folder, "-c", "core.autocrlf=true", "add", "-A")
    git(folder, "commit", "-q", "-m", "start")
    runtime, result, _ = go(
        tmp_path,
        [
            reply(
                tool_use("sh", {"command": "echo changed > t.txt; echo new > c.txt; echo log > ignored.log; rm crlf.txt"})
            ),
            reply(tool_use("sh", {"command": "rm t.txt mixed.txt"})),
            reply(tool_use("sh", {"command": "true"})),
            done(),
        ],
        {"checkpoints": {"shell": True}},
    )
    assert result.status == "delivered"
    marks = [r["data"] for r in records(runtime) if r["kind"] == "checkpoint"]
    assert [m["step"] for m in marks] == [1, 2, 3] and all(m["kind"] == "git" for m in marks)
    assert git(folder, "diff", "--cached", "--name-only") == "" and git(folder, "stash", "list") == ""
    lines = rewind(runtime.run_dir, 1)
    assert lines == [
        f"root step 3: nothing differs from tree {marks[2]['tree'][:12]}",
        "root step 2: restore mixed.txt",
        "root step 2: restore t.txt",
    ]
    assert (folder / "t.txt").read_bytes().strip() == b"changed" and (folder / "c.txt").exists()
    rewind(runtime.run_dir, 0)
    assert (folder / "t.txt").read_bytes() == b"tracked\n"
    assert (folder / "crlf.txt").read_bytes() == b"one\r\ntwo\r\n" and (folder / "mixed.txt").read_bytes() == b"a\nb\r\n"
    assert not (folder / "c.txt").exists() and (folder / "ignored.log").exists()
    assert not (runtime.run_dir / "root" / "checkpoints" / "rewind-now").exists()


def test_a_run_folder_inside_the_work_tree_is_never_captured_or_deleted(tmp_path: Path) -> None:
    folder = project(tmp_path)
    if subprocess.run(["git", "init", "-q"], cwd=folder, capture_output=True).returncode != 0:
        pytest.skip("git is not available")
    (folder / "t.txt").write_text("tracked\n", encoding="utf-8")
    git(folder, "add", "-A")
    git(folder, "commit", "-q", "-m", "start")
    task = {"checkpoints": {"shell": True}, "journal": {"dir": str(folder / "runs")}}
    runtime, _, _ = go(tmp_path, [reply(tool_use("sh", {"command": "echo x > new.txt"})), done()], task)
    tree = next(r["data"]["tree"] for r in records(runtime) if r["kind"] == "checkpoint")
    assert "runs" not in git(folder, "ls-tree", "-r", "--name-only", tree)
    rewind(runtime.run_dir, 0)
    assert not (folder / "new.txt").exists() and (runtime.run_dir / "root" / "journal.jsonl").exists()


def test_cli_rewind_dry_run_and_refusals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _, _ = go(tmp_path, [reply(tool_use("edit", {"path": "a.txt", "old": "", "new": "x"})), done()], EDITS)
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SYNTHIC_ORG_POLICY", raising=False)
    policy = tmp_path / "policy.toml"
    policy.write_text(f'[journal]\ndir = "{(tmp_path / "runs").as_posix()}"\n', encoding="utf-8")
    base = ["--policy", str(policy), "--cwd", str(tmp_path)]
    assert main(["rewind", runtime.run_id, "--dry-run", *base]) == 0
    assert "Would do:\n  root step 1: delete" in capsys.readouterr().out
    assert (project(tmp_path) / "a.txt").exists()
    assert main(["show", runtime.run_id, *base]) == 0
    assert "checkpoint step 1: edit" in capsys.readouterr().out
    beat(runtime.run_dir)
    assert main(["rewind", runtime.run_id, *base]) == 2
    (runtime.run_dir / "alive").unlink()
    assert main(["rewind", runtime.run_id, "--agent", "zz", *base]) == 2
    assert main(["rewind", runtime.run_id, *base]) == 0
    assert not (project(tmp_path) / "a.txt").exists()
