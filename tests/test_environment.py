from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from synthic.environment import Workspace
from synthic.types import EnvironmentSettings

COPY = EnvironmentSettings(mode="copy")


def make_plain(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "src").mkdir(parents=True)
    (root / "src" / "a.txt").write_text("alpha", encoding="utf-8")
    (root / "b.txt").write_text("beta", encoding="utf-8")
    return root


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
    return done.stdout


def test_in_place_returns_root(tmp_path: Path) -> None:
    root = make_plain(tmp_path)
    workspace = Workspace(EnvironmentSettings(), root, tmp_path / "state")

    async def scenario() -> None:
        assert await workspace.start("a1") == root
        assert await workspace.changes("a1") == ""
        await workspace.close("a1")
        assert root.exists()

    asyncio.run(scenario())


def test_plain_copy_isolated_and_removed(tmp_path: Path) -> None:
    root = make_plain(tmp_path)
    workspace = Workspace(COPY, root, tmp_path / "state")

    async def scenario() -> None:
        work = await workspace.start("a1")
        assert work == tmp_path / "state" / "work" / "a1"
        assert (work / "src" / "a.txt").read_text(encoding="utf-8") == "alpha"
        assert await workspace.changes("a1") == ""
        (work / "src" / "a.txt").write_text("alpha changed", encoding="utf-8")
        (work / "new.txt").write_text("new", encoding="utf-8")
        (work / "b.txt").unlink()
        assert (root / "src" / "a.txt").read_text(encoding="utf-8") == "alpha"
        lines = (await workspace.changes("a1")).splitlines()
        assert lines == ["D b.txt", "A new.txt", "M src/a.txt"]
        await workspace.close("a1")
        assert not work.exists()

    asyncio.run(scenario())


def test_keep_leaves_the_copy(tmp_path: Path) -> None:
    root = make_plain(tmp_path)
    workspace = Workspace(COPY, root, tmp_path / "state")

    async def scenario() -> None:
        work = await workspace.start("a1")
        await workspace.close("a1", keep=True)
        assert work.exists()
        await workspace.close("a1")
        assert not work.exists()

    asyncio.run(scenario())


def test_copy_skips_state_dir_inside_root(tmp_path: Path) -> None:
    root = make_plain(tmp_path)
    state = root / ".synthic"
    (state / "runs").mkdir(parents=True)
    (state / "runs" / "log.txt").write_text("x", encoding="utf-8")
    workspace = Workspace(COPY, root, state)

    async def scenario() -> None:
        work = await workspace.start("a1")
        assert (work / "b.txt").exists()
        assert not (work / ".synthic").exists()
        assert await workspace.changes("a1") == ""

    asyncio.run(scenario())


def test_missing_git_falls_back_to_plain_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_plain(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    workspace = Workspace(COPY, root, tmp_path / "state")

    async def scenario() -> None:
        work = await workspace.start("a1")
        assert (work / "b.txt").exists()
        assert not (work / ".git").exists()

    asyncio.run(scenario())


needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "sub").mkdir(parents=True)
    (repo / "sub" / "f.txt").write_text("one", encoding="utf-8")
    git(repo, "init")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "first")
    return repo


def commit_in(work: Path, name: str, text: str, message: str) -> None:
    (work / name).write_text(text, encoding="utf-8")
    git(work, "add", name)
    git(work, "-c", "user.name=Agent", "-c", "user.email=a@example.com", "commit", "-m", message)


@needs_git
def test_git_worktree_lifecycle(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = Workspace(COPY, repo / "sub", tmp_path / "state", "r1")

    async def scenario() -> None:
        work = await workspace.start("a1")
        dest = tmp_path / "state" / "work" / "a1"
        assert work == dest / "sub"
        assert (dest / ".git").exists()
        assert "synthic/r1/a1" in git(repo, "branch", "--list", "synthic/r1/a1")
        assert await workspace.changes("a1") == ""
        assert not await workspace.has_changes("a1")
        (work / "f.txt").write_text("two", encoding="utf-8")
        assert "f.txt" in await workspace.changes("a1")
        assert await workspace.has_changes("a1")
        assert (repo / "sub" / "f.txt").read_text(encoding="utf-8") == "one"
        await workspace.close("a1")
        assert not dest.exists()
        assert git(repo, "branch", "--list", "synthic/*").strip() == ""
        assert "synthic/r1/a1" not in git(repo, "worktree", "list")

    asyncio.run(scenario())


@needs_git
def test_second_run_starts_again_with_the_same_agent_id(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)

    async def one_run(run_id: str, state: str) -> None:
        workspace = Workspace(COPY, repo, tmp_path / state, run_id)
        await workspace.start("a1")
        commit_in(tmp_path / state / "work" / "a1", "n.txt", run_id, f"work of {run_id}")
        await workspace.close("a1")

    async def scenario() -> None:
        await one_run("r1", "s1")
        await one_run("r2", "s2")
        again = Workspace(COPY, repo, tmp_path / "s3")
        await again.start("a1")
        await again.close("a1")

    asyncio.run(scenario())
    branches = git(repo, "branch", "--list", "synthic/*")
    assert "synthic/r1/a1" in branches and "synthic/r2/a1" in branches


@needs_git
def test_committed_work_is_reported_and_its_branch_survives_close(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    workspace = Workspace(COPY, repo, tmp_path / "state", "r1")

    async def scenario() -> None:
        work = await workspace.start("a1")
        commit_in(work, "sub/f.txt", "two", "change f")
        commit_in(work, "g.txt", "g", "add g")
        (work / "h.txt").write_text("h", encoding="utf-8")
        report = await workspace.changes("a1")
        lines = report.splitlines()
        assert "synthic/r1/a1" in lines[0] and "2 commit" in lines[0]
        assert any(line.endswith("change f") for line in lines) and any(line.endswith("add g") for line in lines)
        assert "uncommitted:" in lines and "?? h.txt" in lines
        assert await workspace.has_changes("a1")
        await workspace.close("a1")
        assert not work.exists()
        assert "synthic/r1/a1" in git(repo, "branch", "--list", "synthic/r1/a1")
        assert "add g" in git(repo, "log", "--oneline", "synthic/r1/a1")

    asyncio.run(scenario())


@needs_git
def test_a_new_process_finds_the_copy_on_disk(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    state = tmp_path / "state"

    async def scenario() -> None:
        first = Workspace(COPY, repo / "sub", state, "r1")
        work = await first.start("a1")
        commit_in(work, "g.txt", "g", "add g")
        (work / "f.txt").write_text("dirty", encoding="utf-8")
        second = Workspace(COPY, repo / "sub", state, "r1")
        report = await second.changes("a1")
        assert "add g" in report and "f.txt" in report and "synthic/r1/a1" in report
        assert await second.has_changes("a1")
        await second.close("a1")
        assert not work.exists()
        assert "synthic/r1/a1" in git(repo, "branch", "--list", "synthic/r1/a1")
        assert await second.changes("a1") == ""

        clean = await Workspace(COPY, repo, state, "r1").start("a2")
        assert clean.exists()
        await Workspace(COPY, repo, state, "r1").close("a2")
        assert not clean.exists()
        assert "synthic/r1/a2" not in git(repo, "branch", "--list")

    asyncio.run(scenario())


def test_a_new_process_finds_a_plain_copy_on_disk(tmp_path: Path) -> None:
    root = make_plain(tmp_path)
    state = tmp_path / "state"

    async def scenario() -> None:
        work = await Workspace(COPY, root, state).start("a1")
        (work / "new.txt").write_text("new", encoding="utf-8")
        later = Workspace(COPY, root, state)
        assert await later.changes("a1") == "A new.txt"
        assert await later.has_changes("a1")
        await later.close("a1")
        assert not work.exists()

    asyncio.run(scenario())
