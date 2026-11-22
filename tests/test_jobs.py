from __future__ import annotations

import asyncio
import time
from pathlib import Path

from test_proc import SETTINGS, async_wait_for_pids, survivors, tree_command
from test_sh import make_ctx

from synthic.tools import builtin_tools, to_definition
from synthic.tools.jobs import JobManager, JobTool, _safe_end, decode


def test_kill_all_reaches_background_jobs_and_finished_commands(tmp_path: Path) -> None:
    running_dir, orphan_dir = tmp_path / "running", tmp_path / "orphan"
    running_dir.mkdir()
    orphan_dir.mkdir()

    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        try:
            background = await jobs.start(tree_command(running_dir, "stay"), tmp_path, {}, SETTINGS, background=True)
            finished = await jobs.start(tree_command(orphan_dir, "orphan"), tmp_path, {}, SETTINGS)
            pids = await async_wait_for_pids(running_dir) + await async_wait_for_pids(orphan_dir)
            await asyncio.wait_for(finished.done.wait(), 20)
            assert finished.state == "exited"
            assert background.state == "running"
            await jobs.kill_all()
            assert survivors(pids) == []
            assert background.state == "killed"
            assert jobs.drain_finished() == []
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_job_ids_and_lookup(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        try:
            first = await jobs.start("exit 0", tmp_path, {}, SETTINGS)
            second = await jobs.start("exit 0", tmp_path, {}, SETTINGS)
            assert (first.id, second.id) == ("j1", "j2")
            assert jobs.get("j2") is second
            assert jobs.get("j9") is None
            assert await jobs.kill("j9") is None
            assert [job.id for job in jobs.list()] == ["j1", "j2"]
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_job_tool_errors(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        tool = JobTool(jobs)
        ctx = make_ctx(tmp_path)
        unknown = await tool.run({"action": "status", "job": "j7"}, ctx)
        assert unknown.is_error and "j7" in unknown.text
        bad = await tool.run({"action": "restart", "job": "j1"}, ctx)
        assert bad.is_error
        empty = await tool.run({"action": "list"}, ctx)
        assert empty.text == "No jobs."
        assert tool.resource({"action": "list"}, ctx) is None
        await jobs.close()

    asyncio.run(scenario())


def test_job_wait_wakes_on_cancel(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        tool = JobTool(jobs)
        ctx = make_ctx(tmp_path, timeout=60)
        try:
            job = await jobs.start(tree_command(tmp_path, "stay"), tmp_path, {}, SETTINGS, background=True)
            waiting = asyncio.create_task(tool.run({"action": "wait", "job": job.id, "seconds": 60}, ctx))
            await asyncio.sleep(0.3)
            started = time.monotonic()
            ctx.cancel.set()
            result = await asyncio.wait_for(waiting, 5)
            assert time.monotonic() - started < 2
            assert result.is_error
            assert job.state == "running"
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_close_removes_its_own_folder(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager()
        folder = jobs.output_dir
        job = await jobs.start("echo hi", tmp_path, {}, SETTINGS)
        await asyncio.wait_for(job.done.wait(), 20)
        assert job.read()[0].strip() == "hi"
        assert job.tail(100).strip() == "hi"
        await jobs.close()
        assert not folder.exists()

    asyncio.run(scenario())


def test_tail_starts_at_a_line(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        job = await jobs.start("echo alpha; echo beta; echo gamma", tmp_path, {}, SETTINGS)
        await asyncio.wait_for(job.done.wait(), 20)
        assert job.tail(9).split() == ["gamma"]
        assert job.read(len("alpha\n"))[0].split()[0] == "beta"
        await jobs.close()

    asyncio.run(scenario())


def test_read_cursor_never_splits_a_character_or_crlf() -> None:
    euro = "€".encode()
    assert _safe_end(b"ab" + euro[:1]) == 2
    assert _safe_end(b"ab" + euro[:2]) == 2
    assert _safe_end(b"ab" + euro) == 5
    assert _safe_end(b"line\r") == 4
    assert _safe_end(b"line\r\n") == 6
    assert _safe_end(b"") == 0
    assert decode(b"a\r\nb\xff") == "a\nb�"


def test_builtin_tools_and_definitions(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs = JobManager(tmp_path / "jobs")
        tools = builtin_tools(SETTINGS, jobs)
        assert set(tools) == {"sh", "edit", "job"}
        for name, tool in tools.items():
            definition = to_definition(tool)
            assert definition["name"] == name
            assert set(definition) == {"name", "description", "input_schema"}
            assert definition["input_schema"]["type"] == "object"
        assert tools["sh"].jobs is tools["job"].jobs is jobs
        await jobs.close()

    asyncio.run(scenario())
