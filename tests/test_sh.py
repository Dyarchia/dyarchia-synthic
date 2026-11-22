from __future__ import annotations

import asyncio
import time
from pathlib import Path

from test_proc import (
    SETTINGS,
    async_wait_for_pids,
    py_command,
    survivors,
    tree_command,
    uses_powershell,
    write_script,
)

from synthic.tools.jobs import JobManager, JobTool
from synthic.tools.sh import ShTool
from synthic.types import ToolContext


def make_ctx(cwd: Path, timeout: float = 30.0) -> ToolContext:
    return ToolContext(agent_id="test", cwd=cwd, env={}, timeout_seconds=timeout, cancel=asyncio.Event())


def make_tools(tmp_path: Path) -> tuple[JobManager, ShTool, JobTool]:
    jobs = JobManager(tmp_path / "jobs")
    return jobs, ShTool(SETTINGS, jobs), JobTool(jobs)


def test_foreground_output_and_exit_code(tmp_path: Path) -> None:
    script = write_script(
        tmp_path,
        "fail.py",
        "import sys\nprint('out line')\nprint('err line', file=sys.stderr)\nsys.exit(3)\n",
    )

    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        try:
            result = await sh.run({"command": py_command(script)}, make_ctx(tmp_path))
        finally:
            await jobs.close()
        assert "out line" in result.text and "err line" in result.text
        assert "\r\n" not in result.text
        assert result.text.rstrip().splitlines()[-1].startswith("[exit code ")
        assert result.meta["exit_code"] != 0
        if not uses_powershell():
            assert result.meta["exit_code"] == 3
            assert result.text.endswith("[exit code 3]")
        assert result.meta["seconds"] >= 0
        assert not result.is_error

    asyncio.run(scenario())


def test_success_has_no_exit_line_and_env_overlay(tmp_path: Path) -> None:
    script = write_script(tmp_path, "ok.py", "import os\nprint('value', os.environ['SYNTHIC_SH_PROBE'])\n")

    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        ctx = make_ctx(tmp_path)
        ctx.env["SYNTHIC_SH_PROBE"] = "42"
        try:
            result = await sh.run({"command": py_command(script)}, ctx)
        finally:
            await jobs.close()
        assert result.text == "value 42\n"
        assert result.meta["exit_code"] == 0

    asyncio.run(scenario())


def test_stdin_does_not_hang(tmp_path: Path) -> None:
    script = write_script(tmp_path, "read.py", "import sys\nprint('read', len(sys.stdin.read()))\n")

    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        try:
            result = await sh.run({"command": py_command(script)}, make_ctx(tmp_path, timeout=20))
        finally:
            await jobs.close()
        assert result.text == "read 0\n"
        assert "job" not in result.meta

    asyncio.run(scenario())


def test_large_output_does_not_deadlock(tmp_path: Path) -> None:
    size = 6 * 1024 * 1024
    script = write_script(
        tmp_path,
        "big.py",
        "import sys\nline = b'x' * 1023 + b'\\n'\n"
        f"for _ in range({size // 1024}):\n    sys.stdout.buffer.write(line)\n",
    )

    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        try:
            result = await sh.run({"command": py_command(script)}, make_ctx(tmp_path, timeout=30))
        finally:
            await jobs.close()
        assert result.meta["exit_code"] == 0
        assert len(result.text) == size

    asyncio.run(scenario())


def test_non_utf8_output_is_replaced(tmp_path: Path) -> None:
    script = write_script(
        tmp_path, "bytes.py", "import sys\nsys.stdout.buffer.write('caf\\u00e9 '.encode() + b'\\xff\\r\\n')\n"
    )

    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        try:
            result = await sh.run({"command": py_command(script)}, make_ctx(tmp_path))
        finally:
            await jobs.close()
        assert result.text == "caf\u00e9 \ufffd\n"

    asyncio.run(scenario())


def test_timeout_hands_off_to_background_job(tmp_path: Path) -> None:
    script = write_script(
        tmp_path,
        "slow.py",
        "import time\nprint('first', flush=True)\ntime.sleep(3)\nprint('second', flush=True)\n",
    )

    async def scenario() -> None:
        jobs, sh, job_tool = make_tools(tmp_path)
        ctx = make_ctx(tmp_path, timeout=30)
        try:
            handed = await sh.run({"command": py_command(script), "timeout": 1.5}, ctx)
            job_id = handed.meta["job"]
            assert handed.meta["exit_code"] is None
            assert f"as job {job_id}" in handed.text
            before = handed.text.rsplit("[still running", 1)[0]
            assert jobs.drain_finished() == []

            job = jobs.get(job_id)
            assert job is not None
            await asyncio.wait_for(job.done.wait(), 20)
            status = await job_tool.run({"action": "status", "job": job_id}, ctx)
            assert f"[{job_id} exited with code 0" in status.text
            after = status.text.rsplit(f"[{job_id} ", 1)[0]
            assert (before + after).replace("(no new output)\n", "") == "first\nsecond\n"

            finished = jobs.drain_finished()
            assert [job.id for job in finished] == [job_id]
            assert jobs.drain_finished() == []

            again = await job_tool.run({"action": "status", "job": job_id}, ctx)
            assert again.text.startswith("(no new output)")
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_job_wait_returns_when_the_job_ends(tmp_path: Path) -> None:
    script = write_script(tmp_path, "short.py", "import time\ntime.sleep(1)\nprint('end')\n")

    async def scenario() -> None:
        jobs, sh, job_tool = make_tools(tmp_path)
        ctx = make_ctx(tmp_path, timeout=30)
        try:
            job_id = (await sh.run({"command": py_command(script), "background": True}, ctx)).meta["job"]
            started = time.monotonic()
            waited = await job_tool.run({"action": "wait", "job": job_id, "seconds": 25}, ctx)
            assert time.monotonic() - started < 15
            assert waited.text.startswith("end\n")
            assert f"[{job_id} exited with code 0" in waited.text
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_cancel_during_background_start_kills_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        ctx = make_ctx(tmp_path)
        real_start = jobs.start

        async def start_then_interrupt(*args: object, **kwargs: object) -> object:
            job = await real_start(*args, **kwargs)
            ctx.cancel.set()
            return job

        jobs.start = start_then_interrupt
        try:
            result = await sh.run({"command": tree_command(tmp_path, "stay"), "background": True}, ctx)
            assert result.is_error and "cancelled" in result.text
            job = jobs.list()[0]
            assert job.state == "killed"
            assert not job.tree.alive()
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_start_failure_is_a_tool_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        result = await sh.run({"command": "exit 0"}, make_ctx(tmp_path / "missing"))
        assert result.is_error
        assert "Could not start" in result.text
        await jobs.close()

    asyncio.run(scenario())


def test_background_true_returns_at_once(tmp_path: Path) -> None:
    script = write_script(tmp_path, "sleepy.py", "import time\nprint('up', flush=True)\ntime.sleep(60)\n")

    async def scenario() -> None:
        jobs, sh, job_tool = make_tools(tmp_path)
        ctx = make_ctx(tmp_path)
        try:
            started = time.monotonic()
            result = await sh.run({"command": py_command(script), "background": True}, ctx)
            assert time.monotonic() - started < 5
            job_id = result.meta["job"]
            listed = await job_tool.run({"action": "list"}, ctx)
            assert listed.text.startswith(f"{job_id} running")
            killed = await job_tool.run({"action": "kill", "job": job_id}, ctx)
            assert f"{job_id} killed" in killed.text
            assert jobs.drain_finished() == []
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_cancel_kills_the_whole_tree(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        ctx = make_ctx(tmp_path, timeout=60)
        try:
            running = asyncio.create_task(sh.run({"command": tree_command(tmp_path, "stay")}, ctx))
            pids = await async_wait_for_pids(tmp_path)
            ctx.cancel.set()
            result = await asyncio.wait_for(running, 20)
            assert result.is_error
            assert "cancelled" in result.text
            assert survivors(pids) == []
            assert jobs.list()[0].state == "killed"
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_cancel_already_set_starts_nothing(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        ctx = make_ctx(tmp_path)
        ctx.cancel.set()
        result = await sh.run({"command": py_command(tmp_path / "missing.py")}, ctx)
        assert result.is_error
        assert jobs.list() == []
        await jobs.close()

    asyncio.run(scenario())


def test_task_cancellation_kills_the_tree(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        try:
            running = asyncio.create_task(sh.run({"command": tree_command(tmp_path, "stay")}, make_ctx(tmp_path)))
            pids = await async_wait_for_pids(tmp_path)
            running.cancel()
            try:
                await running
            except asyncio.CancelledError:
                pass
            assert survivors(pids) == []
        finally:
            await jobs.close()

    asyncio.run(scenario())


def test_empty_command_is_an_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        jobs, sh, _ = make_tools(tmp_path)
        result = await sh.run({"command": "  "}, make_ctx(tmp_path))
        assert result.is_error
        await jobs.close()

    asyncio.run(scenario())
