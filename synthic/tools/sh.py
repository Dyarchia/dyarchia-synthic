"""The ``sh`` tool: run a shell command, handing it to the background when it outlives its limit."""

from __future__ import annotations

import asyncio
from typing import Any

from ..types import EnvironmentSettings, ToolContext, ToolOutput
from .jobs import JobManager, wait_job, with_line


class ShTool:
    """Runs one command line through the policy's shell as a job of ``jobs``."""

    name = "sh"
    description = (
        "Run a shell command in the working folder. stdout and stderr are merged; stdin is closed, so never "
        "wait for input. Returns the output, then `[exit code N]` when N is not 0. If the command is still "
        "running when the time limit passes (`timeout` seconds, never more than the tool limit), it is NOT "
        "stopped: it keeps running as a background job and you get its output so far and its id; follow it "
        "with the `job` tool. Set `background: true` to get the job id at once (servers, watchers, long builds)."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "number", "description": "Seconds to wait before handing off to the background."},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
    }

    def __init__(self, settings: EnvironmentSettings, jobs: JobManager) -> None:
        self.settings = settings
        self.jobs = jobs

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        command = tool_input.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolOutput("`command` must be a non-empty string.", is_error=True)
        if ctx.cancel.is_set():
            return ToolOutput("[cancelled before start]", is_error=True, meta={"exit_code": None, "seconds": 0.0})
        background = tool_input.get("background") is True
        try:
            job = await self.jobs.start(command, ctx.cwd, ctx.env, self.settings, background=background)
        except OSError as error:
            return ToolOutput(f"Could not start the command: {error}", is_error=True)
        if ctx.cancel.is_set():
            await self.jobs.kill(job.id)
            text = "[cancelled: the command was killed as it started]"
            return ToolOutput(text, is_error=True, meta={"exit_code": None, "seconds": 0.0})
        if background:
            text = f"[started as job {job.id}; check it with the job tool (status, wait or kill)]"
            return ToolOutput(text, meta={"exit_code": None, "seconds": 0.0, "job": job.id})
        limit = ctx.timeout_seconds
        timeout = tool_input.get("timeout")
        if isinstance(timeout, int | float) and not isinstance(timeout, bool) and timeout > 0:
            limit = min(float(timeout), limit)
        try:
            outcome = await wait_job(job, limit, ctx.cancel)
        except asyncio.CancelledError:
            await asyncio.shield(self.jobs.kill(job.id))
            raise
        match outcome:
            case "done":
                text = job.take_new()
                if job.exit_code:
                    text = with_line(text, f"[exit code {job.exit_code}]")
                self.jobs.discard_output(job)
                meta = {"exit_code": job.exit_code, "seconds": round(job.seconds, 3)}
                return ToolOutput(text or "(no output)", meta=meta)
            case "timeout":
                job.background = True
                note = (
                    f"[still running after {limit:g}s as job {job.id}; "
                    f"use the job tool (status, wait or kill) with job {job.id}]"
                )
                meta = {"exit_code": None, "seconds": round(job.seconds, 3), "job": job.id}
                return ToolOutput(with_line(job.take_new(), note), meta=meta)
            case "cancel":
                await self.jobs.kill(job.id)
                text = with_line(job.take_new(), "[cancelled: the command and every process it started were killed]")
                self.jobs.discard_output(job)
                return ToolOutput(text, is_error=True, meta={"exit_code": None, "seconds": round(job.seconds, 3)})
