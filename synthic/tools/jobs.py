"""Background work: every process the agent starts is a job owned by one ``JobManager``.

``sh`` starts each command as a job. A command that ends within its time limit
is answered in the foreground; one that does not stays running as a background
job the model follows with the ``job`` tool. ``kill_all`` ends every tree,
including what finished commands left running, so an interrupt or the end of
the agent leaves nothing behind.
"""

from __future__ import annotations

import asyncio
import itertools
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .. import proc
from ..sandbox import job_token, of
from ..types import EnvironmentSettings, Sandbox, ToolContext, ToolOutput

JobState = Literal["running", "exited", "killed"]
WaitOutcome = Literal["done", "timeout", "cancel"]


def decode(data: bytes) -> str:
    """Process output as text: UTF-8 with replacement, CRLF folded to LF."""
    return data.decode("utf-8", "replace").replace("\r\n", "\n")


def _safe_end(data: bytes) -> int:
    """Length of the prefix of ``data`` that ends on a whole UTF-8 character and not between CR and LF.

    Used while a job still writes, so a read cursor never splits a character
    or a line ending that the next read would then decode wrongly.
    """
    end = len(data)
    if data.endswith(b"\r"):
        end -= 1
    lead = end - 1
    while lead >= 0 and end - lead <= 3 and data[lead] & 0xC0 == 0x80:
        lead -= 1
    if lead >= 0 and data[lead] >= 0xC0:
        size = 2 if data[lead] < 0xE0 else 3 if data[lead] < 0xF0 else 4
        if end - lead < size:
            end = lead
    return end


@dataclass(eq=False)
class Job:
    """One started command. ``cursor`` is the byte offset the model has already seen."""

    id: str
    command: str
    started: float
    output_path: Path
    tree: proc.ProcessTree
    background: bool = False
    state: JobState = "running"
    exit_code: int | None = None
    ended: float | None = None
    cursor: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    settled: bool = False

    @property
    def seconds(self) -> float:
        return (self.ended or time.time()) - self.started

    def read(self, offset: int = 0) -> tuple[str, int]:
        """Output from byte ``offset`` on, and the offset to continue from next time."""
        try:
            with open(self.output_path, "rb") as source:
                source.seek(offset)
                data = source.read()
        except FileNotFoundError:
            return "", offset
        end = len(data) if self.state != "running" else _safe_end(data)
        return decode(data[:end]), offset + end

    def tail(self, max_bytes: int) -> str:
        """The last ``max_bytes`` bytes of output, starting at a line when one begins inside them."""
        try:
            with open(self.output_path, "rb") as source:
                size = source.seek(0, 2)
                source.seek(max(0, size - max_bytes))
                data = source.read()
        except FileNotFoundError:
            return ""
        if size > max_bytes and b"\n" in data:
            data = data[data.index(b"\n") + 1 :]
        return decode(data)

    def take_new(self) -> str:
        """Output the model has not seen yet; advances ``cursor``."""
        text, self.cursor = self.read(self.cursor)
        return text

    def summary(self) -> str:
        """One line: id, state, duration and command."""
        match self.state:
            case "running":
                status = f"running for {self.seconds:.0f}s"
            case "exited":
                status = f"exited with code {self.exit_code} after {self.seconds:.0f}s"
            case "killed":
                status = f"killed after {self.seconds:.0f}s"
        return f"{self.id} {status}: {self.command}"


async def wait_job(job: Job, seconds: float, cancel: asyncio.Event) -> WaitOutcome:
    """Wait until the job's shell exits, ``cancel`` fires or ``seconds`` pass, whichever comes first."""
    if job.done.is_set():
        return "done"
    if cancel.is_set():
        return "cancel"
    waiters = {asyncio.ensure_future(job.done.wait()), asyncio.ensure_future(cancel.wait())}
    try:
        await asyncio.wait(waiters, timeout=max(seconds, 0.0), return_when=asyncio.FIRST_COMPLETED)
    finally:
        for waiter in waiters:
            waiter.cancel()
    if job.done.is_set():
        return "done"
    return "cancel" if cancel.is_set() else "timeout"


class JobManager:
    """Owns every process tree the agent started.

    ``output_dir`` holds one output file per job; when it is not given, a
    temporary folder is created and ``close`` removes it. ``sandbox`` is where
    the commands run (the host by default); each command gets a token of its
    own, which the Docker sandbox uses to find everything it started.
    """

    def __init__(self, output_dir: Path | None = None, sandbox: Sandbox | None = None) -> None:
        self._own_dir = output_dir is None
        self.output_dir = Path(tempfile.mkdtemp(prefix="synthic-jobs-")) if output_dir is None else output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = of(sandbox)
        self._token = job_token()
        self._jobs: dict[str, Job] = {}
        self._finished: list[Job] = []
        self._watchers: set[asyncio.Task[None]] = set()
        self._ids = itertools.count(1)

    async def start(
        self,
        command: str,
        cwd: Path,
        env: dict[str, str],
        settings: EnvironmentSettings,
        background: bool = False,
    ) -> Job:
        """Start ``command`` in ``cwd``; on the host with ``os.environ`` overlaid by ``env``."""
        self._settle()
        job_id = f"j{next(self._ids)}"
        output = self.output_dir / f"{job_id}.log"
        tree = await self.sandbox.start(command, cwd, env, output, settings, f"{self._token}-{job_id}")
        job = Job(job_id, command, time.time(), output, tree, background=background)
        self._jobs[job_id] = job
        watcher = asyncio.create_task(self._watch(job))
        self._watchers.add(watcher)
        watcher.add_done_callback(self._watchers.discard)
        return job

    async def _watch(self, job: Job) -> None:
        code = await job.tree.wait()
        job.exit_code = code
        job.ended = time.time()
        if job.state == "running":
            job.state = "exited"
            if job.background:
                self._finished.append(job)
        job.done.set()
        """``done`` is set before the refresh, which in a container is one more ``docker exec``: the
        command's result must not wait for the check of what it left behind. On the host the
        refresh never suspends, so nothing can run in between."""
        await job.tree.refresh()
        if not job.settled and not job.tree.alive():
            job.tree.release()
            job.settled = True

    def _settle(self) -> None:
        """Forget trees whose shell exited and that left nothing running.

        Done on every start, so a POSIX process group id is not kept long
        enough to be recycled by an unrelated process before ``kill_all``.
        """
        for job in self._jobs.values():
            if not job.settled and job.done.is_set() and not job.tree.alive():
                job.tree.release()
                job.settled = True

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return list(self._jobs.values())

    async def kill(self, job_id: str) -> Job | None:
        """Kill the job's whole tree. Also reaches what a finished command left running."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.state == "running":
            job.state = "killed"
        if not job.settled:
            await job.tree.kill_tree()
            job.tree.release()
            job.settled = True
        await job.done.wait()
        return job

    async def kill_all(self) -> None:
        """Kill every tree still alive. Used on interrupt and when the agent ends."""
        await asyncio.gather(*(self.kill(job.id) for job in self._jobs.values() if not job.settled))

    def drain_finished(self) -> list[Job]:
        """Background jobs that exited on their own since the last drain."""
        finished, self._finished = self._finished, []
        return finished

    def discard_output(self, job: Job) -> None:
        """Delete a job's output file once nobody will read it again; a file still held open is left."""
        try:
            job.output_path.unlink(missing_ok=True)
        except OSError:
            pass

    async def close(self) -> None:
        """Kill everything and remove the output folder if this manager created it."""
        await self.kill_all()
        if self._own_dir:
            shutil.rmtree(self.output_dir, ignore_errors=True)


def with_line(text: str, line: str) -> str:
    """``text`` followed by ``line`` on a line of its own."""
    if text and not text.endswith("\n"):
        text += "\n"
    return text + line


class JobTool:
    """The ``job`` tool: follow and control background jobs."""

    name = "job"
    description = (
        "Follow or stop background jobs started by `sh` (with background: true, or after outliving their time "
        "limit). Actions: `list` shows jobs; `status` returns the job's state and the output printed since your "
        "last status or wait on it; `wait` blocks until the job ends or `seconds` pass (capped by the tool time "
        "limit), then answers like status; `kill` stops the job and every process it started. Jobs that end on "
        "their own are also announced to you."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "status", "wait", "kill"]},
            "job": {"type": "string", "description": "Job id such as j3; required except for list."},
            "seconds": {"type": "number", "description": "For wait: the longest to wait."},
        },
        "required": ["action"],
    }

    def __init__(self, jobs: JobManager) -> None:
        self.jobs = jobs

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        action = tool_input.get("action")
        if action == "list":
            shown = [job.summary() for job in self.jobs.list() if job.background or job.state == "running"]
            return ToolOutput("\n".join(shown) or "No jobs.")
        if action not in ("status", "wait", "kill"):
            return ToolOutput(f"Unknown action {action!r}: use list, status, wait or kill.", is_error=True)
        job = self.jobs.get(str(tool_input.get("job", "")))
        if job is None:
            known = ", ".join(j.id for j in self.jobs.list() if j.background) or "none"
            return ToolOutput(f"No job {tool_input.get('job')!r}. Background jobs: {known}.", is_error=True)
        meta = {"job": job.id}
        if action == "kill":
            was_running = job.state == "running"
            await self.jobs.kill(job.id)
            note = "Killed with every process it started." if was_running else "Had already exited; leftovers killed."
            return ToolOutput(with_line(job.take_new(), f"[{job.summary()}] {note}"), meta=meta)
        if action == "wait":
            seconds = tool_input.get("seconds")
            limit = ctx.timeout_seconds
            if isinstance(seconds, int | float) and not isinstance(seconds, bool):
                limit = min(max(float(seconds), 0.0), limit)
            if await wait_job(job, limit, ctx.cancel) == "cancel":
                return ToolOutput(with_line(job.take_new(), f"[wait interrupted; {job.summary()}]"), is_error=True, meta=meta)
        new = job.take_new() or "(no new output)\n"
        return ToolOutput(with_line(new, f"[{job.summary()}]"), meta=meta)
