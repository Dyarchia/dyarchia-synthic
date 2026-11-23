"""Verification before delivery: the checks a root agent's ``done`` must pass when it says ``delivered``.

Set by the ``[verify]`` policy table and off while it names nothing. The order
is fixed: the ``before_done`` hook (in the loop), then ``commands``, then the
``review``. A failing command skips the review, which would only cost more.

* Commands run in the working folder through the policy's shell, one after
  another, and each must exit 0 within ``timeout_seconds``. One that outlives
  it is killed and counts as failed: a check that never ends cannot pass.
* The reviewer is a fresh agent with the role ``verify.review``, taken from
  the policy as it is (the human chose it, so it is not narrowed to the
  delivering role, which needs no ``spawn`` right). It is a child of the
  delivering agent for the tree and the ledger: its cost counts in the same
  budget. It reads the human's messages from the journal, not from the
  context, which compaction may have shortened, and every file the human
  attached. Its ``done`` is the verdict: ``delivered`` approves, ``failed``
  rejects with the list of problems. A reviewer that ends any other way (budget,
  stopped, a failed model call or an engine error, which also end an agent as
  ``failed`` but not through its ``done``) gave no verdict, so the delivery is
  accepted as not verified without using up a round. A reviewer is never reviewed: only the
  root agent is verified, since a child's delivery is checked by its parent.
* A rejection returns to the agent as the error result of its ``done``, with
  the failing output or the reviewer's list, and the agent keeps working.
  After ``rounds`` rejections, a delivery that still fails is accepted and its
  summary says it was not verified and why.
* Children still running are cancelled before the checks, as every ``done``
  does: the work under review must not change while it is checked.
* An interrupt during the checks is an error result, never a delivery.

Every check is journaled as a ``verify`` record, so a resumed agent counts
the rejections it already had (those after the last delivery) and a reviewer
cut by a crash is closed as cancelled when the run resumes.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from .context import unpack
from .tools.jobs import wait_job
from .types import Attachment, Record, ToolOutput

if TYPE_CHECKING:
    from .loop import Agent

REVIEWER_ID = re.compile(r"v(\d+)")


def active(agent: Agent) -> bool:
    """Whether ``agent``'s deliveries are verified: it is the root and the policy names a check."""
    settings = agent.policy.verify
    return agent.parent_id is None and bool(settings.commands or settings.review)


def rejections(records: list[Record]) -> int:
    """Rejections since the last delivery: a new task after one starts with all its rounds, while a run that
    ended otherwise (failed, stopped, a crash) and is resumed goes on with the rounds it had left."""
    count = 0
    for record in records:
        if record.kind == "done" and record.data.get("status") == "delivered":
            count = 0
        elif record.kind == "verify" and not record.data.get("accepted"):
            count += 1
    return count


def _git(folder: Path) -> bool:
    return any((parent / ".git").exists() for parent in (folder, *folder.parents))


class Verifier:
    """Runs the checks of one delivery for one agent. Build a new one for every ``done``."""

    def __init__(self, agent: Agent) -> None:
        self.agent = agent
        self.settings = agent.policy.verify

    async def check(self, summary: str) -> ToolOutput | str:
        """The rejection to return as the ``done`` result, or the summary to deliver."""
        agent = self.agent
        rounds = self.settings.rounds
        report, reviewer = await self._commands(), None
        verdict = "failed" if report else "passed"
        if verdict == "passed" and self.settings.review:
            verdict, report, reviewer = await self._review(summary)
        if agent.cancelled():
            return ToolOutput("Not delivered: interrupted during verification.", is_error=True)
        attempt = agent.rejections + 1
        data = {"round": attempt, "passed": verdict == "passed", "report": report, "reviewer": reviewer}
        if verdict == "failed" and agent.rejections < rounds:
            agent.rejections += 1
            agent.journal.append("verify", {**data, "accepted": False})
            agent.runtime.output(agent.id, "verify", f"rejected ({agent.rejections} of {rounds})")
            return ToolOutput(
                f"Not delivered: verification failed (rejection {agent.rejections} of {rounds} allowed). Fix what "
                f"follows, then call done again.\n\n{report}",
                is_error=True,
            )
        agent.journal.append("verify", {**data, "accepted": True})
        if verdict == "passed":
            agent.runtime.output(agent.id, "verify", "passed")
            return summary
        why = " ".join(report.split("\n")[:2]).strip() or "no reason given"
        why = why if len(why) <= 300 else why[:297] + "..."
        if verdict == "failed":
            note = f"NOT VERIFIED: still failing after {rounds} rejection{'s' if rounds != 1 else ''}: {why}"
        else:
            note = f"NOT VERIFIED: {why}"
        agent.runtime.output(agent.id, "verify", note)
        return f"{note}\n\n{summary}"

    async def _commands(self) -> str:
        """The report of the first failing command (the tail of its output), or ``""`` when all passed; it stops there."""
        agent = self.agent
        failures = []
        for command in self.settings.commands:
            if agent.cancelled():
                break
            env = agent.tool_context().env
            try:
                job = await agent.jobs.start(command, agent.cwd, env, agent.policy.environment)
            except OSError as error:
                failures.append(f"$ {command}\nCould not start: {error}")
                break
            outcome = await wait_job(job, self.settings.timeout_seconds, agent.cancel_event)
            if outcome != "done":
                await agent.jobs.kill(job.id)
            text, _ = job.read(0)
            agent.jobs.discard_output(job)
            if outcome == "done" and job.exit_code == 0:
                agent.runtime.output(agent.id, "verify", f"$ {command}: passed")
                continue
            if outcome == "timeout":
                state = f"killed after {self.settings.timeout_seconds:g}s"
            elif outcome == "cancel":
                state = "interrupted"
            else:
                state = f"exit code {job.exit_code}"
            agent.runtime.output(agent.id, "verify", f"$ {command}: {state}")
            tail = agent.policy.context.output_tail
            shown = text if len(text) <= tail else f"[last {tail:,} of {len(text):,} chars]\n{text[-tail:]}"
            failures.append(f"$ {command}\n[{state}]\n{shown.rstrip()}")
            break
        return "\n\n".join(failures)

    async def _review(self, summary: str) -> tuple[str, str, str]:
        """Run the reviewer and return ``(verdict, report, reviewer id)``; verdict is passed, failed or none."""
        agent = self.agent
        runtime = agent.runtime
        role = agent.policy.roles[self.settings.review or ""]
        task, files = self._brief(summary)
        reviewer_id = runtime.reviewer_id()
        reviewer = type(agent)(
            runtime, reviewer_id, role, task, agent.cwd, depth=agent.depth + 1, parent_id=agent.id, attachments=files
        )
        runtime.agents[reviewer_id] = reviewer
        runner = asyncio.ensure_future(reviewer.run())
        waiter = asyncio.ensure_future(agent.cancel_event.wait())
        try:
            await asyncio.wait({runner, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
            if not runner.done():
                reviewer.cancel()
            await asyncio.gather(runner, waiter, return_exceptions=True)
            runtime.agents.pop(reviewer_id, None)
            reviewer.journal.close()
        result = runner.result() if not runner.cancelled() and runner.exception() is None else None
        if result is None:
            return "none", "the reviewer could not run", reviewer_id
        match result.status:
            case "delivered":
                return "passed", result.summary, reviewer_id
            case "failed" if reviewer.ended_by_done:
                return "failed", f"The reviewer ({role.name}) rejected the delivery:\n{result.summary}", reviewer_id
        return "none", f"the review did not finish ({result.status}: {result.summary})", reviewer_id

    def _brief(self, summary: str) -> tuple[str, list[Attachment]]:
        """The reviewer's task: every message of the human, the summary, the checks passed, how to see the work."""
        agent = self.agent
        records = list(agent.journal.records())
        started = next((r.ts for r in records if r.kind == "start"), None)
        messages: list[str] = []
        files: list[Attachment] = []
        asked = {r.data["question_id"]: r.data["text"] for r in records if r.kind == "question"}
        for record in records:
            if record.kind == "answer":
                question = asked.get(record.data["question_id"], record.data["question_id"])
                messages.append(f'{len(messages) + 1}. (answering "{question}") {record.data["text"]}')
                continue
            if record.kind != "user" or record.data.get("source") != "human":
                continue
            attached = record.data.get("attachments", [])
            names = "".join(f"\n(attached: {a['name']})" for a in attached)
            messages.append(f"{len(messages) + 1}. {record.data['text']}{names}")
            files += [Attachment(a["name"], unpack(a["block"], agent.receipts), a["note"]) for a in attached]
        since = datetime.fromtimestamp(started).astimezone().isoformat(timespec="seconds") if started else None
        if _git(agent.cwd):
            commits = f' and `git log --since="{since}"` with `git show` for commits made during the run' if since else ""
            where = f"Run `git status` and `git diff` there for uncommitted changes{commits}."
        else:
            where = "It is not a git repository: look at the files" + (f" changed since {since}." if since else ".")
        lines = [
            "You review work that another agent says it has finished. Judge it against what the human asked; do "
            "not change anything yourself.",
            "",
            "Messages from the human, in order (the first one is the task):",
            *messages,
            "",
            "The delivering agent's summary:",
            summary,
        ]
        if self.settings.commands:
            lines += ["", "These checks already passed: " + "; ".join(f"`{c}`" for c in self.settings.commands)]
        lines += [
            "",
            f"The work is in {agent.runtime.sandbox.shown(agent.cwd)}. {where}",
            "",
            "Answer by calling `done`: status `delivered` when everything asked is done and correct; status "
            "`failed` when anything is missing or wrong, with the summary listing each problem on a line of its own.",
        ]
        return "\n".join(lines), files
