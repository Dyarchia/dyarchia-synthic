"""Job 8: the agent tree, and the runtime that owns everything a run shares.

One ``Runtime`` per run. It holds the policy, the provider, the run folder, the
masker, the hooks, the work copies, the plugins, the budget ledger, the open
questions and every agent. Agents reach it through ``agent.runtime``.

Counting: the root has depth 0 and a child one more than its parent; a spawn
is refused when the child's depth would pass ``agents.depth``. ``agents.concurrent``
counts the spawned agents of the whole tree that have not ended (the root is
not counted). A child keeps one id for its whole life, retries and escalation
included; its attempts get journals ``<id>``, ``<id>.2``, ``<id>.3``... and
share one per-agent allowance (``per_agent_dollars`` and ``max_steps``).

The root agent always works in the project root. Children work where
``Workspace.start`` puts them: the root again in mode ``in_place``, a copy of
their own in mode ``copy``. When an attempt ends, its journal and work copy are
closed (the copy is kept when it holds changes) and the ``Agent`` is dropped,
so a run of many hours does not keep every finished history in memory.

While a run is alive the runtime keeps ``<run_dir>/alive`` fresh (see
``journal.is_alive``) and removes it on ``close``.
"""

from __future__ import annotations

import asyncio
import bisect
import itertools
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import web
from .environment import Workspace
from .hooks import Hooks
from .journal import (
    ALIVE_FILE,
    HEARTBEAT_SECONDS,
    JOURNAL_FILE,
    Masker,
    agents_in,
    beat,
    ended,
    new_run,
    read_records,
)
from .loop import ENGINE_TOOLS, Agent, AgentResult, Question
from .plugins import load_plugins
from .policy import LoadedPolicy, PolicyError, spawn_role
from .provider import AnthropicProvider
from .types import Attachment, Budget, Provider, Role, ToolContext, ToolOutput, Usage

OutputCallback = Callable[[str, str, str], None]

ROOT_ID = "root"
TREE_CAPS = frozenset({"budget.dollars", "budget.wall_clock_hours", "budget.steps"})
"""Caps of the whole tree: a child stopped by one of them is not retried, since a retry cannot pass it."""


def allowance_of(agent_id: str) -> str:
    """The key of the per-agent allowance: a child's attempts ``a1``, ``a1.2``... and its escalation share ``a1``."""
    return agent_id.partition(".")[0]


class Ledger:
    """Usage, steps and time of the whole tree, against ``policy.budget``.

    ``elapsed`` carries the active time of a run before it was interrupted, so
    a resumed run keeps counting from there instead of from zero or from the
    interruption. Per-agent usage is kept by ``allowance_of``, so a retried or
    escalated child does not get a fresh ``per_agent_dollars``.
    """

    def __init__(self, budget: Budget, elapsed: float = 0.0, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self.total = Usage()
        self.per_agent: dict[str, Usage] = {}
        self.steps = 0
        self._clock = clock
        self.started = clock() - elapsed
        self._warned: set[str] = set()

    def add(self, agent_id: str, usage: Usage, step: bool = True) -> None:
        self.total.add(usage)
        self.per_agent.setdefault(allowance_of(agent_id), Usage()).add(usage)
        self.steps += step

    @property
    def hours(self) -> float:
        return (self._clock() - self.started) / 3600

    def _levels(self, agent_id: str | None) -> list[tuple[str, str, float, float]]:
        """``(key, cap name, used, cap)`` for every cap the policy sets."""
        b = self.budget
        found = []
        if b.dollars is not None:
            found.append(("budget.dollars", "budget.dollars", self.total.cost_usd, b.dollars))
        if b.wall_clock_hours is not None:
            found.append(("budget.wall_clock_hours", "budget.wall_clock_hours", self.hours, b.wall_clock_hours))
        if b.steps is not None:
            found.append(("budget.steps", "budget.steps", float(self.steps), float(b.steps)))
        if b.per_agent_dollars is not None and agent_id is not None:
            key = allowance_of(agent_id)
            spent = self.per_agent.get(key, Usage()).cost_usd
            found.append((f"per_agent_dollars:{key}", "budget.per_agent_dollars", spent, b.per_agent_dollars))
        return found

    def check(self, agent_id: str | None = None) -> str | None:
        """The name of the first cap reached (``budget.dollars``, ...), or ``None`` while all are under."""
        for _, name, used, cap in self._levels(agent_id):
            if used >= cap:
                return name
        return None

    def warning_due(self, agent_id: str | None = None) -> str | None:
        """A one-line warning the first time a cap passes ``warn_at`` of itself; ``None`` otherwise.

        Tree caps warn once per run, the per-agent cap once per agent.
        """
        for key, name, used, cap in self._levels(agent_id):
            if key not in self._warned and cap > 0 and used / cap >= self.budget.warn_at:
                self._warned.add(key)
                return f"{name} is {used / cap:.0%} used ({used:g} of {cap:g})"
        return None


@dataclass(eq=False)
class ChildHandle:
    """A child as its parent sees it: one id from spawn to the final notice, whatever retries happen inside.

    ``current`` is the attempt running now, ``None`` between attempts and once
    the child ended: a finished attempt is dropped with its history. What the
    parent still needs is kept here: ``usage`` and ``steps`` of every attempt
    (they share one allowance), the last attempt's work ``copy`` and the result.
    """

    id: str
    role: str
    task: str
    parent: Agent
    started: float = field(default_factory=time.monotonic)
    current: Agent | None = None
    result: AgentResult | None = None
    attempts: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    copy: Path | None = None
    reported: bool = False
    cancelled: bool = False
    done: asyncio.Event = field(default_factory=asyncio.Event)
    runner: asyncio.Task[None] | None = None
    """The supervising task; kept here so the event loop cannot drop it while it runs."""

    async def cancel(self) -> None:
        """Cancel the child and whatever it started, and return once it has ended."""
        self.cancelled = True
        if self.current is not None:
            self.current.cancel()
        await self.done.wait()

    def line(self) -> str:
        state = self.result.status if self.result else f"running for {time.monotonic() - self.started:.0f}s"
        return f"{self.id} ({self.role}) {state}: {' '.join(self.task.split())[:120]}"

    def notice(self) -> str:
        """What the parent reads when the child ends: status, summary, attempts, work copy and changes."""
        assert self.result is not None
        result = self.result
        lines = [f"[child {self.id} ({self.role}) ended: {result.status}]", f"Summary: {result.summary}"]
        if len(self.attempts) > 1:
            lines.append("Attempts: " + "; ".join(self.attempts) + ".")
        if self.copy is not None:
            if result.changes:
                lines.append(f"Work copy: {self.copy}")
                lines.append("Changes:\n" + result.changes)
            else:
                lines.append("Changes: none (the work copy was removed).")
        return "\n".join(lines)


class SpawnTool:
    """``spawn``: start a child agent and return its id at once."""

    name = "spawn"

    def __init__(self, runtime: Runtime, agent: Agent) -> None:
        self._runtime = runtime
        self._agent = agent
        roles = list(agent.role.can_spawn)
        self.description = (
            f"Start a child agent on a task and get its id at once: it works in parallel and does not block you. "
            f"Roles: {', '.join(roles)}. The child sees only `task`, so put everything it needs there. Its result "
            "arrives as a message when it ends; use the `agent` tool to wait for it, message it or cancel it."
        )
        self.input_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"role": {"type": "string", "enum": roles}, "task": {"type": "string"}},
            "required": ["role", "task"],
        }

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        role, task = tool_input.get("role"), tool_input.get("task")
        if not isinstance(role, str) or not isinstance(task, str) or not task.strip():
            return ToolOutput("`role` and a non-empty `task` are required.", is_error=True)
        return await self._runtime.spawn(self._agent, role, task)


class AgentTool:
    """``agent``: list, wait for, message or cancel the caller's children."""

    name = "agent"
    description = (
        "Manage your child agents. `list`: every child and its state. `wait`: block until child `id` ends (any "
        "child without `id`), a message arrives, or `seconds` pass; returns status, summary and changes. "
        "`message`: send `text` to child `id`. `cancel`: stop child `id` and everything it started."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "wait", "message", "cancel"]},
            "id": {"type": "string"},
            "seconds": {"type": "number"},
            "text": {"type": "string"},
        },
        "required": ["action"],
    }

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        children = self._agent.children
        action = tool_input.get("action")
        if action == "list":
            return ToolOutput("\n".join(h.line() for h in children.values()) or "No children.")
        if action not in ("wait", "message", "cancel"):
            return ToolOutput(f"Unknown action {action!r}: use list, wait, message or cancel.", is_error=True)
        child_id = tool_input.get("id")
        handle = children.get(child_id) if isinstance(child_id, str) else None
        if handle is None and (action != "wait" or child_id is not None):
            known = ", ".join(children) or "none"
            return ToolOutput(f"No child {child_id!r}. Your children: {known}.", is_error=True)
        if action == "message":
            text = tool_input.get("text")
            if not isinstance(text, str) or not text.strip():
                return ToolOutput("`text` is required.", is_error=True)
            if handle.result is not None or handle.current is None:
                return ToolOutput(f"Child {handle.id} is not running.", is_error=True)
            handle.current.add(text, source="parent")
            return ToolOutput(f"Sent to {handle.id}.")
        if action == "cancel":
            await handle.cancel()
            handle.reported = True
            return ToolOutput(handle.notice())
        return await self._wait(handle, tool_input.get("seconds"), ctx)

    async def _wait(self, handle: ChildHandle | None, seconds: Any, ctx: ToolContext) -> ToolOutput:
        targets = [handle] if handle else list(self._agent.children.values())
        if not targets:
            return ToolOutput("You have no children to wait for.", is_error=True)
        limit = ctx.timeout_seconds
        if isinstance(seconds, int | float) and not isinstance(seconds, bool) and seconds >= 0:
            limit = float(seconds)
        if not any(h.done.is_set() and (h is handle or not h.reported) for h in targets):
            await self._agent.wait_for(*(h.done.wait() for h in targets), timeout=limit)
        ended = [h for h in targets if h.done.is_set() and (h is handle or not h.reported)]
        for h in ended:
            h.reported = True
        if ended:
            return ToolOutput("\n\n".join(h.notice() for h in ended))
        running = "\n".join(h.line() for h in targets)
        why = "interrupted" if ctx.cancel.is_set() else "a message arrived" if self._agent.has_news() else "time is up"
        return ToolOutput(f"Still running ({why}):\n{running}")


class Runtime:
    """Owns one run: build it, ``await run(task)`` (or ``resume()``), then ``await close()``.

    ``on_output`` callbacks receive ``(agent_id, kind, text)`` with kind one of:

    * ``text``: model text as it streams (a retried stream may repeat itself);
    * ``tool_start`` / ``tool_end``: one compact line per tool call;
    * ``agent_start`` / ``agent_end``: an agent began (role and task) or ended (status and summary);
    * ``question``: a question or permission request is open (``id: text``);
    * ``warning``: a budget warning, a model switch, a failed compaction, a policy or plugin problem.

    Callbacks run inline on the event loop and are not guarded: one that
    raises ends the agent that produced the output as ``failed``.
    """

    def __init__(
        self,
        loaded: LoadedPolicy,
        root: Path,
        provider: Provider | None = None,
        run_id: str | None = None,
    ) -> None:
        self.loaded = loaded
        self.policy = policy = loaded.policy
        self.root = Path(root).resolve()
        self.run_dir = new_run(policy.journal, run_id)
        self.run_id = self.run_dir.name
        self.masker = Masker((*policy.journal.mask, *policy.never_send))
        self.plugins = load_plugins(policy.plugins, self.root, policy.environment.env)
        self.plugin_tools = self.plugins.tools()
        self.hooks = Hooks((*policy.hooks, *self.plugins.hooks), self.root, policy.environment.env)
        self.workspace = Workspace(policy.environment, self.root, self.run_dir, self.run_id)
        self._own_provider = provider is None
        self.provider: Provider = provider or AnthropicProvider(
            cache_ttl=policy.context.cache_ttl,
            retry_seconds=policy.provider.retry_seconds,
            timeout_seconds=policy.provider.timeout_seconds,
            search_price=policy.web.search_price,
        )
        self.ledger = Ledger(policy.budget)
        self.agents: dict[str, Agent] = {}
        """Agents still running, and the root; a child's attempt is dropped once it ends."""
        self.handles: dict[str, ChildHandle] = {}
        self.root_agent: Agent | None = None
        self._root_done = asyncio.Event()
        self._heartbeat: asyncio.Task[None] | None = None
        self._questions: dict[str, Question] = {}
        self._callbacks: list[OutputCallback] = []
        self._ids = itertools.count(1)
        self.warnings = [*loaded.warnings, *self.plugins.problems, *web.warnings(policy)]
        for raw in policy.instructions.files:
            path = Path(raw).expanduser()
            if not (path if path.is_absolute() else self.root / path).is_file():
                self.warnings.append(f"instruction file not found: {raw}")

    def on_output(self, callback: OutputCallback) -> None:
        self._callbacks.append(callback)

    def output(self, agent_id: str, kind: str, text: str) -> None:
        for callback in list(self._callbacks):
            callback(agent_id, kind, text)

    async def run(self, task: str, role: str | None = None, attachments: Sequence[Attachment] = ()) -> AgentResult:
        """Run the root agent on ``task`` with ``role`` (default ``policy.initial_role``) until it ends.

        ``attachments`` are files (``media.load``) sent with the task.
        """
        name = role or self.policy.initial_role
        if name not in self.policy.roles:
            raise PolicyError([f"role '{name}' is not defined"])
        if self.root_agent is not None:
            raise RuntimeError("this runtime has already run; build a new one for another task")
        for warning in self.warnings:
            self.output(ROOT_ID, "warning", warning)
        self.root_agent = Agent(self, ROOT_ID, self.policy.roles[name], task, self.root, attachments=attachments)
        self.agents[ROOT_ID] = self.root_agent
        return await self._run_root()

    async def _run_root(self) -> AgentResult:
        assert self.root_agent is not None
        if self._heartbeat is None:
            self._heartbeat = asyncio.get_running_loop().create_task(self._beat())
        try:
            return await self.root_agent.run()
        finally:
            self._root_done.set()

    async def _beat(self) -> None:
        """Keep ``<run_dir>/alive`` fresh while this runtime works on the run (``journal.is_alive``)."""
        while True:
            beat(self.run_dir)
            await asyncio.sleep(HEARTBEAT_SECONDS)

    async def resume(self, text: str | None = None, attachments: Sequence[Attachment] = ()) -> AgentResult:
        """Continue the run whose folder this runtime was built on, from the root journal.

        A run cut by a crash goes on where it stopped. A root that ended
        ``stopped`` goes on too: the human can always interrupt, so the human
        can always pick the work up again. A root that ended otherwise
        (``delivered``, ``failed``, ``budget``) returns its recorded result,
        unless ``text`` is given. ``text`` enters as a new message from the
        human in every case. The ledger is rebuilt from every journal of the
        run, so the budget keeps counting.
        """
        if self.root_agent is not None:
            raise RuntimeError("this runtime has already run")
        if not (self.run_dir / ROOT_ID / JOURNAL_FILE).is_file():
            raise FileNotFoundError(f"no root journal to resume in {self.run_dir}")
        self._restore_ledger()
        records = list(read_records(self.run_dir, ROOT_ID))
        end = ended(records)
        if end is not None and end.data["status"] != "stopped" and not (text and text.strip()):
            usage = Usage()
            for record in records:
                if record.kind == "model_call":
                    usage.add(Usage(**record.data["usage"]))
            steps = sum(1 for r in records if r.kind == "model_call" and "purpose" not in r.data)
            return AgentResult(ROOT_ID, end.data["status"], end.data["summary"], usage, steps)
        numbers = [int(m.group(1)) for a in agents_in(self.run_dir) if (m := re.fullmatch(r"a(\d+)(?:\.\d+)?", a))]
        self._ids = itertools.count(max(numbers, default=0) + 1)
        self.root_agent = Agent.resume(self, ROOT_ID, text, attachments)
        self.agents[ROOT_ID] = self.root_agent
        return await self._run_root()

    def _restore_ledger(self) -> None:
        """Rebuild the ledger from every journal of the run.

        Active time is summed per process lifetime: each resume writes a
        ``resume`` record in the root journal, and the time between the last
        record before it and the resume itself (the process was dead, or the
        run had ended) is not counted against ``wall_clock_hours``.
        """
        stamps: list[float] = []
        usages: list[tuple[str, Usage, bool]] = []
        for agent_id in agents_in(self.run_dir):
            for record in read_records(self.run_dir, agent_id):
                stamps.append(record.ts)
                if record.kind == "model_call":
                    usages.append((agent_id, Usage(**record.data["usage"]), "purpose" not in record.data))
        stamps.sort()
        marks = [r.ts for r in read_records(self.run_dir, ROOT_ID) if r.kind == "resume"]
        elapsed = 0.0
        for low, high in itertools.pairwise([-math.inf, *sorted(marks), math.inf]):
            inside = stamps[bisect.bisect_left(stamps, low) : bisect.bisect_left(stamps, high)]
            if inside:
                elapsed += inside[-1] - inside[0]
        self.ledger = Ledger(self.policy.budget, elapsed=elapsed)
        for agent_id, usage, step in usages:
            self.ledger.add(agent_id, usage, step)

    def _agent(self, agent_id: str | None) -> Agent:
        agent_id = agent_id or ROOT_ID
        if (handle := self.handles.get(agent_id)) is not None and handle.current is not None:
            return handle.current
        if agent_id not in self.agents:
            raise KeyError(f"no agent {agent_id!r}")
        return self.agents[agent_id]

    def stop(self) -> None:
        """Stop the run: the root stops, and every agent below it is cancelled as it ends."""
        if self.root_agent is not None:
            self.root_agent.stop()

    def add(self, text: str, agent_id: str | None = None, attachments: Sequence[Attachment] = ()) -> None:
        """Queue a human message for an agent (the root by default); it enters at the agent's next step."""
        self._agent(agent_id).add(text, attachments=attachments)

    def redirect(self, text: str, agent_id: str | None = None) -> None:
        """Abort what an agent (the root by default) is doing and continue with ``text`` as the new message."""
        self._agent(agent_id).redirect(text)

    def questions(self) -> list[Question]:
        """Open questions and permission requests of the whole tree, oldest first."""
        return list(self._questions.values())

    def answer(self, question_id: str, text: str) -> None:
        """Answer an open question. If the agent is not waiting on it any more, it gets the answer as a message."""
        question = self._questions.pop(question_id, None)
        if question is None:
            raise KeyError(f"no open question {question_id!r}")
        agent = self.agents[question.agent_id]
        agent.journal.append("answer", {"question_id": question_id, "text": text})
        if question.waiting and not question.future.done():
            question.future.set_result(text)
        elif question.kind == "question":
            agent.add(f"[answer to {question_id}] You asked: {question.text}\nThe human answered: {text}")

    def open_question(self, question: Question) -> None:
        self._questions[question.id] = question

    def close_question(self, question_id: str) -> None:
        self._questions.pop(question_id, None)

    def close_questions(self, agent_id: str) -> None:
        for question_id in [q.id for q in self._questions.values() if q.agent_id == agent_id]:
            del self._questions[question_id]

    def spawn_tools(self, agent: Agent) -> dict[str, Any]:
        return {"spawn": SpawnTool(self, agent), "agent": AgentTool(agent)}

    def running_children(self) -> int:
        return sum(1 for handle in self.handles.values() if not handle.done.is_set())

    async def spawn(self, parent: Agent, role_name: str, task: str) -> ToolOutput:
        """Start a child of ``parent`` and return its id without waiting for it."""
        limits = self.policy.agents
        if parent.depth + 1 > limits.depth:
            return ToolOutput(
                f"Cannot spawn: the agent tree is limited to depth {limits.depth} and you are at depth "
                f"{parent.depth}. Do the work yourself.",
                is_error=True,
            )
        try:
            role = spawn_role(self.policy, parent.role, role_name, always=ENGINE_TOOLS)
        except PolicyError as error:
            return ToolOutput(f"Cannot spawn: {'; '.join(error.problems)}.", is_error=True)
        running = self.running_children()
        if running >= limits.concurrent:
            return ToolOutput(
                f"Cannot spawn now: {running} agents are running and the limit is {limits.concurrent}. "
                "Wait for one to end (agent wait), then try again.",
                is_error=True,
            )
        child_id = f"a{next(self._ids)}"
        handle = ChildHandle(child_id, role_name, task, parent)
        self.handles[child_id] = handle
        outcome = await parent._emit("spawn", child_id=child_id, role=role_name, task=task)
        if outcome.action == "block":
            del self.handles[child_id]
            return ToolOutput(f"Spawn blocked by a hook: {outcome.reason}", is_error=True)
        parent.children[child_id] = handle
        parent.journal.append("spawn", {"child_id": child_id, "role": role_name, "task": task})
        handle.runner = asyncio.get_running_loop().create_task(self._supervise(handle, role))
        return ToolOutput(f"Started {child_id} ({role_name}). Its result will arrive as a message; or use agent wait.")

    def _escalated(self, parent: Agent, role_name: str) -> Role:
        """The escalation role as a child of ``parent``: imposed by the policy, so not limited by ``can_spawn``."""
        allowed = replace(parent.role, can_spawn=(*parent.role.can_spawn, role_name))
        role = spawn_role(self.policy, allowed, role_name, always=ENGINE_TOOLS)
        return replace(role, can_spawn=tuple(r for r in role.can_spawn if r in parent.role.can_spawn))

    async def _supervise(self, handle: ChildHandle, role: Role) -> None:
        """Run the child's attempts, then tell the parent once.

        Every way a child can end (a result, an engine error, the supervising
        task being cancelled) goes through the ``finally`` below: the result is
        set, ``child_done`` is journaled and emitted, and the parent is woken.
        """
        parent = handle.parent
        result: AgentResult | None = None
        try:
            result = await self._attempts(handle, role)
            if result is None:
                result = AgentResult(handle.id, "cancelled", "Cancelled before it started.", Usage(), 0)
        except Exception as error:
            result = AgentResult(handle.id, "failed", f"The child could not run: {type(error).__name__}: {error}", Usage(), 0)
        finally:
            if result is None:
                result = AgentResult(handle.id, "cancelled", "Its task was cancelled.", Usage(), 0)
            final = handle.result = replace(result, agent_id=handle.id, usage=handle.usage, steps=handle.steps)
            try:
                parent.journal.append("child_done", {"child_id": handle.id, "status": final.status, "summary": final.summary})
                await parent._emit("child_done", child_id=handle.id, status=final.status, summary=final.summary)
            finally:
                handle.done.set()
                parent.wake()

    async def _attempts(self, handle: ChildHandle, role: Role) -> AgentResult | None:
        """Retry failures and escalate per ``on_failure``; all attempts share one allowance.

        A child stopped by a budget cap is not retried: its allowance (the
        handle's ``per_agent_dollars`` and ``max_steps``) or the tree's is spent,
        and a retry would only spend more of the tree's.
        """
        parent = handle.parent
        failure = self.policy.roles[handle.role].on_failure
        failures = 0
        escalated = False
        result: AgentResult | None = None
        while not handle.cancelled:
            attempt = len(handle.attempts) + 1
            agent_id = handle.id if attempt == 1 else f"{handle.id}.{attempt}"
            cwd = await self.workspace.start(agent_id)
            agent = Agent(
                self,
                agent_id,
                role,
                handle.task,
                cwd,
                depth=parent.depth + 1,
                parent_id=parent.id,
                own_copy=self.policy.environment.mode == "copy",
                steps_before=handle.steps,
            )
            self.agents[agent_id] = agent
            handle.current = agent
            if handle.cancelled:
                agent.cancel()
            try:
                result = await agent.run()
            finally:
                await self._retire(handle, agent)
            handle.attempts.append(f"{agent_id} as {role.name}: {result.status}")
            if handle.cancelled or result.status != "failed" or escalated or failure is None:
                break
            if self.ledger.check() in TREE_CAPS:
                break
            failures += 1
            if failures >= failure.attempts:
                role = self._escalated(parent, failure.escalate_to)
                escalated = True
        return result

    async def _retire(self, handle: ChildHandle, agent: Agent) -> None:
        """An attempt ended: count it, close its journal and work copy, and drop it with its history.

        The copy is kept when it holds changes, or when they are unknown (the
        attempt has no result, or listing them failed), so work is never lost.
        """
        handle.current = None
        self.agents.pop(agent.id, None)
        self.close_questions(agent.id)
        handle.usage.add(agent.usage)
        handle.steps += agent.steps
        agent.journal.close()
        if agent.own_copy:
            handle.copy = agent.cwd
            keep = agent.result is None or bool(agent.result.changes)
            await self.workspace.close(agent.id, keep=keep)

    async def close(self) -> None:
        """Cancel whatever is still alive, close plugins and work copies (kept when they hold changes) and journals."""
        if self.root_agent is not None and self.root_agent.running:
            self.root_agent.stop()
            await self._root_done.wait()
        await asyncio.gather(*(h.cancel() for h in self.handles.values() if not h.done.is_set()))
        await self.plugins.close()
        for agent_id, agent in self.agents.items():
            if agent.own_copy:
                keep = agent.result is None or bool(agent.result.changes)
                await self.workspace.close(agent_id, keep=keep)
            agent.journal.close()
        if self._own_provider and isinstance(self.provider, AnthropicProvider):
            await self.provider.aclose()
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            await asyncio.gather(self._heartbeat, return_exceptions=True)
            (self.run_dir / ALIVE_FILE).unlink(missing_ok=True)
