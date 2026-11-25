"""Job 2: the loop of one agent.

An ``Agent`` asks the model, runs the tools it requests and repeats until the
model calls ``done``, a cap of the policy is reached, or the human interrupts.
It owns its context, its journal, its receipts and its background jobs; the
``Runtime`` (``agents.py``) owns what the whole tree shares and is reached
through ``self.runtime``.

Rules that shape the code:

* Nothing waits for a human unless ``policy.human`` says so, and every wait of
  the loop (a question, a permission, an idle parent, ``agent wait``) also ends
  when the agent is interrupted.
* Errors go back to the model as tool results; only a failed model call or an
  engine bug ends the agent, with status ``failed``.
* Every string that enters the context is masked first (tool output, human
  text, hook text, notices, the system prompt, and the model's own text and
  tool inputs), and the very same data is journaled. The journal masks with
  the same function, which is then a no-op, so replaying the journal rebuilds
  the context byte for byte. A tool still runs on the input the model wrote:
  only the copy kept in the context is masked. Thinking blocks, signatures and
  image data are never masked (see ``Masker.mask_obj``).
* A resumed agent reuses the system blocks and tool definitions of its
  ``start`` record instead of rebuilding them from today's policy, files and
  plugins. The replayed history is byte for byte the one the process had, so
  its thinking blocks stay and the prompt cache still matches; they are dropped
  (``Context.drop_thinking``) only when the model changed, when the policy says
  so, or when the API rejects one, and the journal records each drop so that
  every later replay sends the same history.
* Tool results enter the journal in the order they complete, and the context in
  that same order, which is the order ``Context.replay`` re-adds them.
* Tools that act only on the agent's own state (``SELF_TOOLS``) skip the
  permission check; hooks still see them.

Hook payloads (every one also has ``agent_id``):

* ``before_tool``: ``tool``, ``call_id``, ``input``. ``replace = {"input": {...}}``
  substitutes the input; a block becomes the tool result.
* ``after_tool``: the same plus ``output`` and ``is_error``. ``replace =
  {"output": "..."}`` substitutes the output text.
* ``before_model``: ``model``, ``step``. ``after_model``: ``stop_reason``, ``usage``.
* ``before_compaction``: a block skips the compaction.
* ``budget_warning``: ``warning``. ``question``: ``question_id``, ``text``, ``assumption``.
* ``before_done``: ``summary``, ``status``; a block becomes an error result.
* ``start``: ``role``, ``task``, ``parent``. ``resume``: ``role``. ``end``: ``status``, ``summary``.

``add_text`` from ``before_tool`` and ``after_tool`` enters the context after
the results of the turn; from other events it is ignored.
"""

from __future__ import annotations

import asyncio
import copy
import itertools
import json
import os
import time
from collections.abc import Awaitable, Coroutine, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from . import verify, web
from .checkpoints import Recorder
from .schema import validate as validate_result
from .context import PINNED_SOURCES, THINKING, Context, ReceiptStore, ReceiptTool, ToolResultEntry, pack
from .journal import FileJournal, ended, pending_intents
from .policy import decide, deep_merge
from .prompt import build_system
from .tools import builtin_tools, to_definition
from .tools.jobs import JobManager
from .tools.todo import TodoTool
from .tools.todo import render as render_todo
from .types import (
    Attachment,
    Block,
    Decision,
    HookOutcome,
    ModelReply,
    ModelRequest,
    ModelSpec,
    ProviderError,
    Record,
    Role,
    Tool,
    ToolCall,
    ToolContext,
    ToolOutput,
    Usage,
)

if TYPE_CHECKING:
    from .agents import ChildHandle, Runtime

Status = Literal["delivered", "failed", "budget", "stopped", "cancelled"]

ENGINE_TOOLS = frozenset({"done", "ask", "receipt"})
"""Tools every agent has whatever its role says; ``spawn_role`` never removes them."""

SELF_TOOLS = frozenset({*ENGINE_TOOLS, "agent", "job", "todo"})
"""Tools that act only on the agent's own state (its delivery, questions, receipts, children, jobs, task list).

Permission rules guard actions on the world; running these through
``decide`` would let ``default_permission = "deny"`` or a ``"*: *"`` rule stop
an agent from even delivering. ``job`` only follows or kills processes the
agent's ``sh`` already started, which ``sh``'s own rules admitted.
"""

ENDED_NORMALLY = frozenset({"end_turn", "stop_sequence"})
REREAD = (
    "Not delivered yet. First read again what you were asked, word for word:\n\n{asked}\n\n"
    "Go through it against the work as it is now, line by line: every requirement, exact name, constraint and "
    "rule, including those given before the current task. Fix whatever is missing or differs. Then call done "
    "again; if nothing was, call it with the same summary."
)
COMPACTION_CHECK = (
    "Before the conversation is replaced, check your summary against it. Go through the conversation above and "
    "look for anything you would need to continue and finish the work that the summary lacks or states wrongly: a "
    "requirement or constraint, a decision and its reason, a fact you found out about the system, an exact name, "
    "path, id, value or command, an error still open, a step still to do. Then reply with the complete final "
    "summary, in the same sections, corrected and completed. It replaces the first one, so do not answer with "
    "only the differences. Do not call any tool."
)
"""Stop reasons after which a compaction summary is complete and can replace the history."""

UNKNOWN_OUTCOME = (
    "The process died during this action, so its outcome is unknown. Check its effects before repeating it."
)
CANCELLED_BEFORE_START = "[cancelled before start: the agent was interrupted]"
YES = frozenset({"y", "yes", "allow", "ok", "approve", "approved", "true"})


@dataclass
class AgentResult:
    """How an agent ended. ``usage`` is this agent's own; ``changes`` is set only for agents that worked in a copy.

    ``result`` is the structured delivery of a role with a ``result_schema``, already validated against it.
    """

    agent_id: str
    status: Status
    summary: str
    usage: Usage
    steps: int
    changes: str = ""
    result: Any = None


@dataclass(frozen=True)
class _Prepared:
    """A background summary: its text, how many messages it covers, their digest, and the prompt it came from."""

    summary: str
    covered: int
    fingerprint: str
    tokens: int


@dataclass(eq=False)
class Question:
    """An open question to the human, from ``ask`` (``kind="question"``) or from a permission rule (``"permission"``).

    ``waiting`` is true while the agent blocks on it; an answer that arrives
    otherwise is delivered to the agent as a message.
    """

    id: str
    agent_id: str
    kind: Literal["question", "permission"]
    text: str
    assumption: str = ""
    future: asyncio.Future[str] = field(default_factory=lambda: asyncio.get_running_loop().create_future(), repr=False)
    waiting: bool = False


def _short(text: str, limit: int = 160) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _yes(text: str) -> bool:
    return text.strip().lower().rstrip(".!") in YES


def _unbound_thinking(error: ProviderError) -> bool:
    """A request the API refused because a thinking block no longer matches the conversation before it.

    The fixed text of that 400 is quoted in the preserved thinking docs.
    https://platform.claude.com/docs/en/build-with-claude/preserved-thinking
    """
    return error.status == 400 and "bound to a different conversation" in str(error).lower()


def _too_long(error: ProviderError) -> bool:
    """A request the API refused for its size: 413, or a 400 whose message says the prompt is too long."""
    return error.status == 413 or (error.status == 400 and "too long" in str(error).lower())


async def wait_any(*waits: Awaitable[Any], timeout: float | None = None) -> None:
    """Return when the first of ``waits`` finishes or ``timeout`` passes; the others are cancelled."""
    tasks = [asyncio.ensure_future(w) for w in waits]
    try:
        await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _result_schema(tool_defs: list[Block]) -> dict[str, Any] | None:
    """The ``result`` schema inside the journaled definition of ``done``, if it has one."""
    for definition in tool_defs:
        if definition.get("name") == "done":
            result = definition.get("input_schema", {}).get("properties", {}).get("result")
            return result if isinstance(result, dict) else None
    return None


class _DoneTool:
    name = "done"
    description = (
        "Finish your work: call it once the task is delivered, or once it cannot be. `summary` says what was done, "
        "how it was verified and what is left open. `status` is `delivered` (default) or `failed`. Any child agent "
        "still running is cancelled first."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "status": {"type": "string", "enum": ["delivered", "failed"]},
        },
        "required": ["summary"],
    }

    def __init__(self, agent: Agent) -> None:
        self._agent = agent
        if (schema := agent.result_schema) is not None:
            """Only a role with a schema pays for it: the schema becomes the ``result`` property, so the
            model reads it where it writes the value. It is not in ``required``, since a ``failed``
            delivery has no result; the engine requires it for ``delivered``. ``input_schema`` is a JSON
            Schema object: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools"""
            self.description += " This role must also give `result`, a value matching the schema of that property."
            properties = {**self.input_schema["properties"], "result": schema}
            self.input_schema = {**self.input_schema, "properties": properties}

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._agent._done(tool_input)


class _AskTool:
    name = "ask"
    description = (
        "Ask the human about a decision that only the human can make. Give the `assumption` you will act on "
        "without an answer. Usually nobody answers at once: continue on your assumption; a later answer arrives "
        "as a message."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"question": {"type": "string"}, "assumption": {"type": "string"}},
        "required": ["question", "assumption"],
    }

    def __init__(self, agent: Agent) -> None:
        self._agent = agent

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return await self._agent._ask(tool_input)


class Agent:
    """One agent of the tree. Build it, then ``await run()``; interrupt it with ``stop``, ``add``, ``redirect``.

    ``role`` is the role as narrowed for this agent (``policy.spawn_role``),
    so its tools and ``can_spawn`` are what this agent really has. ``depth`` is
    0 for the root. ``own_copy`` marks an agent working in its own copy of the
    project, whose ``changes`` are reported when it ends. ``steps_before``
    counts the steps earlier attempts of the same child already spent against
    ``max_steps``. ``system`` and ``tool_defs`` are given only on resume, from
    the ``start`` record. ``attachments`` go with the task, in its first message.
    """

    def __init__(
        self,
        runtime: Runtime,
        agent_id: str,
        role: Role,
        task: str,
        cwd: Path,
        *,
        depth: int = 0,
        parent_id: str | None = None,
        own_copy: bool = False,
        journal: FileJournal | None = None,
        steps_before: int = 0,
        system: list[str] | None = None,
        tool_defs: list[Block] | None = None,
        attachments: Sequence[Attachment] = (),
    ) -> None:
        self.runtime = runtime
        self.policy = runtime.policy
        self.id = agent_id
        self.role = role
        self.task = task
        self.cwd = Path(cwd)
        self.depth = depth
        self.parent_id = parent_id
        self.own_copy = own_copy
        self.journal = journal or FileJournal(runtime.run_dir, agent_id, runtime.masker)
        self.receipts = ReceiptStore(self.journal.dir / "receipts")
        self.jobs = JobManager(sandbox=runtime.sandbox)
        checkpoints = self.policy.checkpoints
        self.recorder = Recorder(self.journal, lambda: self.steps) if checkpoints.edits or checkpoints.shell else None
        self.children: dict[str, ChildHandle] = {}
        self.result_schema = role.result_schema if tool_defs is None else _result_schema(tool_defs)
        """The schema a delivery's ``result`` must match; a resumed agent keeps the one its ``done`` was shown."""
        self.tools = self._build_tools()
        masker = runtime.masker
        if tool_defs is None:
            """Server tools go first, so the cache mark of the tool list lands on a client tool
            definition: the docs never say that a server tool definition takes ``cache_control``."""
            client = [to_definition(tool) for tool in self.tools.values()]
            tool_defs = masker.mask_obj([*web.definitions(self.policy, role.tools), *client])
        if system is None:
            texts = build_system(
                self.policy,
                role,
                self.cwd,
                self.tools,
                runtime.plugins,
                runtime.root,
                parent_id is None,
                sandbox=runtime.sandbox,
            )
            system = [masker.mask(text) for text in texts]
        self.tool_defs: list[Block] = tool_defs
        self.system: list[str] = system
        self.context = Context(self.policy.context, self.receipts, self.system, can_view="view" in self.tools)
        self._attachments = list(attachments)
        self.usage = Usage()
        self.steps = 0
        self.steps_before = steps_before
        self.result: AgentResult | None = None
        self.running = False
        self._model = role.model
        self._next_model: str | None = None
        self._compact_retry_at: int | None = None
        self._overflow_retried = False
        self._thinking_retried = False
        self._announced: set[str] = set()
        self._cancel = asyncio.Event()
        self._mode: Literal["stop", "cancel", "redirect"] | None = None
        self._redirects: list[str] = []
        self._inbox: list[tuple[str, str, Sequence[Attachment]]] = []
        self._wake = asyncio.Event()
        self._finish: tuple[Status, str] | None = None
        self._idle_turns = 0
        self._cut_turns = 0
        self._limit_override: int | None = None
        self._last_limit = 0
        self._force_compact = False
        self._prepared: _Prepared | None = None
        self._preparing: asyncio.Task[None] | None = None
        self._prepare_cancel = asyncio.Event()
        self._repeat: tuple[str, str] | None = None
        self._repeat_count = 0
        self._repeat_warned = False
        self._questions = itertools.count(1)
        self._resumed = False
        self._paused = False
        self.rejections = 0
        self._reread = False
        self._result: tuple[Any] | None = None
        """The delivered ``result``, as a 1-tuple so that a JSON ``null`` result is still a result."""
        self.ended_by_done = False
        """Whether the agent ended by calling ``done``, rather than by a cap, an interrupt or a failure."""
        """Deliveries ``verify`` rejected since the last ``done``; restored from the journal on resume."""
        self._next_task: tuple[str, Sequence[Attachment]] | None = None

    def _build_tools(self) -> dict[str, Tool]:
        """The fixed tool list: engine tools, then what the role grants. It never changes during the run (cache).

        An agent at the maximum depth gets no ``spawn`` or ``agent``: every
        spawn it tried would be refused.
        """
        granted = set(self.role.tools)
        tools: dict[str, Tool] = {
            "done": _DoneTool(self),
            "ask": _AskTool(self),
            "receipt": ReceiptTool(self.receipts, self.policy.context),
        }
        builtin = builtin_tools(self.policy.environment, self.jobs)
        if "sh" in granted:
            tools["sh"] = builtin["sh"]
            tools["job"] = builtin["job"]
        if "edit" in granted:
            tools["edit"] = builtin["edit"]
        for name in ("view", "read", "grep"):
            if name in granted:
                tools[name] = builtin[name]
        if "todo" in granted:
            tools["todo"] = TodoTool(self._todo_changed)
        if "spawn" in granted and self.role.can_spawn and self.depth < self.policy.agents.depth:
            tools.update(self.runtime.spawn_tools(self))
        for name, tool in self.runtime.plugin_tools.items():
            if name in granted:
                tools[name] = tool
        return tools

    @classmethod
    def resume(
        cls, runtime: Runtime, agent_id: str, text: str | None = None, attachments: Sequence[Attachment] = ()
    ) -> Agent:
        """Rebuild an agent from its journal, ready to ``run()`` on: after a crash, or after it ended.

        Tool calls whose intent was journaled without a result are answered as
        "outcome unknown"; calls that never got an intent are answered as
        cancelled. Children that were running are reported as cancelled, and
        their journals get a ``done`` record saying so. ``text`` enters as a
        new message from the human, with ``attachments``; without it, an agent that had ended (or a
        history that ends on the model's turn) gets an engine notice, since a
        request must not end on an assistant message.
        """
        journal = FileJournal(runtime.run_dir, agent_id, runtime.masker)
        records = list(journal.records())
        start = next((r for r in records if r.kind == "start"), None)
        if start is None:
            journal.close()
            raise ValueError(f"the journal of {agent_id} has no start record")
        role = runtime.policy.roles[start.data["role"]]
        agent = cls(
            runtime,
            agent_id,
            role,
            start.data["task"],
            Path(start.data["cwd"]),
            parent_id=start.data.get("parent"),
            journal=journal,
            system=start.data.get("system"),
            tool_defs=start.data.get("tools"),
        )
        agent._restore(records, text, attachments)
        return agent

    def _keeps_thinking(self, records: list[Record]) -> bool:
        """Whether a resumed history keeps its thinking blocks.

        It does when the policy allows it (``context.resume_keep_thinking``)
        and the model is the one that made the last call: a block is readable
        only by the model that produced it. Keeping them is what lets the
        first call after a resume read the prompt cache instead of writing the
        whole conversation again.
        """
        if not self.runtime.policy.context.resume_keep_thinking:
            return False
        calls = [r.data.get("model_id", "") for r in records if r.kind == "model_call" and "purpose" not in r.data]
        if not calls:
            return True
        last, now = str(calls[-1]), self.spec().id
        return last.startswith(now) or now.startswith(last)

    def _restore(self, records: list[Record], text: str | None, attachments: Sequence[Attachment] = ()) -> None:
        self._resumed = True
        kept = self._keeps_thinking(records)
        self.journal.append("resume", {"thinking_kept": kept})
        self.context.replay(records)
        if not kept:
            self.context.drop_thinking()
        self.rejections = verify.rejections(records)
        for record in records:
            match record.kind:
                case "model_call":
                    self.usage.add(Usage(**record.data["usage"]))
                    self.steps += "purpose" not in record.data
                case "question":
                    next(self._questions)
                case "todo" if isinstance(todo := self.tools.get("todo"), TodoTool):
                    todo.items = list(record.data["items"])
        for intent in pending_intents(records):
            entry = ToolResultEntry(intent.data["call_id"], UNKNOWN_OUTCOME, True)
            self.context.add_tool_results([entry])
            self._journal_result(intent.data["call_id"], intent.data["name"], entry, 0.0)
        self._repair("the process died before it started")
        reported = {r.data["child_id"] for r in records if r.kind == "child_done"}
        for record in records:
            child_id = record.data.get("child_id")
            if record.kind != "spawn" or child_id in reported:
                continue
            status, summary = "cancelled", "Cancelled: the process died while it ran."
            if (self.runtime.run_dir / child_id).is_dir():
                child = FileJournal(self.runtime.run_dir, child_id, self.runtime.masker)
                ended_at = next((r for r in child.records() if r.kind == "done"), None)
                if ended_at is None:
                    child.append("done", {"status": status, "summary": summary})
                else:
                    status, summary = ended_at.data["status"], ended_at.data["summary"]
                child.close()
            self.journal.append("child_done", {"child_id": child_id, "status": status, "summary": summary})
            advice = " Spawn it again if its work is still needed." if status == "cancelled" else ""
            self._add_user(f"[child {child_id} ({record.data['role']}) ended: {status}]\nSummary: {summary}{advice}", "engine")
        _, messages = self.context.build()
        end = ended(records)
        if text and text.strip() and end is not None and end.data["status"] == "delivered":
            """A new task for an agent that had delivered: a task boundary. The text waits for the first
            step, which may compact the finished work before the new task enters the history."""
            calls = [r for r in records if r.kind == "model_call" and "purpose" not in r.data]
            if calls:
                self.context.restore_measure(Usage(**calls[-1].data["usage"]).context_tokens)
            self._next_task = (text, attachments)
        elif text and text.strip():
            self._add_user(text, "human", attachments)
        elif end is not None or (messages and messages[-1]["role"] == "assistant"):
            self._add_user(
                "[The human resumed this run after it was interrupted. Continue the task, or call `done` if it is "
                "finished.]",
                "engine",
            )

    async def run(self) -> AgentResult:
        """Run until the agent ends. Whatever happens, no process or child of this agent is left running."""
        self.running = True
        try:
            if self._resumed:
                await self._emit("resume", role=self.role.name)
            else:
                self.journal.append(
                    "start",
                    {
                        "run_id": self.runtime.run_id,
                        "role": self.role.name,
                        "task": self.task,
                        "cwd": str(self.cwd),
                        "parent": self.parent_id,
                        "system": self.system,
                        "tools": self.tool_defs,
                    },
                )
                self._add_user(self.task, "parent" if self.parent_id else "human", self._attachments)
                await self._emit("start", role=self.role.name, task=self.task, parent=self.parent_id)
            self.runtime.output(self.id, "agent_start", f"{self.role.name}: {_short(self.task)}")
            while self._finish is None:
                try:
                    await self._step()
                except Exception as error:
                    self._finish = ("failed", f"Engine error: {type(error).__name__}: {error}")
            return await self._end()
        finally:
            self.running = False
            await self._drop_prepared()
            await self._cancel_children()
            await self.jobs.close()

    def stop(self) -> None:
        """The human stops this agent: the model call and tools in progress abort and it ends as ``stopped``."""
        if not self.running or self._mode in ("stop", "cancel"):
            return
        self._mode = "stop"
        self.journal.append("interrupt", {"mode": "stop"})
        self._cancel.set()
        self._wake.set()

    def cancel(self) -> None:
        """The parent (or the runtime) cancels this agent: like ``stop``, but it ends as ``cancelled``.

        It also works before ``run()`` starts: the agent then ends at its first step.
        """
        if self.result is not None or self._mode in ("stop", "cancel"):
            return
        self._mode = "cancel"
        self._cancel.set()
        self._wake.set()

    def add(self, text: str, source: str = "human", attachments: Sequence[Attachment] = ()) -> None:
        """Queue a message; it enters the context at the next step boundary. Empty text is ignored."""
        if not text.strip() or not self.running:
            return
        if source == "human":
            self.journal.append("interrupt", {"mode": "add"})
        self._inbox.append((source, text, attachments))
        self._wake.set()

    def redirect(self, text: str) -> None:
        """Abort what is in progress, as ``stop`` does, then continue with ``text`` as a new user message."""
        if not text.strip() or not self.running or self._mode in ("stop", "cancel"):
            return
        self._mode = "redirect"
        self._redirects.append(text)
        self.journal.append("interrupt", {"mode": "redirect"})
        self._cancel.set()
        self._wake.set()

    def wake(self) -> None:
        """Called by the runtime when something arrived for this agent (a child ended, an answer came)."""
        self._wake.set()

    def spec(self) -> ModelSpec:
        """The model of the step in progress: the role's own, or the fallback for one call after a failure."""
        return self.policy.models[self._model]

    async def _emit(self, event: str, **payload: Any) -> HookOutcome:
        """Emit to the hooks; an interrupt that arrives meanwhile cuts them short.

        Cancelling the emit makes the hooks module kill a running hook
        command, so a stop does not wait for every hook's own timeout. Events
        emitted after the interrupt (``end``, ``child_done``...) run in full.
        A cut emit counts as a block: an interrupted ``before_done`` gate must
        not let a delivery through. The event is captured at the start because
        a redirect replaces it.
        """
        cancel = self._cancel
        emit = self.runtime.hooks.emit(event, {"agent_id": self.id, **payload})
        if cancel.is_set():
            outcome = await emit
        else:
            task = asyncio.ensure_future(emit)
            waiter = asyncio.ensure_future(cancel.wait())
            try:
                await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, waiter, return_exceptions=True)
            if task.cancelled():
                outcome = HookOutcome("block", f"interrupted before the {event} hooks finished")
            else:
                outcome = task.result()
        if outcome.action == "block" or outcome.reason:
            self.journal.append("hook", {"event": event, "action": outcome.action, "reason": outcome.reason})
        return outcome

    def _add_user(self, text: str, source: str, attachments: Sequence[Attachment] = ()) -> None:
        """Mask, add and journal a user message; attachment data is journaled by reference (``context.pack``)."""
        mask = self.runtime.masker.mask
        text = mask(text)
        attachments = [replace(a, name=mask(a.name), note=mask(a.note)) for a in attachments]
        self.context.add_user(text, attachments, pinned=source in PINNED_SOURCES)
        if source in PINNED_SOURCES:
            self._reread = False
        data: dict[str, Any] = {"text": text, "source": source}
        if attachments:
            data["attachments"] = [
                {"name": a.name, "note": a.note, "block": pack(a.block, self.receipts)} for a in attachments
            ]
        self.journal.append("user", data)

    def _journal_result(self, call_id: str, name: str, entry: ToolResultEntry, seconds: float) -> None:
        data: dict[str, Any] = {
            "call_id": call_id,
            "name": name,
            "content": pack(entry.content, self.receipts),
            "is_error": entry.is_error,
            "receipt": entry.receipt,
            "seconds": round(seconds, 3),
        }
        if entry.media is not None:
            data["media"] = entry.media
        self.journal.append("tool_result", data)

    def _repair(self, reason: str) -> None:
        for entry in self.context.repair_dangling(reason):
            self._journal_result(entry.call_id, "", entry, 0.0)

    async def _step(self) -> None:
        if self._cancel.is_set():
            self._interrupt()
            return
        if cap := self._over_cap():
            self._finish = ("budget", f"Stopped: the {cap} cap was reached.")
            return
        paused, self._paused = self._paused, False
        if not paused:
            await self._arrivals()
        self._model, self._next_model = self._next_model or self.role.model, None
        if not paused and not self.context.server_pending():
            await self._maintain_context()
        reply = await self._call_model()
        if reply is not None:
            await self._act(reply)

    async def _arrivals(self) -> None:
        """What enters before a step: a budget warning, the next task, then messages, notices and child results.

        Skipped on the step that continues a ``pause_turn`` reply, which the
        docs say to resend as it is, with no new user message; whatever came
        meanwhile waits for the step after it. Maintenance (clean, compaction)
        is skipped then too, and while a server tool call is open, since both
        edit or extend the turn the provider is still working on.
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
        """
        if warning := self.runtime.ledger.warning_due(self.id):
            await self._emit("budget_warning", warning=warning)
            self.runtime.output(self.id, "warning", warning)
            self._add_user(f"[budget] {warning}. Plan to finish within the budget.", "engine")
        if self._next_task is not None:
            (text, attachments), self._next_task = self._next_task, None
            due = self.context.boundary_due() and not self.context.server_pending()
            if due and (await self._emit("before_compaction")).action != "block":
                await self._compact()
            self._add_user(text, "human", attachments)
        self._deliver()

    def _over_cap(self) -> str | None:
        if cap := self.runtime.ledger.check(self.id):
            return cap
        if self.role.max_steps is not None and self.steps_before + self.steps >= self.role.max_steps:
            return f"role.{self.role.name}.max_steps"
        return None

    def _interrupt(self) -> None:
        """Act on a set cancel event at a step boundary: end the agent, or take the redirect and go on."""
        if self._mode == "redirect":
            texts, self._redirects = self._redirects, []
            self._mode = None
            self._cancel = asyncio.Event()
            self._repair("interrupted by the human")
            self._add_user("\n\n".join(texts), "human")
        elif self._mode == "cancel":
            self._finish = ("cancelled", "Cancelled before it finished.")
        else:
            self._finish = ("stopped", "Stopped by the human.")

    def has_news(self) -> bool:
        """Whether something waits to be delivered at the next step: a message, a child result, a finished job."""
        return (
            bool(self._inbox)
            or any(h.result is not None and not h.reported for h in self.children.values())
            or any(j.background and j.state == "exited" and j.id not in self._announced for j in self.jobs.list())
        )

    async def wait_for(self, *waits: Awaitable[Any], timeout: float | None = None) -> None:
        """Wait for the first of ``waits``, news for this agent (a background job ending included), an interrupt,
        or ``timeout``."""
        self._wake.clear()
        if self.has_news() or self._cancel.is_set():
            for pending in waits:
                if isinstance(pending, Coroutine):
                    pending.close()
            return
        jobs = [j.done.wait() for j in self.jobs.list() if j.background and not j.done.is_set()]
        await wait_any(*waits, *jobs, self._wake.wait(), self._cancel.wait(), timeout=timeout)

    def _deliver(self) -> None:
        """Everything that arrived since the last step enters the context as user text."""
        for job in self.jobs.drain_finished():
            self._add_user(
                f"[background job finished] {job.summary()}. Read its output with the job tool (status, job {job.id}).",
                "engine",
            )
        self._announced.update(j.id for j in self.jobs.list() if j.background and j.state == "exited")
        for handle in self.children.values():
            if handle.result is not None and not handle.reported:
                handle.reported = True
                self._add_user(handle.notice(), "engine")
        inbox, self._inbox = self._inbox, []
        for source, text, attachments in inbox:
            self._add_user(text, source, attachments)

    async def _maintain_context(self) -> None:
        cleaned = self.context.clean_if_cold()
        attachments = self.context.clean_attachments_if_cold()
        if cleaned or attachments:
            record: dict[str, Any] = {"call_ids": cleaned}
            if attachments:
                record["attachments"] = attachments
            self.journal.append("clean", record)
        due = self.context.should_compact(self.spec()) and (
            self._compact_retry_at is None or self.context.last_prompt_tokens >= self._compact_retry_at
        )
        if self._force_compact or due:
            self._force_compact = False
            if (await self._emit("before_compaction")).action != "block":
                await self._compact()

    async def _compact(self) -> None:
        """Ask the model for a summary with the same system, tools and extra (cache), then replace the history
        with it, keeping the newest turns as they are (``Context.compaction_keep``).

        The request text never enters the context; only the summary does. A
        summary counts only when the reply ended normally and has text: a reply
        cut at ``max_tokens`` or one that called a tool is not a summary. Its
        cost is in the ledger either way.

        A failed attempt leaves the history as it is and holds back the next
        one until the measured prompt has passed halfway from its size at the
        failure to the context window. Every step makes the prompt grow, so
        "until it has grown" alone would retry at full price on every step;
        halving the remaining room spaces the retries out without a new knob.
        A compaction forced by an overflow ignores the hold.

        A summary prepared in the background is used instead when there is a
        valid one (``_use_prepared``), and any other is dropped afterwards.
        """
        if await self._use_prepared():
            return
        await self._drop_prepared()
        spec = self.spec()
        system, messages = self.context.build()
        block = {"type": "text", "text": self.context.compaction_request()}
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"].append(block)
        else:
            messages.append({"role": "user", "content": [block]})
        request = ModelRequest(
            spec, system, messages, self.tool_defs, self.role.effort, self.role.max_output_tokens, self._extra(spec)
        )
        try:
            reply = await self.runtime.provider.complete(request, cancel=self._cancel)
        except asyncio.CancelledError:
            if self._cancel.is_set():
                return
            raise
        except ProviderError as error:
            self._compaction_failed(f"compaction failed: {error}")
            return
        self._account(reply, purpose="compaction")
        summary = "\n".join(b["text"] for b in reply.content if b.get("type") == "text").strip()
        if reply.stop_reason not in ENDED_NORMALLY or not summary:
            self._compaction_failed(f"compaction gave no usable summary (stop reason {reply.stop_reason})")
            return
        self._compact_retry_at = None
        if self.policy.context.compact_check:
            summary = await self._checked_summary(request, reply, summary)
        summary = self.runtime.masker.mask(self._with_todo(summary))
        keep = self.context.compaction_keep()
        self.context.apply_compaction(summary, keep)
        self.journal.append("compaction", {"summary": summary, "keep": keep})

    async def _checked_summary(self, request: ModelRequest, draft: ModelReply, summary: str) -> str:
        """The summary after the agent has read it against the conversation it replaces (``context.compact_check``).

        One more call on the same request, with the draft as the assistant turn
        and ``COMPACTION_CHECK`` after it: the model still has the whole
        conversation, so this is the last moment it can notice what the draft
        lost. The reply replaces the draft. A check that fails, is cut or
        calls a tool leaves the draft as it is: a summary nobody reread is
        better than none.
        """
        messages = [
            *request.messages,
            {"role": "assistant", "content": draft.content},
            {"role": "user", "content": [{"type": "text", "text": COMPACTION_CHECK}]},
        ]
        try:
            reply = await self.runtime.provider.complete(replace(request, messages=messages), cancel=self._cancel)
        except asyncio.CancelledError:
            if self._cancel.is_set():
                return summary
            raise
        except ProviderError:
            return summary
        self._account(reply, purpose="compaction_check")
        checked = "\n".join(b["text"] for b in reply.content if b.get("type") == "text").strip()
        return checked if reply.stop_reason in ENDED_NORMALLY and checked else summary

    def _maybe_prepare(self, request: ModelRequest, prompt_tokens: int) -> None:
        """Start a background summary after a step, when ``context.compact_background`` asks for one now.

        Due once the prompt reaches ``compact_prepare_at`` of ``compaction_window``, and
        again each time it has grown ``compact_refresh_tokens`` past the prompt
        a ready summary was made from; one at a time. The request is the one
        this step just sent, with the compaction template appended to its last
        user message, exactly as ``_compact`` builds it: its prefix is the one
        this step wrote to the cache, and the docs say an entry can be read once
        the response that wrote it has started, which it has
        (https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
        Never while a server tool call is open: a text after it would be refused.
        """
        settings = self.policy.context
        if not settings.compact_background or self._finish is not None:
            return
        if self._preparing is not None and not self._preparing.done():
            return
        if prompt_tokens < settings.compact_prepare_at * self.context.compaction_window(request.model):
            return
        ready = self._prepared
        if ready is not None and self.context.fingerprint(ready.covered) != ready.fingerprint:
            self._prepared = ready = None
        if ready is not None and prompt_tokens - ready.tokens < settings.compact_refresh_tokens:
            return
        if not request.messages or request.messages[-1]["role"] != "user" or self.context.server_pending():
            return
        covered = len(request.messages)
        if covered != self.context.message_count:
            return
        messages = copy.deepcopy(request.messages)
        messages[-1]["content"].append({"type": "text", "text": self.context.compaction_request()})
        summary_request = replace(request, messages=messages)
        prepared = _Prepared("", covered, self.context.fingerprint(covered), prompt_tokens)
        self._preparing = asyncio.get_running_loop().create_task(self._prepare(summary_request, prepared))

    async def _prepare(self, request: ModelRequest, prepared: _Prepared) -> None:
        """The background summary call. Its cost is in the ledger and the journal (``purpose``), used or not."""
        try:
            reply = await self.runtime.provider.complete(request, cancel=self._prepare_cancel)
        except Exception as error:
            self.runtime.output(self.id, "warning", f"background summary failed: {_short(str(error))}")
            return
        self._account(reply, purpose="compaction_prepare")
        summary = "\n".join(b["text"] for b in reply.content if b.get("type") == "text").strip()
        if reply.stop_reason in ENDED_NORMALLY and summary:
            self._prepared = replace(prepared, summary=summary)

    async def _use_prepared(self) -> bool:
        """Swap in the background summary, keeping verbatim every message after the part it covers.

        An unfinished one is waited for: it started earlier than a new call
        would. It is valid only while the messages it covers are byte for byte
        the ones it was made from (a clean, a compaction, a thinking strip or a
        message merged into its last turn make it stale) and the tail starts at
        an assistant turn that splits no server tool call. The ``compaction``
        record is the same as a blocking compaction's, ``keep`` being the tail,
        so replay rebuilds the history exactly; a summary not yet used when the
        process dies is lost, its cost already counted.
        """
        if self._preparing is not None and not self._preparing.done():
            await wait_any(asyncio.shield(self._preparing), self._cancel.wait())
            if self._cancel.is_set():
                return False
        prepared, self._prepared = self._prepared, None
        if prepared is None or self.context.fingerprint(prepared.covered) != prepared.fingerprint:
            return False
        keep = self.context.tail_from(prepared.covered)
        if keep is None:
            return False
        self._compact_retry_at = None
        summary = self.runtime.masker.mask(self._with_todo(prepared.summary))
        self.context.apply_compaction(summary, keep)
        self.journal.append("compaction", {"summary": summary, "keep": keep})
        return True

    async def _drop_prepared(self) -> None:
        """Cancel a background summary in flight and forget a ready one (the agent ends, or compacts without it)."""
        self._prepared = None
        task, self._preparing = self._preparing, None
        if task is not None and not task.done():
            self._prepare_cancel.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._prepare_cancel = asyncio.Event()

    def _todo_changed(self, items: list[dict[str, str]]) -> None:
        self.journal.append("todo", {"items": items})
        self.runtime.output(self.id, "todo", render_todo(items))

    def _with_todo(self, summary: str) -> str:
        """A compaction summary with the task list appended, since the calls that built it are summarized away."""
        todo = self.tools.get("todo")
        if not isinstance(todo, TodoTool) or not todo.items:
            return summary
        return f"{summary}\n\nYour task list (`todo`) at this point:\n{render_todo(todo.items)}"

    def _compaction_failed(self, warning: str) -> None:
        measured = self.context.last_prompt_tokens
        self._compact_retry_at = measured + max(0, self.spec().context_window - measured) // 2
        self.runtime.output(self.id, "warning", warning)

    def _extra(self, spec: ModelSpec) -> dict[str, Any]:
        """Vendor fields for the request: the model's ``extra``, then the role's over it."""
        return deep_merge(spec.extra, self.role.extra)

    async def _call_model(self) -> ModelReply | None:
        spec = self.spec()
        system, messages = self.context.build()
        limit = self._limit_override or self.role.max_output_tokens
        self._limit_override = None
        self._last_limit = limit or spec.max_output_tokens
        request = ModelRequest(spec, system, messages, self.tool_defs, self.role.effort, limit, self._extra(spec))
        await self._emit("before_model", model=spec.alias, step=self.steps + 1)
        try:
            reply = await self.runtime.provider.complete(
                request, on_text=lambda text: self.runtime.output(self.id, "text", text), cancel=self._cancel
            )
        except asyncio.CancelledError:
            if self._cancel.is_set():
                return None
            raise
        except ProviderError as error:
            if _too_long(error) and not self._overflow_retried:
                self._overflow_retried = True
                self._force_compact = True
                self.runtime.output(self.id, "warning", f"the prompt is too long, compacting: {_short(str(error))}")
            elif _unbound_thinking(error) and not self._thinking_retried:
                """Drop every thinking block and try once more. The record makes each later replay drop
                them at this same point, so the blocks the API refused are never sent again."""
                self._thinking_retried = True
                self.context.drop_thinking()
                self.journal.append("thinking_dropped", {"reason": _short(str(error))})
                self.runtime.output(self.id, "warning", "the API rejected earlier thinking: dropped it and trying again")
            elif error.retryable and spec.fallback:
                self._switch(spec.fallback, str(error))
            else:
                self._finish = ("failed", f"The model call failed: {error}")
            return None
        self._overflow_retried = False
        self._thinking_retried = False
        self.steps += 1
        self.context.note_model_call(reply.usage)
        self._account(reply)
        self._maybe_prepare(request, reply.usage.context_tokens)
        await self._emit("after_model", stop_reason=reply.stop_reason, usage=asdict(reply.usage))
        return reply

    def _account(self, reply: ModelReply, purpose: str | None = None) -> None:
        self.usage.add(reply.usage)
        self.runtime.ledger.add(self.id, reply.usage, step=purpose is None)
        data: dict[str, Any] = {
            "model_id": reply.model_id,
            "stop_reason": reply.stop_reason,
            "usage": asdict(reply.usage),
            "cost_usd": reply.usage.cost_usd,
        }
        if purpose:
            data["purpose"] = purpose
        self.journal.append("model_call", data)

    def _switch(self, alias: str, reason: str) -> None:
        """Send the next call to the fallback model; once a call succeeds there, the role's model is used again."""
        self.journal.append("fallback", {"from": self._model, "to": alias, "reason": reason})
        self.runtime.output(self.id, "warning", f"model {self._model} -> {alias} for the next call: {_short(reason)}")
        self._next_model = alias

    def _append(self, content: list[Block]) -> bool:
        """Mask the reply and add it to the context and the journal.

        Masking a block is an edit of what the model wrote, and a thinking
        block placed after it in the same reply was made on the unmasked text,
        so such thinking blocks are dropped (the omission the docs allow; see
        ``Context.clean_if_cold``). Thinking before the first edit is kept.
        """
        masker = self.runtime.masker
        blocks: list[Block] = []
        edited = False
        for block in content:
            if block.get("type") in THINKING:
                if not edited:
                    blocks.append(block)
                continue
            masked = masker.mask_obj(block)
            edited = edited or masked != block
            blocks.append(masked)
        if not blocks:
            return False
        released = self.context.add_assistant(blocks)
        self.journal.append("assistant", {"content": pack(blocks, self.receipts)})
        return released

    async def _act(self, reply: ModelReply) -> None:
        """Do what the provider docstring prescribes for each stop reason."""
        reason = reply.stop_reason
        calls = [b for b in reply.content if b.get("type") == "tool_use"]
        if reason == "refusal":
            if fallback := self.spec().fallback:
                self._switch(fallback, "the model refused")
            else:
                self._finish = ("failed", "The model refused to continue and no fallback model is set.")
            return
        if reason in ("max_tokens", "model_context_window_exceeded"):
            self._force_compact = reason == "model_context_window_exceeded"
            await self._truncated(reply, calls)
            return
        self._cut_turns = 0
        released = self._append(reply.content)
        if reason in ("pause_turn", "compaction"):
            self._paused = reason == "pause_turn"
            return
        if calls:
            self._idle_turns = 0
            await self._run_calls(calls)
        else:
            await self._idle(reply, released)

    async def _truncated(self, reply: ModelReply, calls: list[Block]) -> None:
        """A reply cut by a limit: never run or keep a cut tool call; keep cut text and let the model go on.

        A thinking block cut before its signature arrived (empty signature) is
        dropped from the end of the kept text: the API would reject it. So is a
        trailing ``server_tool_use`` without its result, whose input may be cut
        too: kept, the provider would run it on the next request.
        """
        if not calls:
            self._cut_turns = 0
            content = list(reply.content)
            answered = {b.get("tool_use_id") for b in content if b.get("type") != "tool_result"}
            while content and (
                (content[-1].get("type") == "thinking" and not content[-1].get("signature"))
                or (content[-1].get("type") == "server_tool_use" and content[-1].get("id") not in answered)
            ):
                content.pop()
            if content:
                self._append(content)
            self._add_user("[Your reply was cut off at the output limit. Continue from where it stopped.]", "engine")
            return
        if reply.stop_reason == "max_tokens" and self._last_limit < self.spec().max_output_tokens:
            self._limit_override = self.spec().max_output_tokens
            return
        self._cut_turns += 1
        if self._cut_turns > 1:
            self._finish = ("failed", "Two replies in a row were cut off before their tool calls were complete.")
            return
        self._add_user(
            "[Your last reply was cut off before its tool call was complete and was discarded. "
            "Work in smaller steps: shorter text and smaller tool inputs.]",
            "engine",
        )

    async def _idle(self, reply: ModelReply, released: bool = False) -> None:
        """The model ended its turn without calling a tool.

        With children running this is a parent waiting for them: it sleeps
        until something arrives. Otherwise it is reminded once to continue or
        call ``done``; a second time in a row ends the agent as ``failed``.
        Messages held during a server tool call that entered with this reply
        (``released``, see ``Context.add_user``) are news as well.
        """
        if self.has_news() or released:
            return
        if any(not h.done.is_set() for h in self.children.values()):
            while not self.has_news() and not self._cancel.is_set():
                await self.wait_for()
            return
        self._idle_turns += 1
        if self._idle_turns > 1:
            last = " ".join(b["text"] for b in reply.content if b.get("type") == "text").strip()
            self._finish = ("failed", "Ended its turn twice without calling done." + (f" Last words: {last}" if last else ""))
            return
        self._add_user("[You ended your turn without calling a tool. Continue the task, or call `done` if it is finished.]", "engine")

    def _target(self, call: ToolCall) -> str:
        """What permission rules match against for this call."""
        tool_input = call.input
        match call.name:
            case "sh":
                return str(tool_input.get("command", ""))
            case "edit" | "view" | "read" | "grep":
                raw = tool_input.get("path") or ("." if call.name == "grep" else None)
                if not isinstance(raw, str) or not raw:
                    return ""
                location = self.runtime.sandbox.locate(raw, self.cwd)
                if location.host is None:
                    return location.shown
                try:
                    return location.host.relative_to(self.cwd.resolve()).as_posix()
                except ValueError:
                    return location.shown if self.runtime.sandbox.kind == "docker" else location.host.as_posix()
            case "mcp":
                return f"{tool_input.get('server', '')}/{tool_input.get('tool', '')}"
            case "spawn":
                return str(tool_input.get("role", ""))
        return ""

    def tool_context(self) -> ToolContext:
        env = dict(self.policy.environment.env)
        if dirs := self.runtime.plugins.path_dirs:
            current = env.get("PATH") or os.environ.get("PATH", "")
            env["PATH"] = os.pathsep.join([*(str(d) for d in dirs), current])
        before_write = self.recorder.before_write if self.recorder and self.policy.checkpoints.edits else None
        return ToolContext(
            self.id,
            self.cwd,
            env,
            self.policy.environment.tool_timeout_seconds,
            self._cancel,
            self.runtime.sandbox,
            before_write,
        )

    async def _run_calls(self, blocks: list[Block]) -> None:
        """Admit each call in order, run the admitted ones (same resource in order, the rest at once), record all."""
        calls = [
            ToolCall(b["id"], b["name"], b["input"] if isinstance(b.get("input"), dict) else {}) for b in blocks
        ]
        done: list[tuple[ToolCall, ToolResultEntry]] = []
        texts: list[str] = []
        admitted: list[tuple[ToolCall, Tool, dict[str, Any]]] = []
        for call in calls:
            self.journal.append("tool_intent", {"call_id": call.id, "name": call.name, "input": call.input})
            self.runtime.output(self.id, "tool_start", f"{call.name} {_short(json.dumps(call.input, ensure_ascii=False))}")
            verdict = await self._admit(call, texts)
            if isinstance(verdict, ToolOutput):
                done.append((call, self._record(call, verdict, 0.0)))
            else:
                admitted.append((call, *verdict))
        ctx = self.tool_context()
        if self.recorder and self.policy.checkpoints.shell and any(call.name == "sh" for call, _, _ in admitted):
            try:
                await self.recorder.snapshot(self.cwd)
            except Exception as error:
                self.runtime.output(self.id, "warning", f"checkpoint failed: {_short(str(error))}")
        lanes: dict[str, list[tuple[ToolCall, Tool, dict[str, Any]]]] = {}
        groups: list[list[tuple[ToolCall, Tool, dict[str, Any]]]] = []
        for item in admitted:
            try:
                key = item[1].resource(item[2], ctx)
            except Exception:
                key = None
            if key is None:
                groups.append([item])
            else:
                if key not in lanes:
                    lanes[key] = []
                    groups.append(lanes[key])
                lanes[key].append(item)

        async def lane(items: list[tuple[ToolCall, Tool, dict[str, Any]]]) -> None:
            for call, tool, tool_input in items:
                done.append((call, await self._execute(call, tool, tool_input, ctx, texts)))

        await asyncio.gather(*(lane(group) for group in groups))
        self.context.add_tool_results([entry for _, entry in done])
        for text in texts:
            self._add_user(text, "hook")
        order = {call.id: index for index, call in enumerate(calls)}
        self._guard_repeats(sorted(done, key=lambda pair: order[pair[0].id]))

    async def _admit(self, call: ToolCall, texts: list[str]) -> ToolOutput | tuple[Tool, dict[str, Any]]:
        """The tool and input to run, or the result that replaces running it (interrupted, unknown, denied, blocked)."""
        if self._cancel.is_set():
            return ToolOutput(CANCELLED_BEFORE_START, is_error=True)
        tool = self.tools.get(call.name)
        if tool is None:
            return ToolOutput(f"Unknown tool {call.name!r}. Available tools: {', '.join(self.tools)}.", is_error=True)
        if denied := await self._permit(call):
            return denied
        outcome = await self._emit("before_tool", tool=call.name, call_id=call.id, input=call.input)
        if outcome.add_text:
            texts.append(outcome.add_text)
        if outcome.action == "block":
            return ToolOutput(f"Blocked by a hook: {outcome.reason}", is_error=True)
        replaced = (outcome.replace or {}).get("input")
        if not isinstance(replaced, dict) or replaced == call.input:
            return tool, call.input
        if denied := await self._permit(replace(call, input=replaced)):
            return denied
        return tool, replaced

    async def _permit(self, call: ToolCall) -> ToolOutput | None:
        """``None`` when the permission rules let ``call`` run, else the denial to return as its result."""
        if call.name in SELF_TOOLS:
            return None
        target = self._target(call)
        decision, rule = decide(self.policy, call.name, target)
        how = f"the rule '{rule.tool}: {rule.pattern}' of the {rule.level} policy" if rule else "the default permission"
        if decision == "ask":
            decision, how = await self._permission(call, target, how)
        if decision == "deny":
            shown = f" {target}" if target else ""
            return ToolOutput(f"Denied by {how}: {call.name}{shown}. Do not retry it; find another way.", is_error=True)
        return None

    async def _permission(self, call: ToolCall, target: str, how: str) -> tuple[Decision, str]:
        """Resolve an ``ask`` rule through ``policy.human``: wait for a yes or no, else the fallback."""
        human = self.policy.human
        if human.wait_seconds > 0:
            question = self._open("permission", f"Allow {' '.join(filter(None, (call.name, target)))}?", "")
            answer = await self._await(question, human.wait_seconds)
            self.runtime.close_question(question.id)
            if self._cancel.is_set():
                return "deny", "an interrupt (the agent was stopped while waiting for approval)"
            if answer is not None:
                verdict: Decision = "allow" if _yes(answer) else "deny"
                return verdict, f"the human ({how} asked for approval)"
        return human.permission_fallback, f"the permission fallback ({how} asked for approval, nobody answered)"

    def _open(self, kind: Literal["question", "permission"], text: str, assumption: str) -> Question:
        question = Question(f"{self.id}-q{next(self._questions)}", self.id, kind, text, assumption)
        self.runtime.open_question(question)
        self.runtime.output(self.id, "question", f"{question.id}: {text}")
        return question

    async def _await(self, question: Question, seconds: float) -> str | None:
        """Wait for an answer, at most ``seconds`` and never past an interrupt."""
        question.waiting = True
        try:
            await wait_any(asyncio.shield(question.future), self._cancel.wait(), timeout=seconds)
        finally:
            question.waiting = False
        return question.future.result() if question.future.done() else None

    async def _execute(
        self, call: ToolCall, tool: Tool, tool_input: dict[str, Any], ctx: ToolContext, texts: list[str]
    ) -> ToolResultEntry:
        if ctx.cancel.is_set():
            return self._record(call, ToolOutput(CANCELLED_BEFORE_START, is_error=True), 0.0)
        started = time.monotonic()
        try:
            output = await tool.run(tool_input, ctx)
        except Exception as error:
            output = ToolOutput(f"The tool failed: {type(error).__name__}: {error}", is_error=True)
        seconds = time.monotonic() - started
        outcome = await self._emit(
            "after_tool", tool=call.name, call_id=call.id, input=tool_input, output=output.text, is_error=output.is_error
        )
        replaced = (outcome.replace or {}).get("output")
        if isinstance(replaced, str):
            output = replace(output, text=replaced)
        if outcome.add_text:
            texts.append(outcome.add_text)
        return self._record(call, output, seconds)

    def _record(self, call: ToolCall, output: ToolOutput, seconds: float) -> ToolResultEntry:
        """Mask, cap and journal one result; the context gets it with the rest of the turn."""
        output = replace(output, text=self.runtime.masker.mask(output.text))
        entry = self.context.cap(call, output)
        self._journal_result(call.id, call.name, entry, seconds)
        code = output.meta.get("exit_code")
        state = "error" if entry.is_error else f"exit {code}" if code else "ok"
        self.runtime.output(self.id, "tool_end", f"{call.name} {state} in {seconds:.1f}s")
        return entry

    def _guard_repeats(self, results: list[tuple[ToolCall, ToolResultEntry]]) -> None:
        """Same call, same result, ``repeat_limit`` times in a row: warn once; as many times again: fail."""
        limit = self.policy.agents.repeat_limit
        if limit <= 0 or not results:
            return
        for call, entry in results:
            key = (
                f"{call.name}\n{json.dumps(call.input, sort_keys=True, ensure_ascii=False)}",
                json.dumps([entry.content, entry.is_error], ensure_ascii=False),
            )
            if key == self._repeat:
                self._repeat_count += 1
            else:
                self._repeat, self._repeat_count, self._repeat_warned = key, 1, False
        if self._repeat_count >= 2 * limit:
            self._finish = ("failed", f"Repeated the same {call.name} call {self._repeat_count} times with the same result.")
        elif self._repeat_count >= limit and not self._repeat_warned:
            self._repeat_warned = True
            self._add_user(
                f"[You made the same {call.name} call {self._repeat_count} times in a row and got the same result "
                "each time. Repeating it will not change the result: try something different.]",
                "engine",
            )

    async def _done(self, tool_input: dict[str, Any]) -> ToolOutput:
        summary = str(tool_input.get("summary") or "").strip()
        if not summary:
            return ToolOutput("`summary` is required: say what was done and how it was verified.", is_error=True)
        status: Status = "failed" if tool_input.get("status") == "failed" else "delivered"
        schema = self.result_schema
        if schema is not None and status == "delivered":
            if "result" not in tool_input:
                return ToolOutput("Not finished: `result` is required, matching the schema in this tool.", is_error=True)
            if errors := validate_result(tool_input["result"], schema):
                listed = "\n".join(f"- {e}" for e in errors)
                return ToolOutput(f"Not finished: `result` does not match the schema:\n{listed}", is_error=True)
        if status == "delivered" and self._reread_due():
            return ToolOutput(REREAD.format(asked=self.context.asked()), is_error=True)
        outcome = await self._emit("before_done", summary=summary, status=status)
        if outcome.action == "block":
            return ToolOutput(
                f"Not finished: a hook blocked the delivery: {outcome.reason}. Deal with it, then call done again.",
                is_error=True,
            )
        await self._cancel_children()
        if status == "delivered" and verify.active(self):
            checked = await verify.Verifier(self).check(summary)
            if isinstance(checked, ToolOutput):
                return checked
            summary = checked
        self._finish = (status, summary)
        if schema is not None and status == "delivered":
            self._result = (self.runtime.masker.mask_obj(tool_input["result"]),)
        self.ended_by_done = True
        return ToolOutput("Finished.")

    def _reread_due(self) -> bool:
        """Whether this ``done`` is answered with what was asked instead of delivering (``verify.reread``).

        Once per delivery of the root agent, and again after each new message
        from the human. It is not a check by someone else: late in a long
        task the agent's attention is on its last steps, and reading the
        request again is what brings back a requirement it left behind.
        """
        if self.parent_id is not None or not self.policy.verify.reread or self._reread or not self.context.asked():
            return False
        self._reread = True
        return True

    @property
    def cancel_event(self) -> asyncio.Event:
        """The event an interrupt sets; a redirect replaces it, so read it when it is needed."""
        return self._cancel

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    async def _ask(self, tool_input: dict[str, Any]) -> ToolOutput:
        text = str(tool_input.get("question") or "").strip()
        assumption = str(tool_input.get("assumption") or "").strip()
        if not text or not assumption:
            return ToolOutput("Both `question` and `assumption` are required.", is_error=True)
        question = self._open("question", text, assumption)
        self.journal.append("question", {"question_id": question.id, "text": text, "assumption": assumption})
        await self._emit("question", question_id=question.id, text=text, assumption=assumption)
        human = self.policy.human
        if human.wait_seconds > 0:
            answer = await self._await(question, human.wait_seconds)
            if answer is not None:
                self.runtime.close_question(question.id)
                return ToolOutput(f"The human answered: {answer}")
            if self._cancel.is_set():
                return ToolOutput(f"Interrupted while waiting for an answer to {question.id}.", is_error=True)
        if human.question_fallback == "stop":
            self._finish = ("stopped", f"Stopped on a question for the human ({question.id}): {text}")
            return ToolOutput("No answer. The rules say to stop here.")
        return ToolOutput(
            f"No answer now ({question.id}). Proceed on your assumption: {assumption}. "
            "If the human answers later, the answer arrives as a message."
        )

    async def _cancel_children(self) -> None:
        running = [h for h in self.children.values() if not h.done.is_set()]
        if running:
            await asyncio.gather(*(h.cancel() for h in running))

    async def _end(self) -> AgentResult:
        status, summary = self._finish or ("failed", "")
        await self._drop_prepared()
        self._repair("the agent ended")
        await self._cancel_children()
        await self.jobs.close()
        self.runtime.close_questions(self.id)
        changes = ""
        if self.own_copy:
            try:
                changes = await self.runtime.workspace.changes(self.id)
            except Exception as error:
                changes = f"(changes unknown: {error})"
        given = self._result if status == "delivered" else None
        result = given[0] if given else None
        record: dict[str, Any] = {"status": status, "summary": summary}
        if given:
            record["result"] = result
        self.journal.append("done", record)
        await self._emit("end", status=status, summary=summary)
        self.runtime.output(self.id, "agent_end", f"{status}: {_short(summary)}")
        self.result = AgentResult(self.id, status, summary, self.usage, self.steps, changes, result)
        return self.result
