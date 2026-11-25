"""What a run has done so far, kept in one place for every view of the interface.

A ``Board`` holds the agents of a run and their steps in the order they happened. It is filled live from
the output callback of the runtime (``feed``) or from a stored journal (``load``), and it tells its
listeners about each change, so the views never read the engine themselves. Nothing here imports Textual.
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthic.journal import agents_in, read_records
from synthic.transcript import entries

ROOT = "root"
ENDED = ("delivered", "failed", "stopped", "cancelled", "blocked")
Listener = Callable[[str, Any], None]


@dataclass
class Step:
    """One thing an agent did or was told: a row of the ledger and an entry of a conversation.

    ``kind`` is ``user``, ``text`` (the model speaking), ``tool``, ``question``, ``note`` or ``end``.
    ``result`` is ``running``, ``ok``, ``error`` or ``exit N`` for a tool, ``waiting`` or ``answered``
    for a question, and the final status for an ``end``.
    """

    n: int
    ts: float
    agent: str
    kind: str
    text: str
    result: str = ""
    seconds: float | None = None
    cost: float = 0.0
    error: bool = False
    name: str = ""
    input: dict[str, Any] | None = None
    detail: str = ""
    ref: str = ""
    """The id of a question, so the answer can be written without looking it up."""


@dataclass
class AgentView:
    id: str
    role: str = ""
    task: str = ""
    parent: str | None = None
    state: str = "working"
    steps: list[Step] = field(default_factory=list)
    cost: float = 0.0
    calls: int = 0
    context: int = 0
    depth: int = 0


class Board:
    def __init__(self) -> None:
        self.agents: dict[str, AgentView] = {}
        self.steps: list[Step] = []
        self.spend: list[tuple[float, float]] = []
        self.started = 0.0
        self.run_dir: Path | None = None
        self._listeners: list[Listener] = []
        self._speaking: dict[str, Step] = {}
        self._pending: dict[tuple[str, str], list[Step]] = {}
        self._billed: dict[str, float] = {}

    def listen(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def tell(self, event: str, subject: Any = None) -> None:
        """Events: ``reset``, ``agent`` (an ``AgentView`` appeared or changed), ``step`` (added), ``change``."""
        for listener in list(self._listeners):
            listener(event, subject)

    def reset(self) -> None:
        self.agents.clear()
        self.steps.clear()
        self.spend.clear()
        self._speaking.clear()
        self._pending.clear()
        self._billed.clear()
        self.started, self.run_dir = 0.0, None
        self.tell("reset")

    @property
    def total(self) -> float:
        return sum(agent.cost for agent in self.agents.values())

    def agent(self, agent_id: str) -> AgentView:
        view = self.agents.get(agent_id)
        if view is None:
            view = self.agents[agent_id] = AgentView(agent_id, depth=0 if agent_id == ROOT else 1)
            self.tell("agent", view)
        return view

    def waiting(self) -> list[Step]:
        return [step for step in self.steps if step.kind == "question" and step.result == "waiting"]

    def _add(self, agent_id: str, kind: str, text: str, ts: float | None = None, **more: Any) -> Step:
        step = Step(len(self.steps) + 1, time.time() if ts is None else ts, agent_id, kind, text, **more)
        if not self.steps:
            self.started = step.ts
        self.steps.append(step)
        self.agent(agent_id).steps.append(step)
        self.tell("step", step)
        return step

    def _close(self, agent_id: str) -> None:
        """What the model was saying ends when the same agent does anything else."""
        step = self._speaking.pop(agent_id, None)
        if step is not None:
            step.text = step.text.strip()
            self.tell("change", step)

    def _state(self, view: AgentView, state: str) -> None:
        if view.state != state:
            view.state = state
            self.tell("agent", view)

    """Live: the output callback of the runtime."""

    def said(self, text: str, agent_id: str = ROOT, note: str = "") -> Step:
        """A message of the human, as it enters the conversation of an agent."""
        self._close(agent_id)
        view = self.agent(agent_id)
        if view.state in ENDED or not view.steps:
            self._state(view, "working")
        return self._add(agent_id, "user", f"({note}) {text}" if note else text)

    def answered(self, question_id: str) -> None:
        for step in self.waiting():
            if step.ref == question_id:
                step.result = "answered"
                self.tell("change", step)
                view = self.agent(step.agent)
                if view.state == "waiting":
                    self._state(view, "working")

    def measure(self, agent_id: str, cost: float, calls: int, context: int) -> None:
        """Take the figures of the ledger for one agent; what was spent since the last step goes to a step.

        Usage is added when a model call ends, after its text streamed, so the spend belongs to what the
        agent was saying when there is such a step, and to its newest step otherwise.
        """
        view = self.agent(agent_id)
        fresh = cost - self._billed.get(agent_id, 0.0)
        changed = (view.cost, view.calls, view.context) != (cost, calls, context)
        view.cost, view.calls, view.context = cost, calls, context
        if fresh > 1e-9 and view.steps:
            self._billed[agent_id] = cost
            target = self._speaking.get(agent_id) or view.steps[-1]
            target.cost += fresh
            self.spend.append((time.time(), self.total))
            self.tell("change", target)
        if changed:
            self.tell("agent", view)

    def feed(self, agent_id: str, kind: str, text: str) -> None:
        view = self.agent(agent_id)
        name = text.partition(":")[0].partition(" ")[0]
        if kind == "text":
            step = self._speaking.get(agent_id)
            if step is None:
                self._speaking[agent_id] = self._add(agent_id, "text", text)
            else:
                step.text += text
                self.tell("change", step)
            return
        self._close(agent_id)
        match kind:
            case "tool_start" if name != "done":
                step = self._add(agent_id, "tool", text, name=name, result="running")
                self._pending.setdefault((agent_id, name), []).append(step)
            case "tool_end" if name != "done":
                waiting = self._pending.get((agent_id, name))
                if waiting:
                    step = waiting.pop(0)
                    outcome, _, took = text.partition(" ")[2].rpartition(" in ")
                    step.result = outcome or "ok"
                    step.error = not step.result.startswith("ok")
                    step.seconds = float(took.rstrip("s")) if took.rstrip("s").replace(".", "").isdigit() else None
                    self.tell("change", step)
            case "tool_start" | "tool_end":
                pass
            case "agent_start":
                view.role, _, task = text.partition(": ")
                view.task = view.task or task
                self._state(view, "working")
                self.tell("agent", view)
            case "agent_end":
                status, _, summary = text.partition(": ")
                for step in self.waiting():
                    if step.agent == agent_id:
                        step.result = "closed"
                if agent_id != ROOT:
                    self._add(agent_id, "end", summary, result=status, error=status != "delivered")
                self._state(view, status)
            case "question":
                question_id, _, asked = text.partition(": ")
                self._add(agent_id, "question", asked, result="waiting", ref=question_id)
                self._state(view, "waiting")
            case "warning":
                self._add(agent_id, "note", text)
            case "todo":
                self._add(agent_id, "note", "Task list\n" + text)
            case _:
                self._add(agent_id, "note", f"{kind}: {text}")

    def ended(self, status: str, summary: str) -> None:
        """The run ended: the root has no ``agent_end`` step of its own, its delivery is written here."""
        self._close(ROOT)
        self._add(ROOT, "end", summary, result=status, error=status != "delivered")
        for view in self.agents.values():
            if view.state not in ENDED:
                self._state(view, status if view.id == ROOT else "stopped")

    """Stored: a run read back from its journal."""

    def load(self, run_dir: Path) -> None:
        self.reset()
        self.run_dir = run_dir
        rows: list[tuple[float, int, str, Any]] = []
        bills: dict[str, list[tuple[float, float]]] = {}
        order = 0
        for agent_id in sorted(agents_in(run_dir), key=lambda name: (name != ROOT, name)):
            records = list(read_records(run_dir, agent_id))
            view = self.agent(agent_id)
            for record in records:
                if record.kind == "start":
                    view.role = str(record.data.get("role", ""))
                    view.task = str(record.data.get("task", ""))
                    view.parent = record.data.get("parent")
                elif record.kind == "model_call":
                    cost = float(record.data.get("cost_usd", 0.0))
                    bills.setdefault(agent_id, []).append((record.ts, cost))
                    view.cost += cost
                    view.calls += "purpose" not in record.data
                    usage = record.data.get("usage") or {}
                    view.context = sum(int(usage.get(key, 0)) for key in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
            for entry in entries(records):
                rows.append((entry.ts, order, agent_id, entry))
                order += 1
            view.state = "stopped"
        rows.sort(key=lambda row: (row[0], row[1]))
        total = 0.0
        for ts, _, agent_id, entry in rows:
            due = [bill for bill in bills.get(agent_id, []) if bill[0] <= ts]
            bills[agent_id] = bills.get(agent_id, [])[len(due) :]
            cost = sum(bill[1] for bill in due)
            view = self.agents[agent_id]
            match entry.kind:
                case "tool":
                    result = "error" if entry.error else "ok"
                    step = self._add(
                        agent_id, "tool", entry.text, ts, name=entry.name, input=entry.input, detail=entry.detail,
                        result=result, error=entry.error, seconds=entry.seconds, cost=cost,
                    )
                case "end":
                    step = self._add(agent_id, "end", entry.detail, ts, result=entry.text, error=entry.error, cost=cost)
                    view.state = entry.text
                case _:
                    step = self._add(agent_id, entry.kind, entry.text, ts, cost=cost)
            if cost:
                total += cost
                self.spend.append((step.ts, total))
        for view in self.agents.values():
            self._billed[view.id] = view.cost
            parent = self.agents.get(view.parent or "")
            view.depth = 0 if view.id == ROOT else (parent.depth + 1 if parent else 1)
            self.tell("agent", view)

    """The full text of one step, for the pane that shows it."""

    def detail(self, step: Step) -> list[tuple[str, str]]:
        """Lines of ``(tone, text)`` with tone ``ink``, ``muted``, ``up`` or ``down``."""
        if step.kind != "tool":
            return [("ink", line) for line in step.text.splitlines()]
        if self.run_dir is not None and (step.input is None or (not step.detail and step.result != "running")):
            self._fetch(step)
        lines = [("muted", f"● {step.text}"), ("muted", "")]
        tool_input = step.input or {}
        if step.name == "edit" and tool_input.get("old"):
            old, new = str(tool_input.get("old", "")).splitlines(), str(tool_input.get("new", "")).splitlines()
            for line in list(difflib.unified_diff(old, new, lineterm="", n=2))[2:]:
                tone = "up" if line.startswith("+") else "down" if line.startswith("-") else "muted"
                lines.append((tone, line))
        elif step.name == "edit":
            lines += [("up", "+" + line) for line in str(tool_input.get("new", "")).splitlines()]
        if step.detail.strip():
            if len(lines) > 2:
                lines.append(("muted", ""))
            lines += [("down" if step.error else "ink", line) for line in step.detail.splitlines()]
        elif step.result == "running":
            lines.append(("muted", "running"))
        return lines

    def _fetch(self, step: Step) -> None:
        """A live step knows its line and not its input or output: read them from the journal, by position."""
        assert self.run_dir is not None
        mine = [s for s in self.agents[step.agent].steps if s.kind == "tool"]
        stored = [e for e in entries(read_records(self.run_dir, step.agent)) if e.kind == "tool"]
        offset = len(stored) - len(mine)
        index = mine.index(step) + offset
        entry = stored[index] if 0 <= index < len(stored) and stored[index].text == step.text else None
        if entry is None:
            entry = next((e for e in reversed(stored) if e.text == step.text), None)
        if entry is not None:
            step.input = entry.input
            if step.result != "running":
                step.detail = entry.detail
