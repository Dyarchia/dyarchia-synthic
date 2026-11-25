"""A run as a person reads it: one line per tool call, and a whole run as a conversation.

The engine, ``synthic show`` and every interface word a tool call with ``tool_line``, so a live run and a
stored one read the same. Nothing here imports an interface.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .types import Record


def one_line(text: str, limit: int = 160) -> str:
    line = " ".join(text.split())
    return line if len(line) <= limit else line[: limit - 3] + "..."


def _lines(text: str) -> int:
    return len(text.splitlines())


def tool_target(name: str, tool_input: dict[str, Any]) -> str:
    """What a tool call acts on, in the words a person would use for it."""
    get = tool_input.get
    match name:
        case "sh":
            return str(get("command", ""))
        case "edit":
            old, new = str(get("old", "")), str(get("new", ""))
            change = f"new file, {_lines(new)} lines" if not old else f"+{_lines(new)} -{_lines(old)}"
            return f"{get('path', '')} ({change})"
        case "view" | "read":
            lines = f" lines {get('lines')}" if get("lines") else ""
            return f"{get('path', '')}{lines}"
        case "grep":
            where = f" in {get('path')}" if get("path") else ""
            return f"{get('pattern', '')}{where}"
        case "spawn":
            return f"{get('role', '')}: {get('task', '')}"
        case "mcp":
            return f"{get('server', '')}/{get('tool', '')}"
        case "skill":
            return str(get("name", ""))
        case "receipt":
            detail = get("search") or get("lines") or ""
            return f"{get('id', '')} {detail}".strip()
        case "done":
            return f"{get('status', '')}: {get('summary', '')}"
        case "ask":
            return str(get("question", ""))
        case "todo":
            return f"{len(get('items', []))} items"
        case "job":
            return " ".join(str(v) for v in tool_input.values())
    return ", ".join(f"{key}={value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)}" for key, value in tool_input.items())


def tool_line(name: str, tool_input: dict[str, Any], limit: int | None = 160) -> str:
    """``sh: git status``, ``edit: a.txt (+3 -1)``: the tool and what it acts on, on one line."""
    target = tool_target(name, tool_input)
    target = " ".join(target.split()) if limit is None else one_line(target, limit)
    return f"{name}: {target}" if target else name


def content_text(content: Any) -> str:
    """The text of a tool result, whose content is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")) if block.get("type") == "text" else f"[{block.get('type')}]")
    return "\n".join(parts)


@dataclass(frozen=True)
class Entry:
    """One thing that happened in a run.

    ``kind`` is ``user``, ``text`` (the model speaking), ``tool`` (a call, with ``detail`` holding its
    result and ``error`` saying whether it failed), ``note`` or ``end``.
    """

    kind: str
    text: str
    detail: str = ""
    error: bool = False
    ts: float = 0.0


def entries(records: Iterable[Record]) -> Iterator[Entry]:
    """The records of one agent as the conversation it was: what was asked, said, done, and how it ended."""
    records = list(records)
    results = {r.data.get("call_id"): r for r in records if r.kind == "tool_result"}
    for record in records:
        data = record.data
        match record.kind:
            case "user" if data.get("source", "human") in ("human", "parent"):
                yield Entry("user", str(data.get("text", "")), ts=record.ts)
            case "assistant":
                for block in data.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text", "")).strip():
                        yield Entry("text", str(block["text"]).strip(), ts=record.ts)
            case "tool_intent":
                name = str(data.get("name"))
                tool_input = data.get("input") or {}
                if name == "done":
                    continue
                result = results.get(data.get("call_id"))
                if result is None:
                    yield Entry("tool", tool_line(name, tool_input), "no result", True, record.ts)
                    continue
                failed = bool(result.data.get("is_error"))
                yield Entry("tool", tool_line(name, tool_input), content_text(result.data.get("content", "")), failed, record.ts)
            case "question":
                yield Entry("note", f"question {data.get('question_id')}: {data.get('text', '')}", ts=record.ts)
            case "answer":
                yield Entry("user", f"(answer {data.get('question_id')}) {data.get('text', '')}", ts=record.ts)
            case "compaction":
                yield Entry("note", "compaction: the conversation so far was replaced by a summary", str(data.get("summary", "")), ts=record.ts)
            case "fallback":
                yield Entry("note", f"model {data.get('from')} -> {data.get('to')}: {data.get('reason', '')}", ts=record.ts)
            case "thinking_dropped":
                yield Entry("note", "the API rejected earlier thinking; it was dropped", ts=record.ts)
            case "done":
                failed = data.get("status") != "delivered"
                yield Entry("end", str(data.get("status")), str(data.get("summary", "")), failed, record.ts)


def result_summary(entry: Entry) -> str:
    """A tool result in a few words: ``ok, 12 lines`` or ``error: the first line``."""
    lines = [line for line in entry.detail.splitlines() if line.strip()]
    if entry.error:
        return "error: " + (one_line(lines[0], 120) if lines else "no output")
    return f"ok, {len(lines)} line{'s' if len(lines) != 1 else ''}" if lines else "ok"


def _fence(text: str, limit: int | None) -> str:
    lines = text.splitlines()
    if limit is not None and len(lines) > limit:
        lines = [*lines[:limit], f"... {len(lines) - limit} more lines"]
    body = "\n".join(lines)
    ticks = "```"
    while ticks in body:
        ticks += "`"
    return f"{ticks}text\n{body}\n{ticks}"


def markdown(run_id: str, by_agent: dict[str, list[Record]], full: bool = False) -> str:
    """A run as a Markdown document: each agent's conversation, then what it cost.

    Tool results show their first lines; ``full`` shows them whole.
    """
    limit = None if full else 12
    out = [f"# Run {run_id}", ""]
    total = 0.0
    costs: list[tuple[str, int, float]] = []
    for agent_id, records in by_agent.items():
        if len(by_agent) > 1:
            out += [f"## Agent {agent_id}", ""]
        for entry in entries(records):
            match entry.kind:
                case "user":
                    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.ts))
                    out += [f"### You ({stamp})", "", entry.text, ""]
                case "text":
                    out += [entry.text, ""]
                case "tool":
                    out += [f"- **{entry.text.replace('*', chr(92) + '*')}** ({result_summary(entry)})", ""]
                    if entry.detail.strip() and (full or entry.error):
                        out += [_fence(entry.detail, limit), ""]
                case "note":
                    out += [f"> {entry.text}", ""]
                case "end":
                    out += [f"**Ended {entry.text}.** {entry.detail}", ""]
        calls = [r for r in records if r.kind == "model_call"]
        cost = sum(float(r.data.get("cost_usd", 0.0)) for r in calls)
        costs.append((agent_id, sum("purpose" not in r.data for r in calls), cost))
        total += cost
    out += ["## Cost", ""]
    out += [f"- {agent_id}: {steps} steps, ${cost:.4f}" for agent_id, steps, cost in costs]
    if len(costs) > 1:
        out.append(f"- total: ${total:.4f}")
    return "\n".join(out) + "\n"
