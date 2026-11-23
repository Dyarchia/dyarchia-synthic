"""The ``todo`` tool: a short list of steps the agent keeps, each with a state.

It is built to cost little. The list enters the context only through the
agent's own calls: a result never repeats the whole list, only the counts and
the step now active, and no reminder is ever injected. Every change is
journaled as a ``todo`` record holding the whole list, which is what the CLI
shows and what a resumed agent starts from; compaction appends the list to its
summary, so it survives the history being replaced.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..types import ToolContext, ToolOutput

STATES = ("pending", "active", "done", "dropped")
MARKS = {"pending": "[ ]", "active": "[>]", "done": "[x]", "dropped": "[-]"}

Item = dict[str, str]


def render(items: list[Item]) -> str:
    """The whole list, one numbered line per step."""
    return "\n".join(f"{MARKS[i['state']]} {n}. {i['text']}" for n, i in enumerate(items, 1)) or "(empty)"


def tally(items: list[Item]) -> str:
    """One line: how many steps are done, and which are active."""
    counted = [i for i in items if i["state"] != "dropped"]
    done = sum(i["state"] == "done" for i in counted)
    active = [f"{n}. {i['text']}" for n, i in enumerate(items, 1) if i["state"] == "active"]
    line = f"{done}/{len(counted)} done"
    return line + (f"; active: {'; '.join(active)}" if active else "")


class TodoTool:
    name = "todo"
    description = (
        "Keep a short list of the steps of your task, each `pending`, `active`, `done` or `dropped`. `set` "
        "replaces the whole list (use it once to plan, or to replan); `update` changes steps by number "
        "(1-based); `add` appends steps. The result gives only the counts and the active step: keep the list in "
        "mind, it is not shown to you again."
    )
    _item = {
        "type": "object",
        "properties": {"text": {"type": "string"}, "state": {"type": "string", "enum": list(STATES)}},
        "required": ["text"],
    }
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "set": {"type": "array", "items": _item},
            "update": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 1},
                        "state": {"type": "string", "enum": list(STATES)},
                        "text": {"type": "string"},
                    },
                    "required": ["n"],
                },
            },
            "add": {"type": "array", "items": _item},
        },
    }

    def __init__(self, on_change: Callable[[list[Item]], None], items: list[Item] | None = None) -> None:
        self.items: list[Item] = list(items or [])
        self._on_change = on_change

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return self.name

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            items = self.apply(tool_input)
        except ValueError as error:
            return ToolOutput(f"{error} The list is unchanged.", is_error=True)
        if items == self.items:
            return ToolOutput(f"No change. {tally(items)}.")
        self.items = items
        self._on_change(items)
        return ToolOutput(f"Saved. {tally(items)}.")

    def apply(self, tool_input: dict[str, Any]) -> list[Item]:
        """The list after ``set``, then ``update``, then ``add``; ``ValueError`` names the first bad entry."""
        if not any(key in tool_input for key in ("set", "update", "add")):
            raise ValueError("Give `set`, `update` or `add`.")
        items = [dict(i) for i in self.items]
        if "set" in tool_input:
            items = [_item(entry, "set") for entry in _entries(tool_input["set"], "set")]
        for entry in _entries(tool_input.get("update", []), "update"):
            n = entry.get("n")
            if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= len(items):
                raise ValueError(f"`update` names step {n!r}, but the list has {len(items)} steps.")
            changed = _item({"text": items[n - 1]["text"], "state": items[n - 1]["state"], **entry}, "update")
            items[n - 1] = changed
        items += [_item(entry, "add") for entry in _entries(tool_input.get("add", []), "add")]
        return items


def _entries(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(e, dict) for e in value):
        raise ValueError(f"`{key}` must be a list of objects.")
    return value


def _item(entry: dict[str, Any], key: str) -> Item:
    text, state = entry.get("text"), entry.get("state", "pending")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Every step in `{key}` needs a non-empty `text`.")
    if state not in STATES:
        raise ValueError(f"`state` must be one of {', '.join(STATES)}, got {state!r}.")
    return {"text": " ".join(text.split()), "state": state}
