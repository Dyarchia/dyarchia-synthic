"""Offline stand-ins used by the test suites of every module.

``ScriptedProvider`` replays a fixed list of replies, so the loop, the agents
and the context can be exercised without network access or an API key.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable
from typing import Any

from .types import Block, ModelReply, ModelRequest, ModelSpec, Usage

TEST_MODEL = ModelSpec(
    alias="test",
    id="test-model",
    context_window=200_000,
    max_output_tokens=8_000,
    price_input=1.0,
    price_output=5.0,
    price_cache_write=1.25,
    price_cache_read=0.1,
)

_ids = itertools.count(1)


def text(value: str) -> Block:
    return {"type": "text", "text": value}


def tool_use(name: str, tool_input: dict[str, Any], call_id: str | None = None) -> Block:
    return {"type": "tool_use", "id": call_id or f"toolu_{next(_ids)}", "name": name, "input": tool_input}


def reply(*blocks: Block, stop_reason: str | None = None, usage: Usage | None = None) -> ModelReply:
    """Build a reply; the stop reason defaults to ``tool_use`` when any block asks for a tool."""
    reason = stop_reason or ("tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn")
    return ModelReply(
        content=list(blocks),
        stop_reason=reason,
        usage=usage or Usage(input_tokens=100, output_tokens=20, cost_usd=0.0002),
        model_id=TEST_MODEL.id,
    )


class ScriptedProvider:
    """A ``Provider`` that answers from a script.

    Each script item is a ``ModelReply``, an exception to raise, or a function
    that receives the ``ModelRequest`` and returns a ``ModelReply``. Requests
    are kept in ``requests`` so a test can assert on what the model was shown.
    """

    def __init__(self, script: list[ModelReply | Exception | Callable[[ModelRequest], ModelReply]]) -> None:
        self.script = list(script)
        self.requests: list[ModelRequest] = []

    async def complete(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> ModelReply:
        self.requests.append(request)
        await asyncio.sleep(0)
        if cancel is not None and cancel.is_set():
            raise asyncio.CancelledError
        if not self.script:
            raise AssertionError("ScriptedProvider ran out of replies")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        result = item(request) if callable(item) else item
        if on_text is not None:
            for block in result.content:
                if block["type"] == "text":
                    on_text(block["text"])
        return result
