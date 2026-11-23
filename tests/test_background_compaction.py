"""Background summaries for compaction (``context.compact_background``)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from test_loop import assert_replays, done, kinds, make_policy, project, records

from synthic.agents import Runtime
from synthic.loop import AgentResult
from synthic.testing import ScriptedProvider, reply, text, tool_use
from synthic.types import ModelReply, ModelRequest, Usage

TEMPLATE_MARK = "Use exactly these sections"
SETTINGS = {
    "models": {"test": {"context_window": 1000}},
    "context": {
        "compact_background": True,
        "compact_prepare_at": 0.3,
        "compact_at": 0.8,
        "compact_refresh_tokens": 10_000,
    },
}


def tokens(n: int) -> Usage:
    return Usage(input_tokens=n, output_tokens=10, cost_usd=0.001)


class SplitProvider:
    """Main steps from a script; summary requests (they end with the compaction template) from another one."""

    def __init__(self, main: list[Any], summaries: list[ModelReply], hang: bool = False) -> None:
        self.main = ScriptedProvider(main)
        self.summaries = list(summaries)
        self.summary_requests: list[ModelRequest] = []
        self.hang = hang
        self.cancelled = 0

    @property
    def requests(self) -> list[ModelRequest]:
        return self.main.requests

    async def complete(self, request: ModelRequest, on_text: Any = None, cancel: Any = None) -> ModelReply:
        last = request.messages[-1]["content"][-1]
        if last.get("type") == "text" and TEMPLATE_MARK in last["text"]:
            self.summary_requests.append(request)
            if self.hang:
                try:
                    await cancel.wait()
                finally:
                    self.cancelled += 1
                raise asyncio.CancelledError
            await asyncio.sleep(0)
            return self.summaries.pop(0)
        if self.hang and not self.main.script:
            await cancel.wait()
            raise asyncio.CancelledError
        return await self.main.complete(request, on_text, cancel)


def run(tmp_path: Path, provider: SplitProvider, task: dict[str, Any] | None = None) -> tuple[Runtime, AgentResult]:
    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path, {**SETTINGS, **(task or {})}), project(tmp_path), provider)
        try:
            return runtime, await runtime.run("Do the task.")
        finally:
            await runtime.close()

    return asyncio.run(main())


def test_a_prepared_summary_is_swapped_in_without_a_blocking_call(tmp_path: Path) -> None:
    provider = SplitProvider(
        [
            reply(tool_use("receipt", {"id": "r1"}), usage=tokens(400)),
            reply(tool_use("receipt", {"id": "r2"}), usage=tokens(900)),
            done(),
        ],
        [reply(text("SUMMARY-A"), usage=Usage(cache_read_tokens=400, output_tokens=30, cost_usd=0.0005))],
    )
    runtime, result = run(tmp_path, provider)
    assert result.status == "delivered" and result.steps == 3
    assert len(provider.summary_requests) == 1
    background, first = provider.summary_requests[0], provider.requests[0]
    assert background.system == first.system and background.tools == first.tools
    assert background.messages[0]["content"][:-1] == first.messages[0]["content"]
    after = provider.requests[2].messages
    assert after[0]["content"][0]["text"].endswith("SUMMARY-A") and len(after) == 5
    assert after[1]["role"] == "assistant" and after[1] == provider.requests[1].messages[1]
    log = records(runtime)
    compaction = next(r for r in log if r["kind"] == "compaction")
    assert compaction["data"]["keep"] == 4
    prepared = [r for r in log if r["kind"] == "model_call" and r["data"].get("purpose") == "compaction_prepare"]
    assert len(prepared) == 1
    assert abs(runtime.ledger.total.cost_usd - 0.0027) < 1e-12 and runtime.ledger.steps == 3
    assert_replays(runtime)


def test_a_stale_summary_is_dropped_and_compaction_blocks(tmp_path: Path) -> None:
    cut = reply(tool_use("receipt", {"id": "r1"}), stop_reason="max_tokens", usage=tokens(400))
    provider = SplitProvider(
        [cut, reply(tool_use("receipt", {"id": "r2"}), usage=tokens(900)), done()],
        [reply(text("STALE")), reply(text("FRESH"))],
    )
    runtime, result = run(tmp_path, provider, {"role": {"lead": {"max_output_tokens": 8000}}})
    assert result.status == "delivered" and len(provider.summary_requests) == 2
    after = provider.requests[2].messages
    assert after[0]["content"][0]["text"].endswith("FRESH")
    assert "STALE" not in str(after)
    assert_replays(runtime)


def test_off_by_default_and_below_the_threshold(tmp_path: Path) -> None:
    provider = SplitProvider([reply(tool_use("receipt", {"id": "r1"}), usage=tokens(200)), done()], [])
    run(tmp_path, provider)
    assert provider.summary_requests == []
    other = tmp_path / "off"
    other.mkdir()
    provider = SplitProvider([reply(tool_use("receipt", {"id": "r1"}), usage=tokens(700)), done()], [])
    run(other, provider, {"context": {"compact_background": False}})
    assert provider.summary_requests == []


def test_the_summary_refreshes_as_the_prompt_grows(tmp_path: Path) -> None:
    provider = SplitProvider(
        [
            reply(tool_use("receipt", {"id": "r1"}), usage=tokens(400)),
            reply(tool_use("receipt", {"id": "r2"}), usage=tokens(450)),
            reply(tool_use("receipt", {"id": "r3"}), usage=tokens(600)),
            done(),
        ],
        [reply(text("ONE")), reply(text("TWO"))],
    )
    runtime, result = run(tmp_path, provider, {"context": {**SETTINGS["context"], "compact_refresh_tokens": 150}})
    assert result.status == "delivered" and len(provider.summary_requests) == 2
    assert len(provider.summary_requests[1].messages) == 5
    assert "compaction" not in kinds(runtime)


def test_a_stop_cancels_the_summary_in_flight(tmp_path: Path) -> None:
    provider = SplitProvider([reply(tool_use("receipt", {"id": "r1"}), usage=tokens(400))], [], hang=True)

    async def main() -> tuple[Runtime, AgentResult]:
        runtime = Runtime(make_policy(tmp_path, SETTINGS), project(tmp_path), provider)
        task = asyncio.create_task(runtime.run("Do the task."))
        while not provider.summary_requests:
            await asyncio.sleep(0.01)
        runtime.stop()
        result = await task
        await runtime.close()
        return runtime, result

    runtime, result = asyncio.run(main())
    assert result.status == "stopped" and provider.cancelled == 1
    calls = [r["data"] for r in records(runtime) if r["kind"] == "model_call"]
    assert not any(c.get("purpose") == "compaction_prepare" for c in calls)
