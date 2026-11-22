from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from synthic.hooks import Hooks
from synthic.types import HookOutcome, HookSpec

SCRIPT = """
import json, os, sys, time

mode = sys.argv[1]
payload = json.load(sys.stdin)
if mode == "exit2":
    sys.stderr.write("nope, not allowed")
    sys.exit(2)
if mode == "exit3":
    sys.stderr.write("boom")
    sys.exit(3)
if mode == "sleep":
    time.sleep(60)
if mode == "beat":
    while True:
        with open(sys.argv[2], "a", encoding="utf-8") as beat:
            beat.write(".")
        time.sleep(0.05)
if mode == "print":
    sys.stdout.write(open(sys.argv[2], encoding="utf-8").read())
if mode == "env":
    sys.stdout.write(json.dumps({
        "add_text": "|".join([os.environ["SYNTHIC_EVENT"], os.environ["SYNTHIC_AGENT"], payload["tool"], payload["blob"][-3:]])
    }))
"""


@pytest.fixture
def script(tmp_path: Path) -> Path:
    path = tmp_path / "hook.py"
    path.write_text(SCRIPT, encoding="utf-8")
    return path


def command(script: Path, *args: str | Path) -> str:
    argv = [sys.executable, str(script), *map(str, args)]
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def printing(script: Path, tmp_path: Path, name: str, text: str) -> str:
    out = tmp_path / f"{name}.txt"
    out.write_text(text, encoding="utf-8")
    return command(script, "print", out)


def run(hooks: Hooks, event: str = "before_tool", payload: dict | None = None) -> HookOutcome:
    return asyncio.run(hooks.emit(event, payload if payload is not None else {"tool": "sh", "agent_id": "a1"}))


def make(tmp_path: Path, *specs: HookSpec) -> Hooks:
    return Hooks(tuple(specs), tmp_path)


def test_unknown_event_raises(tmp_path: Path) -> None:
    hooks = make(tmp_path)
    with pytest.raises(ValueError):
        run(hooks, "nope")
    with pytest.raises(ValueError):
        hooks.subscribe("nope", lambda e, p: None)


def test_no_listeners_passes(tmp_path: Path) -> None:
    assert run(make(tmp_path)) == HookOutcome()


def test_pass_on_exit_zero(tmp_path: Path, script: Path) -> None:
    outcome = run(make(tmp_path, HookSpec("before_tool", command(script, "ok"))))
    assert outcome == HookOutcome()


def test_block_by_exit_code_two(tmp_path: Path, script: Path) -> None:
    outcome = run(make(tmp_path, HookSpec("before_tool", command(script, "exit2"))))
    assert outcome.action == "block"
    assert outcome.reason == "nope, not allowed"


def test_block_by_json(tmp_path: Path, script: Path) -> None:
    body = json.dumps({"action": "block", "reason": "policy says no"})
    outcome = run(make(tmp_path, HookSpec("before_tool", printing(script, tmp_path, "b", body))))
    assert (outcome.action, outcome.reason) == ("block", "policy says no")


def test_replace_merge_and_add_text_join(tmp_path: Path, script: Path) -> None:
    first = json.dumps({"replace": {"a": 1, "b": 1}, "add_text": "one"})
    second = json.dumps({"replace": {"b": 2, "c": 3}, "add_text": "two"})
    outcome = run(
        make(
            tmp_path,
            HookSpec("before_tool", printing(script, tmp_path, "f", first)),
            HookSpec("before_tool", printing(script, tmp_path, "s", second)),
        )
    )
    assert outcome.action == "pass"
    assert outcome.replace == {"a": 1, "b": 2, "c": 3}
    assert outcome.add_text == "one\n\ntwo"


def test_plain_stdout_becomes_add_text(tmp_path: Path, script: Path) -> None:
    outcome = run(make(tmp_path, HookSpec("before_tool", printing(script, tmp_path, "p", "remember this\n"))))
    assert outcome.add_text == "remember this"


def test_first_block_stops_the_rest(tmp_path: Path, script: Path) -> None:
    hooks = make(
        tmp_path,
        HookSpec("before_tool", command(script, "exit2")),
        HookSpec("before_tool", printing(script, tmp_path, "late", "never")),
    )
    outcome = run(hooks)
    assert outcome.action == "block"
    assert outcome.add_text is None


def test_match_glob_and_event_filter(tmp_path: Path, script: Path) -> None:
    hooks = make(
        tmp_path,
        HookSpec("before_tool", command(script, "exit2"), match="edit*"),
        HookSpec("after_tool", command(script, "exit2")),
    )
    assert run(hooks, payload={"tool": "sh"}).action == "pass"
    assert run(hooks, payload={"tool": "edit_file"}).action == "block"
    assert run(hooks, payload={}).action == "pass"


def test_stdin_payload_and_environment(tmp_path: Path, script: Path) -> None:
    hooks = make(tmp_path, HookSpec("before_tool", command(script, "env")))
    blob = "x" * 2_000_000 + "end"
    outcome = run(hooks, payload={"tool": "sh", "agent_id": "a7", "blob": blob})
    assert outcome.add_text == "before_tool|a7|sh|end"


def test_failures_pass_and_are_reported(tmp_path: Path, script: Path) -> None:
    hooks = make(
        tmp_path,
        HookSpec("before_tool", command(script, "exit3")),
        HookSpec("before_tool", "definitely-not-a-real-command-xyz"),
    )
    outcome = run(hooks)
    assert outcome.action == "pass"
    assert "exited with code 3" in outcome.reason
    assert "boom" in outcome.reason
    assert outcome.reason.count("\n") == 1


def test_timeout_does_not_hang(tmp_path: Path, script: Path) -> None:
    hooks = make(tmp_path, HookSpec("before_tool", command(script, "sleep"), timeout_seconds=0.5))
    start = time.monotonic()
    outcome = run(hooks)
    assert time.monotonic() - start < 20
    assert outcome.action == "pass"
    assert "timed out" in outcome.reason


def test_subscribers_run_in_order_before_commands(tmp_path: Path, script: Path) -> None:
    calls: list[str] = []

    async def named(name: str, text: str | None):
        async def listener(event: str, payload: dict) -> HookOutcome | None:
            calls.append(f"{name}:{event}")
            return HookOutcome(add_text=text) if text else None

        return listener

    async def setup() -> Hooks:
        hooks = make(tmp_path, HookSpec("before_tool", printing(script, tmp_path, "c", "from-command")))
        hooks.subscribe("*", await named("star", "from-star"))
        hooks.subscribe("before_tool", await named("exact", None))
        hooks.subscribe("end", await named("other", "never"))
        return hooks

    hooks = asyncio.run(setup())
    outcome = run(hooks)
    assert calls == ["star:before_tool", "exact:before_tool"]
    assert outcome.add_text == "from-star\n\nfrom-command"
    run(hooks, "end", {})
    assert calls[-2:] == ["star:end", "other:end"]


def test_subscriber_block_stops_commands_and_exceptions_are_contained(tmp_path: Path, script: Path) -> None:
    hooks = make(tmp_path, HookSpec("before_tool", printing(script, tmp_path, "c", "late")))

    async def broken(event: str, payload: dict) -> HookOutcome | None:
        raise RuntimeError("oops")

    async def blocker(event: str, payload: dict) -> HookOutcome | None:
        return HookOutcome("block", "stop here")

    hooks.subscribe("*", broken)
    hooks.subscribe("before_tool", blocker)
    outcome = run(hooks)
    assert (outcome.action, outcome.reason, outcome.add_text) == ("block", "stop here", None)


def test_cancelling_emit_kills_the_running_hook(tmp_path: Path, script: Path) -> None:
    beat = tmp_path / "beat.txt"
    hooks = make(tmp_path, HookSpec("before_tool", command(script, "beat", beat), timeout_seconds=60))

    async def scenario() -> None:
        task = asyncio.ensure_future(hooks.emit("before_tool", {"tool": "sh", "agent_id": "a1"}))
        for _ in range(200):
            if beat.exists() and beat.stat().st_size > 2:
                break
            await asyncio.sleep(0.05)
        assert beat.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.3)
        size = beat.stat().st_size
        await asyncio.sleep(0.5)
        assert beat.stat().st_size == size

    asyncio.run(scenario())


def test_hook_does_not_inherit_the_api_key(tmp_path: Path, script: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\nsys.stdout.write(os.environ.get('ANTHROPIC_API_KEY', 'absent'))\n", encoding="utf-8"
    )
    argv = [sys.executable, str(probe)]
    line = subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)
    assert run(make(tmp_path, HookSpec("before_tool", line))).add_text == "absent"
    hooks = Hooks((HookSpec("before_tool", line),), tmp_path, {"ANTHROPIC_API_KEY": "explicit"})
    assert run(hooks).add_text == "explicit"
