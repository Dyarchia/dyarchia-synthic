"""Job 6: an event for everything that happens; the human hangs commands from events.

Two kinds of listeners run on each event, in this order: in-process subscribers
(the CLI, add-ons) and then the shell commands of the policy. A hook never
hangs or crashes the engine: a listener that raises, a command that fails, times
out or is missing counts as a pass, and the failure text travels in
``HookOutcome.reason`` so the caller can journal it.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import json
import os
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from synthic.proc import base_env
from synthic.types import EVENTS, HookOutcome, HookSpec

Listener = Callable[[str, dict[str, Any]], Awaitable[HookOutcome | None]]

_WINDOWS = sys.platform == "win32"
_REAP_SECONDS = 5.0


def _failed(reason: str) -> HookOutcome:
    return HookOutcome(reason=reason)


async def _kill_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the shell and everything it started, then reap it."""
    if proc.returncode is None:
        if _WINDOWS:
            with contextlib.suppress(OSError):
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(proc.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
        else:
            with contextlib.suppress(OSError):
                os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            proc.kill()
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(proc.wait(), _REAP_SECONDS)


def _interpret(command: str, code: int, stdout: str, stderr: str) -> HookOutcome:
    """Apply the hook command protocol to a finished process."""
    if code == 2:
        return HookOutcome("block", stderr.strip() or f"blocked by hook {command!r}")
    if code != 0:
        return _failed(f"hook {command!r} exited with code {code}: {' '.join(stderr.split())[:500]}")
    text = stdout.strip()
    if not text:
        return HookOutcome()
    parsed: Any = None
    if text.startswith("{"):
        with contextlib.suppress(ValueError):
            parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return HookOutcome(add_text=text)
    replace = parsed.get("replace")
    add_text = parsed.get("add_text")
    outcome = HookOutcome(
        replace=replace if isinstance(replace, dict) else None,
        add_text=add_text if isinstance(add_text, str) and add_text else None,
    )
    if parsed.get("action") == "block":
        outcome.action = "block"
        outcome.reason = str(parsed.get("reason") or f"blocked by hook {command!r}")
    return outcome


class Hooks:
    """Emits events to subscribers and to the policy's shell commands. ``emit`` satisfies ``Emit``."""

    def __init__(self, specs: tuple[HookSpec, ...], cwd: Path, env: dict[str, str] | None = None) -> None:
        self._specs = tuple(specs)
        self._cwd = cwd
        self._env = dict(env or {})
        self._listeners: list[tuple[str, Listener]] = []

    def subscribe(self, event: str, listener: Listener) -> None:
        """Register an in-process listener. ``event`` may be ``"*"`` for every event."""
        if event != "*" and event not in EVENTS:
            raise ValueError(f"unknown event {event!r}")
        self._listeners.append((event, listener))

    async def emit(self, event: str, payload: dict[str, Any]) -> HookOutcome:
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}")
        reasons: list[str] = []
        replace: dict[str, Any] | None = None
        texts: list[str] = []

        def combine(outcome: HookOutcome) -> bool:
            nonlocal replace
            if outcome.replace:
                replace = {**(replace or {}), **outcome.replace}
            if outcome.add_text:
                texts.append(outcome.add_text)
            if outcome.action == "pass" and outcome.reason:
                reasons.append(outcome.reason)
            return outcome.action == "block"

        def result(blocker: HookOutcome | None) -> HookOutcome:
            return HookOutcome(
                action="block" if blocker else "pass",
                reason=blocker.reason if blocker else "\n".join(reasons),
                replace=replace,
                add_text="\n\n".join(texts) or None,
            )

        for name, listener in list(self._listeners):
            if name not in ("*", event):
                continue
            try:
                outcome = await listener(event, payload) or HookOutcome()
            except Exception as exc:
                outcome = _failed(f"subscriber for {name!r} raised {type(exc).__name__}: {exc}")
            if combine(outcome):
                return result(outcome)
        tool = str(payload.get("tool", ""))
        for spec in self._specs:
            if spec.event != event or (spec.match and not fnmatch.fnmatchcase(tool, spec.match)):
                continue
            outcome = await self._run_command(spec, event, payload)
            if combine(outcome):
                return result(outcome)
        return result(None)

    async def _run_command(self, spec: HookSpec, event: str, payload: dict[str, Any]) -> HookOutcome:
        env = {
            **base_env(),
            **self._env,
            "SYNTHIC_EVENT": event,
            "SYNTHIC_AGENT": str(payload.get("agent_id", "")),
        }
        try:
            stdin = json.dumps(payload, default=str).encode()
            proc = await asyncio.create_subprocess_shell(
                spec.command,
                cwd=self._cwd,
                env=env,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=not _WINDOWS,
            )
        except (OSError, ValueError, TypeError) as exc:
            return _failed(f"hook {spec.command!r} could not start: {exc}")
        try:
            out, err = await asyncio.wait_for(proc.communicate(stdin), spec.timeout_seconds)
        except TimeoutError:
            await _kill_tree(proc)
            return _failed(f"hook {spec.command!r} timed out after {spec.timeout_seconds:g}s and was killed")
        except asyncio.CancelledError:
            await _kill_tree(proc)
            raise
        return _interpret(
            spec.command,
            proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )
