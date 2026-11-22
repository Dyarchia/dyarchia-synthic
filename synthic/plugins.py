"""Job 10: what gets plugged into the engine.

Four kinds of plugin: command folders on PATH, skills, MCP servers over stdio and
packages that bundle the other three. The rule that shapes everything here is that
a capability costs one line of prompt until it is used. The engine's own tool list
stays fixed (one ``skill`` tool and one ``mcp`` tool), so adding a server or a
skill never invalidates the prompt cache.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from synthic.proc import merged_env
from synthic.types import (
    EVENTS,
    HookSpec,
    ImageData,
    McpServerSpec,
    PluginSettings,
    Tool,
    ToolContext,
    ToolOutput,
)

MCP_PROTOCOL_VERSION = "2024-11-05"
DESCRIPTION_LIMIT = 200
TOOL_LINE_LIMIT = 160
IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
"""The only image types the model API accepts; any other one in a request makes the whole call fail with a 400."""
LINE_LIMIT_BYTES = 64 * 1024 * 1024
FOLDED_MARKERS = frozenset({">", "|", ">-", "|-", ">+", "|+"})


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``SKILL.md`` into its ``key: value`` header and its body.

    The header sits between two ``---`` lines. A value may continue on the
    following indented lines. Without a closed header the whole text is the body.
    """
    lines = text.lstrip("\ufeff").split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    key: str | None = None
    for index in range(1, len(lines)):
        line = lines[index]
        if line.strip() == "---":
            return meta, "\n".join(lines[index + 1 :]).lstrip("\n")
        if line[:1].isspace() and key is not None:
            meta[key] = f"{meta[key]} {line.strip()}".strip()
        elif ":" in line and not line.lstrip().startswith("#"):
            name, _, value = line.partition(":")
            key = name.strip()
            value = value.strip()
            meta[key] = "" if value in FOLDED_MARKERS else value.strip("\"'")
        else:
            key = None
    return {}, text


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _first_sentence(text: str) -> str:
    text = _one_line(text)
    match = re.match(r"(.+?[.!?])(?:\s|$)", text)
    return _cut(match.group(1) if match else text, TOOL_LINE_LIMIT)


def _read_skill(folder: Path) -> Skill | str:
    try:
        meta, _ = parse_front_matter((folder / "SKILL.md").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        return f"skill {folder}: cannot read SKILL.md ({error})"
    if not meta.get("name") or not meta.get("description"):
        return f"skill {folder}: SKILL.md needs a front matter with name and description"
    return Skill(meta["name"], _one_line(meta["description"]), folder)


class McpError(Exception):
    """A server could not be reached or answered with an error. Always turned into a tool error."""


class _McpClient:
    """The smallest MCP client over stdio: newline-delimited JSON-RPC 2.0 messages.

    The server starts on first use and restarts if it died. Requests are
    serialised, so replies cannot be mixed up. A server that does not answer in
    time, or whose request is cancelled halfway, is killed, since its stream can
    no longer be trusted. Each request races ``cancel`` on its own, so a stop
    never waits behind a running call or behind calls queued on the lock.
    """

    def __init__(self, spec: McpServerSpec, cwd: Path | None, env: Mapping[str, str]) -> None:
        self.spec = spec
        self.cwd = cwd
        self.env = dict(env)
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._tools: list[dict[str, Any]] | None = None

    async def list_tools(self, timeout: float, cancel: asyncio.Event | None = None) -> list[dict[str, Any]]:
        if self._tools is None:
            tools: list[dict[str, Any]] = []
            cursor: str | None = None
            while True:
                result = await self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout, cancel)
                tools.extend(result.get("tools", []))
                cursor = result.get("nextCursor")
                if not cursor:
                    break
            self._tools = tools
        return self._tools

    async def call_tool(
        self, name: str, arguments: dict[str, Any], timeout: float, cancel: asyncio.Event | None = None
    ) -> dict[str, Any]:
        return await self.request("tools/call", {"name": name, "arguments": arguments}, timeout, cancel)

    async def request(
        self, method: str, params: dict[str, Any], timeout: float, cancel: asyncio.Event | None = None
    ) -> dict[str, Any]:
        """Send one request, waiting at most ``timeout`` seconds, and give up at once when ``cancel`` is set."""
        work = asyncio.ensure_future(self._locked(method, params, timeout))
        if cancel is None:
            return await work
        waiter = asyncio.ensure_future(cancel.wait())
        try:
            await asyncio.wait({work, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pending in (work, waiter):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(work, waiter, return_exceptions=True)
        if work.cancelled():
            raise McpError(f"mcp call to '{self.spec.name}' was cancelled")
        return work.result()

    async def _locked(self, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        async with self._lock:
            try:
                async with asyncio.timeout(timeout):
                    await self._ensure_started()
                    return await self._rpc(method, params)
            except TimeoutError:
                await self.stop()
                raise McpError(f"mcp server '{self.spec.name}' did not answer {method} within {timeout:g}s") from None
            except OSError as error:
                await self.stop()
                raise McpError(f"mcp server '{self.spec.name}' connection failed: {error}") from error
            except asyncio.CancelledError:
                await self.stop()
                raise

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        self._tools = None
        if proc is None or proc.returncode is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            await asyncio.wait_for(proc.wait(), 1)
        except (TimeoutError, OSError):
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), 3)
            except TimeoutError:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

    async def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        await self.stop()
        command = list(self.spec.command)
        if not command:
            raise McpError(f"mcp server '{self.spec.name}' has an empty command")
        program = Path(command[0])
        if self.cwd is not None and not program.is_absolute() and len(program.parts) > 1:
            program = self.cwd / program
        executable = shutil.which(str(program))
        if executable is None:
            raise McpError(f"mcp server '{self.spec.name}': command not found: {command[0]}")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                executable,
                *command[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=self.cwd,
                env=merged_env({**self.env, **self.spec.env}),
                limit=LINE_LIMIT_BYTES,
            )
        except OSError as error:
            raise McpError(f"mcp server '{self.spec.name}' cannot start: {error}") from error
        try:
            await self._rpc(
                "initialize",
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "synthic", "version": "0.1"},
                },
            )
            await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except BaseException:
            await self.stop()
            raise

    async def _send(self, message: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read(self) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                await self.stop()
                raise McpError(f"mcp server '{self.spec.name}' exited")
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                return message

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        await self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
        while True:
            message = await self._read()
            if "method" in message:
                if "id" in message:
                    reply = {"code": -32601, "message": "Method not found"}
                    await self._send({"jsonrpc": "2.0", "id": message["id"], "error": reply})
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                detail = message["error"]
                text = detail.get("message", detail) if isinstance(detail, dict) else detail
                raise McpError(f"mcp server '{self.spec.name}': {text}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}


class _SkillTool:
    name = "skill"
    description = (
        "Load a skill by name. Returns the folder of the skill, then its instructions. "
        "The folder holds any other files the instructions mention."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Name of the skill, as listed in the prompt."}},
        "required": ["name"],
    }

    def __init__(self, skills: dict[str, Skill]) -> None:
        self._skills = skills

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        skill = self._skills.get(str(tool_input.get("name", "")))
        if skill is None:
            valid = ", ".join(self._skills)
            return ToolOutput(f"unknown skill '{tool_input.get('name', '')}'. Valid names: {valid}", is_error=True)
        try:
            text = await asyncio.to_thread((skill.path / "SKILL.md").read_text, encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            return ToolOutput(f"cannot read skill '{skill.name}': {error}", is_error=True)
        _, body = parse_front_matter(text)
        return ToolOutput(f"{skill.path.resolve()}\n{body}")


class _McpTool:
    name = "mcp"
    description = (
        "Use a tool of an MCP server. With only 'server': list its tools. With 'server' and 'tool': "
        "show that tool's description and input schema. With 'server', 'tool' and 'arguments': call it."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "Name of the MCP server, as listed in the prompt."},
            "tool": {"type": "string", "description": "Name of one of the server's tools."},
            "arguments": {"type": "object", "description": "Arguments of the call, following the tool's input schema."},
        },
        "required": ["server"],
    }

    def __init__(self, plugins: Plugins) -> None:
        self._plugins = plugins

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return f"mcp:{tool_input.get('server', '')}"

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        server = str(tool_input.get("server", ""))
        if server not in self._plugins.servers:
            valid = ", ".join(self._plugins.servers)
            return ToolOutput(f"unknown mcp server '{server}'. Valid names: {valid}", is_error=True)
        client = self._plugins.client(server)
        tool = tool_input.get("tool")
        try:
            tools = await client.list_tools(ctx.timeout_seconds, ctx.cancel)
            if not tool:
                lines = [f"{t.get('name', '')}: {_first_sentence(str(t.get('description', '')))}" for t in tools]
                return ToolOutput("\n".join(lines) or f"mcp server '{server}' has no tools")
            found = next((t for t in tools if t.get("name") == tool), None)
            if found is None:
                names = ", ".join(str(t.get("name", "")) for t in tools)
                return ToolOutput(f"unknown tool '{tool}' on '{server}'. Valid names: {names}", is_error=True)
            if "arguments" not in tool_input:
                shown = {k: found[k] for k in ("name", "description", "inputSchema") if k in found}
                return ToolOutput(json.dumps(shown, separators=(",", ":"), ensure_ascii=False))
            arguments = tool_input["arguments"]
            if not isinstance(arguments, dict):
                return ToolOutput("'arguments' must be an object", is_error=True)
            result = await client.call_tool(str(tool), arguments, ctx.timeout_seconds, ctx.cancel)
        except McpError as error:
            return ToolOutput(str(error), is_error=True)
        return _to_output(result)


def _to_output(result: dict[str, Any]) -> ToolOutput:
    texts: list[str] = []
    images: list[ImageData] = []
    for block in result.get("content", []):
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text", "")))
        elif kind == "image":
            media_type = str(block.get("mimeType", "image/png")).split(";")[0].strip().lower()
            data = str(block.get("data", ""))
            if media_type in IMAGE_TYPES:
                images.append(ImageData(media_type, data))
            else:
                texts.append(f"[image omitted: unsupported type {media_type or 'unknown'}, about {len(data) * 3 // 4} bytes]")
        else:
            texts.append(json.dumps(block, separators=(",", ":"), ensure_ascii=False))
    return ToolOutput("\n".join(texts), images, bool(result.get("isError", False)))


class Plugins:
    """Everything plugged in, as found from the settings. Built by ``load_plugins``."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        self.env = dict(env or {})
        self.skills: dict[str, Skill] = {}
        self.path_dirs: list[Path] = []
        self.hooks: tuple[HookSpec, ...] = ()
        self.servers: dict[str, McpServerSpec] = {}
        self.problems: list[str] = []
        self._cwds: dict[str, Path] = {}
        self._clients: dict[str, _McpClient] = {}

    def index_text(self) -> str:
        """One line per skill and one per server: the whole cost of a capability until it is used."""
        parts: list[str] = []
        if self.skills:
            parts.append("Skills (read one with the skill tool):")
            parts.extend(f"{s.name}: {_cut(s.description, DESCRIPTION_LIMIT)}" for s in self.skills.values())
        if self.servers:
            parts.append("MCP servers (use them with the mcp tool):")
            parts.extend(self.servers)
        return "\n".join(parts)

    def tools(self) -> dict[str, Tool]:
        found: dict[str, Tool] = {}
        if self.skills:
            found["skill"] = _SkillTool(self.skills)
        if self.servers:
            found["mcp"] = _McpTool(self)
        return found

    def client(self, name: str) -> _McpClient:
        if name not in self._clients:
            self._clients[name] = _McpClient(self.servers[name], self._cwds.get(name), self.env)
        return self._clients[name]

    async def close(self) -> None:
        clients, self._clients = list(self._clients.values()), {}
        await asyncio.gather(*(c.stop() for c in clients))

    def _add_skills(self, entry: Path) -> None:
        if (entry / "SKILL.md").is_file():
            folders = [entry]
        else:
            folders = sorted(p for p in entry.iterdir() if (p / "SKILL.md").is_file())
        for folder in folders:
            skill = _read_skill(folder)
            if isinstance(skill, str):
                self.problems.append(skill)
            elif skill.name in self.skills:
                self.problems.append(f"skill '{skill.name}' at {folder} ignored: the name is already taken")
            else:
                self.skills[skill.name] = skill

    def _add_package(self, folder: Path) -> None:
        if (folder / "skills").is_dir():
            self._add_skills(folder / "skills")
        if (folder / "bin").is_dir():
            self.path_dirs.append(folder / "bin")
        manifest = folder / "package.toml"
        if not manifest.is_file():
            return
        try:
            data = tomllib.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
            self.problems.append(f"package {folder}: cannot read package.toml ({error})")
            return
        for name, table in (data.get("mcp") or {}).items():
            table = table if isinstance(table, dict) else {}
            command = table.get("command")
            env = table.get("env", {})
            if not (isinstance(command, list) and command and all(isinstance(c, str) for c in command)):
                self.problems.append(f"package {folder}: mcp '{name}' needs a non-empty command list")
            elif name in self.servers:
                self.problems.append(f"package {folder}: mcp '{name}' ignored: the name is already taken")
            else:
                self.servers[name] = McpServerSpec(name, tuple(command), {str(k): str(v) for k, v in env.items()})
                self._cwds[name] = folder
        hooks = list(self.hooks)
        for table in data.get("hooks") or []:
            if table.get("event") not in EVENTS or not table.get("command"):
                self.problems.append(f"package {folder}: hook {table!r} needs a valid event and a command")
                continue
            hooks.append(
                HookSpec(
                    table["event"],
                    str(table["command"]),
                    str(table.get("match", "")),
                    float(table.get("timeout_seconds", 30.0)),
                )
            )
        self.hooks = tuple(hooks)


def load_plugins(settings: PluginSettings, base: Path, env: Mapping[str, str] | None = None) -> Plugins:
    """Find every plugin named in the settings. Relative paths resolve against ``base``.

    ``env`` (the policy's environment variables) is added to every MCP server's
    environment, under the server's own ``env``.
    """
    plugins = Plugins(env)

    def locate(kind: str, raw: str) -> Path | None:
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else Path(base) / path
        if path.is_dir():
            return path
        plugins.problems.append(f"{kind} folder not found: {path}")
        return None

    for raw in settings.skills:
        if (folder := locate("skills", raw)) is not None:
            plugins._add_skills(folder)
    for raw in settings.path:
        if (folder := locate("path", raw)) is not None:
            plugins.path_dirs.append(folder)
    for spec in settings.mcp:
        plugins.servers[spec.name] = spec
    for raw in settings.packages:
        if (folder := locate("package", raw)) is not None:
            plugins._add_package(folder)
    return plugins
