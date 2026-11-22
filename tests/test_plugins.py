from __future__ import annotations

import asyncio
import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

import synthic.plugins as plugins_module
from synthic.plugins import Plugins, load_plugins, parse_front_matter
from synthic.types import McpServerSpec, PluginSettings, ToolContext

FAKE_SERVER = '''
import json
import os
import sys
import time

PAGE_ONE = [
    {"name": "echo", "description": "Echo the text back. Also prints a key. Extra detail.",
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
    {"name": "pic", "description": "Return an image.", "inputSchema": {"type": "object"}},
]
PAGE_TWO = [
    {"name": "boom", "description": "Always fails.", "inputSchema": {"type": "object"}},
    {"name": "where", "description": "Report the working folder.", "inputSchema": {"type": "object"}},
    {"name": "svg", "description": "Return an unsupported image.", "inputSchema": {"type": "object"}},
    {"name": "slow", "description": "Sleep then answer.", "inputSchema": {"type": "object"}},
    {"name": "key", "description": "Report the API key.", "inputSchema": {"type": "object"}},
    {"name": "die", "description": "Exit the process.", "inputSchema": {"type": "object"}},
]


def reply(request_id, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": request_id}
    message["error" if error else "result"] = error or result
    sys.stdout.write("not json noise\\n")
    sys.stdout.write(json.dumps(message) + "\\n")
    sys.stdout.flush()


for line in sys.stdin:
    message = json.loads(line)
    method, request_id = message.get("method"), message.get("id")
    if request_id is None:
        continue
    params = message.get("params", {})
    if method == "initialize":
        reply(request_id, {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                           "serverInfo": {"name": "fake", "version": "1"}})
    elif method == "tools/list":
        if params.get("cursor") == "p2":
            reply(request_id, {"tools": PAGE_TWO})
        else:
            reply(request_id, {"tools": PAGE_ONE, "nextCursor": "p2"})
    elif method == "tools/call":
        name, arguments = params["name"], params["arguments"]
        if name == "echo":
            text = arguments.get("text", "") + "|" + os.environ.get("FAKE_KEY", "")
            reply(request_id, {"content": [{"type": "text", "text": text}]})
        elif name == "pic":
            reply(request_id, {"content": [{"type": "image", "data": "QUJD", "mimeType": "image/png"}]})
        elif name == "svg":
            reply(request_id, {"content": [{"type": "image", "data": "QUJDREVG", "mimeType": "image/svg+xml"}]})
        elif name == "slow":
            time.sleep(float(arguments.get("seconds", 5)))
            reply(request_id, {"content": [{"type": "text", "text": "slept"}]})
        elif name == "key":
            reply(request_id, {"content": [{"type": "text", "text": os.environ.get("ANTHROPIC_API_KEY", "absent")}]})
        elif name == "boom":
            reply(request_id, {"content": [{"type": "text", "text": "it broke"}], "isError": True})
        elif name == "where":
            reply(request_id, {"content": [{"type": "text", "text": os.getcwd()}]})
        elif name == "die":
            sys.exit(1)
    else:
        reply(request_id, error={"code": -32601, "message": "no such method"})
'''


def ctx(tmp_path: Path, timeout: float = 30.0, cancel: asyncio.Event | None = None) -> ToolContext:
    return ToolContext("a1", tmp_path, {}, timeout, cancel or asyncio.Event())


def write_skill(folder: Path, front: str, body: str = "Do the thing.\n") -> None:
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\n{front}\n---\n{body}", encoding="utf-8")


def fake_server(tmp_path: Path) -> Path:
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return script


def mcp_plugins(tmp_path: Path, **env: str) -> Plugins:
    spec = McpServerSpec("fake", (sys.executable, str(fake_server(tmp_path))), env)
    plugins = load_plugins(PluginSettings(mcp=(spec,)), tmp_path)
    return plugins


def test_front_matter_parsing() -> None:
    text = "---\nname: 'quoted'\ndescription: first line\n  second line\nother: \"x\"\n---\n\nBody\n"
    meta, body = parse_front_matter(text)
    assert meta == {"name": "quoted", "description": "first line second line", "other": "x"}
    assert body == "Body\n"
    assert parse_front_matter("no header") == ({}, "no header")
    assert parse_front_matter("---\nname: x\n")[0] == {}


def test_skill_discovery_and_index(tmp_path: Path) -> None:
    library = tmp_path / "skills"
    write_skill(library / "one", "name: one\ndescription: First skill")
    write_skill(library / "two", "name: two\ndescription: " + "long " * 80)
    write_skill(tmp_path / "single", "name: single\ndescription: A skill folder itself")
    (library / "broken").mkdir()
    (library / "broken" / "SKILL.md").write_text("no front matter", encoding="utf-8")
    settings = PluginSettings(skills=("skills", "single", "missing"))
    plugins = load_plugins(settings, tmp_path)
    assert list(plugins.skills) == ["one", "two", "single"]
    assert len(plugins.problems) == 2
    assert any("missing" in p for p in plugins.problems)
    lines = plugins.index_text().splitlines()
    skill_lines = [line for line in lines if line.startswith(("one:", "two:", "single:"))]
    assert len(skill_lines) == 3
    assert lines[0].startswith("Skills")
    assert all(len(line) <= 200 + len("two: ") for line in skill_lines)


def test_empty_plugins_have_no_text_and_no_tools(tmp_path: Path) -> None:
    plugins = load_plugins(PluginSettings(), tmp_path)
    assert plugins.index_text() == ""
    assert plugins.tools() == {}


def test_skill_tool(tmp_path: Path) -> None:
    write_skill(tmp_path / "skills" / "one", "name: one\ndescription: First", "Step 1\nStep 2\n")
    plugins = load_plugins(PluginSettings(skills=("skills",)), tmp_path)
    tool = plugins.tools()["skill"]
    assert "mcp" not in plugins.tools()
    assert tool.resource({"name": "one"}, ctx(tmp_path)) is None

    async def scenario() -> None:
        ok = await tool.run({"name": "one"}, ctx(tmp_path))
        first, _, rest = ok.text.partition("\n")
        assert Path(first) == (tmp_path / "skills" / "one").resolve()
        assert rest == "Step 1\nStep 2\n"
        assert "name:" not in ok.text
        bad = await tool.run({"name": "nope"}, ctx(tmp_path))
        assert bad.is_error and "one" in bad.text

    asyncio.run(scenario())


def test_package_loading(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    write_skill(package / "skills" / "packed", "name: packed\ndescription: From a package")
    (package / "bin").mkdir()
    script = fake_server(tmp_path)
    manifest = f"""
        [mcp.fake]
        command = {json.dumps([sys.executable, str(script)])}
        env = {{ FAKE_KEY = "secret" }}

        [[hooks]]
        event = "before_tool"
        command = "check.sh"
        match = "sh"
        timeout_seconds = 5

        [[hooks]]
        event = "not_an_event"
        command = "x"
    """
    (package / "package.toml").write_text(textwrap.dedent(manifest), encoding="utf-8")
    plugins = load_plugins(PluginSettings(packages=("pkg",)), tmp_path)
    assert list(plugins.skills) == ["packed"]
    assert plugins.path_dirs == [package / "bin"]
    assert [(h.event, h.command, h.match, h.timeout_seconds) for h in plugins.hooks] == [
        ("before_tool", "check.sh", "sh", 5.0)
    ]
    assert len(plugins.problems) == 1
    assert plugins.servers["fake"].env == {"FAKE_KEY": "secret"}
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        where = await tool.run({"server": "fake", "tool": "where", "arguments": {}}, ctx(tmp_path))
        assert Path(where.text).resolve() == package.resolve()
        echo = await tool.run({"server": "fake", "tool": "echo", "arguments": {"text": "hi"}}, ctx(tmp_path))
        assert echo.text == "hi|secret"
        await plugins.close()

    asyncio.run(scenario())


def test_settings_path_and_index_for_servers(tmp_path: Path) -> None:
    (tmp_path / "tools").mkdir()
    plugins = load_plugins(PluginSettings(path=("tools", "gone"), mcp=(McpServerSpec("srv", ("x",)),)), tmp_path)
    assert plugins.path_dirs == [tmp_path / "tools"]
    assert plugins.problems == [f"path folder not found: {tmp_path / 'gone'}"]
    assert plugins.index_text().splitlines()[-1] == "srv"
    assert set(plugins.tools()) == {"mcp"}
    assert plugins.tools()["mcp"].resource({"server": "srv"}, ctx(tmp_path)) == "mcp:srv"


def test_mcp_client(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path, FAKE_KEY="k1")
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        listing = await tool.run({"server": "fake"}, ctx(tmp_path))
        lines = listing.text.splitlines()
        assert lines[0] == "echo: Echo the text back."
        assert [line.split(":")[0] for line in lines] == ["echo", "pic", "boom", "where", "svg", "slow", "key", "die"]

        described = await tool.run({"server": "fake", "tool": "echo"}, ctx(tmp_path))
        info = json.loads(described.text)
        assert info["name"] == "echo" and "properties" in info["inputSchema"]
        assert " " not in described.text.split('"description"')[0]

        called = await tool.run({"server": "fake", "tool": "echo", "arguments": {"text": "hi"}}, ctx(tmp_path))
        assert called.text == "hi|k1" and not called.is_error

        image = await tool.run({"server": "fake", "tool": "pic", "arguments": {}}, ctx(tmp_path))
        assert [(i.media_type, i.data) for i in image.images] == [("image/png", "QUJD")]

        failed = await tool.run({"server": "fake", "tool": "boom", "arguments": {}}, ctx(tmp_path))
        assert failed.is_error and failed.text == "it broke"

        unknown = await tool.run({"server": "fake", "tool": "zzz", "arguments": {}}, ctx(tmp_path))
        assert unknown.is_error and "echo" in unknown.text
        no_server = await tool.run({"server": "zzz"}, ctx(tmp_path))
        assert no_server.is_error and "fake" in no_server.text

        proc = plugins.client("fake")._proc
        assert proc is not None and proc.returncode is None
        await plugins.close()
        assert proc.returncode is not None

    asyncio.run(scenario())


def test_dead_server_gives_error_then_restarts(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        died = await tool.run({"server": "fake", "tool": "die", "arguments": {}}, ctx(tmp_path))
        assert died.is_error and "exited" in died.text
        again = await tool.run({"server": "fake", "tool": "echo", "arguments": {"text": "x"}}, ctx(tmp_path))
        assert again.text == "x|" and not again.is_error
        await plugins.close()

    asyncio.run(scenario())


def test_missing_command_is_a_tool_error(tmp_path: Path) -> None:
    spec = McpServerSpec("ghost", ("definitely-not-a-real-command-xyz",))
    plugins = load_plugins(PluginSettings(mcp=(spec,)), tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        out = await tool.run({"server": "ghost"}, ctx(tmp_path))
        assert out.is_error and "command not found" in out.text
        await plugins.close()

    asyncio.run(scenario())


def test_silent_server_times_out(tmp_path: Path) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import sys\nfor _ in sys.stdin:\n    pass\n", encoding="utf-8")
    plugins = load_plugins(PluginSettings(mcp=(McpServerSpec("mute", (sys.executable, str(script))),)), tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        out = await tool.run({"server": "mute"}, ctx(tmp_path, timeout=0.5))
        assert out.is_error and "did not answer" in out.text
        proc = plugins.client("mute")._proc
        assert proc is None
        await plugins.close()

    asyncio.run(scenario())


def test_unsupported_image_type_becomes_a_text_note(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        out = await tool.run({"server": "fake", "tool": "svg", "arguments": {}}, ctx(tmp_path))
        assert out.images == [] and not out.is_error
        assert "image/svg+xml" in out.text and "6 bytes" in out.text
        await plugins.close()

    asyncio.run(scenario())


def test_call_uses_the_policy_timeout_not_a_fixed_one(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        slow = {"server": "fake", "tool": "slow", "arguments": {"seconds": 1.5}}
        out = await tool.run(slow, ctx(tmp_path, timeout=0.5))
        assert out.is_error and "did not answer" in out.text
        done = await tool.run(slow, ctx(tmp_path, timeout=20.0))
        assert done.text == "slept" and not done.is_error
        await plugins.close()

    asyncio.run(scenario())


def test_cancel_interrupts_a_running_call_and_the_server_restarts(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        await tool.run({"server": "fake", "tool": "echo", "arguments": {}}, ctx(tmp_path))
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.3, cancel.set)
        began = time.monotonic()
        slow = {"server": "fake", "tool": "slow", "arguments": {"seconds": 30}}
        out = await tool.run(slow, ctx(tmp_path, timeout=60.0, cancel=cancel))
        assert time.monotonic() - began < 10
        assert out.is_error and "cancelled" in out.text
        assert plugins.client("fake")._proc is None
        again = await tool.run({"server": "fake", "tool": "echo", "arguments": {"text": "x"}}, ctx(tmp_path))
        assert again.text == "x|" and not again.is_error
        await plugins.close()

    asyncio.run(scenario())


def test_cancel_does_not_wait_behind_a_queued_call(tmp_path: Path) -> None:
    plugins = mcp_plugins(tmp_path)
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        await tool.run({"server": "fake", "tool": "echo", "arguments": {}}, ctx(tmp_path))
        holder = asyncio.ensure_future(
            tool.run({"server": "fake", "tool": "slow", "arguments": {"seconds": 3}}, ctx(tmp_path))
        )
        await asyncio.sleep(0.3)
        cancel = asyncio.Event()
        asyncio.get_running_loop().call_later(0.2, cancel.set)
        began = time.monotonic()
        queued = await tool.run(
            {"server": "fake", "tool": "echo", "arguments": {}}, ctx(tmp_path, cancel=cancel)
        )
        assert time.monotonic() - began < 2.0
        assert queued.is_error and "cancelled" in queued.text
        assert (await holder).text == "slept"
        await plugins.close()

    asyncio.run(scenario())


def test_policy_env_reaches_servers_under_their_own_env(tmp_path: Path) -> None:
    spec = McpServerSpec("fake", (sys.executable, str(fake_server(tmp_path))), {"FAKE_KEY": "own"})
    plugins = load_plugins(PluginSettings(mcp=(spec,)), tmp_path, {"FAKE_KEY": "policy", "OTHER": "1"})
    tool = plugins.tools()["mcp"]

    async def scenario() -> None:
        out = await tool.run({"server": "fake", "tool": "echo", "arguments": {"text": "t"}}, ctx(tmp_path))
        assert out.text == "t|own"
        await plugins.close()

    asyncio.run(scenario())
    bare = McpServerSpec("fake", (sys.executable, str(fake_server(tmp_path))))
    plugins = load_plugins(PluginSettings(mcp=(bare,)), tmp_path, {"FAKE_KEY": "policy"})

    async def second() -> None:
        out = await plugins.tools()["mcp"].run(
            {"server": "fake", "tool": "echo", "arguments": {"text": "t"}}, ctx(tmp_path)
        )
        assert out.text == "t|policy"
        await plugins.close()

    asyncio.run(second())


def test_servers_do_not_inherit_the_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    plugins = mcp_plugins(tmp_path)
    explicit = load_plugins(PluginSettings(), tmp_path, {"ANTHROPIC_API_KEY": "explicit"})
    explicit.servers.update(plugins.servers)

    async def scenario() -> None:
        call = {"server": "fake", "tool": "key", "arguments": {}}
        assert (await plugins.tools()["mcp"].run(call, ctx(tmp_path))).text == "absent"
        assert (await explicit.tools()["mcp"].run(call, ctx(tmp_path))).text == "explicit"
        await plugins.close()
        await explicit.close()

    asyncio.run(scenario())


def test_source_has_no_invisible_characters() -> None:
    source = Path(plugins_module.__file__).read_text(encoding="utf-8")
    assert "﻿" not in source
