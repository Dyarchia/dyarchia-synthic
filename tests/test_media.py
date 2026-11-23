"""Multimodal input: attachments on human messages and the ``view`` tool, offline.

Images and PDFs are built byte by byte here, so nothing depends on files from
elsewhere. The loop scenarios reuse the helpers of ``test_loop.py``.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import pytest
from test_loop import assert_replays, done, go, make_policy, project, records, results

import synthic.media as media
from synthic import cli
from synthic.agents import Runtime
from synthic.context import Context, ReceiptStore, pack, unpack
from synthic.media import MediaError, block_tokens, from_bytes, image_size, sniff
from synthic.testing import ScriptedProvider, reply, tool_use
from synthic.types import ContextSettings, ModelRequest, Record, ToolCall, ToolOutput, Usage


def png(width: int = 2, height: int = 3) -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    rows = b"".join(b"\x00" + b"\xff\x00\x00" * width for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def jpeg(width: int, height: int) -> bytes:
    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


PDF = b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n2 0 obj << /Type /Pages >> endobj\n" + (
    b"3 0 obj << /Type /Page >> endobj\n4 0 obj << /Type /Page >> endobj\n%%EOF\n"
)


def write(folder: Path, name: str, data: bytes) -> Path:
    path = folder / name
    path.write_bytes(data)
    return path


def test_types_and_sizes_come_from_the_bytes_not_the_name() -> None:
    assert sniff(png()) == "image/png" and image_size(png(7, 5), "image/png") == (7, 5)
    assert sniff(jpeg(640, 480)) == "image/jpeg" and image_size(jpeg(640, 480), "image/jpeg") == (640, 480)
    gif = b"GIF89a" + struct.pack("<HH", 30, 20) + b"\x00" * 10
    assert sniff(gif) == "image/gif" and image_size(gif, "image/gif") == (30, 20)
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8X" + b"\x00" * 8 + (99).to_bytes(3, "little") + (49).to_bytes(3, "little")
    assert sniff(webp) == "image/webp" and image_size(webp, "image/webp") == (100, 50)
    assert sniff(PDF) == "application/pdf"
    assert sniff(b"BM....") is None
    attachment = from_bytes(PDF, "doc.pdf")
    assert attachment.block["type"] == "document" and attachment.note.startswith("PDF, 2 pages")
    assert attachment.block["source"]["media_type"] == "application/pdf"


def test_what_the_api_would_refuse_is_refused_with_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(MediaError, match="not an image"):
        from_bytes(b"plain text", "notes.txt")
    with pytest.raises(MediaError, match="9000x10 px"):
        from_bytes(png(9000, 10), "wide.png")
    with pytest.raises(MediaError, match="encrypted"):
        from_bytes(PDF.replace(b"/Catalog", b"/Catalog /Encrypt 9 0 R"), "secret.pdf")
    monkeypatch.setattr(media, "MAX_IMAGE_BASE64", 100)
    with pytest.raises(MediaError, match="at most"):
        from_bytes(png(40, 40), "big.png")


def test_images_and_pdfs_are_weighed_in_tokens_not_base64() -> None:
    big = from_bytes(jpeg(5152, 2898), "photo.jpg").block
    assert block_tokens(big) == 4784
    small = from_bytes(png(56, 28), "icon.png").block
    assert block_tokens(small) == 2
    assert block_tokens(from_bytes(PDF, "doc.pdf").block) == 2 * media.PAGE_TOKENS


def test_pack_moves_base64_to_the_store_and_unpack_restores_it(tmp_path: Path) -> None:
    store = ReceiptStore(tmp_path / "receipts")
    block = from_bytes(png(), "a.png").block
    packed = pack([{"type": "text", "text": "x"}, block], store)
    assert "data" not in packed[1]["source"] and packed[1]["source"]["file"].endswith(".png")
    assert unpack(packed, store) == [{"type": "text", "text": "x"}, block]
    assert pack(block, store) == packed[1]
    assert len(list((store.dir / "media").iterdir())) == 1


def first_user(request: ModelRequest) -> list[dict[str, Any]]:
    return request.messages[0]["content"]


def test_attachment_goes_with_the_task_is_journaled_by_reference_and_replays(tmp_path: Path) -> None:
    image = media.load(write(tmp_path, "shot.png", png(4, 4)))
    provider = ScriptedProvider([done()])

    async def main() -> Runtime:
        runtime = Runtime(make_policy(tmp_path), project(tmp_path), provider)
        try:
            assert (await runtime.run("Describe it.", attachments=[image])).status == "delivered"
            assert_replays(runtime)
            return runtime
        finally:
            await runtime.close()

    runtime = asyncio.run(main())
    label, block, task = first_user(provider.requests[0])
    assert label["text"] == f"[attachment 1: {image.name} (image/png, 4x4, 1 KB)]"
    assert block == image.block and task == {"type": "text", "text": "Describe it."}
    user = next(r for r in records(runtime) if r["kind"] == "user")
    source = user["data"]["attachments"][0]["block"]["source"]
    assert "data" not in source and (runtime.run_dir / "root" / "receipts" / "media" / source["file"]).is_file()
    assert image.block["source"]["data"] not in (runtime.run_dir / "root" / "journal.jsonl").read_text(encoding="utf-8")


def test_message_and_resume_text_carry_attachments(tmp_path: Path) -> None:
    first = media.load(write(tmp_path, "one.png", png(2, 2)))
    second = media.load(write(tmp_path, "two.pdf", PDF))
    box: dict[str, Runtime] = {}

    def queue(request: ModelRequest) -> Any:
        box["runtime"].add("Look at this too.", attachments=[first])
        return reply(tool_use("receipt", {"id": "r9"}))

    def prepare(runtime: Runtime) -> None:
        box["runtime"] = runtime

    runtime, result, provider = go(tmp_path, [queue, done()], prepare=prepare)
    assert result.status == "delivered"
    sent = provider.requests[1].messages[-1]["content"]
    assert [b["type"] for b in sent] == ["tool_result", "text", "image", "text"]

    resumed = ScriptedProvider([done("Read it.")])

    async def main() -> None:
        again = Runtime(make_policy(tmp_path), project(tmp_path), resumed, run_id=runtime.run_id)
        try:
            assert (await again.resume("And this PDF.", [second])).status == "delivered"
            assert_replays(again)
        finally:
            await again.close()

    asyncio.run(main())
    last = resumed.requests[0].messages[-1]["content"]
    assert [b["type"] for b in last[-3:]] == ["text", "document", "text"] and last[-1]["text"] == "And this PDF."


def view_task(tmp_path: Path) -> dict[str, Any]:
    return {"role": {"lead": {"tools": ["sh", "edit", "view"]}}}


def test_view_shows_an_image_or_a_pdf_and_refuses_what_cannot_be_sent(tmp_path: Path) -> None:
    folder = project(tmp_path)
    write(folder, "pic.png", png(3, 3))
    write(folder, "doc.pdf", PDF)
    write(folder, "wide.png", png(9001, 1))
    script = [
        reply(tool_use("view", {"path": "pic.png"}), tool_use("view", {"path": "doc.pdf"})),
        reply(tool_use("view", {"path": "wide.png"}), tool_use("view", {"path": "notes.txt"})),
        done(),
    ]
    write(folder, "notes.txt", b"hello")
    runtime, result, provider = go(tmp_path, script, view_task(tmp_path))
    assert result.status == "delivered"
    shown = {r["content"][1]["type"]: r["content"] for r in results(provider.requests[1])}
    assert [b["type"] for b in shown["image"]] == ["text", "image"]
    assert [b["type"] for b in shown["document"]] == ["text", "document"]
    assert shown["image"][0]["text"].endswith("pic.png (image/png, 3x3, 1 KB)")
    refused = sorted(results(provider.requests[2]), key=lambda r: "not an image" in r["content"])
    assert all(r["is_error"] for r in refused)
    assert "9001x1 px" in refused[0]["content"] and "not an image" in refused[1]["content"]
    assert "view" in [t["name"] for t in provider.requests[0].tools]
    assert_replays(runtime)


def test_view_is_not_granted_by_default_roles(tmp_path: Path) -> None:
    runtime, _, provider = go(tmp_path, [done()])
    assert "view" not in [t["name"] for t in provider.requests[0].tools]
    defaults = make_policy(tmp_path).policy.roles
    assert all("view" not in defaults[name].tools for name in ("explore", "mechanic", "build"))


def cold_clock(request: ModelRequest, box: dict[str, Runtime]) -> None:
    """Every reading of the context clock is 1,000 s after the previous one: the cache is always cold."""
    agent = box["runtime"].root_agent
    assert agent is not None
    agent.context._clock = itertools.count(0, 1000).__next__


def test_cold_cache_stubs_old_attachments_and_view_results_and_replays(tmp_path: Path) -> None:
    folder = project(tmp_path)
    write(folder, "pic.png", png(3, 3))
    image = media.load(write(tmp_path, "task.png", png(5, 5)))
    box: dict[str, Runtime] = {}

    def first(request: ModelRequest) -> Any:
        cold_clock(request, box)
        return reply(tool_use("view", {"path": "pic.png"}))

    script = [first, reply(tool_use("receipt", {"id": "r9"})), reply(tool_use("receipt", {"id": "r8"})), done()]
    task = {**view_task(tmp_path), "context": {"clean_keep_recent": 1}}
    provider = ScriptedProvider(script)

    async def main() -> Runtime:
        runtime = Runtime(make_policy(tmp_path, task), folder, provider)
        box["runtime"] = runtime
        try:
            assert (await runtime.run("Compare.", attachments=[image])).status == "delivered"
            assert_replays(runtime)
            return runtime
        finally:
            await runtime.close()

    runtime = asyncio.run(main())
    messages = provider.requests[-1].messages
    task_message = messages[0]["content"]
    assert task_message[1]["text"] == f"[image {image.name} removed to save context: view the file again to see it]"
    viewed = messages[2]["content"][0]["content"]
    assert viewed[1]["text"].startswith("[image ") and viewed[1]["text"].endswith("view the file again to see it]")
    cleans = [r["data"] for r in records(runtime) if r["kind"] == "clean"]
    assert cleans[0]["attachments"] == [1] and any(c["call_ids"] for c in cleans)


def test_without_view_an_attachment_is_never_stubbed(tmp_path: Path) -> None:
    settings = ContextSettings(clean_after_seconds=0)
    clock = itertools.count(0, 1000).__next__
    ctx = Context(settings, ReceiptStore(tmp_path / "r"), ["system"], clock)
    ctx.add_user("look", [from_bytes(png(), "a.png")])
    ctx.add_assistant([{"type": "tool_use", "id": "c1", "name": "sh", "input": {}}])
    ctx.add_tool_results([ctx.cap(ToolCall("c1", "sh", {}), ToolOutput("ok"))])
    ctx.note_model_call(Usage(input_tokens=10))
    assert ctx.clean_attachments_if_cold() == []
    viewer = Context(settings, ReceiptStore(tmp_path / "v"), ["system"], itertools.count(0, 1000).__next__, can_view=True)
    viewer.add_user("look", [from_bytes(png(), "a.png")])
    viewer.add_assistant([{"type": "tool_use", "id": "c1", "name": "sh", "input": {}}])
    viewer.add_tool_results([viewer.cap(ToolCall("c1", "sh", {}), ToolOutput("ok"))])
    viewer.note_model_call(Usage(input_tokens=10))
    assert viewer.clean_attachments_if_cold() == [1]
    replayed = Context(settings, ReceiptStore(tmp_path / "v"), ["system"])
    replayed.replay(
        [
            Record(1, 0, "x", "user", {"text": "look", "source": "human", "attachments": [
                {"name": "a.png", "note": "n", "block": from_bytes(png(), "a.png").block}
            ]}),
        ]
    )
    assert replayed.build()[1][0]["content"][1]["type"] == "image"


def test_cli_attach_checks_files_before_anything_starts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write(tmp_path, "notes.txt", b"hello")
    code = cli.main(["run", "x", "--attach", str(tmp_path / "notes.txt"), "--set", f"journal.dir={tmp_path.as_posix()}"],
                    provider_factory=lambda: ScriptedProvider([]))
    assert code == 2 and "cannot attach" in capsys.readouterr().err
    code = cli.main(["run", "x", "--attach", str(tmp_path / "missing.png"), "--set", f"journal.dir={tmp_path.as_posix()}"],
                    provider_factory=lambda: ScriptedProvider([]))
    assert code == 2 and "cannot be read" in capsys.readouterr().err


def test_cli_run_and_say_send_attachments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    picture = write(tmp_path, "p.png", png(2, 2))
    provider = ScriptedProvider([done()])
    opts = ["--set", f"journal.dir={(tmp_path / 'runs').as_posix()}", "--set", "memory.enabled=false",
            "--cwd", str(project(tmp_path))]
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))
    assert cli.main(["run", "Look.", "--attach", str(picture), "--quiet", *opts], provider_factory=lambda: provider) == 0
    assert [b["type"] for b in first_user(provider.requests[0])] == ["text", "image", "text"]
    run_dir = next((tmp_path / "runs").iterdir())
    (run_dir / "alive").write_text("1", encoding="utf-8")
    assert cli.main(["say", run_dir.name, "and this", "--attach", str(picture), *opts]) == 0
    line = json.loads((run_dir / cli.CONTROL_FILE).read_text(encoding="utf-8"))
    assert line["attach"] == [str(picture.resolve())] and line["text"] == "and this"
