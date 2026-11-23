"""Images and PDFs the model can read: type sniffing, the API's limits and the content blocks, standard library only.

What the docs state (``anthropic-version: 2026-10-21``):

* Images go as ``image`` blocks with a base64 source. The API accepts JPEG, PNG,
  GIF and WebP, at most 10 MB of base64 per image and 8000x8000 px. An image
  costs about ``ceil(w / 28) * ceil(h / 28)`` visual tokens once scaled to the
  model's tier (long edge 2,576 px and 4,784 tokens on the high resolution
  tier). Above 20 images in one request a stricter per-image limit applies;
  that depends on the whole request, so it is not checked here.
  https://platform.claude.com/docs/en/build-with-claude/vision
* PDFs go as ``document`` blocks with a base64 source of ``application/pdf``.
  A request is at most 128 MB and 600 pages, and PDFs with a password or
  encryption are refused. A page costs its text (1,500 to 3,000 tokens) plus at
  most 1,568 visual tokens.
  https://platform.claude.com/docs/en/build-with-claude/pdf-support
* A ``tool_result`` may hold ``text``, ``image`` and ``document`` blocks, so the
  ``view`` tool returns a PDF the same way an attachment sends it.
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls

"MB" is read as 10^6 bytes, the stricter reading, so nothing passed here is
refused by the API for its size.
"""

from __future__ import annotations

import base64
import math
import re
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

from .types import Attachment, Block

IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})
PDF = "application/pdf"
MEDIA_BLOCKS = frozenset({"image", "document"})
MAX_IMAGE_BASE64 = 10_000_000
MAX_IMAGE_SIDE = 8000
MAX_REQUEST_BYTES = 128_000_000
MAX_PDF_PAGES = 600
IMAGE_TOKENS_CAP = 4784
IMAGE_LONG_EDGE = 2576
PAGE_TOKENS = 1568 + 2250
_PAGE = re.compile(rb"/Type\s*/Page(?![a-zA-Z])")


class MediaError(ValueError):
    """A file the API would refuse; the message says why, for the human or the model."""


def sniff(data: bytes) -> str | None:
    """The media type from the first bytes of a file, or ``None`` when it is not one the API reads."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if b"%PDF-" in data[:1024]:
        return PDF
    return None


def image_size(data: bytes, media_type: str) -> tuple[int, int] | None:
    """Width and height read from the image header, or ``None`` when the header cannot be read."""
    try:
        match media_type:
            case "image/png":
                return struct.unpack(">II", data[16:24])
            case "image/gif":
                return struct.unpack("<HH", data[6:10])
            case "image/webp":
                return _webp_size(data)
            case "image/jpeg":
                return _jpeg_size(data)
    except struct.error:
        return None
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Walk the JPEG markers to the first start-of-frame segment, which holds the size."""
    position = 2
    while position + 9 < len(data):
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker == 0xFF:
            position += 1
            continue
        if marker in (0x01, *range(0xD0, 0xDA)):
            position += 2
            continue
        length = struct.unpack(">H", data[position + 2 : position + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[position + 5 : position + 9])
            return width, height
        position += 2 + length
    return None


def _pages(data: bytes) -> int:
    """Page objects counted in the PDF; 0 when they are hidden in compressed object streams."""
    return len(_PAGE.findall(data))


def _size(count: int) -> str:
    return f"{count / 1_000_000:.1f} MB" if count >= 1_000_000 else f"{max(1, round(count / 1000))} KB"


def _check(data: bytes, media_type: str) -> str:
    """Raise ``MediaError`` for what the API would refuse; otherwise return the short description."""
    encoded = 4 * math.ceil(len(data) / 3)
    if media_type == PDF:
        if encoded > MAX_REQUEST_BYTES:
            raise MediaError(f"is {_size(len(data))}; a request may carry at most {_size(MAX_REQUEST_BYTES)}")
        if b"/Encrypt" in data:
            raise MediaError("is an encrypted or password-protected PDF, which the API does not read")
        pages = _pages(data)
        if pages > MAX_PDF_PAGES:
            raise MediaError(f"has {pages} pages; a request may carry at most {MAX_PDF_PAGES}")
        shown = f", {pages} pages" if pages else ""
        return f"PDF{shown}, {_size(len(data))}"
    if encoded > MAX_IMAGE_BASE64:
        raise MediaError(
            f"is {_size(len(data))} ({_size(encoded)} in base64); the API takes at most {_size(MAX_IMAGE_BASE64)} "
            "of base64 per image: make it smaller first"
        )
    dims = image_size(data, media_type)
    if dims is not None and max(dims) > MAX_IMAGE_SIDE:
        raise MediaError(
            f"is {dims[0]}x{dims[1]} px; the API takes at most {MAX_IMAGE_SIDE}x{MAX_IMAGE_SIDE}: make it smaller first"
        )
    shown = f", {dims[0]}x{dims[1]}" if dims else ""
    return f"{media_type}{shown}, {_size(len(data))}"


def from_bytes(data: bytes, name: str) -> Attachment:
    """The content block for ``data``, or ``MediaError`` naming ``name`` and the reason."""
    media_type = sniff(data)
    if media_type is None:
        raise MediaError(f"{name}: not an image (JPEG, PNG, GIF, WebP) or a PDF, the only files the model can see")
    try:
        note = _check(data, media_type)
    except MediaError as error:
        raise MediaError(f"{name} {error}") from None
    kind = "document" if media_type == PDF else "image"
    source = {"type": "base64", "media_type": media_type, "data": base64.b64encode(data).decode("ascii")}
    return Attachment(name, {"type": kind, "source": source}, note)


def load(path: Path | str) -> Attachment:
    """Read and check one file. ``name`` is the absolute path, so a stub can tell the model where it was."""
    target = Path(path).expanduser().resolve()
    try:
        data = target.read_bytes()
    except OSError as error:
        raise MediaError(f"{target}: cannot be read: {error.strerror or error}") from None
    return from_bytes(data, str(target))


@lru_cache(maxsize=256)
def _tokens(media_type: str, data: str) -> int:
    """Estimated input tokens of one image or PDF; cached, since the context weighs its history every step."""
    try:
        raw = base64.b64decode(data, validate=False)
    except ValueError:
        return IMAGE_TOKENS_CAP
    if media_type == PDF:
        return max(1, _pages(raw)) * PAGE_TOKENS
    dims = image_size(raw, media_type)
    if dims is None or min(dims) <= 0:
        """No readable header: guess from the size, about 100 bytes of compressed image per visual
        token (a 1000x1000 JPEG of some 130 KB is 1,296 tokens)."""
        return min(IMAGE_TOKENS_CAP, max(1, len(raw) // 100))
    width, height = dims
    scale = min(1.0, IMAGE_LONG_EDGE / max(width, height))
    return min(IMAGE_TOKENS_CAP, math.ceil(width * scale / 28) * math.ceil(height * scale / 28))


def block_tokens(block: Block) -> int:
    """Estimated input tokens of an ``image`` or ``document`` block; text documents count their characters."""
    source: Any = block.get("source")
    if not isinstance(source, dict):
        return 0
    data = source.get("data")
    if source.get("type") != "base64" or not isinstance(data, str):
        return len(data) // 4 if isinstance(data, str) else 0
    return _tokens(str(source.get("media_type", "")), data)
