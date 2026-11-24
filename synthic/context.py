"""Job 3: assemble what the model sees on each step, spending the minimum.

The whole context is paid for on every step, and the vendor discounts the part
of the prompt that is byte-identical to the previous call (the prompt cache)
for as long as the cache stays warm. Everything here follows from that:

* The history only grows at the end. Old messages are never rewritten while
  the cache is warm, and the system blocks never change during a run.
* A long tool output enters the context as head, tail and a receipt id; the
  full text stays on disk in a ``ReceiptStore`` and the ``receipt`` tool lets
  the model search or slice it without loading it whole.
* When the cache has already expired (more than ``clean_after_seconds`` and
  the cache TTL since the last call), the next call is paid in full whatever we send, so that is
  the one moment when shrinking old tool results costs nothing extra.
* Compaction replaces the history with a summary, keeping the newest turns as
  they are. It runs when the context is nearly full, and between two tasks of
  one session once the prompt is large enough to be worth it.

History blocks are kept in the provider's native shape (see ``types.py``).
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthic.media import MEDIA_BLOCKS, block_tokens
from synthic.types import (
    Attachment,
    Block,
    ContextSettings,
    Message,
    ModelSpec,
    Record,
    ToolCall,
    ToolContext,
    ToolOutput,
    Usage,
)

_SNAP = 0.2
"""Share of the head or tail we are willing to give up to cut on a line boundary instead of mid-line."""

_STUB_FIRST_LINE = 120
_RECEIPT_ID = re.compile(r"r([1-9]\d*)")
_MEDIA_FILE = re.compile(r"m-[0-9a-f]{32}\.[a-z]+")
_MEDIA_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp", "application/pdf": "pdf"}
CHARS_PER_TOKEN = 4
"""How many characters of text weigh as much as one token; media is weighed in estimated tokens times this."""
THINKING = frozenset({"thinking", "redacted_thinking"})
_TTL_SECONDS = {"5m": 300.0, "1h": 3600.0}
_COMPACTION_TEMPLATE = Path(__file__).with_name("compaction.md")
PINNED_SOURCES = ("human", "parent")


def _lines(text: str) -> list[str]:
    """Split on ``\\n`` only, so line numbers agree on every platform; a final newline opens no extra line."""
    parts = text.split("\n")
    if parts[-1] == "":
        parts.pop()
    return parts


def _numbered(lines: list[str], first: int) -> list[str]:
    return [f"{first + i}: {line.rstrip("\r")}" for i, line in enumerate(lines)]


class ReceiptStore:
    """Full tool outputs on disk, one UTF-8 file per receipt, ids ``r1``, ``r2``, ...

    Files are written and read with ``newline=""`` so that the stored text is
    exactly the tool output on every platform: character counts, line slices
    and stubs must not depend on Windows newline translation.
    """

    def __init__(self, dir: Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        numbers = (
            int(m.group(1))
            for p in self.dir.glob("r*.txt")
            if (m := _RECEIPT_ID.fullmatch(p.stem))
        )
        self._next = max(numbers, default=0) + 1

    def _path(self, receipt_id: str) -> Path | None:
        if not isinstance(receipt_id, str) or not _RECEIPT_ID.fullmatch(receipt_id):
            return None
        return self.dir / f"{receipt_id}.txt"

    def put(self, text: str) -> str:
        while True:
            receipt_id = f"r{self._next}"
            self._next += 1
            try:
                with open(self.dir / f"{receipt_id}.txt", "x", encoding="utf-8", newline="") as f:
                    f.write(text)
            except FileExistsError:
                continue
            return receipt_id

    def exists(self, receipt_id: str) -> bool:
        path = self._path(receipt_id)
        return path is not None and path.is_file()

    def get(self, receipt_id: str) -> str:
        path = self._path(receipt_id)
        if path is None or not path.is_file():
            raise KeyError(receipt_id)
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()

    def info(self, receipt_id: str) -> tuple[int, int]:
        """Return ``(chars, lines)`` of a stored output."""
        text = self.get(receipt_id)
        return len(text), len(_lines(text))

    def put_media(self, media_type: str, data: str) -> str:
        """Store base64 ``data`` once under ``media/``, named by its hash, and return the file name."""
        digest = hashlib.sha256(data.encode("ascii", "replace")).hexdigest()[:32]
        name = f"m-{digest}.{_MEDIA_EXT.get(media_type, 'bin')}"
        folder = self.dir / "media"
        folder.mkdir(exist_ok=True)
        target = folder / name
        if not target.is_file():
            temp = folder / f".{name}.tmp"
            temp.write_text(data, encoding="ascii", errors="replace", newline="")
            temp.replace(target)
        return name

    def get_media(self, name: str) -> str:
        if not isinstance(name, str) or not _MEDIA_FILE.fullmatch(name):
            raise KeyError(name)
        try:
            with open(self.dir / "media" / name, encoding="ascii", newline="") as f:
                return f.read()
        except FileNotFoundError:
            raise KeyError(name) from None


def pack(value: Any, receipts: ReceiptStore) -> Any:
    """A copy of ``value`` for the journal: each base64 source's data goes to the receipt store, by file name.

    Media can be megabytes, and a journal line that size would make every
    reader and the crash repair of the file slow. ``unpack`` restores the very
    same strings, so a replayed history stays byte for byte the live one.
    """
    if isinstance(value, list):
        return [pack(item, receipts) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "base64" and isinstance(value.get("data"), str):
        media_type = str(value.get("media_type", ""))
        return {**{k: v for k, v in value.items() if k != "data"}, "file": receipts.put_media(media_type, value["data"])}
    return {key: pack(item, receipts) for key, item in value.items()}


def unpack(value: Any, receipts: ReceiptStore) -> Any:
    """The inverse of ``pack``; values journaled before packing existed pass through unchanged."""
    if isinstance(value, list):
        return [unpack(item, receipts) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") == "base64" and "file" in value and "data" not in value:
        try:
            data = receipts.get_media(value["file"])
        except KeyError:
            raise FileNotFoundError(f"media file {value['file']!r} is missing from {receipts.dir / 'media'}") from None
        return {**{k: v for k, v in value.items() if k != "file"}, "data": data}
    return {key: unpack(item, receipts) for key, item in value.items()}


@dataclass(frozen=True)
class ToolResultEntry:
    """One tool result as it enters the context.

    ``content`` is already capped: a string, or a list of text, image and
    document blocks. ``receipt`` names the stored full output, or is ``None``
    when the output was too short to be worth one. ``media`` names the file
    its images or documents show when the agent can show it again (``view``).
    Only results with a receipt or with ``media`` are ever cleaned.
    """

    call_id: str
    content: str | list[Block]
    is_error: bool = False
    receipt: str | None = None
    media: str | None = None


def _weight(value: Any) -> int:
    """About the length of ``value`` as JSON, with every image or document weighed by its estimated tokens."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, dict):
        if value.get("type") in MEDIA_BLOCKS:
            return block_tokens(value) * CHARS_PER_TOKEN
        return sum(len(key) + 4 + _weight(item) for key, item in value.items())
    if isinstance(value, list):
        return sum(_weight(item) + 1 for item in value)
    return len(str(value))


def _block_chars(block: Any) -> int:
    """Characters the model reads in a block. Images and PDFs are weighed in estimated tokens
    (``media.block_tokens``) times ``CHARS_PER_TOKEN``, never by their base64 size."""
    if isinstance(block, str):
        return len(block)
    match block.get("type"):
        case "text":
            return len(block["text"])
        case "thinking":
            return len(block.get("thinking", ""))
        case "redacted_thinking":
            return 0
        case "image" | "document":
            return block_tokens(block) * CHARS_PER_TOKEN
        case "tool_use":
            return len(block["name"]) + _weight(block["input"])
        case "tool_result":
            content = block.get("content", "")
            if isinstance(content, str):
                return len(content)
            return sum(_block_chars(b) for b in content)
        case _:
            return _weight(block)


class Context:
    """The conversation history of one agent and every decision about its size.

    Invariants kept by every method:

    * Roles alternate: consecutive user-side additions (results, a human
      message, a hook's text, an engine notice) merge into one user message,
      and consecutive assistant content merges into one assistant message.
    * All results of one assistant turn sit in the single user message that
      follows it, before any text in that message: text added while results
      are still pending is kept after them, and a later result is inserted
      ahead of it. ``build()`` is meant to be called only when no result is
      pending; between two such calls the history only grows at the end.
    """

    def __init__(
        self,
        settings: ContextSettings,
        receipts: ReceiptStore,
        system: list[str],
        clock: Callable[[], float] = time.monotonic,
        *,
        can_view: bool = False,
    ) -> None:
        self._settings = settings
        self._receipts = receipts
        self._system: list[Block] = [{"type": "text", "text": text} for text in system]
        self._clock = clock
        self._can_view = can_view
        self._messages: list[Message] = []
        self._receipt_of: dict[str, str] = {}
        self._media_of: dict[str, str] = {}
        self._cleaned: set[str] = set()
        self._attachments: dict[int, tuple[str, Block]] = {}
        self._attached = 0
        self._held: list[tuple[str, Sequence[Attachment], bool]] = []
        self._pinned: list[str] = []
        self._last_call_at: float | None = None
        self._last_prompt_tokens = 0
        self._chars_at_call = 0

    def add_user(self, text: str, attachments: Sequence[Attachment] = (), *, pinned: bool = False) -> None:
        """Add a user message; ``attachments`` go before the text, each after a one-line label naming it.

        ``pinned`` marks what the agent was asked: the task and every later
        message from the human or the parent. A compaction carries those texts
        over word for word (see ``apply_compaction``) and never leaves them to
        the summary, which a model writes and may shorten.

        Attachments are numbered in the order they enter, which replay
        repeats, so a ``clean`` record can name them by number.

        While a server tool call is open (``server_pending``) the message is
        held and enters right after the next assistant turn: the docs require
        the user message that follows such a turn to hold only ``tool_result``
        blocks, and a text there ends the turn with a 400. Holding depends on
        the history alone, so replay holds and releases at the same points.
        https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
        """
        if not text:
            raise ValueError("empty user text")
        if self.server_pending():
            self._held.append((text, attachments, pinned))
            return
        if pinned:
            self._pinned.append(text)
        blocks: list[Block] = []
        for attachment in attachments:
            self._attached += 1
            media = copy.deepcopy(attachment.block)
            label = f"[attachment {self._attached}: {attachment.name} ({attachment.note})]"
            blocks += [{"type": "text", "text": label}, media]
            self._attachments[self._attached] = (attachment.name, media)
        blocks.append({"type": "text", "text": text})
        if self._messages and self._messages[-1]["role"] == "user":
            self._messages[-1]["content"].extend(blocks)
        else:
            self._messages.append({"role": "user", "content": blocks})

    def add_assistant(self, content: list[Block]) -> bool:
        """Add an assistant turn; True when it closed a server tool call and released held user messages."""
        if not content:
            raise ValueError("empty assistant content")
        if self._pending():
            raise ValueError("the previous assistant turn has tool calls without results")
        blocks = copy.deepcopy(content)
        if self._messages and self._messages[-1]["role"] == "assistant":
            self._messages[-1]["content"].extend(blocks)
        else:
            self._messages.append({"role": "assistant", "content": blocks})
        if not self._held or self.server_pending():
            return False
        held, self._held = self._held, []
        for text, attachments, pinned in held:
            self.add_user(text, attachments, pinned=pinned)
        return True

    def server_pending(self) -> bool:
        """Whether a ``server_tool_use`` of the history still has no result block.

        That happens when the model called a server tool and a client tool in
        one turn (the reply stops at ``tool_use``) and after some ``pause_turn``
        replies; the provider runs the call at the start of the next request.
        """
        called: set[str] = set()
        for message in self._messages:
            if message["role"] != "assistant":
                continue
            for block in message["content"]:
                kind = block.get("type")
                if kind == "server_tool_use":
                    called.add(block.get("id"))
                elif kind != "tool_result" and "tool_use_id" in block:
                    called.discard(block["tool_use_id"])
        return bool(called)

    def add_tool_results(self, results: list[ToolResultEntry]) -> None:
        pending = self._pending()
        for entry in results:
            if entry.call_id not in pending:
                raise ValueError(f"no pending tool call with id {entry.call_id!r}")
            pending.remove(entry.call_id)
            block = {
                "type": "tool_result",
                "tool_use_id": entry.call_id,
                "content": copy.deepcopy(entry.content),
                "is_error": entry.is_error,
            }
            if self._messages[-1]["role"] == "assistant":
                self._messages.append({"role": "user", "content": [block]})
            else:
                content = self._messages[-1]["content"]
                at = 0
                while at < len(content) and content[at]["type"] == "tool_result":
                    at += 1
                content.insert(at, block)
            if entry.receipt is not None:
                self._receipt_of[entry.call_id] = entry.receipt
            if entry.media is not None:
                self._media_of[entry.call_id] = entry.media

    def _pending(self) -> list[str]:
        """Ids of the last assistant message's ``tool_use`` blocks that have no result yet, in order."""
        for index in range(len(self._messages) - 1, -1, -1):
            if self._messages[index]["role"] == "assistant":
                break
        else:
            return []
        calls = [b["id"] for b in self._messages[index]["content"] if b["type"] == "tool_use"]
        if index + 1 < len(self._messages):
            answered = {
                b["tool_use_id"]
                for b in self._messages[index + 1]["content"]
                if b["type"] == "tool_result"
            }
            calls = [c for c in calls if c not in answered]
        return calls

    def cap(self, call: ToolCall, output: ToolOutput) -> ToolResultEntry:
        """Turn a tool output into what enters the context, storing the full text under a receipt when worth it.

        Outputs of the ``receipt`` tool are views of an existing receipt: they
        are never stored again (no receipt chains); they point back at the
        receipt they came from, which ``ReceiptTool`` puts in ``meta``.
        """
        s = self._settings
        text = output.text
        over = len(text) > s.output_cap
        receipt: str | None = None
        if over or len(text) > s.clean_min_chars:
            if call.name == ReceiptTool.name:
                source = output.meta.get("receipt")
                receipt = source if isinstance(source, str) and self._receipts.exists(source) else None
            else:
                receipt = self._receipts.put(text)
        body = self._truncate(text, receipt, call.name == ReceiptTool.name) if over else text
        content: str | list[Block]
        media = [("image", i.media_type, i.data) for i in output.images]
        media += [("document", d.media_type, d.data) for d in output.documents]
        if media:
            content = [{"type": "text", "text": body}] if body else []
            content += [
                {"type": kind, "source": {"type": "base64", "media_type": media_type, "data": data}}
                for kind, media_type, data in media
            ]
        else:
            content = body or "(no output)"
        name = output.meta.get("media") if media else None
        return ToolResultEntry(call.id, content, output.is_error, receipt, name if isinstance(name, str) else None)

    def _truncate(self, text: str, receipt: str | None, is_view: bool) -> str:
        s = self._settings
        head_end = s.output_head
        newline = text.rfind("\n", 0, head_end)
        if newline >= head_end - int(head_end * _SNAP) and newline > 0:
            head_end = newline + 1
        tail_start = max(len(text) - s.output_tail, head_end)
        if text[tail_start - 1] != "\n":
            newline = text.find("\n", tail_start, tail_start + int(s.output_tail * _SNAP))
            if newline != -1:
                tail_start = newline + 1
        head, tail = text[:head_end], text[tail_start:]
        total_lines = len(_lines(text))
        last_head_line = head.count("\n") + (0 if head.endswith("\n") else 1)
        first_tail_line = text.count("\n", 0, tail_start) + 1
        shown = f"1-{last_head_line} and {first_tail_line}-{total_lines}" if tail else f"1-{last_head_line}"
        if is_view and receipt:
            where = f"This is a view of receipt {receipt}: ask for a narrower search or line range."
        elif receipt:
            where = f"Full output: receipt {receipt}, search it or read a line range with the receipt tool."
        else:
            where = "Ask for a narrower search or line range."
        notice = (
            f"[output truncated: {len(text):,} chars, {total_lines:,} lines; showing lines {shown}. {where}]"
        )
        separator = "" if head.endswith("\n") or not head else "\n"
        return f"{head}{separator}{notice}\n{tail}" if tail else f"{head}{separator}{notice}"

    def build(self) -> tuple[list[Block], list[Message]]:
        """System blocks and messages for the next call, as deep copies: the caller cannot alter the history."""
        return copy.deepcopy(self._system), copy.deepcopy(self._messages)

    def note_model_call(self, usage: Usage) -> None:
        """Record a model call; call it when the reply arrives, before the reply is appended.

        The time is taken after the reply, while the cache lifetime runs from
        the start of the request, so "more than the TTL since this moment"
        always means the cache entry has expired.
        """
        self._last_call_at = self._clock()
        self._last_prompt_tokens = usage.context_tokens
        self._chars_at_call = self._chars()

    @property
    def last_prompt_tokens(self) -> int:
        """Prompt size measured on the last model call (0 before any)."""
        return self._last_prompt_tokens

    def _chars(self) -> int:
        return sum(_block_chars(b) for b in self._system) + sum(
            _block_chars(b) for m in self._messages for b in m["content"]
        )

    def clean_if_cold(self) -> list[str]:
        """Replace old tool results that have a receipt with a one-line stub, but only when the cache is cold.

        What is cleaned: every tool result with a receipt, not cleaned yet and
        not among the ``clean_keep_recent`` most recent tool results (all
        results count for that, with or without receipt). Results without a
        receipt are never touched. Nothing happens before the first model call
        or while the cache may still be warm: less than ``clean_after_seconds``
        or the ``cache_ttl``, whichever is longer, since the last call.
        Image and document blocks of a cleaned result are kept, since the
        receipt holds text only, unless the result names the file they show
        (``ToolResultEntry.media``, from ``view``): those become a stub telling
        the model to view the file again. A ``view`` result counts like any
        other for ``clean_keep_recent``.

        Editing an earlier message has a second consequence besides the cache.
        What the fact sheet (section 6) states: on models with preserved
        thinking (Opus 5.5, Sonnet 5.5, Haiku 5.5, Fable and Mythos 5.x) each
        ``thinking`` block's prefix (``system``, ``tools``, earlier messages)
        must be unchanged, and edits to earlier messages make later thinking
        invalid: 400 on Fable 5.2, Mythos 5.2 and Mythos 5.1, a possible 400
        on the others; thinking blocks may be omitted from the oldest end, the
        newest end or entirely, never one in the middle; and within a tool-use
        turn they must be passed back. Sources:
        https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/prefix-checks
        and https://platform.claude.com/docs/en/build-with-claude/thinking

        The sheet does not say what to do after an edit, so the same docs were
        read in the local crawl the sheet was made from. They give the recovery
        for an edited history: remove every ``thinking`` and
        ``redacted_thinking`` block after the edit and keep every other block,
        with no exception for the turn in progress, and they state that in
        adaptive mode (the default of every current model) no assistant turn
        needs to start with a thinking block. The documented cost is extra
        output tokens, as the model works again through reasoning it can no
        longer read. Sources:
        https://platform.claude.com/docs/en/build-with-claude/preserved-thinking/fix-bound-to-a-different-conversation
        ("Remove the invalid thinking by hand", "What dropping thinking costs"),
        https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting
        and https://platform.claude.com/docs/en/build-with-claude/thinking
        (thinking mode within a turn).

        So a clean strips both kinds of thinking block from every assistant
        message after the first edited one, the turn in progress included,
        which overrides the "pass back within a tool-use turn" rule as the
        recovery procedure does, and keeps the blocks before the edit, whose
        prefix is unchanged. The removed blocks form the newest contiguous run,
        which the omission rule allows. An assistant message left empty is
        dropped and the user messages around it merge, so roles keep
        alternating (the docs warn that two same-role messages in a row make
        the API count positions differently).
        """
        if not self._cold():
            return []
        order = [
            b["tool_use_id"]
            for m in self._messages
            if m["role"] == "user"
            for b in m["content"]
            if b["type"] == "tool_result"
        ]
        keep = self._settings.clean_keep_recent
        protected = set(order[-keep:]) if keep > 0 else set()
        ids = [
            c
            for c in order
            if (c in self._receipt_of or c in self._media_of) and c not in self._cleaned and c not in protected
        ]
        if ids:
            self._clean(ids)
        return ids

    def _clean(self, call_ids: list[str]) -> None:
        """Apply a clean; a pure function of the history and ``call_ids``, so replay re-derives it exactly."""
        targets = set(call_ids)
        first: int | None = None
        for index, message in enumerate(self._messages):
            if message["role"] != "user":
                continue
            for block in message["content"]:
                if block["type"] == "tool_result" and block["tool_use_id"] in targets:
                    receipt = self._receipt_of.get(block["tool_use_id"])
                    name = self._media_of.get(block["tool_use_id"])
                    content = block["content"]
                    if isinstance(content, list):
                        kept = [{"type": "text", "text": self._stub(receipt)}] if receipt else []
                        for part in content:
                            if part["type"] in MEDIA_BLOCKS:
                                kept.append(_media_stub(part, name) if name else part)
                            elif receipt is None:
                                kept.append(part)
                        block["content"] = kept
                    elif receipt:
                        block["content"] = self._stub(receipt)
                    if first is None:
                        first = index
        self._cleaned |= targets
        if first is not None:
            self._strip_thinking(first)

    def _cold(self) -> bool:
        """Whether the cache has surely expired: a model call happened, and longer than its TTL ago."""
        if self._last_call_at is None:
            return False
        wait = max(self._settings.clean_after_seconds, _TTL_SECONDS[self._settings.cache_ttl])
        return self._clock() - self._last_call_at > wait

    def clean_attachments_if_cold(self) -> list[int]:
        """Replace old attachments with a one-line stub when the cache is cold, if the agent can ``view`` them.

        An attachment is a file the human sent; its label (the line before it)
        stays and names the file, and the stub says to view it again. Those in
        the last user message are kept, and nothing is touched for an agent
        without ``view``: it could never see the file again. The edit has the
        same effect on thinking blocks as ``clean_if_cold``.
        """
        if not self._can_view or not self._cold() or not self._messages:
            return []
        last = self._messages[-1]
        recent = {id(b) for b in last["content"]} if last["role"] == "user" else set()
        numbers = [n for n, (_, block) in self._attachments.items() if id(block) not in recent]
        if numbers:
            self._clean_attachments(numbers)
        return numbers

    def _clean_attachments(self, numbers: Iterable[int]) -> None:
        """Apply an attachment clean; a pure function of the history and ``numbers``, as ``_clean`` is."""
        targets = {id(self._attachments[n][1]): n for n in numbers if n in self._attachments}
        first: int | None = None
        for index, message in enumerate(self._messages):
            for position, block in enumerate(message["content"]):
                if id(block) in targets:
                    name, _ = self._attachments.pop(targets[id(block)])
                    message["content"][position] = _media_stub(block, name)
                    first = index if first is None else first
        if first is not None:
            self._strip_thinking(first)

    def drop_thinking(self) -> None:
        """Remove every thinking and redacted thinking block, as a resumed agent does after ``replay``.

        A resumed history is rebuilt from the journal, and nothing proves its
        prefix is byte for byte what each thinking block was bound to (an
        older journal, a masked string, a changed engine). The docs allow
        omitting all thinking blocks, so dropping them makes any resumed
        history acceptable to preserved-thinking checks; the cost is the
        model working again through reasoning it can no longer read.
        """
        self._strip_thinking(-1)

    def _strip_thinking(self, after: int) -> None:
        """Remove thinking blocks from assistant messages past index ``after``; drop emptied ones and merge roles."""
        kept: list[Message] = []
        for index, message in enumerate(self._messages):
            if index > after and message["role"] == "assistant":
                message["content"] = [b for b in message["content"] if b["type"] not in THINKING]
                if not message["content"]:
                    continue
            if kept and kept[-1]["role"] == message["role"]:
                kept[-1]["content"].extend(message["content"])
            else:
                kept.append(message)
        self._messages = kept

    def _stub(self, receipt_id: str) -> str:
        text = self._receipts.get(receipt_id)
        lines = _lines(text)
        first_line = lines[0].rstrip("\r")[:_STUB_FIRST_LINE] if lines else ""
        return f"[output cleared; receipt {receipt_id}, {len(lines):,} lines, {len(text):,} chars: {first_line}]"

    def compaction_window(self, model: ModelSpec) -> int:
        """The prompt size ``compact_at`` and ``compact_prepare_at`` are shares of.

        The context window, or the model's ``long_prompt_threshold`` when it has
        one and ``compact_within_price_tier`` is on: past that size every token
        of every later step costs ``long_prompt_factor`` times more, which a
        summary written just before avoids.
        """
        if self._settings.compact_within_price_tier and model.long_prompt_threshold:
            return min(model.context_window, model.long_prompt_threshold)
        return model.context_window

    def should_compact(self, model: ModelSpec) -> bool:
        """True when the next prompt is estimated to reach ``compact_at`` of ``compaction_window``.

        The estimate is the prompt measured on the last call, scaled down by
        how much the history shrank since (a clean), in characters. It is never
        scaled up: tool definitions are in the measured tokens but not in the
        characters, so growth would be overestimated early in a run; the
        headroom left by ``compact_at`` absorbs one step of growth. After a
        compaction nothing is measured until the next call.
        """
        if self._last_call_at is None or self._chars_at_call == 0:
            return False
        ratio = min(1.0, self._chars() / self._chars_at_call)
        return self._last_prompt_tokens * ratio >= self._settings.compact_at * self.compaction_window(model)

    def compaction_request(self) -> str:
        """The compaction template, to be sent to the model as a user message."""
        template = self._settings.compaction_template
        path = Path(template).expanduser() if template else _COMPACTION_TEMPLATE
        return path.read_text(encoding="utf-8")

    def restore_measure(self, prompt_tokens: int) -> None:
        """Tell a rebuilt context how large its prompt was on the last call of the run it continues."""
        self._last_prompt_tokens = prompt_tokens

    def boundary_due(self) -> bool:
        """True when a finished task leaves a prompt of at least ``compact_boundary_tokens`` (0 turns it off).

        Between two tasks nothing is in flight, so a summary loses the least
        there, and the new task then starts on a short prompt instead of
        paying for the whole finished one on every step.
        """
        limit = self._settings.compact_boundary_tokens
        return limit > 0 and self._last_prompt_tokens >= limit

    def compaction_keep(self) -> int:
        """How many messages at the end of the history a compaction keeps as they are.

        The newest ``compact_keep_turns`` assistant turns with their results,
        fewer when they exceed ``compact_keep_chars``. 0 when nothing would be
        left to summarize before them.
        """
        s = self._settings
        starts = [i for i, m in enumerate(self._messages) if m["role"] == "assistant"]
        for start in starts[-s.compact_keep_turns :] if s.compact_keep_turns > 0 else []:
            chars = sum(_block_chars(b) for m in self._messages[start:] for b in m["content"])
            if start > 1 and chars <= s.compact_keep_chars and self._self_contained(start):
                return len(self._messages) - start
        return 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def fingerprint(self, count: int) -> str:
        """A digest of the first ``count`` messages: equal digests mean that prefix is byte for byte the same."""
        data = json.dumps(self._messages[:count], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def tail_from(self, start: int) -> int | None:
        """How many messages a compaction keeps when a summary covers the first ``start``: ``None`` when the
        message at ``start`` is not an assistant turn or the tail would split a server tool call."""
        if start > len(self._messages):
            return None
        if start < len(self._messages) and self._messages[start]["role"] != "assistant":
            return None
        return len(self._messages) - start if self._self_contained(start) else None

    def _self_contained(self, start: int) -> bool:
        """Whether every server tool result from ``start`` on has its ``server_tool_use`` there too.

        In a turn that mixed server and client tools, the call and its result
        arrive in two replies, so a tail cut between them would send a result
        without its call.
        """
        called: set[str] = set()
        for message in self._messages[start:]:
            for block in message["content"]:
                if block.get("type") == "server_tool_use":
                    called.add(block.get("id"))
                elif block.get("type") != "tool_result" and "tool_use_id" in block and block["tool_use_id"] not in called:
                    return False
        return True

    def apply_compaction(self, summary: str, keep: int = 0) -> None:
        """Replace the history with ``summary``, followed by its last ``keep`` messages as they were.

        The pinned texts go first, word for word (``_pinned_block``). The kept
        messages lose their thinking blocks: their prefix changed, and
        thinking bound to another prefix is rejected (see ``clean_if_cold``).
        A pure function of the history, ``summary`` and ``keep``, so a replay
        gives the same bytes.
        """
        tail = self._messages[-keep:] if keep > 0 else []
        follows = " The most recent steps follow it unchanged." if tail else ""
        text = self._pinned_block(tail) + (
            "[Summary of earlier work. The conversation before this point was replaced by this summary "
            "to save context. Every receipt it mentions (r1, r2, ...) is still stored: search it or read "
            f"a line range with the receipt tool.{follows}]\n\n" + summary
        )
        self._messages = [{"role": "user", "content": [{"type": "text", "text": text}]}, *tail]
        self._strip_thinking(0)
        alive = {
            b["tool_use_id"] for m in self._messages for b in m["content"] if b["type"] == "tool_result"
        }
        self._receipt_of = {c: r for c, r in self._receipt_of.items() if c in alive}
        self._media_of = {c: m for c, m in self._media_of.items() if c in alive}
        self._cleaned &= alive
        present = {id(b) for m in self._messages for b in m["content"]}
        self._attachments = {n: a for n, a in self._attachments.items() if id(a[1]) in present}
        self._chars_at_call = 0

    def _pinned_block(self, tail: list[Message]) -> str:
        """The pinned texts a compaction carries over, as the opening of the message that holds the summary.

        Texts still present in ``tail`` are left out, since they follow
        unchanged. The whole block stays within ``compact_pin_chars`` (0 pins
        nothing): the first text is the task and always stays, cut in the
        middle when it alone is longer; then the newest ones that fit. What is
        left out is said, so the agent knows the summary is all it has of it.
        """
        limit = self._settings.compact_pin_chars
        kept_as_is = {b["text"] for m in tail for b in m["content"] if b.get("type") == "text"}
        texts = [(n, t) for n, t in enumerate(self._pinned, 1) if t not in kept_as_is]
        if limit <= 0 or not texts:
            return ""
        (number, first), later = texts[0], texts[1:]
        if len(first) > limit:
            half = limit // 2
            first = f"{first[:half]}\n[... {len(first) - 2 * half:,} chars left out ...]\n{first[-half:]}"
        chosen, room = [], limit - len(first)
        for n, t in reversed(later):
            if len(t) > room:
                break
            chosen.insert(0, (n, t))
            room -= len(t)
        left_out = len(later) - len(chosen)
        note = f" {left_out} of them did not fit and are only in the summary." if left_out else ""
        body = "\n\n".join(f"Message {n}:\n{t}" for n, t in [(number, first), *chosen])
        return (
            "[What you were asked, word for word and in order: the task, then each later message. A "
            f"compaction never rewrites these.{note}]\n\n{body}\n\n"
        )

    def repair_dangling(self, reason: str) -> list[ToolResultEntry]:
        """Answer every tool call of the last assistant message that has no result, as cancelled.

        Returns the entries added, so the loop can journal them as
        ``tool_result`` records and a replay stays identical.
        """
        entries = [
            ToolResultEntry(call_id, f"Cancelled, the action did not complete: {reason}", True)
            for call_id in self._pending()
        ]
        self.add_tool_results(entries)
        return entries

    def replay(self, records: Iterable[Record]) -> None:
        """Rebuild the history from a journal; records of other kinds are ignored.

        The context starts empty. Model call times are not restored, so no
        clean can happen until the next model call.
        """
        self._messages = []
        self._receipt_of.clear()
        self._media_of.clear()
        self._cleaned.clear()
        self._attachments.clear()
        self._attached = 0
        self._held = []
        self._pinned = []
        receipts = self._receipts
        for record in records:
            data = record.data
            match record.kind:
                case "user":
                    attachments = [
                        Attachment(a["name"], unpack(a["block"], receipts), a["note"])
                        for a in data.get("attachments", [])
                    ]
                    self.add_user(data["text"], attachments, pinned=data.get("source") in PINNED_SOURCES)
                case "assistant":
                    self.add_assistant(unpack(data["content"], receipts))
                case "tool_result":
                    content = unpack(data["content"], receipts)
                    entry = ToolResultEntry(
                        data["call_id"], content, data["is_error"], data.get("receipt"), data.get("media")
                    )
                    self.add_tool_results([entry])
                case "clean":
                    self._clean(list(data["call_ids"]))
                    self._clean_attachments(data.get("attachments", []))
                case "compaction":
                    self.apply_compaction(data["summary"], data.get("keep", 0))
                case "resume":
                    self._strip_thinking(-1)


def _media_stub(block: Block, name: str) -> Block:
    kind = "PDF" if block.get("type") == "document" else "image"
    return {"type": "text", "text": f"[{kind} {name} removed to save context: view the file again to see it]"}


class ReceiptTool:
    """The ``receipt`` tool: query a stored tool output without loading it whole."""

    name = "receipt"
    description = (
        "Read a stored tool output by its receipt id (r1, r2, ...), as named in truncated or cleared "
        "outputs. With `search`, list the lines matching a regular expression, with `context` lines "
        "around each. With `lines` (\"120-180\" or \"120\"), return that line range. With both, search "
        "inside the range. With neither, return the size and the first lines."
    )
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Receipt id, such as r41."},
            "search": {"type": "string", "description": "Python regular expression, matched per line."},
            "lines": {"type": "string", "description": "1-based inclusive line range, such as 120-180."},
            "context": {"type": "integer", "minimum": 0, "description": "Lines shown around each match (default 0)."},
        },
        "required": ["id"],
    }

    def __init__(self, receipts: ReceiptStore, settings: ContextSettings) -> None:
        self._receipts = receipts
        self._settings = settings

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        return None

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return self.query(tool_input)

    def query(self, tool_input: dict[str, Any]) -> ToolOutput:
        receipt_id = tool_input.get("id")
        if not isinstance(receipt_id, str) or not self._receipts.exists(receipt_id):
            return ToolOutput(
                text=f"Unknown receipt id {receipt_id!r}. Receipt ids look like r41 and appear in "
                "truncated or cleared tool outputs.",
                is_error=True,
            )
        text = self._receipts.get(receipt_id)
        lines = _lines(text)
        total = len(lines)
        first, last = 1, total
        spec = tool_input.get("lines")
        if spec is not None:
            m = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", str(spec))
            if not m:
                return ToolOutput(text=f"Invalid line range {spec!r}: use \"120-180\" or \"120\".", is_error=True)
            first = int(m.group(1))
            last = int(m.group(2)) if m.group(2) else first
            if first < 1 or last < first:
                return ToolOutput(text=f"Invalid line range {spec!r}: lines start at 1, end >= start.", is_error=True)
            if first > total:
                return ToolOutput(text=f"Receipt {receipt_id} has only {total:,} lines.", is_error=True)
            last = min(last, total)
        context = tool_input.get("context", 0)
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            return ToolOutput(text="`context` must be a non-negative integer.", is_error=True)
        pattern = tool_input.get("search")
        meta = {"receipt": receipt_id}
        if pattern is not None:
            try:
                regex = re.compile(str(pattern))
            except re.error as e:
                return ToolOutput(text=f"Invalid regular expression {pattern!r}: {e}.", is_error=True)
            return ToolOutput(text=self._search(receipt_id, lines, regex, first, last, context), meta=meta)
        if spec is not None:
            body = "\n".join(_numbered(lines[first - 1 : last], first))
            return ToolOutput(text=f"Receipt {receipt_id}, lines {first}-{last} of {total:,}:\n{body}", meta=meta)
        shown: list[str] = []
        size = 0
        for line in _numbered(lines, 1):
            size += len(line) + 1
            if shown and size > self._settings.output_head:
                break
            shown.append(line)
        header = f"Receipt {receipt_id}: {total:,} lines, {len(text):,} chars. First lines:"
        footer = "" if len(shown) == total else "\nUse `search` or `lines` to read more."
        return ToolOutput(text=f"{header}\n" + "\n".join(shown) + footer, meta=meta)

    def _search(
        self, receipt_id: str, lines: list[str], regex: re.Pattern[str], first: int, last: int, context: int
    ) -> str:
        """Matching lines with context, grouped; stops before the output would pass ``output_cap``."""
        hits = [i for i in range(first - 1, last) if regex.search(lines[i])]
        scope = f"lines {first}-{last}" if (first, last) != (1, len(lines)) else f"{len(lines):,} lines"
        if not hits:
            return f"No line matches {regex.pattern!r} in receipt {receipt_id} ({scope})."
        groups: list[list[int]] = []
        for i in hits:
            lo, hi = max(first - 1, i - context), min(last - 1, i + context)
            if groups and lo <= groups[-1][1] + 1:
                groups[-1][1] = hi
            else:
                groups.append([lo, hi])
        header = f"Receipt {receipt_id}: {len(hits):,} matching lines for {regex.pattern!r} ({scope})."
        out: list[str] = []
        size = len(header)
        hit_set = set(hits)
        shown = 0
        full = False
        for lo, hi in groups:
            if out:
                out.append("--")
                size += 3
            for index in range(lo, hi + 1):
                line = f"{index + 1}: {lines[index].rstrip("\r")}"
                if out and size + len(line) + 1 > self._settings.output_cap:
                    full = True
                    break
                out.append(line)
                size += len(line) + 1
                shown += index in hit_set
            if full:
                break
        if out and out[-1] == "--":
            out.pop()
        body = "\n".join(out)
        rest = len(hits) - shown
        footer = f"\n[{rest:,} more matching lines not shown: narrow the pattern or add a line range.]" if rest else ""
        return f"{header}\n{body}{footer}"
