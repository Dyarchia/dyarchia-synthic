"""Job 1: talk to the model vendor (Anthropic Messages API over raw HTTP).

Every wire detail here (headers, body fields, SSE events, cache marks, effort,
stop reasons, error types) follows the fact sheet extracted from the official
Claude API docs. Where the docs were silent, the chosen behaviour is stated in
the docstring of the function that implements it.
"""

from __future__ import annotations

import asyncio
import codecs
import copy
import json
import os
import random
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from synthic.types import Block, Message, ModelReply, ModelRequest, ModelSpec, ProviderError, Usage

API_VERSION = "2026-10-21"

CACHEABLE_BLOCKS = frozenset({"text", "image", "document", "tool_use", "tool_result"})
"""Message block types the docs allow to carry ``cache_control``. Thinking, redacted thinking,
server tool blocks and anything unknown are left alone, and so are empty text blocks."""

ERROR_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}
"""Error type to HTTP status, so an ``error`` event inside a stream is judged like the HTTP error."""

BACKOFF_BASE = 1.0
BACKOFF_CAP = 60.0
BACKOFF_MAX_DOUBLINGS = 10


def cost(spec: ModelSpec, usage: Usage, ttl: str, search_price: float = 0.0) -> float:
    """Dollar cost of ``usage`` on ``spec``.

    Cache writes are priced at the 1 hour rate when ``ttl`` is ``"1h"``. A spec
    that leaves ``price_cache_write_1h`` at its ``0.0`` default would make those
    writes free, so in that case the documented multiplier (2x base input) is
    used instead. Above ``long_prompt_threshold`` prompt tokens (strictly
    greater) every price is multiplied by ``long_prompt_factor``.

    Web searches add ``search_price`` dollars per 1,000, outside the token
    factor; fetches cost tokens only. A failed search is not billed and is not
    counted in ``web_search_requests`` either.
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool (Usage and pricing)
    """
    if ttl == "1h":
        write_price = spec.price_cache_write_1h or 2 * spec.price_input
    else:
        write_price = spec.price_cache_write
    factor = 1.0
    if spec.long_prompt_threshold is not None and usage.context_tokens > spec.long_prompt_threshold:
        factor = spec.long_prompt_factor
    total = (
        usage.input_tokens * spec.price_input
        + usage.output_tokens * spec.price_output
        + usage.cache_write_tokens * write_price
        + usage.cache_read_tokens * spec.price_cache_read
    )
    return total * factor / 1_000_000 + usage.web_search_requests * search_price / 1000


class _Transient(Exception):
    """A failure worth another attempt. Never leaves this module: it becomes a ``ProviderError``."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _classify(status: int, payload: Any, retry_after: float | None, request_id: str | None) -> Exception:
    """Turn an API error into ``_Transient`` (retry) or a final ``ProviderError``.

    Retried: 429 except the tier spend cap, and every 5xx (500, 504 and 529 are
    documented; other 5xx are treated the same way because the official SDK
    policy retries "5xx"). Everything else (400, 401, 402, 403, 404, 409, 413,
    the spend-cap 429) fails at once with ``retryable=False``. The body may not
    be JSON (Cloudflare answers 413), so it is parsed defensively.
    """
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    kind = error.get("type") or "unknown_error"
    text = error.get("message") or (payload if isinstance(payload, str) else "") or "no message"
    request_id = request_id or (payload.get("request_id") if isinstance(payload, dict) else None)
    message = f"{status} {kind}: {text}" + (f" (request {request_id})" if request_id else "")
    if status == 429 and details.get("error_code") == "enforced_spend_limit_reached":
        return ProviderError(message, status, retryable=False)
    if status == 429 or status >= 500:
        return _Transient(message, status, retry_after)
    return ProviderError(message, status, retryable=False)


def _retry_after(headers: httpx.Headers) -> float | None:
    """Seconds from a ``retry-after`` header, given either as a number or as an HTTP date."""
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        return max(0.0, (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds())
    except (TypeError, ValueError):
        return None


def _http_error(response: httpx.Response) -> Exception:
    try:
        payload: Any = response.json()
    except ValueError:
        payload = response.text[:500]
    return _classify(response.status_code, payload, _retry_after(response.headers), response.headers.get("request-id"))


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge ``extra`` into ``base``: dicts merge key by key, anything else replaces.

    Merging dicts lets ``extra={"output_config": {"format": ...}}`` coexist with
    the effort the engine sets in ``output_config``; a flat replace would drop it.
    """
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)


def _mark(block: Block, ttl: str) -> None:
    block["cache_control"] = {"type": "ephemeral", "ttl": ttl}


def _cacheable(block: Any) -> bool:
    if not isinstance(block, dict) or block.get("type") not in CACHEABLE_BLOCKS:
        return False
    return not (block["type"] == "text" and not block.get("text"))


def _place_cache_marks(body: dict[str, Any], ttl: str) -> None:
    """Put up to three cache breakpoints on ``body`` (already a deep copy of the caller's data).

    The docs allow four per request; using three leaves one slot free for a
    policy that adds top-level automatic caching through ``extra``.

    1. Last tool definition that can carry one (``defer_loading`` tools cannot):
       caches the tool list, which never changes within a conversation.
    2. Last non-empty system text block: caches tools plus system.
    3. Last cacheable block of the conversation, walking back from the end
       over thinking, redacted thinking, server tool blocks and empty text, and
       across earlier messages if needed. Only ``user`` and ``assistant``
       messages are walked, since the docs say nothing about marks on
       mid-conversation ``system`` messages. Every call writes the cache up to
       here and the next call reads it back through the 20-block lookback, so
       the cached prefix grows with the conversation.

    All marks share one TTL, so the rule that longer TTLs must come first holds.
    When nothing qualifies at a position, that position gets no mark.
    """
    for tool in reversed(body.get("tools", [])):
        if isinstance(tool, dict) and not tool.get("defer_loading"):
            _mark(tool, ttl)
            break
    for block in reversed(body.get("system", [])):
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            _mark(block, ttl)
            break
    for message in reversed(body["messages"]):
        if message.get("role") not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            if not content:
                continue
            message["content"] = content = [{"type": "text", "text": content}]
        for block in reversed(content or []):
            if _cacheable(block):
                _mark(block, ttl)
                return


class _SSEDecoder:
    """Incremental Server-Sent Events parser.

    Accepts LF, CRLF and lone CR line endings, including a CRLF split across two
    chunks, joins multi-line ``data`` with newlines, and skips comment lines. An
    event still pending when the stream ends is dispatched, which tolerates a
    server that omits the final blank line.
    """

    _EOL = re.compile(r"\r\n|\r|\n")

    def __init__(self) -> None:
        self._buffer = ""
        self._data: list[str] = []

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        events: list[str] = []
        position = 0
        while match := self._EOL.search(self._buffer, position):
            if match.group() == "\r" and match.end() == len(self._buffer):
                break
            self._line(self._buffer[position : match.start()], events)
            position = match.end()
        self._buffer = self._buffer[position:]
        return events

    def close(self) -> list[str]:
        events: list[str] = []
        if self._buffer:
            self._line(self._buffer.rstrip("\r"), events)
            self._buffer = ""
        self._line("", events)
        return events

    def _line(self, line: str, events: list[str]) -> None:
        if not line:
            if self._data:
                events.append("\n".join(self._data))
                self._data = []
            return
        if line.startswith(":"):
            return
        name, _, value = line.partition(":")
        if name == "data":
            self._data.append(value.removeprefix(" "))


class _Assembler:
    """Rebuilds the assistant turn from stream events, exactly as it must go back into history.

    Blocks are keyed by their ``index`` and returned in that order. Blocks that
    arrive whole (redacted thinking, server tool results, compaction, fallback)
    are kept verbatim; thinking keeps its text and signature even when both are
    empty. Unknown event and delta types are ignored, as the docs require.
    """

    def __init__(self, on_text: Callable[[str], None] | None) -> None:
        self._on_text = on_text
        self._blocks: dict[int, Block] = {}
        self._json: dict[int, list[str]] = {}
        self._bad_json: list[int] = []
        self._usage: dict[str, int] = {}
        self.model: str | None = None
        self.stop_reason: str | None = None
        self.finished = False

    def _handle(self, event: dict[str, Any]) -> str | None:
        """Apply one event; returns the visible text it carried, if any."""
        match event.get("type"):
            case "message_start":
                message = event.get("message") or {}
                self.model = message.get("model")
                self._take_usage(message.get("usage"))
            case "content_block_start":
                self._blocks[event["index"]] = copy.deepcopy(event["content_block"])
            case "content_block_delta":
                return self._delta(self._blocks[event["index"]], event["index"], event.get("delta") or {})
            case "content_block_stop":
                self._close_block(event["index"])
            case "message_delta":
                self.stop_reason = (event.get("delta") or {}).get("stop_reason") or self.stop_reason
                self._take_usage(event.get("usage"))
            case "message_stop":
                self.finished = True
            case "error":
                error = event.get("error") or {}
                raise _classify(ERROR_STATUS.get(error.get("type"), 500), event, None, event.get("request_id"))
        return None

    def _delta(self, block: Block, index: int, delta: dict[str, Any]) -> str | None:
        match delta.get("type"):
            case "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
                return delta["text"]
            case "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            case "signature_delta":
                block["signature"] = block.get("signature", "") + delta["signature"]
            case "input_json_delta":
                self._json.setdefault(index, []).append(delta.get("partial_json", ""))
            case "citations_delta":
                block.setdefault("citations", []).append(delta["citation"])
        return None

    def _close_block(self, index: int) -> None:
        raw = "".join(self._json.pop(index, []))
        if not raw.strip():
            return
        try:
            self._blocks[index]["input"] = json.loads(raw)
        except json.JSONDecodeError:
            self._bad_json.append(index)

    def _take_usage(self, usage: Any) -> None:
        """Message_delta counts are cumulative, so each value overwrites the earlier one.

        Server tool counts come nested, as ``server_tool_use.web_search_requests``
        and ``server_tool_use.web_fetch_requests``, and are kept the same way.
        """
        if not isinstance(usage, dict):
            return
        self._usage.update({key: value for key, value in usage.items() if isinstance(value, int)})
        server = usage.get("server_tool_use")
        if isinstance(server, dict):
            self._usage.update({key: value for key, value in server.items() if isinstance(value, int)})

    def usage(self) -> Usage:
        return Usage(
            input_tokens=self._usage.get("input_tokens", 0),
            output_tokens=self._usage.get("output_tokens", 0),
            cache_write_tokens=self._usage.get("cache_creation_input_tokens", 0),
            cache_read_tokens=self._usage.get("cache_read_input_tokens", 0),
            web_search_requests=self._usage.get("web_search_requests", 0),
            web_fetch_requests=self._usage.get("web_fetch_requests", 0),
        )

    def finish(self) -> tuple[list[Block], str]:
        """The finished blocks and stop reason, or ``_Transient`` if the stream did not end cleanly.

        A tool input that is not valid JSON is a corrupt stream (retried) unless
        the reply stopped at ``max_tokens``, where the docs say the last
        ``tool_use`` may be truncated: that block keeps the ``{}`` input it
        started with, and the loop must not run it (see ``AnthropicProvider``).
        """
        if not self.finished or self.stop_reason is None:
            raise _Transient("stream ended before message_stop")
        if self._bad_json and self.stop_reason != "max_tokens":
            raise _Transient(f"invalid tool input JSON in blocks {self._bad_json}")
        return [self._blocks[index] for index in sorted(self._blocks)], self.stop_reason

    def feed(self, data: str) -> None:
        """Handle one SSE data payload; anything malformed counts as a broken stream.

        ``on_text`` runs outside the guarded block, so a bug in the caller's
        callback propagates as is instead of being retried as a broken stream.
        """
        try:
            event = json.loads(data)
            if not isinstance(event, dict):
                raise TypeError("event is not an object")
            text = self._handle(event)
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError) as exc:
            raise _Transient(f"malformed stream event {data[:200]!r}: {exc!r}") from exc
        if text and self._on_text:
            self._on_text(text)


class _Response:
    """Sends a request in streaming mode and maps every failure into ``_Transient`` or ``ProviderError``.

    Connection errors, timeouts and streams that break midway are all
    ``httpx.TransportError`` and are retried; a non-200 status is classified.
    The response is always closed on exit, which is also how cancellation
    releases the connection.
    """

    def __init__(self, client: httpx.AsyncClient, request: httpx.Request) -> None:
        self._client = client
        self._request = request
        self._response: httpx.Response | None = None

    async def __aenter__(self) -> httpx.Response:
        try:
            self._response = await self._client.send(self._request, stream=True)
            if self._response.status_code != 200:
                await self._response.aread()
                raise _http_error(self._response)
        except httpx.TransportError as exc:
            await self._close()
            raise _Transient(f"connection failed: {exc!r}") from exc
        except BaseException:
            await self._close()
            raise
        return self._response

    async def __aexit__(self, kind: type[BaseException] | None, exc: BaseException | None, tb: Any) -> None:
        await self._close()
        if isinstance(exc, httpx.TransportError):
            raise _Transient(f"stream broke: {exc!r}") from exc

    async def _close(self) -> None:
        if self._response is not None:
            await self._response.aclose()


class AnthropicProvider:
    """``Provider`` for the Anthropic Messages API.

    The reply's ``stop_reason`` is returned unchanged. What the loop must do with
    each value, per the docs:

    * ``end_turn``: natural finish, the reply is final.
    * ``tool_use``: run every ``tool_use`` block, then send ONE user message whose
      content starts with one ``tool_result`` per call (failed or skipped calls
      included, with ``is_error``), text only after the results, same tools.
    * ``max_tokens``: output hit ``max_tokens``. A trailing ``tool_use`` may be
      truncated; its input is then ``{}``. Never run tools from such a reply and
      never append one that contains ``tool_use`` as is (a ``tool_use`` without
      its ``tool_result`` makes the next request fail with 400): retry with a
      higher limit. A text-only reply can be appended and continued.
    * ``stop_sequence``: a requested stop sequence fired; the reply is final.
    * ``pause_turn``: a server tool loop hit its iteration limit. Append the reply
      as is and resend with the same tools and NO new user message, plus the
      container id when one was used (``ModelReply`` does not carry it yet).
      A ``tool_use`` reply can also hold a ``server_tool_use`` with no result
      yet (server and client tools called together): the next user message
      must then hold only the ``tool_result`` blocks, and the API runs the
      server tool on that request.
      https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
    * ``refusal``: HTTP 200, content may be partial or empty. Discard it and
      retry on the fallback model. It is a normal reply, not a ``ProviderError``.
    * ``model_context_window_exceeded``: the reply filled the context window;
      treat it as truncated (compact, then continue).
    * ``compaction``: only when server compaction is enabled through ``extra``;
      append the reply (the compaction block) and continue.

    Thinking and redacted thinking blocks must go back in history unchanged.

    A stream that breaks is retried from scratch, so ``on_text`` can receive the
    same text again after a retry; a consumer that prints it should treat a
    retry as a restart. ``aclose()`` closes the HTTP client only if this object
    created it.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com",
        cache_ttl: str = "5m",
        retry_seconds: float = 3600.0,
        timeout_seconds: float = 600.0,
        client: httpx.AsyncClient | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        search_price: float = 0.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.search_price = search_price
        self._base_url = base_url.rstrip("/")
        self.cache_ttl = cache_ttl
        self.retry_seconds = retry_seconds
        self._own_client = client is None
        self._client = client or httpx.AsyncClient()
        self._sleep = sleep
        self._clock = clock
        self._timeout = httpx.Timeout(timeout_seconds, connect=10.0)
        """Long thinking can leave the stream silent for minutes; pings reset the read timer."""

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()

    async def complete(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> ModelReply:
        """Stream one reply, retrying transient failures, racing the whole call against ``cancel``."""
        headers = self._headers(request)
        body = self._body(request)
        work = self._retrying(lambda: self._stream_once(request, body, headers, on_text))
        if cancel is None:
            return await work
        if cancel.is_set():
            work.close()
            raise asyncio.CancelledError
        task = asyncio.ensure_future(work)
        waiter = asyncio.ensure_future(cancel.wait())
        try:
            done, _ = await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for pending in (task, waiter):
                if not pending.done():
                    pending.cancel()
            await asyncio.gather(task, waiter, return_exceptions=True)
        if task in done:
            return task.result()
        raise asyncio.CancelledError

    def _headers(self, request: ModelRequest) -> dict[str, str]:
        if not self._api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not set and no api_key was given", retryable=False)
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        beta = request.extra.get("anthropic_beta")
        if beta:
            headers["anthropic-beta"] = beta if isinstance(beta, str) else ",".join(beta)
        return headers

    def _body(self, request: ModelRequest) -> dict[str, Any]:
        """The request body on deep copies of the caller's data, with cache marks placed.

        Marks are placed before ``extra`` is merged, so a policy can still
        override anything through ``extra``.
        """
        spec = request.model
        max_tokens = request.max_output_tokens if request.max_output_tokens is not None else spec.max_output_tokens
        body: dict[str, Any] = {"model": spec.id, "max_tokens": max_tokens}
        if request.system:
            body["system"] = copy.deepcopy(request.system)
        messages: list[Message] = copy.deepcopy(request.messages)
        body["messages"] = messages
        if request.tools:
            body["tools"] = copy.deepcopy(request.tools)
        if request.effort is not None:
            body["output_config"] = {"effort": request.effort}
        body["stream"] = True
        _place_cache_marks(body, self.cache_ttl)
        _merge(body, {key: value for key, value in request.extra.items() if key != "anthropic_beta"})
        return body

    def _send(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> _Response:
        request = self._client.build_request(
            "POST", self._base_url + path, json=body, headers=headers, timeout=self._timeout
        )
        return _Response(self._client, request)

    async def _stream_once(
        self,
        request: ModelRequest,
        body: dict[str, Any],
        headers: dict[str, str],
        on_text: Callable[[str], None] | None,
    ) -> ModelReply:
        assembler = _Assembler(on_text)
        decoder = _SSEDecoder()
        utf8 = codecs.getincrementaldecoder("utf-8")()
        async with self._send("/v1/messages", body, headers) as response:
            async for chunk in response.aiter_bytes():
                for data in decoder.feed(utf8.decode(chunk)):
                    assembler.feed(data)
            for data in decoder.feed(utf8.decode(b"", final=True)) + decoder.close():
                assembler.feed(data)
        content, stop_reason = assembler.finish()
        usage = assembler.usage()
        usage.cost_usd = cost(request.model, usage, self.cache_ttl, self.search_price)
        return ModelReply(content=content, stop_reason=stop_reason, usage=usage, model_id=assembler.model or request.model.id)

    async def _retrying[T](self, attempt: Callable[[], Awaitable[T]]) -> T:
        """Run ``attempt`` until it succeeds, fails for good, or ``retry_seconds`` have passed.

        An outage can last hours and the agent should ride it out, so the limit
        is a time window and not a count. The docs give no backoff formula: this
        uses exponential backoff from 1 s, capped at 60 s, with jitter between
        half and all of the delay, and never sleeps past the end of the window.
        A ``retry-after`` header is slept exactly, unless it is longer than what
        is left of the window, which ends the retries. A cancel stops the sleep
        at once because ``complete`` races this whole coroutine against it.
        """
        deadline = self._clock() + self.retry_seconds
        retries = 0
        while True:
            try:
                return await attempt()
            except _Transient as exc:
                remaining = deadline - self._clock()
                if exc.retry_after is not None:
                    delay = exc.retry_after
                else:
                    doublings = min(retries, BACKOFF_MAX_DOUBLINGS)
                    delay = random.uniform(0.5, 1.0) * min(BACKOFF_CAP, BACKOFF_BASE * 2**doublings)
                    delay = min(delay, max(remaining, 0.0))
                if remaining <= 0 or delay > remaining:
                    raise ProviderError(
                        f"gave up after {retries + 1} attempts over {self.retry_seconds:g}s: {exc}",
                        exc.status,
                        retryable=True,
                    ) from exc
                await self._sleep(delay)
                retries += 1
