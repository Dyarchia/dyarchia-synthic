"""Shared contracts of the Synthic engine.

Every module of the engine depends on this file and on nothing else from its
siblings, so each of the ten jobs (model, loop, context, tools, policy, hooks,
journal, agents, environment, plugins) stays replaceable.

Conventions that hold across the whole engine:

* Conversation history is kept in the provider's native shape: a list of
  ``{"role": "user" | "assistant", "content": [block, ...]}`` dicts whose blocks
  are Anthropic content blocks (``text``, ``thinking``, ``tool_use``,
  ``tool_result``, ``image``, ``document``). Nothing is translated back and forth, which keeps
  the cached prefix byte-stable.
* The engine has no opinions. Every number and every choice lives in the policy
  (see ``defaults.toml``); code never hardcodes a model, a limit or a permission.
* Once the policy is set the agent runs on its own. Nothing in the engine waits
  for a human unless the policy says so, and a human can interrupt at any time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

Block = dict[str, Any]
Message = dict[str, Any]
Decision = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class ModelSpec:
    """One entry of the ``[models.<alias>]`` policy table. Prices are in US dollars per million tokens.

    Some models charge more once the prompt passes a size: above
    ``long_prompt_threshold`` prompt tokens every price is multiplied by
    ``long_prompt_factor``.
    """

    alias: str
    id: str
    context_window: int
    max_output_tokens: int
    price_input: float
    price_output: float
    price_cache_write: float
    price_cache_read: float
    price_cache_write_1h: float = 0.0
    long_prompt_threshold: int | None = None
    long_prompt_factor: float = 1.0
    fallback: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    """Vendor fields the engine does not know, merged into ``ModelRequest.extra`` (the role's ``extra`` wins)."""


@dataclass
class Usage:
    """Token counts and cost of one model call, or the running sum of many.

    ``web_search_requests`` and ``web_fetch_requests`` count the server tool
    calls the provider ran during the call; searches are billed per use.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    web_search_requests: int = 0
    web_fetch_requests: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cost_usd += other.cost_usd
        self.web_search_requests += other.web_search_requests
        self.web_fetch_requests += other.web_fetch_requests

    @property
    def context_tokens(self) -> int:
        """Size of the prompt the model just read: fresh, cache-written and cache-read input together."""
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens


@dataclass
class ModelRequest:
    """Everything the provider needs for one call.

    ``system`` is a list of text blocks, most stable first. ``tools`` is the list
    of Anthropic tool definitions. The provider adds cache marks itself; callers
    never put ``cache_control`` in these structures.
    """

    model: ModelSpec
    system: list[Block]
    messages: list[Message]
    tools: list[Block]
    effort: str | None = None
    max_output_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelReply:
    """The assistant turn exactly as it must be appended to history, plus accounting."""

    content: list[Block]
    stop_reason: str
    usage: Usage
    model_id: str


class Provider(Protocol):
    """Job 1: talks to the model vendor."""

    async def complete(
        self,
        request: ModelRequest,
        on_text: Callable[[str], None] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> ModelReply:
        """Send one request and return the full reply.

        ``on_text`` receives visible text as it streams. If ``cancel`` is set
        while streaming, the call stops and raises ``asyncio.CancelledError``.
        Transient failures are retried inside; a failure that survives the
        retries raises ``ProviderError``.
        """
        ...


class ProviderError(Exception):
    """A model call failed for good. ``retryable`` tells the loop whether trying a fallback model makes sense."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True)
class ToolCall:
    """One ``tool_use`` block requested by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ImageData:
    """An image returned by a tool. ``data`` is base64 text."""

    media_type: str
    data: str


@dataclass
class DocumentData:
    """A PDF returned by a tool. ``data`` is base64 text."""

    media_type: str
    data: str


@dataclass
class ToolOutput:
    """What a tool returns to the loop, before the context module caps it.

    ``meta["media"]`` names the file that ``images`` or ``documents`` came from
    when the tool can show it again (``view``): only such media is ever removed
    from an old result to save context.
    """

    text: str = ""
    images: list[ImageData] = field(default_factory=list)
    is_error: bool = False
    meta: dict[str, Any] = field(default_factory=dict)
    documents: list[DocumentData] = field(default_factory=list)


@dataclass
class Attachment:
    """A file sent with a human message: its absolute path, its ``image`` or ``document`` block, a short note."""

    name: str
    block: Block
    note: str


@dataclass
class ToolContext:
    """What the loop hands to a tool on every call.

    ``cancel`` is set when the human interrupts or the agent is cancelled; a
    tool that started processes must kill all of them when it fires.
    ``timeout_seconds`` is the foreground limit from the policy: a tool that can
    keep working past it must move the work to the background and return.
    """

    agent_id: str
    cwd: Path
    env: dict[str, str]
    timeout_seconds: float
    cancel: asyncio.Event


class Tool(Protocol):
    """Job 4: one action the model can request."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def resource(self, tool_input: dict[str, Any], ctx: ToolContext) -> str | None:
        """Return a key for what this call writes to, or ``None`` if it can run next to anything.

        Calls of one turn that share a key run one after another in the order
        the model gave; all others run at the same time.
        """
        ...

    async def run(self, tool_input: dict[str, Any], ctx: ToolContext) -> ToolOutput: ...


@dataclass(frozen=True)
class Escalation:
    """``on_failure`` of a role: after ``attempts`` failed tries the task is handed to ``escalate_to``."""

    escalate_to: str
    attempts: int = 1


@dataclass(frozen=True)
class Role:
    """A named kind of work. Everything about it is imposed by the policy, never chosen by the agent."""

    name: str
    model: str
    effort: str | None = None
    tools: tuple[str, ...] = ()
    can_spawn: tuple[str, ...] = ()
    on_failure: Escalation | None = None
    max_steps: int | None = None
    max_output_tokens: int | None = None
    instructions: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Budget:
    """Caps for the whole agent tree. ``None`` means no cap. ``warn_at`` is a fraction of any cap."""

    dollars: float | None = None
    wall_clock_hours: float | None = None
    steps: int | None = None
    per_agent_dollars: float | None = None
    warn_at: float = 0.8


@dataclass(frozen=True)
class AgentLimits:
    """``repeat_limit``: the same tool call with the same result this many times in a row draws a warning,
    twice as many ends the agent as failed; 0 turns the guard off."""

    concurrent: int = 4
    depth: int = 2
    repeat_limit: int = 3


@dataclass(frozen=True)
class HumanPolicy:
    """What happens where a human would be needed.

    The engine is autonomous by default: with ``wait_seconds = 0`` nothing ever
    waits. A permission that resolves to ``ask`` becomes ``permission_fallback``
    and a question from the agent is answered by ``question_fallback``
    (``"assume"``: go on with the stated assumption, ``"stop"``: end the agent).
    A human answer that arrives later is delivered as a message.
    """

    wait_seconds: float = 0.0
    permission_fallback: Literal["allow", "deny"] = "deny"
    question_fallback: Literal["assume", "stop"] = "assume"


@dataclass(frozen=True)
class ContextSettings:
    """Knobs of job 3. Sizes are in characters unless the name says tokens."""

    cache_ttl: Literal["5m", "1h"] = "5m"
    output_cap: int = 12000
    output_head: int = 4000
    output_tail: int = 4000
    clean_after_seconds: float = 300.0
    clean_keep_recent: int = 6
    clean_min_chars: int = 600
    compact_at: float = 0.85
    compact_keep_turns: int = 2
    compact_keep_chars: int = 24000
    compact_boundary_tokens: int = 80000
    compaction_template: str | None = None


@dataclass(frozen=True)
class JournalSettings:
    dir: str = "~/.synthic/runs"
    retention_days: int | None = None
    mask: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemorySettings:
    """Notes the agent keeps between runs, one folder per project under ``dir``.

    ``MEMORY.md`` of that folder is read once when an agent starts and enters
    its system prompt, cut at ``max_chars``. The agent keeps the folder up to
    date with its ordinary ``edit`` tool.
    """

    enabled: bool = True
    dir: str = "~/.synthic/memory"
    max_chars: int = 8000


@dataclass(frozen=True)
class EnvironmentSettings:
    """``mode`` is ``"in_place"`` (the folder as it is) or ``"copy"`` (a separate work copy per agent)."""

    mode: Literal["in_place", "copy"] = "in_place"
    shell: tuple[str, ...] = ()
    tool_timeout_seconds: float = 120.0
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderSettings:
    """``retry_seconds``: how long transient failures of one call are retried before it fails.
    ``timeout_seconds``: how long a stream may stay silent before the call counts as broken."""

    retry_seconds: float = 3600.0
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class WebSettings:
    """The server tools ``web_search`` and ``web_fetch``, offered to a role that lists them in its tools.

    ``search_tool`` and ``fetch_tool`` are the tool versions sent as ``type``.
    ``*_max_uses`` caps the calls of one turn (``None``: no cap). The domain
    lists apply to both tools, together with the permission rules (see
    ``web.py``). ``search_price`` is in US dollars per 1,000 searches.
    """

    search_tool: str = "web_search_20250305"
    fetch_tool: str = "web_fetch_20250910"
    search_max_uses: int | None = None
    fetch_max_uses: int | None = None
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    fetch_max_content_tokens: int | None = None
    fetch_citations: bool = False
    search_price: float = 10.0


@dataclass(frozen=True)
class VerifySettings:
    """Checks a delivery must pass, run when the root agent calls ``done`` as delivered. Off while both are empty.

    ``commands`` run in the working folder, in order, and must exit 0 within
    ``timeout_seconds`` each. ``review`` names the role of a reviewer agent
    that judges the work with a fresh context. ``rounds`` is how many times a
    delivery may be rejected; one that still fails after that is accepted and
    marked as not verified.
    """

    commands: tuple[str, ...] = ()
    review: str | None = None
    rounds: int = 2
    timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class HookSpec:
    """A command of the human that runs on an event. ``match`` is a glob on the tool name, or empty."""

    event: str
    command: str
    match: str = ""
    timeout_seconds: float = 30.0


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: tuple[str, ...]
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PluginSettings:
    skills: tuple[str, ...] = ()
    path: tuple[str, ...] = ()
    packages: tuple[str, ...] = ()
    mcp: tuple[McpServerSpec, ...] = ()


@dataclass(frozen=True)
class InstructionSettings:
    """Texts loaded into the system prompt, in order. Files inside the repo load only if ``load_repo_files``."""

    files: tuple[str, ...] = ()
    repo_files: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
    load_repo_files: bool = False


@dataclass(frozen=True)
class PermissionRule:
    """``"sh: git push*" = "deny"`` in the policy file.

    ``tool`` is a tool name. ``pattern`` is a glob matched against the tool's
    target: the command line for ``sh``, the path relative to the working
    folder (forward slashes) for ``edit`` and ``view``, ``server/tool`` for
    ``mcp``, the role for ``spawn``, a domain for ``web_search`` and
    ``web_fetch`` (applied when the request is built, see ``web.py``).
    """

    tool: str
    pattern: str
    decision: Decision
    level: str = "default"


@dataclass(frozen=True)
class Policy:
    """The merged result of all policy levels. Immutable: the agent cannot change it."""

    initial_role: str
    roles: dict[str, Role]
    models: dict[str, ModelSpec]
    budget: Budget = Budget()
    agents: AgentLimits = AgentLimits()
    human: HumanPolicy = HumanPolicy()
    context: ContextSettings = ContextSettings()
    journal: JournalSettings = JournalSettings()
    memory: MemorySettings = MemorySettings()
    environment: EnvironmentSettings = EnvironmentSettings()
    provider: ProviderSettings = ProviderSettings()
    instructions: InstructionSettings = InstructionSettings()
    plugins: PluginSettings = PluginSettings()
    web: WebSettings = WebSettings()
    verify: VerifySettings = VerifySettings()
    permissions: tuple[PermissionRule, ...] = ()
    default_permission: Decision = "allow"
    hooks: tuple[HookSpec, ...] = ()
    never_send: tuple[str, ...] = ()
    visible_to_agent: bool = True


@dataclass
class Record:
    """One line of the journal.

    Kinds and the shape of ``data``:

    * ``start``: ``{run_id, role, task, cwd, parent, system, tools}`` where
      ``system`` is the list of system texts and ``tools`` the tool definitions
      exactly as sent; a resumed agent reuses both instead of rebuilding them
    * ``user``: ``{text, source}`` with source ``human``, ``parent``, ``engine`` or ``hook``, plus
      ``attachments: [{name, note, block}]`` when files came with it
    * every base64 ``source`` inside ``user``, ``assistant`` and ``tool_result`` records is journaled as
      ``{type: "base64", media_type, file}``: the data itself is stored once under ``receipts/media``
      and read back on replay (``context.pack`` and ``context.unpack``)
    * ``model_call``: ``{model_id, stop_reason, usage, cost_usd}``, plus
      ``purpose: "compaction"`` for a call that is not a step of the loop
    * ``fallback``: ``{from, to, reason}`` the next model call goes to the
      fallback model; the call after a successful one goes back to the role's model
    * ``assistant``: ``{content}`` the reply blocks as appended to history,
      text and tool inputs already masked
    * ``tool_intent``: ``{call_id, name, input}`` written before the tool runs
    * ``tool_result``: ``{call_id, name, content, is_error, receipt, seconds}``
      where ``content`` is exactly what entered the context; ``name`` is empty
      for a call answered by a repair (interrupted or dead process); plus
      ``media`` (the file shown) for a result whose media can be shown again
    * ``clean``: ``{call_ids}`` tool results replaced by their receipt stub, plus ``attachments``
      (their numbers) when old attachments were replaced by a stub
    * ``compaction``: ``{summary, keep}`` history replaced by this summary plus its last ``keep`` messages
    * ``spawn``: ``{child_id, role, task}``
    * ``child_done``: ``{child_id, status, summary}``
    * ``question``: ``{question_id, text, assumption}``
    * ``answer``: ``{question_id, text}``
    * ``interrupt``: ``{mode}`` with mode ``stop``, ``add`` or ``redirect``
    * ``hook``: ``{event, action, reason}``
    * ``verify``: ``{round, passed, accepted, report, reviewer}`` one check of a
      delivery (``verify`` policy): ``accepted`` false is a rejection, which
      returns to the agent as the error result of its ``done``; ``reviewer`` is
      the id of the reviewer agent, or ``None``
    * ``done``: ``{status, summary}`` with status ``delivered``, ``failed``,
      ``budget``, ``stopped`` or ``cancelled``
    * ``resume``: ``{}`` a new process took the agent up again (after a crash or
      after a ``done``). A ``done`` is in force only while no ``resume`` follows
      it (``journal.ended``); the time before a ``resume`` since the previous
      record is not counted against ``wall_clock_hours``.

    A ``tool_intent`` without a matching ``tool_result`` means the process died
    in the middle of that action.
    """

    seq: int
    ts: float
    agent_id: str
    kind: str
    data: dict[str, Any]


class Journal(Protocol):
    """Job 7: the ordered, on-disk record of one agent."""

    agent_id: str
    dir: Path

    def append(self, kind: str, data: dict[str, Any]) -> Record: ...

    def records(self) -> Iterator[Record]: ...


EVENTS = (
    "start",
    "before_model",
    "after_model",
    "before_tool",
    "after_tool",
    "spawn",
    "child_done",
    "question",
    "before_compaction",
    "budget_warning",
    "resume",
    "before_done",
    "end",
)
"""Every moment at which the engine emits an event. A hook can hang from any of them."""


@dataclass
class HookOutcome:
    """What the hooks of one event decided, already combined.

    ``action`` is ``"pass"`` or ``"block"``. ``replace`` substitutes the payload
    field the event allows to change (the tool input before a tool, the tool
    output after it). ``add_text`` is appended to the conversation for the model.
    """

    action: Literal["pass", "block"] = "pass"
    reason: str = ""
    replace: dict[str, Any] | None = None
    add_text: str | None = None


Emit = Callable[[str, dict[str, Any]], Awaitable[HookOutcome]]
"""Job 6 as seen by the other modules: ``await emit(event, payload)``."""
