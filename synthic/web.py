"""Server tools: ``web_search`` and ``web_fetch``, which the provider runs on its side during a model call.

What the docs state (``anthropic-version: 2026-10-21``), and what follows here:

* A role gets them by listing them in ``role.<name>.tools``; the request then
  carries ``{"type": <version>, "name": "web_search" | "web_fetch", ...}`` in
  ``tools``. No beta header is needed. The versions default to the basic ones,
  ``web_search_20250305`` and ``web_fetch_20250910``: later versions filter
  results with code execution, which brings a container id the engine does not
  carry, so a paused turn of theirs would restart in a new container.
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
* Options: ``max_uses`` per turn, ``allowed_domains`` or ``blocked_domains``
  (never both in one request, or the API answers 400), and for fetch
  ``max_content_tokens`` and ``citations``. A domain entry is a bare domain,
  optionally with a path; subdomains are included; a wildcard is allowed only in
  the path; web fetch ignores entries with a path.
  https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools
* Search costs 10 USD per 1,000 searches on top of tokens (``usage.server_tool_use
  .web_search_requests``); fetch costs tokens only. The price is policy
  (``web.search_price``).

The limit the human must know: the engine never sees a search or a fetch
before it happens. The provider runs it inside the model call, so no hook,
``ask`` rule or human approval can stop one call. Permissions act where they
can, when the request is built:

* A rule whose pattern is ``*`` decides whether the tool is offered at all,
  and a deny of that kind always holds. ``ask`` becomes
  ``human.permission_fallback``, since nobody can be asked mid-call. When no
  such rule matches, the default permission decides; a default deny still
  lets the allowed domains through.
* Every other rule names a domain: ``"web_fetch: example.com" = "deny"`` adds it
  to ``blocked_domains``; an ``allow`` rule adds it to ``allowed_domains`` when
  the tool is otherwise denied. ``*.example.com`` is read as ``example.com``,
  which also covers its subdomains. A pattern with any other wildcard in the
  domain cannot be sent: a deny of that kind turns the tool off, an allow of
  that kind is ignored, and both are reported as warnings when the run starts.
* As the API takes one list only, an allowed list wins and every blocked domain
  is removed from it; an allowed list left empty turns the tool off.

The results come back as assistant content (``server_tool_use`` and
``*_tool_result`` blocks) and are journaled and replayed with the rest of the
reply. ``never_send`` cannot mask them: their encrypted parts must go back
byte for byte (see ``journal.Masker``).
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field

from .policy import decide
from .types import Block, Policy

SERVER_TOOLS = ("web_search", "web_fetch")


@dataclass
class _Plan:
    offered: bool
    allowed: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _domain(pattern: str) -> str | None:
    """The domain entry a permission pattern stands for, or ``None`` when the API cannot express it."""
    entry = pattern.strip().lower()
    if entry.startswith("*."):
        entry = entry[2:]
    for scheme in ("https://", "http://"):
        entry = entry.removeprefix(scheme)
    host = entry.partition("/")[0]
    if not host or any(c in host for c in "*?[]"):
        return None
    return entry


def _covers(blocked: str, domain: str) -> bool:
    """Whether a blocked entry covers an allowed one: the same host or a subdomain of it, and its path."""
    b_host, _, b_path = blocked.partition("/")
    d_host, _, d_path = domain.partition("/")
    same = d_host == b_host or d_host.endswith("." + b_host)
    return same and (not b_path or d_path.startswith(b_path.rstrip("*")))


def _plan(policy: Policy, name: str) -> _Plan:
    web = policy.web
    decision, matched = decide(policy, name, "*")
    if decision == "ask":
        decision = policy.human.permission_fallback
    if decision == "deny" and matched is not None:
        return _Plan(offered=False)
    plan = _Plan(offered=True, allowed=list(web.allowed_domains), blocked=list(web.blocked_domains))
    for rule in policy.permissions:
        if rule.tool != name or rule.pattern.strip() == "*":
            continue
        verdict = policy.human.permission_fallback if rule.decision == "ask" else rule.decision
        domain = _domain(rule.pattern)
        if domain is None:
            if verdict == "deny":
                plan.offered = False
                plan.warnings.append(
                    f"{name} is not offered: the rule '{rule.tool}: {rule.pattern}' cannot be sent as a domain "
                    "filter, and a server tool cannot be checked call by call"
                )
            else:
                plan.warnings.append(f"the rule '{rule.tool}: {rule.pattern}' is ignored: not a domain the API takes")
            continue
        if verdict == "deny":
            plan.blocked.append(domain)
        elif decision != "allow":
            plan.allowed.append(domain)
    if decision == "deny" and not plan.allowed:
        plan.offered = False
    if plan.allowed:
        plan.allowed = [d for d in dict.fromkeys(plan.allowed) if not any(_covers(b, d) for b in plan.blocked)]
        plan.blocked = []
        plan.offered = plan.offered and bool(plan.allowed)
    plan.blocked = list(dict.fromkeys(plan.blocked))
    return plan


def definitions(policy: Policy, tools: Collection[str]) -> list[Block]:
    """The server tool definitions for a role whose tools are ``tools``, as the request sends them."""
    web = policy.web
    found: list[Block] = []
    for name in SERVER_TOOLS:
        if name not in tools:
            continue
        plan = _plan(policy, name)
        if not plan.offered:
            continue
        search = name == "web_search"
        definition: Block = {"type": web.search_tool if search else web.fetch_tool, "name": name}
        uses = web.search_max_uses if search else web.fetch_max_uses
        if uses is not None:
            definition["max_uses"] = uses
        if plan.allowed:
            definition["allowed_domains"] = plan.allowed
        elif plan.blocked:
            definition["blocked_domains"] = plan.blocked
        if not search:
            if web.fetch_max_content_tokens is not None:
                definition["max_content_tokens"] = web.fetch_max_content_tokens
            if web.fetch_citations:
                definition["citations"] = {"enabled": True}
        found.append(definition)
    return found


def warnings(policy: Policy) -> list[str]:
    """What the human should hear when a run starts: rules that could not be applied to a granted server tool."""
    granted = {tool for role in policy.roles.values() for tool in role.tools}
    return [w for name in SERVER_TOOLS if name in granted for w in _plan(policy, name).warnings]
