"""Job 5: the policy. Loads, merges and validates what the human imposes, and answers permission questions.

Principle: everything that can be imposed is imposed by the human, through
policy. The agent only decides where the human said nothing, and it can never
change the policy. The engine has no opinions: the factory values in the
packaged ``defaults.toml`` are just the lowest level.

Levels, from most to least authority:

1. ``organization``: the file named by the ``SYNTHIC_ORG_POLICY`` environment
   variable, if set. If the variable is set and the file is missing, loading fails.
2. ``user``: ``~/.synthic/policy.toml``.
3. ``project``: ``<project root>/.synthic/policy.toml``. It lives inside the repo,
   so whoever wrote the repo wrote it.
4. ``task``: what the human says for this one run, a TOML file and/or a nested
   dict of overrides from the CLI (the dict wins over the file).

Below all of them: ``defaults``.

Merge rules (the contract with the human):

* Scalars and tables of scalars: the value from the lowest level that sets it
  wins (task over project over user over organization over defaults), unless a
  higher level fixed it. Any table may carry ``fixed = ["key", ...]`` naming keys
  of that table that lower levels cannot change; at top level,
  ``fixed = ["initial_role", ...]`` works the same for top-level keys, and
  ``fixed = ["*"]`` fixes the whole table (at top level: the whole policy). A
  lower level's attempt to change a fixed key is ignored and recorded as a
  warning. Merging is per key: a project that sets ``role.build.effort`` does
  not erase ``role.build.model``. ``fixed`` is a reserved key name in every table.
* Lists (``role.<name>.tools``, ``can_spawn``, ``plugins.*``,
  ``instructions.files`` and every other list value) are replaced as a whole by
  the lowest level that sets them, subject to ``fixed`` like any scalar. So is
  ``role.<name>.result_schema``, a table holding a JSON Schema.
* ``models.<alias>.extra`` and ``role.<name>.extra`` are free-form tables of
  vendor fields the engine does not know (beta headers, service tier, context
  management...). They merge per key like any table, ``fixed`` included, and
  the loop sends them in ``ModelRequest.extra``, the role's over the model's.
* Prohibitions always hold. A permission rule with decision ``deny`` from any
  level stays in force whatever other levels say, and ``never_send`` patterns
  accumulate across levels. ``fixed`` never applies to prohibitions or hooks:
  ``permissions``, ``never_send`` and ``hooks`` cannot be named in ``fixed``, and
  a top-level ``fixed = ["*"]`` does not stop a lower level from adding them.
* Permissions and anything that widens power count only from trusted levels.
  ``organization``, ``user`` and ``task`` are always trusted. ``project`` is
  trusted only if the caller says so (``trust_project=True``): the human marks
  a repo as trusted, the engine never decides this. From an untrusted project
  file only prohibitions are read (``deny`` rules and ``never_send``); everything
  else in it, including tightening caps, hooks and its own problems, is ignored
  and summed up in one warning naming the file.
* Numeric caps can only be tightened by a lower level, never loosened. For
  ``budget.dollars``, ``budget.wall_clock_hours``, ``budget.steps``,
  ``budget.per_agent_dollars``, ``agents.concurrent``, ``agents.depth`` and
  ``role.<name>.max_steps`` the effective value is the minimum of the values set
  by the organization, user, project and task levels. The task level is the
  human speaking, but the organization still outranks it, so the same minimum
  applies. An absent value means "no cap" and never loosens a cap set above.
  The ``defaults`` value is not a cap: it is used only when no other level sets
  the key, so any level may raise a factory value such as ``agents.concurrent``.
* ``never_send`` and ``journal.mask`` hold regular expressions (Python ``re`` syntax), not globs: the masker
  compiles them, so each one is checked with ``re.compile`` when the policy loads and a bad one is a problem
  that names the key (in an untrusted project file it is dropped with a warning).
* Hooks accumulate: hooks from all trusted levels run, in level order from
  organization down.

A ``None`` anywhere in a level (only possible in the task overrides dict) means
"not set". Every problem in a trusted level (unknown key, wrong type, unknown
hook event, dangling role or model name, fallback loop) is collected, and all of
them are raised together as one ``PolicyError``.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from importlib.resources import files
from pathlib import Path
from typing import Any

from .schema import problems as schema_problems
from .types import (
    EVENTS,
    AgentLimits,
    Budget,
    ContextSettings,
    Decision,
    DockerSettings,
    EnvironmentSettings,
    Escalation,
    HookSpec,
    HumanPolicy,
    InstructionSettings,
    JournalSettings,
    MemorySettings,
    McpServerSpec,
    ModelSpec,
    PermissionRule,
    PluginSettings,
    Policy,
    ProviderSettings,
    Role,
    VerifySettings,
    WebSettings,
)

ROLE_TOOLS = ("sh", "edit", "view", "todo", "spawn", "skill", "mcp", "web_search", "web_fetch")
"""Every tool a role can grant. The engine tools (``done``, ``ask``, ``receipt``) need no grant; ``sh`` brings
``job`` and ``spawn`` brings ``agent``. ``web_search`` and ``web_fetch`` are server tools: the provider runs them."""

APPLICATION_ORDER = ("defaults", "organization", "user", "project", "task")
"""Order in which levels are applied: each one overrides the previous ones unless they fixed the key."""

AUTHORITY = ("organization", "user", "project", "task", "defaults")
"""Levels from most to least authority, used to rank permission rules and to order hooks."""

_RANK = {name: index for index, name in enumerate(AUTHORITY)}


class PolicyError(Exception):
    """The policy cannot be used. ``problems`` lists every problem found, so the human fixes them in one go."""

    def __init__(self, problems: str | list[str]) -> None:
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("invalid policy:\n" + "\n".join(f"- {p}" for p in self.problems))


@dataclass(frozen=True)
class Level:
    """One parsed policy file (or dict) with its level name, trust flag and origin for messages."""

    name: str
    data: dict[str, Any]
    trusted: bool = True
    origin: str = ""

    @property
    def label(self) -> str:
        return f"{self.name} ({self.origin})" if self.origin else self.name


@dataclass(frozen=True)
class _Each:
    """A table whose keys are free names (model aliases, role names) and whose values share one schema."""

    item: Any


_DECISIONS = ("allow", "ask", "deny")

_MODEL = {
    "id": "str",
    "context_window": "int",
    "max_output_tokens": "int",
    "price_input": "number",
    "price_output": "number",
    "price_cache_write": "number",
    "price_cache_read": "number",
    "price_cache_write_1h": "number",
    "long_prompt_threshold": "int",
    "long_prompt_factor": "number",
    "fallback": "str",
    "extra": _Each("any"),
}
_MODEL_REQUIRED = (
    "id",
    "context_window",
    "max_output_tokens",
    "price_input",
    "price_output",
    "price_cache_write",
    "price_cache_read",
)

_ROLE = {
    "model": "str",
    "effort": "str",
    "tools": "strings",
    "can_spawn": "strings",
    "on_failure": {"escalate_to": "str", "attempts": "int"},
    "max_steps": "int",
    "max_output_tokens": "int",
    "instructions": "str",
    "extra": _Each("any"),
    "result_schema": _Each("any"),
}

_HOOK = {"event": EVENTS, "command": "str", "match": "str", "timeout_seconds": "number"}

_SCHEMA: dict[str, Any] = {
    "initial_role": "str",
    "default_permission": _DECISIONS,
    "visible_to_agent": "bool",
    "never_send": "regexes",
    "models": _Each(_MODEL),
    "role": _Each(_ROLE),
    "budget": {
        "dollars": "number",
        "wall_clock_hours": "number",
        "steps": "int",
        "per_agent_dollars": "number",
        "warn_at": "number",
    },
    "agents": {"concurrent": "int", "depth": "int", "repeat_limit": "int"},
    "human": {
        "wait_seconds": "number",
        "permission_fallback": ("allow", "deny"),
        "question_fallback": ("assume", "stop"),
    },
    "context": {
        "cache_ttl": ("5m", "1h"),
        "output_cap": "int",
        "output_head": "int",
        "output_tail": "int",
        "clean_after_seconds": "number",
        "clean_keep_recent": "int",
        "clean_min_chars": "int",
        "compact_at": "number",
        "compact_keep_turns": "int",
        "compact_keep_chars": "int",
        "compact_boundary_tokens": "int",
        "compaction_template": "str",
    },
    "memory": {"enabled": "bool", "dir": "str", "max_chars": "int"},
    "journal": {"dir": "str", "retention_days": "int", "mask": "regexes"},
    "environment": {
        "mode": ("in_place", "copy", "docker"),
        "shell": "strings",
        "tool_timeout_seconds": "number",
        "env": _Each("str"),
        "docker": {
            "command": "strings",
            "image": "str",
            "workdir": "str",
            "shell": "strings",
            "mounts": _Each("str"),
            "pass_env": "strings",
            "network": "bool",
            "user": "str",
            "options": "strings",
        },
    },
    "provider": {"retry_seconds": "number", "timeout_seconds": "number"},
    "instructions": {"files": "strings", "repo_files": "strings", "load_repo_files": "bool"},
    "plugins": {
        "skills": "strings",
        "path": "strings",
        "packages": "strings",
        "mcp": _Each({"command": "strings", "env": _Each("str")}),
    },
    "web": {
        "search_tool": "str",
        "fetch_tool": "str",
        "search_max_uses": "int",
        "fetch_max_uses": "int",
        "allowed_domains": "strings",
        "blocked_domains": "strings",
        "fetch_max_content_tokens": "int",
        "fetch_citations": "bool",
        "search_price": "number",
    },
    "verify": {"commands": "strings", "review": "str", "rounds": "int", "timeout_seconds": "number"},
    "permissions": "rules",
    "hooks": "hooks",
}

_UNFIXABLE = frozenset({"permissions", "never_send", "hooks"})
_PROHIBITIONS = frozenset({"permissions", "never_send"})

_CAPS = frozenset({
    ("budget", "dollars"),
    ("budget", "wall_clock_hours"),
    ("budget", "steps"),
    ("budget", "per_agent_dollars"),
    ("agents", "concurrent"),
    ("agents", "depth"),
})

_BAD = object()

KeyPath = tuple[str, ...]


def _dotted(path: KeyPath) -> str:
    return ".".join(path)


def _is_cap(path: KeyPath) -> bool:
    return path in _CAPS or (len(path) == 3 and path[0] == "role" and path[2] == "max_steps")


def _parse_rule(key: str) -> tuple[str, str] | None:
    """Split ``"<tool>: <pattern>"`` at the first colon; ``None`` if either side is empty."""
    tool, sep, pattern = key.partition(":")
    tool, pattern = tool.strip(), pattern.strip()
    if not sep or not tool or not pattern or any(c.isspace() for c in tool):
        return None
    return tool, pattern


_EXPECTED = {
    "str": "a string",
    "int": "an integer",
    "number": "a number",
    "bool": "true or false",
    "strings": "a list of strings",
    "regexes": "a list of regular expressions",
    "rules": "a table of rules",
    "hooks": "a list of hook tables",
}


def _describe(schema: Any) -> str:
    if isinstance(schema, tuple):
        return "one of " + ", ".join(schema)
    return _EXPECTED[schema] if isinstance(schema, str) else "a table"


def _clean(value: Any, schema: Any, path: KeyPath, problems: list[str]) -> Any:
    """Return ``value`` checked against ``schema`` (ints widened to float for numbers), or ``_BAD``.

    Invalid entries are dropped and described in ``problems`` so that the merge
    and the cross-reference checks can still run on the rest, and every problem
    of every level reaches the human in one report.
    """
    def bad() -> Any:
        shown = repr(value) if isinstance(value, (str, int, float, bool)) else type(value).__name__
        problems.append(f"{_dotted(path)} must be {_describe(schema)}, got {shown}")
        return _BAD

    if isinstance(schema, (dict, _Each)):
        return _clean_table(value, schema, path, problems) if isinstance(value, dict) else bad()
    if isinstance(schema, tuple):
        return value if isinstance(value, str) and value in schema else bad()
    match schema:
        case "any":
            return value
        case "str":
            return value if isinstance(value, str) else bad()
        case "bool":
            return value if isinstance(value, bool) else bad()
        case "int":
            return value if isinstance(value, int) and not isinstance(value, bool) else bad()
        case "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            return float(value) if ok else bad()
        case "strings":
            ok = isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)
            return list(value) if ok else bad()
        case "regexes":
            if not (isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)):
                return bad()
            good = []
            for pattern in value:
                try:
                    re.compile(pattern)
                except re.error as error:
                    problems.append(
                        f"{_dotted(path)} pattern {pattern!r} is not a valid regular expression ({error}): "
                        "never_send and journal.mask hold regular expressions, not globs"
                    )
                else:
                    good.append(pattern)
            return good
        case "rules":
            return _clean_rules(value, path, problems) if isinstance(value, dict) else bad()
        case "hooks":
            return _clean_hooks(value, path, problems) if isinstance(value, (list, tuple)) else bad()
    raise AssertionError(schema)


def _clean_table(data: dict[str, Any], schema: dict[str, Any] | _Each, path: KeyPath, problems: list[str]) -> dict:
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            continue
        here = (*path, str(key))
        if key == "fixed":
            names = _clean(value, "strings", here, problems)
            if names is _BAD:
                continue
            if isinstance(schema, dict):
                allowed = set(schema) - (_UNFIXABLE if not path else set())
                for name in names:
                    if name != "*" and name not in allowed:
                        problems.append(f"{_dotted(here)} names '{name}', which cannot be fixed here")
                names = [n for n in names if n == "*" or n in allowed]
            out["fixed"] = names
            continue
        if isinstance(schema, _Each):
            sub = schema.item
        elif key in schema:
            sub = schema[key]
        else:
            problems.append(f"unknown key '{_dotted(here)}'")
            continue
        cleaned = _clean(value, sub, here, problems)
        if cleaned is not _BAD:
            out[key] = cleaned
    return out


def _clean_rules(data: dict[str, Any], path: KeyPath, problems: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in data.items():
        if value is None:
            continue
        if _parse_rule(key) is None:
            problems.append(f"permission rule '{key}' must look like '<tool>: <pattern>'")
        elif _clean(value, _DECISIONS, (*path, key), problems) is not _BAD:
            out[key] = value
    return out


def _clean_hooks(data: list[Any] | tuple[Any, ...], path: KeyPath, problems: list[str]) -> list[dict]:
    out: list[dict] = []
    for index, item in enumerate(data):
        here = (*path[:-1], f"{path[-1]}[{index}]")
        if not isinstance(item, dict):
            problems.append(f"{_dotted(here)} must be a table")
            continue
        if "fixed" in item:
            problems.append(f"unknown key '{_dotted((*here, 'fixed'))}'")
            item = {k: v for k, v in item.items() if k != "fixed"}
        hook = _clean_table(item, _HOOK, here, problems)
        missing = [k for k in ("event", "command") if k not in hook and k not in item]
        if missing:
            problems.append(f"{_dotted(here)} needs {' and '.join(missing)}")
        elif "event" in hook and "command" in hook:
            out.append(hook)
    return out


@dataclass
class _Merge:
    """Running state of one merge: effective values, who set each leaf, and what is fixed by whom."""

    tree: dict[str, Any] = field(default_factory=dict)
    sources: dict[KeyPath, str] = field(default_factory=dict)
    fixed: dict[KeyPath, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def fixed_by(self, path: KeyPath) -> str | None:
        for size in range(len(path) + 1):
            owner = self.fixed.get(path[:size])
            if owner is not None:
                return owner
        return None

    def apply(self, data: dict[str, Any], level: str) -> None:
        pending: list[tuple[KeyPath, list[str]]] = []
        self._walk(self.tree, data, (), level, pending)
        for table, names in pending:
            for name in names:
                self.fixed.setdefault(table if name == "*" else (*table, name), level)

    def _walk(self, target: dict, data: dict, path: KeyPath, level: str, pending: list) -> None:
        for key, value in data.items():
            if key == "fixed":
                pending.append((path, value))
                continue
            if not path and key in _UNFIXABLE:
                continue
            here = (*path, key)
            owner = self.fixed_by(here)
            whole = len(here) == 3 and here[0] == "role" and here[2] == "result_schema"
            """A result schema is replaced as a whole, as a list is: two levels' schemas merged key by key
            would describe a value neither level asked for."""
            if isinstance(value, dict) and not whole and (key in target or owner is None):
                self._walk(target.setdefault(key, {}), value, here, level, pending)
                continue
            if owner is not None:
                if target.get(key) != value:
                    self.warnings.append(f"{level} tried to change {_dotted(here)}, fixed by {owner}: ignored")
                continue
            current = self.sources.get(here)
            if _is_cap(here) and current not in (None, "defaults") and value >= target[key]:
                if value > target[key]:
                    self.warnings.append(
                        f"{level} asked for {_dotted(here)} = {value}, kept {target[key]} set by {current}:"
                        " caps can only be tightened"
                    )
                continue
            target[key] = value
            self.sources[here] = level


def merge_levels(levels: list[Level]) -> LoadedPolicy:
    """Merge already-parsed levels into one policy, following the rules of the module docstring.

    Level names must be among ``APPLICATION_ORDER`` and unique; the list may come
    in any order. No environment is read; the only file read is the compaction
    template, to check at load time that it can be. Raises ``PolicyError``
    listing every problem of every trusted level plus the cross-reference problems.
    """
    problems: list[str] = []
    names = [level.name for level in levels]
    for name in names:
        if name not in APPLICATION_ORDER:
            problems.append(f"unknown policy level '{name}'")
        elif names.count(name) > 1:
            problems.append(f"policy level '{name}' given more than once")
    if problems:
        raise PolicyError(sorted(set(problems)))
    ordered = sorted(levels, key=lambda lv: APPLICATION_ORDER.index(lv.name))
    by_authority = sorted(ordered, key=lambda lv: _RANK[lv.name])

    state = _Merge()
    cleaned: dict[str, dict[str, Any]] = {}
    for level in ordered:
        local: list[str] = []
        data = _clean_table(level.data, _SCHEMA, (), local)
        if level.trusted:
            problems.extend(f"{level.label}: {p}" for p in local)
            cleaned[level.name] = data
            state.apply(data, level.name)
            continue
        kept = {"permissions": {k: v for k, v in data.get("permissions", {}).items() if v == "deny"}}
        kept["never_send"] = data.get("never_send", [])
        cleaned[level.name] = kept
        ignored = [k for k in data if k not in _PROHIBITIONS]
        ignored += [f"'{k}' = {v}" for k, v in data.get("permissions", {}).items() if v != "deny"]
        if ignored or local:
            text = f"{level.label} is not trusted: only deny rules and never_send were read"
            if ignored:
                text += "; ignored: " + ", ".join(ignored)
            if local:
                text += "; problems: " + "; ".join(local)
            state.warnings.append(text)

    sources = {_dotted(path): level for path, level in state.sources.items()}
    rules: list[PermissionRule] = []
    never_send: list[str] = []
    hooks: list[HookSpec] = []
    for level in by_authority:
        data = cleaned[level.name]
        for key, decision in data.get("permissions", {}).items():
            tool, pattern = _parse_rule(key) or ("", "")
            sources[f"permissions[{len(rules)}]"] = level.name
            rules.append(PermissionRule(tool, pattern, decision, level.name))
        for pattern in data.get("never_send", []):
            if pattern not in never_send:
                sources[f"never_send[{len(never_send)}]"] = level.name
                never_send.append(pattern)
        for hook in data.get("hooks", []) if level.trusted else []:
            sources[f"hooks[{len(hooks)}]"] = level.name
            hooks.append(HookSpec(**hook))

    policy = _build(state.tree, tuple(rules), tuple(never_send), tuple(hooks), problems)
    if problems or policy is None:
        raise PolicyError(problems)
    values: dict[str, Any] = {}
    for path in state.sources:
        node: Any = state.tree
        for part in path:
            node = node[part]
        values[_dotted(path)] = node
    fixed = frozenset(_dotted(path) for path in state.sources if state.fixed_by(path) is not None)
    return LoadedPolicy(policy, state.warnings, sources, fixed, values)


def _build(
    tree: dict[str, Any],
    rules: tuple[PermissionRule, ...],
    never_send: tuple[str, ...],
    hooks: tuple[HookSpec, ...],
    problems: list[str],
) -> Policy | None:
    """Turn the merged tree into a ``Policy``, adding every cross-reference problem to ``problems``."""
    models: dict[str, ModelSpec] = {}
    for alias, spec in tree.get("models", {}).items():
        missing = [k for k in _MODEL_REQUIRED if k not in spec]
        if missing:
            problems.append(f"models.{alias} is missing {', '.join(missing)}")
        else:
            models[alias] = ModelSpec(alias=alias, **spec)
    for alias, spec in models.items():
        if spec.fallback is not None and spec.fallback not in models:
            problems.append(f"models.{alias}.fallback '{spec.fallback}' is not a defined model")
    problems.extend(_fallback_loops(models))

    roles: dict[str, Role] = {}
    for name, spec in tree.get("role", {}).items():
        if "model" not in spec:
            problems.append(f"role.{name} has no model")
            continue
        escalation = None
        if (failure := spec.get("on_failure")) is not None:
            if "escalate_to" not in failure:
                problems.append(f"role.{name}.on_failure has no escalate_to")
            else:
                escalation = Escalation(**failure)
        roles[name] = Role(
            name=name,
            model=spec["model"],
            effort=spec.get("effort"),
            tools=tuple(spec.get("tools", ())),
            can_spawn=tuple(spec.get("can_spawn", ())),
            on_failure=escalation,
            max_steps=spec.get("max_steps"),
            max_output_tokens=spec.get("max_output_tokens"),
            instructions=spec.get("instructions"),
            extra=dict(spec.get("extra", {})),
            result_schema=spec.get("result_schema"),
        )
        if "result_schema" in spec:
            problems.extend(schema_problems(spec["result_schema"], f"role.{name}.result_schema"))
    declared = set(tree.get("models", {}))
    for name, role in roles.items():
        if role.model not in declared:
            problems.append(f"role.{name}.model '{role.model}' is not a defined model")
        for tool in role.tools:
            if tool not in ROLE_TOOLS:
                problems.append(f"role.{name}.tools names '{tool}', which is not one of {', '.join(ROLE_TOOLS)}")
        for child in role.can_spawn:
            if child not in roles:
                problems.append(f"role.{name}.can_spawn names '{child}', which is not a defined role")
        if role.on_failure and role.on_failure.escalate_to not in roles:
            problems.append(f"role.{name}.on_failure.escalate_to '{role.on_failure.escalate_to}' is not a defined role")

    initial = tree.get("initial_role")
    if initial is None:
        problems.append("initial_role is not set")
    elif initial not in roles:
        problems.append(f"initial_role '{initial}' is not a defined role")

    mcp: list[McpServerSpec] = []
    for name, spec in tree.get("plugins", {}).get("mcp", {}).items():
        if "command" not in spec:
            problems.append(f"plugins.mcp.{name} has no command")
        else:
            mcp.append(McpServerSpec(name, tuple(spec["command"]), dict(spec.get("env", {}))))
    if (template := tree.get("context", {}).get("compaction_template")) is not None:
        try:
            Path(template).expanduser().read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            problems.append(f"context.compaction_template '{template}' is not a readable text file: {error}")
    verify = tree.get("verify", {})
    if (reviewer := verify.get("review")) is not None and reviewer not in roles:
        problems.append(f"verify.review '{reviewer}' is not a defined role")
    if verify.get("rounds", 0) < 0:
        problems.append("verify.rounds must not be negative")
    problems.extend(_docker_problems(tree.get("environment", {})))
    provider = tree.get("provider", {})
    if provider.get("retry_seconds", 0.0) < 0:
        problems.append("provider.retry_seconds must not be negative")
    if provider.get("timeout_seconds", 1.0) <= 0:
        problems.append("provider.timeout_seconds must be positive")
    if problems:
        return None

    def section(name: str) -> dict[str, Any]:
        return {k: tuple(v) if isinstance(v, list) else v for k, v in tree.get(name, {}).items()}

    environment = section("environment")
    environment["env"] = dict(tree.get("environment", {}).get("env", {}))
    docker = {k: tuple(v) if isinstance(v, list) else v for k, v in environment.get("docker", {}).items()}
    docker["mounts"] = dict(docker.get("mounts", {}))
    environment["docker"] = DockerSettings(**docker)
    plugins = section("plugins")
    plugins["mcp"] = tuple(mcp)
    return Policy(
        initial_role=initial,
        roles=roles,
        models=models,
        budget=Budget(**section("budget")),
        agents=AgentLimits(**section("agents")),
        human=HumanPolicy(**section("human")),
        context=ContextSettings(**section("context")),
        journal=JournalSettings(**section("journal")),
        memory=MemorySettings(**section("memory")),
        environment=EnvironmentSettings(**environment),
        provider=ProviderSettings(**section("provider")),
        instructions=InstructionSettings(**section("instructions")),
        plugins=PluginSettings(**plugins),
        web=WebSettings(**section("web")),
        verify=VerifySettings(**section("verify")),
        permissions=rules,
        default_permission=tree.get("default_permission", "allow"),
        hooks=hooks,
        never_send=never_send,
        visible_to_agent=tree.get("visible_to_agent", True),
    )


def _docker_problems(environment: dict[str, Any]) -> list[str]:
    """What makes ``environment.mode = "docker"`` unusable; nothing is checked in the other modes."""
    if environment.get("mode") != "docker":
        return []
    docker = environment.get("docker", {})
    found = []
    if not docker.get("image"):
        found.append('environment.docker.image must name an image when environment.mode = "docker"')
    if not docker.get("command"):
        found.append("environment.docker.command must not be empty")
    if not docker.get("shell"):
        found.append("environment.docker.shell must not be empty")
    for path in [docker.get("workdir", "/workspace"), *docker.get("mounts", {})]:
        if not path.startswith("/"):
            found.append(f"environment.docker: the container path {path!r} must be absolute")
    secret = "ANTHROPIC_API_KEY"
    if any(name.upper() == secret for name in docker.get("pass_env", [])):
        found.append(f"environment.docker.pass_env must not name {secret}: the container never gets the engine's key")
    if any(name.upper() == secret for name in environment.get("env", {})):
        found.append(f'environment.env must not set {secret} when environment.mode = "docker"')
    return found


def _fallback_loops(models: dict[str, ModelSpec]) -> list[str]:
    """Report each fallback cycle once, as the chain of aliases that closes on itself."""
    found: list[str] = []
    seen_cycles: set[frozenset[str]] = set()
    for start in models:
        chain = [start]
        while (nxt := models[chain[-1]].fallback) in models:
            if nxt in chain:
                cycle = chain[chain.index(nxt):]
                if frozenset(cycle) not in seen_cycles:
                    seen_cycles.add(frozenset(cycle))
                    found.append("model fallback loops: " + " -> ".join([*cycle, nxt]))
                break
            chain.append(nxt)
    return found


@dataclass
class LoadedPolicy:
    """The merged policy plus what the human needs to understand it.

    ``sources`` maps each effective dotted key (``role.build.model``) to the level
    that set it; accumulated entries use an index (``permissions[0]``,
    ``hooks[1]``, ``never_send[0]``) matching their position in the policy.
    ``fixed`` holds the dotted keys that lower levels could not change, and
    ``values`` the effective value of every dotted key in ``sources`` that is not
    an indexed entry.
    """

    policy: Policy
    warnings: list[str]
    sources: dict[str, str]
    fixed: frozenset[str] = frozenset()
    values: dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        """Every effective value with the level it came from, fixed keys marked, unset caps shown as no cap."""
        lines = []
        for key in sorted(self.values):
            mark = ", fixed" if key in self.fixed else ""
            lines.append(f"{key} = {self.values[key]!r}  ({self.sources[key]}{mark})")
        caps = {_dotted(c) for c in _CAPS} | {f"role.{name}.max_steps" for name in self.policy.roles}
        lines += [f"{key} = no cap" for key in sorted(caps - set(self.sources))]
        for index, rule in enumerate(self.policy.permissions):
            lines.append(f"permission '{rule.tool}: {rule.pattern}' = {rule.decision}  ({self.sources[f'permissions[{index}]']})")
        for index, hook in enumerate(self.policy.hooks):
            target = f" [{hook.match}]" if hook.match else ""
            lines.append(f"hook {hook.event}{target}: {hook.command!r}  ({self.sources[f'hooks[{index}]']})")
        for index, pattern in enumerate(self.policy.never_send):
            lines.append(f"never_send {pattern!r}  ({self.sources[f'never_send[{index}]']})")
        return "\n".join(lines)


def decide(policy: Policy, tool: str, target: str) -> tuple[Decision, PermissionRule | None]:
    """Answer whether ``tool`` may act on ``target``, and which rule said so.

    A rule matches when its tool is the tool name or ``*`` and its pattern
    matches the target with ``fnmatch.fnmatchcase`` (case-sensitive, ``*`` also
    crosses ``/``). Any matching ``deny`` wins. Otherwise the matching rule of the
    most authoritative level wins; within one level, the longest pattern, then a
    named tool over ``*``. No match: ``policy.default_permission``.

    Warning: text patterns on shell commands are easy to get around. The same
    command can be written in many ways (other spacing, quoting, a variable, a
    script that runs it), so ``"sh: git push*" = "deny"`` stops honest mistakes,
    not a determined agent. The serious protection for something valuable is not
    giving the agent the credential at all, or a hook that inspects the action.
    """
    matches = [
        rule for rule in policy.permissions
        if rule.tool in (tool, "*") and fnmatchcase(target, rule.pattern)
    ]
    if not matches:
        return policy.default_permission, None
    denies = [rule for rule in matches if rule.decision == "deny"]
    best = min(
        denies or matches,
        key=lambda r: (_RANK.get(r.level, len(AUTHORITY)), -len(r.pattern), r.tool == "*"),
    )
    return best.decision, best


def spawn_role(policy: Policy, parent: Role, child_role: str, always: frozenset[str] = frozenset()) -> Role:
    """Return the role a child of ``parent`` runs with, never with more than its parent.

    ``always`` names the engine tools every agent has. The child's tools are
    intersected with the parent's tools plus ``always``, and its ``can_spawn`` with
    the parent's. ``parent`` is used as given, since it may already be narrowed.
    """
    if child_role not in policy.roles:
        raise PolicyError([f"role '{child_role}' is not defined"])
    if child_role not in parent.can_spawn:
        raise PolicyError([f"role '{parent.name}' may not spawn '{child_role}'"])
    child = policy.roles[child_role]
    allowed_tools = set(parent.tools) | always
    return replace(
        child,
        tools=tuple(t for t in child.tools if t in allowed_tools),
        can_spawn=tuple(r for r in child.can_spawn if r in parent.can_spawn),
    )


def _read_toml(path: Path, problems: list[str], label: str) -> dict[str, Any] | None:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        problems.append(f"{label}: file not found: {path}")
    except (OSError, tomllib.TOMLDecodeError) as error:
        problems.append(f"{label}: cannot read {path}: {error}")
    return None


def factory_defaults() -> dict[str, Any]:
    """The packaged ``defaults.toml``, parsed."""
    return tomllib.loads(files("synthic").joinpath("defaults.toml").read_text(encoding="utf-8"))


def deep_merge(base: Mapping[str, Any], top: Mapping[str, Any]) -> dict[str, Any]:
    """``top`` over ``base``: tables merge key by key, any other value replaces."""
    out = dict(base)
    for key, value in top.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = dict(value) if isinstance(value, Mapping) else value
    return out


def load_policy(
    project_root: Path | None = None,
    task_file: Path | None = None,
    task_overrides: dict[str, Any] | None = None,
    trust_project: bool = False,
    env: Mapping[str, str] | None = None,
    home: Path | None = None,
) -> LoadedPolicy:
    """Read every level from disk and merge them. ``env`` and ``home`` default to the real ones.

    ``task_overrides`` is a nested dict shaped like the file
    (``{"role": {"build": {"model": "sonnet"}}}``) and wins over ``task_file``.
    A missing organization file (when the variable is set) or task file is an
    error; missing user and project files are normal. An unreadable untrusted
    project file is only a warning.
    """
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    problems: list[str] = []
    warnings: list[str] = []
    levels = [Level("defaults", factory_defaults(), True, "defaults.toml")]

    if org := env.get("SYNTHIC_ORG_POLICY"):
        if (data := _read_toml(Path(org), problems, "organization")) is not None:
            levels.append(Level("organization", data, True, org))
    user_file = home / ".synthic" / "policy.toml"
    if user_file.is_file() and (data := _read_toml(user_file, problems, "user")) is not None:
        levels.append(Level("user", data, True, str(user_file)))
    if project_root is not None:
        project_file = project_root / ".synthic" / "policy.toml"
        if project_file.is_file():
            sink: list[str] = problems if trust_project else []
            if (data := _read_toml(project_file, sink, "project")) is not None:
                levels.append(Level("project", data, trust_project, str(project_file)))
            elif not trust_project:
                warnings.append(f"project policy not read, its deny rules are not in force: {sink[0]}")
    if task_file is not None or task_overrides:
        task: dict[str, Any] = {}
        origin = []
        if task_file is not None:
            origin.append(str(task_file))
            task = _read_toml(task_file, problems, "task") or {}
        if task_overrides:
            origin.append("overrides")
            task = deep_merge(task, task_overrides)
        levels.append(Level("task", task, True, " + ".join(origin)))

    try:
        loaded = merge_levels(levels)
    except PolicyError as error:
        raise PolicyError(problems + error.problems) from None
    if problems:
        raise PolicyError(problems)
    loaded.warnings[:0] = warnings
    return loaded
