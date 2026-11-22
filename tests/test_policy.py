from __future__ import annotations

from pathlib import Path
import pytest

from synthic.policy import (
    Level,
    LoadedPolicy,
    PolicyError,
    decide,
    factory_defaults,
    load_policy,
    merge_levels,
    spawn_role,
)
from synthic.types import PermissionRule, Policy, Role


def merge(*levels: Level) -> LoadedPolicy:
    return merge_levels([Level("defaults", factory_defaults()), *levels])


def problems_of(*levels: Level) -> list[str]:
    with pytest.raises(PolicyError) as info:
        merge(*levels)
    return info.value.problems


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_fallback_loop_from_a_level_is_reported() -> None:
    problems = problems_of(Level("user", {"models": {"opus": {"fallback": "sonnet"}}}))
    assert problems == ["model fallback loops: sonnet -> opus -> sonnet"]


def test_factory_defaults_load_cleanly() -> None:
    loaded = merge()
    p = loaded.policy
    assert loaded.warnings == []
    assert p.initial_role == "build"
    assert p.roles["build"].model == "opus"
    assert p.roles["build"].tools == ("sh", "edit", "spawn")
    assert p.roles["mechanic"].on_failure is not None
    assert p.roles["mechanic"].on_failure.escalate_to == "build"
    assert p.models["haiku"].long_prompt_factor == 5.0
    assert p.models["sonnet"].fallback == "opus"
    assert p.human.wait_seconds == 0.0 and isinstance(p.human.wait_seconds, float)
    assert p.environment.tool_timeout_seconds == 120.0
    assert p.instructions.repo_files == ("AGENTS.md", "CLAUDE.md")
    assert p.budget.dollars is None
    assert loaded.sources["role.build.model"] == "defaults"


def test_repeat_limit_is_a_plain_knob() -> None:
    assert merge().policy.agents.repeat_limit == 3
    loaded = merge(Level("organization", {"agents": {"repeat_limit": 2}}), Level("task", {"agents": {"repeat_limit": 0}}))
    assert loaded.policy.agents.repeat_limit == 0
    assert loaded.warnings == []
    assert problems_of(Level("task", {"agents": {"repeat_limit": "many"}})) == [
        "task: agents.repeat_limit must be an integer, got 'many'"
    ]


def test_lowest_level_wins() -> None:
    loaded = merge(
        Level("user", {"role": {"build": {"model": "sonnet"}}}),
        Level("project", {"role": {"build": {"model": "haiku"}}}, trusted=True),
        Level("task", {"role": {"build": {"model": "fable"}}}),
    )
    assert loaded.policy.roles["build"].model == "fable"
    assert loaded.sources["role.build.model"] == "task"


def test_higher_level_applies_when_lower_says_nothing() -> None:
    loaded = merge(Level("organization", {"role": {"build": {"model": "sonnet"}}}), Level("task", {}))
    assert loaded.policy.roles["build"].model == "sonnet"
    assert loaded.sources["role.build.model"] == "organization"


def test_levels_may_come_in_any_order() -> None:
    loaded = merge_levels([
        Level("task", {"role": {"build": {"model": "haiku"}}}),
        Level("defaults", factory_defaults()),
        Level("user", {"role": {"build": {"model": "sonnet"}}}),
    ])
    assert loaded.policy.roles["build"].model == "haiku"


def test_fixed_key_in_table() -> None:
    loaded = merge(
        Level("organization", {"role": {"build": {"model": "sonnet", "fixed": ["model"]}}}),
        Level("task", {"role": {"build": {"model": "haiku", "effort": "low"}}}),
    )
    build = loaded.policy.roles["build"]
    assert build.model == "sonnet"
    assert build.effort == "low"
    assert any("role.build.model" in w and "organization" in w for w in loaded.warnings)
    assert "role.build.model" in loaded.fixed
    assert "role.build.effort" not in loaded.fixed


def test_fixed_level_can_still_set_its_own_value() -> None:
    loaded = merge(Level("user", {"role": {"build": {"model": "haiku", "fixed": ["model"]}}}))
    assert loaded.policy.roles["build"].model == "haiku"
    assert loaded.warnings == []


def test_fixed_top_level_key() -> None:
    loaded = merge(
        Level("organization", {"initial_role": "explore", "fixed": ["initial_role"]}),
        Level("user", {"initial_role": "build", "visible_to_agent": False}),
    )
    assert loaded.policy.initial_role == "explore"
    assert loaded.policy.visible_to_agent is False
    assert len(loaded.warnings) == 1 and "initial_role" in loaded.warnings[0]


def test_fixed_star_fixes_the_whole_table() -> None:
    loaded = merge(
        Level("organization", {"budget": {"warn_at": 0.5, "fixed": ["*"]}}),
        Level("user", {"budget": {"warn_at": 0.9, "dollars": 5}}),
    )
    assert loaded.policy.budget.warn_at == 0.5
    assert loaded.policy.budget.dollars is None
    assert len(loaded.warnings) == 2


def test_fixed_role_table_blocks_new_roles() -> None:
    loaded = merge(
        Level("organization", {"role": {"fixed": ["*"]}}),
        Level("task", {"role": {"extra": {"model": "haiku"}}}),
    )
    assert "extra" not in loaded.policy.roles


def test_top_level_star_does_not_block_prohibitions() -> None:
    loaded = merge(
        Level("organization", {"fixed": ["*"]}),
        Level("user", {
            "role": {"build": {"model": "haiku"}},
            "permissions": {"sh: rm -rf*": "deny"},
            "never_send": ["pem$"],
        }),
    )
    assert loaded.policy.roles["build"].model == "opus"
    assert decide(loaded.policy, "sh", "rm -rf /")[0] == "deny"
    assert loaded.policy.never_send == ("pem$",)


def test_prohibitions_cannot_be_fixed() -> None:
    problems = problems_of(Level("organization", {"fixed": ["permissions", "never_send", "hooks"]}))
    assert len(problems) == 3


def test_per_key_merge_inside_role() -> None:
    loaded = merge(Level("project", {"role": {"build": {"effort": "medium"}}}, trusted=True))
    build = loaded.policy.roles["build"]
    assert (build.model, build.effort) == ("opus", "medium")
    assert loaded.sources["role.build.model"] == "defaults"
    assert loaded.sources["role.build.effort"] == "project"


def test_effort_passes_through_unchecked() -> None:
    loaded = merge(Level("task", {"role": {"build": {"effort": "whatever-the-vendor-adds"}}}))
    assert loaded.policy.roles["build"].effort == "whatever-the-vendor-adds"


def test_lists_are_replaced_whole() -> None:
    loaded = merge(
        Level("user", {"role": {"build": {"tools": ["sh", "edit", "spawn", "skill"]}}}),
        Level("task", {"role": {"build": {"tools": ["sh"]}}, "instructions": {"files": ["a.md"]}}),
    )
    assert loaded.policy.roles["build"].tools == ("sh",)
    assert loaded.policy.instructions.files == ("a.md",)


def test_network_allow_is_gone() -> None:
    assert problems_of(Level("task", {"environment": {"network_allow": ["pypi.org"]}})) == [
        "task: unknown key 'environment.network_allow'"
    ]


def test_role_tools_must_be_known_tools() -> None:
    problems = problems_of(Level("task", {"role": {"build": {"tools": ["sh", "web"]}}}))
    assert problems == ["role.build.tools names 'web', which is not one of sh, edit, spawn, skill, mcp"]


def test_compaction_template_must_be_readable(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.md")
    problems = problems_of(Level("task", {"context": {"compaction_template": missing}}))
    assert len(problems) == 1 and problems[0].startswith(f"context.compaction_template '{missing}'")
    good = write(tmp_path / "ok.md", "summarize")
    assert merge(Level("task", {"context": {"compaction_template": str(good)}})).policy.context.compaction_template


def test_provider_table() -> None:
    p = merge().policy.provider
    assert (p.retry_seconds, p.timeout_seconds) == (3600.0, 600.0)
    loaded = merge(Level("task", {"provider": {"retry_seconds": 60, "timeout_seconds": 30}}))
    assert (loaded.policy.provider.retry_seconds, loaded.policy.provider.timeout_seconds) == (60.0, 30.0)
    problems = problems_of(Level("task", {"provider": {"retry_seconds": -1, "timeout_seconds": 0, "tries": 3}}))
    assert sorted(problems) == [
        "provider.retry_seconds must not be negative",
        "provider.timeout_seconds must be positive",
        "task: unknown key 'provider.tries'",
    ]


def test_extra_tables_merge_per_key_and_obey_fixed() -> None:
    loaded = merge(
        Level("organization", {"models": {"opus": {"extra": {"service_tier": "auto", "fixed": ["service_tier"]}}}}),
        Level("user", {"role": {"build": {"extra": {"context_management": {"edits": []}}}}}),
        Level("task", {
            "models": {"opus": {"extra": {"service_tier": "standard_only", "anthropic_beta": ["b-1"]}}},
            "role": {"build": {"extra": {"context_management": {"keep": 3}}}},
        }),
    )
    p = loaded.policy
    assert p.models["opus"].extra == {"service_tier": "auto", "anthropic_beta": ["b-1"]}
    assert p.roles["build"].extra == {"context_management": {"edits": [], "keep": 3}}
    assert any("models.opus.extra.service_tier" in w for w in loaded.warnings)


def test_fixed_list_is_not_replaced() -> None:
    loaded = merge(
        Level("organization", {"plugins": {"skills": ["approved"], "fixed": ["skills"]}}),
        Level("task", {"plugins": {"skills": ["approved", "random"]}}),
    )
    assert loaded.policy.plugins.skills == ("approved",)
    assert loaded.warnings


def test_deny_from_lowest_level_survives_allow_from_highest() -> None:
    loaded = merge(
        Level("organization", {"permissions": {"sh: git push*": "allow"}}),
        Level("task", {"permissions": {"sh: git push*": "deny"}}),
    )
    decision, rule = decide(loaded.policy, "sh", "git push origin main")
    assert decision == "deny"
    assert rule is not None and rule.level == "task"


def test_untrusted_project_contributes_only_prohibitions() -> None:
    project = {
        "initial_role": "explore",
        "fixed": ["initial_role"],
        "role": {"build": {"model": "haiku", "tools": ["sh", "edit", "spawn", "web"]}},
        "budget": {"dollars": 1},
        "hooks": [{"event": "start", "command": "curl evil"}],
        "permissions": {"sh: git push*": "deny", "sh: *": "allow"},
        "never_send": ["secrets/*"],
    }
    loaded = merge(Level("project", project, trusted=False, origin="repo/.synthic/policy.toml"))
    p = loaded.policy
    assert p.initial_role == "build"
    assert p.roles["build"].model == "opus"
    assert p.budget.dollars is None
    assert p.hooks == ()
    assert p.permissions == (PermissionRule("sh", "git push*", "deny", "project"),)
    assert p.never_send == ("secrets/*",)
    assert len(loaded.warnings) == 1
    warning = loaded.warnings[0]
    assert "repo/.synthic/policy.toml" in warning
    assert "role" in warning and "'sh: *' = allow" in warning


def test_untrusted_project_with_only_prohibitions_gives_no_warning() -> None:
    loaded = merge(Level("project", {"permissions": {"edit: .github/**": "deny"}, "never_send": ["key$"]}, trusted=False))
    assert loaded.warnings == []
    assert decide(loaded.policy, "edit", ".github/workflows/ci.yml")[0] == "deny"


def test_untrusted_project_problems_are_part_of_the_warning() -> None:
    loaded = merge(Level("project", {"rol": {}, "budget": {"dollars": "lots"}}, trusted=False, origin="p.toml"))
    assert len(loaded.warnings) == 1
    assert "unknown key 'rol'" in loaded.warnings[0]
    assert "budget.dollars" in loaded.warnings[0]


def test_trusted_project_behaves_like_a_level() -> None:
    loaded = merge(Level("project", {
        "role": {"build": {"model": "sonnet"}},
        "permissions": {"sh: npm *": "allow"},
        "hooks": [{"event": "before_tool", "command": "lint", "match": "edit"}],
    }, trusted=True))
    assert loaded.policy.roles["build"].model == "sonnet"
    assert loaded.policy.permissions[0].decision == "allow"
    assert loaded.policy.hooks[0].command == "lint"
    assert loaded.warnings == []


def test_cap_cannot_be_loosened_by_lower_level() -> None:
    loaded = merge(Level("user", {"budget": {"dollars": 50}}), Level("project", {"budget": {"dollars": 100}}, trusted=True))
    assert loaded.policy.budget.dollars == 50.0
    assert loaded.sources["budget.dollars"] == "user"
    assert any("budget.dollars" in w and "tightened" in w for w in loaded.warnings)


def test_cap_can_be_tightened_by_lower_level() -> None:
    loaded = merge(Level("user", {"budget": {"dollars": 50}}), Level("project", {"budget": {"dollars": 20}}, trusted=True))
    assert loaded.policy.budget.dollars == 20.0
    assert loaded.sources["budget.dollars"] == "project"
    assert loaded.warnings == []


def test_organization_cap_outranks_task() -> None:
    loaded = merge(
        Level("organization", {"budget": {"steps": 100}, "agents": {"depth": 1}}),
        Level("task", {"budget": {"steps": 500}, "agents": {"depth": 3}}),
    )
    assert loaded.policy.budget.steps == 100
    assert loaded.policy.agents.depth == 1
    tighter = merge(Level("organization", {"budget": {"steps": 100}}), Level("task", {"budget": {"steps": 10}}))
    assert tighter.policy.budget.steps == 10


def test_defaults_are_not_a_cap() -> None:
    loaded = merge(Level("user", {"agents": {"concurrent": 8}}), Level("project", {"agents": {"concurrent": 16}}, trusted=True))
    assert loaded.policy.agents.concurrent == 8


def test_absent_cap_never_loosens() -> None:
    loaded = merge(Level("organization", {"budget": {"per_agent_dollars": 3}}), Level("task", {"budget": {"per_agent_dollars": None}}))
    assert loaded.policy.budget.per_agent_dollars == 3.0


def test_role_max_steps_is_a_cap() -> None:
    loaded = merge(
        Level("user", {"role": {"build": {"max_steps": 40}}}),
        Level("task", {"role": {"build": {"max_steps": 400, "max_output_tokens": 999}}}),
    )
    assert loaded.policy.roles["build"].max_steps == 40
    assert loaded.policy.roles["build"].max_output_tokens == 999


def test_fixed_cap_ignores_even_a_tighter_value() -> None:
    loaded = merge(
        Level("organization", {"budget": {"dollars": 10, "fixed": ["dollars"]}}),
        Level("task", {"budget": {"dollars": 5}}),
    )
    assert loaded.policy.budget.dollars == 10.0
    assert len(loaded.warnings) == 1


def test_hooks_accumulate_in_level_order() -> None:
    loaded = merge(
        Level("task", {"hooks": [{"event": "end", "command": "t"}]}),
        Level("organization", {"hooks": [{"event": "start", "command": "o", "timeout_seconds": 5}]}),
        Level("project", {"hooks": [{"event": "end", "command": "p"}]}, trusted=False),
        Level("user", {"hooks": [{"event": "before_tool", "command": "u1"}, {"event": "after_tool", "command": "u2"}]}),
    )
    assert [h.command for h in loaded.policy.hooks] == ["o", "u1", "u2", "t"]
    assert loaded.policy.hooks[0].timeout_seconds == 5.0
    assert loaded.sources["hooks[0]"] == "organization"


def test_never_send_accumulates() -> None:
    loaded = merge(
        Level("organization", {"never_send": ["pem$"]}),
        Level("user", {"never_send": ["pem$", ".env"]}),
        Level("task", {"never_send": []}),
    )
    assert loaded.policy.never_send == ("pem$", ".env")


def test_decide_deny_wins_over_more_authoritative_allow() -> None:
    loaded = merge(
        Level("organization", {"permissions": {"edit: *": "allow"}}),
        Level("project", {"permissions": {"edit: .github/**": "deny"}}, trusted=False),
    )
    assert decide(loaded.policy, "edit", ".github/workflows/release.yml")[0] == "deny"
    assert decide(loaded.policy, "edit", "force-app/main/default/classes/A.cls")[0] == "allow"


def test_decide_level_authority() -> None:
    loaded = merge(
        Level("user", {"permissions": {"sh: git push*": "ask"}}),
        Level("task", {"permissions": {"sh: git push origin*": "allow"}}),
    )
    decision, rule = decide(loaded.policy, "sh", "git push origin main")
    assert decision == "ask"
    assert rule is not None and rule.level == "user"


def test_decide_specificity_within_level() -> None:
    loaded = merge(Level("user", {"permissions": {"sh: git *": "allow", "sh: git push*": "ask"}}))
    assert decide(loaded.policy, "sh", "git push --force")[0] == "ask"
    assert decide(loaded.policy, "sh", "git status")[0] == "allow"


def test_decide_default_and_wildcard_tool() -> None:
    loaded = merge(Level("user", {"default_permission": "ask", "permissions": {"*: *secret*": "deny"}}))
    assert decide(loaded.policy, "sh", "ls") == ("ask", None)
    assert decide(loaded.policy, "edit", "config/secret.txt")[0] == "deny"
    assert decide(loaded.policy, "sh", "cat SECRET")[0] == "ask"


def test_decide_ranks_unknown_level_names_last() -> None:
    policy = Policy(
        initial_role="r",
        roles={},
        models={},
        permissions=(
            PermissionRule("sh", "make*", "allow", "default"),
            PermissionRule("sh", "make", "ask", "user"),
        ),
    )
    assert decide(policy, "sh", "make")[0] == "ask"


def test_validation_errors_are_reported_together() -> None:
    problems = problems_of(Level("user", {
        "initial_role": "nobody",
        "unknown_top": 1,
        "budget": {"dolars": 5, "steps": True},
        "agents": {"concurrent": "four"},
        "models": {
            "loop_a": {**factory_defaults()["models"]["haiku"], "fallback": "loop_b"},
            "loop_b": {**factory_defaults()["models"]["haiku"], "fallback": "loop_a"},
            "lost": {**factory_defaults()["models"]["haiku"], "fallback": "ghost"},
            "partial": {"id": "x"},
        },
        "role": {
            "build": {"model": "gpt", "can_spawn": ["ghost"], "on_failure": {"escalate_to": "nobody"}},
            "empty": {"effort": "low"},
        },
        "permissions": {"sh git push": "deny", "edit: x": "maybe"},
        "hooks": [{"event": "on_boot", "command": "x"}, {"event": "start"}],
        "human": {"permission_fallback": "ask"},
    }))
    expected = [
        "initial_role 'nobody'",
        "unknown key 'unknown_top'",
        "unknown key 'budget.dolars'",
        "budget.steps must be an integer",
        "agents.concurrent must be an integer",
        "fallback loops: loop_a -> loop_b -> loop_a",
        "fallback 'ghost'",
        "models.partial is missing",
        "role.build.model 'gpt'",
        "can_spawn names 'ghost'",
        "escalate_to 'nobody'",
        "role.empty has no model",
        "'sh git push' must look like",
        "permissions.edit: x must be one of allow, ask, deny",
        "hooks[0].event must be one of",
        "hooks[1] needs command",
        "human.permission_fallback must be one of allow, deny",
    ]
    text = "\n".join(problems)
    for fragment in expected:
        assert fragment in text, fragment


def test_valid_references_produce_no_errors() -> None:
    loaded = merge(Level("user", {
        "role": {"review": {"model": "sonnet", "can_spawn": ["explore"], "on_failure": {"escalate_to": "build"}}},
        "models": {"cheap": {**factory_defaults()["models"]["haiku"], "fallback": "haiku"}},
    }))
    assert loaded.policy.roles["review"].on_failure is not None
    assert loaded.policy.models["cheap"].fallback == "haiku"


def test_full_schema_is_accepted(tmp_path: Path) -> None:
    template = str(write(tmp_path / "summary.md", "summarize"))
    loaded = merge(Level("user", {
        "plugins": {"mcp": {"sf": {"command": ["sf-mcp", "--stdio"], "env": {"SF_ORG": "dev"}}}},
        "environment": {"env": {"LANG": "C"}, "shell": ["bash", "-c"]},
        "journal": {"retention_days": 30, "mask": ["token=*"]},
        "context": {"compaction_template": template},
        "role": {"build": {"instructions": "be brief", "max_output_tokens": 4000}},
    }))
    p = loaded.policy
    assert p.plugins.mcp[0].command == ("sf-mcp", "--stdio")
    assert p.plugins.mcp[0].env == {"SF_ORG": "dev"}
    assert p.environment.env == {"LANG": "C"}
    assert p.journal.mask == ("token=*",)
    assert p.context.compaction_template == template
    assert p.roles["build"].instructions == "be brief"


def test_unknown_level_name_is_an_error() -> None:
    with pytest.raises(PolicyError):
        merge_levels([Level("defaults", factory_defaults()), Level("team", {})])


def test_explain_names_level_and_fixed() -> None:
    loaded = merge(
        Level("organization", {"role": {"build": {"model": "sonnet", "fixed": ["model"]}}}),
        Level("user", {"permissions": {"sh: git push*": "deny"}, "never_send": ["pem$"]}),
    )
    text = loaded.explain()
    assert "role.build.model = 'sonnet'  (organization, fixed)" in text
    assert "role.build.effort = 'high'  (defaults)" in text
    assert "budget.dollars = no cap" in text
    assert "permission 'sh: git push*' = deny  (user)" in text
    assert "never_send 'pem$'  (user)" in text


def test_explain_handles_dotted_names() -> None:
    loaded = merge(Level("organization", {
        "models": {"sonnet-4.5": {**factory_defaults()["models"]["sonnet"], "fixed": ["*"]}},
        "role": {"v1.2": {"model": "sonnet-4.5"}},
        "environment": {"env": {"MY.VAR": "1"}},
    }))
    text = loaded.explain()
    assert "models.sonnet-4.5.id = 'claude-sonnet-5-5'  (organization, fixed)" in text
    assert "role.v1.2.model = 'sonnet-4.5'  (organization)" in text
    assert "environment.env.MY.VAR = '1'  (organization)" in text


def test_caps_hold_without_a_defaults_level() -> None:
    loaded = merge_levels([
        Level("organization", {**factory_defaults(), "budget": {"dollars": 10}}),
        Level("task", {"budget": {"dollars": 99}}),
    ])
    assert loaded.policy.budget.dollars == 10.0


def test_setting_a_fixed_key_to_the_same_value_is_silent() -> None:
    loaded = merge(
        Level("organization", {"role": {"build": {"model": "opus", "fixed": ["model"]}}}),
        Level("task", {"role": {"build": {"model": "opus"}}}),
    )
    assert loaded.warnings == []


def test_policy_error_accepts_a_string() -> None:
    assert PolicyError("broken").problems == ["broken"]


def test_spawn_role_narrows_to_parent() -> None:
    loaded = merge(Level("task", {"role": {
        "build": {"tools": ["sh", "spawn"], "can_spawn": ["explore", "mechanic"]},
        "explore": {"tools": ["sh", "edit", "mcp", "spawn"], "can_spawn": ["mechanic", "build"]},
    }}))
    p = loaded.policy
    child = spawn_role(p, p.roles["build"], "explore", frozenset({"mcp"}))
    assert child.tools == ("sh", "mcp", "spawn")
    assert child.can_spawn == ("mechanic",)
    assert child.model == "haiku"
    assert spawn_role(p, p.roles["build"], "explore").tools == ("sh", "spawn")


def test_spawn_role_uses_parent_as_given() -> None:
    p = merge().policy
    narrowed = Role(name="build", model="opus", tools=("sh",), can_spawn=("mechanic",))
    child = spawn_role(p, narrowed, "mechanic")
    assert child.tools == ("sh",)


def test_spawn_role_errors() -> None:
    p = merge().policy
    with pytest.raises(PolicyError):
        spawn_role(p, p.roles["build"], "ghost")
    with pytest.raises(PolicyError):
        spawn_role(p, p.roles["explore"], "mechanic")


def test_load_policy_reads_every_level(tmp_path: Path) -> None:
    home, project = tmp_path / "home", tmp_path / "repo"
    org = write(tmp_path / "org.toml", '[budget]\ndollars = 20\n\n[permissions]\n"sh: git push*" = "deny"\n')
    write(home / ".synthic" / "policy.toml", '[role.build]\nmodel = "sonnet"\neffort = "medium"\n')
    write(project / ".synthic" / "policy.toml", '[role.build]\neffort = "low"\n')
    task = write(tmp_path / "task.toml", '[budget]\ndollars = 5\n')
    loaded = load_policy(
        project_root=project,
        task_file=task,
        task_overrides={"role": {"build": {"model": "haiku"}}},
        trust_project=True,
        env={"SYNTHIC_ORG_POLICY": str(org)},
        home=home,
    )
    p = loaded.policy
    assert p.roles["build"].model == "haiku"
    assert p.roles["build"].effort == "low"
    assert p.budget.dollars == 5.0
    assert decide(p, "sh", "git push")[0] == "deny"
    assert loaded.sources["role.build.model"] == "task"
    assert loaded.sources["role.build.effort"] == "project"


def test_load_policy_without_files(tmp_path: Path) -> None:
    loaded = load_policy(project_root=tmp_path, env={}, home=tmp_path)
    assert loaded.policy.roles["build"].model == "opus"
    assert loaded.warnings == []


def test_load_policy_untrusted_project(tmp_path: Path) -> None:
    write(tmp_path / ".synthic" / "policy.toml", '[role.build]\nmodel = "haiku"\n\n[permissions]\n"edit: .github/**" = "deny"\n')
    loaded = load_policy(project_root=tmp_path, env={}, home=tmp_path / "home")
    assert loaded.policy.roles["build"].model == "opus"
    assert decide(loaded.policy, "edit", ".github/x.yml")[0] == "deny"
    assert len(loaded.warnings) == 1 and "policy.toml" in loaded.warnings[0]


def test_load_policy_unreadable_untrusted_project_is_a_warning(tmp_path: Path) -> None:
    write(tmp_path / ".synthic" / "policy.toml", "this is = = not toml\n")
    loaded = load_policy(project_root=tmp_path, env={}, home=tmp_path / "home")
    assert len(loaded.warnings) == 1 and "deny rules are not in force" in loaded.warnings[0]


def test_load_policy_missing_files_are_errors(tmp_path: Path) -> None:
    with pytest.raises(PolicyError) as info:
        load_policy(
            task_file=tmp_path / "nope.toml",
            env={"SYNTHIC_ORG_POLICY": str(tmp_path / "org.toml")},
            home=tmp_path,
        )
    assert len(info.value.problems) == 2


def test_load_policy_reports_file_and_merge_problems_together(tmp_path: Path) -> None:
    write(tmp_path / ".synthic" / "policy.toml", 'initial_role = "ghost"\n')
    with pytest.raises(PolicyError) as info:
        load_policy(task_file=tmp_path / "nope.toml", env={}, home=tmp_path)
    text = "\n".join(info.value.problems)
    assert "file not found" in text and "initial_role 'ghost'" in text


def test_load_policy_loads_packaged_defaults_cleanly(tmp_path: Path) -> None:
    loaded = load_policy(env={}, home=tmp_path)
    assert loaded.warnings == []
    assert loaded.policy.models["opus"].fallback is None
    assert set(loaded.sources.values()) == {"defaults"}


def test_never_send_and_mask_must_be_regular_expressions() -> None:
    problems = problems_of(Level("task", {"never_send": ["*.pem", "ok"], "journal": {"mask": ["[unclosed"]}}))
    assert len(problems) == 2
    assert "never_send pattern '*.pem'" in problems[0] and "regular expression" in problems[0]
    assert "journal.mask pattern '[unclosed'" in problems[1]
