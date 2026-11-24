"""Structured delivery: the validator, the policy, ``done`` with a ``result``, and how it is exposed."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from test_agents import RoutedProvider, resume_run
from test_loop import MODEL, done, go, make_policy, project, records, results

from synthic.agents import Runtime
from synthic.cli import main
from synthic.policy import Level, PolicyError, factory_defaults, merge_levels
from synthic.schema import problems, validate
from synthic.testing import ScriptedProvider, reply, tool_use

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string", "pattern": r"\.py$"}, "minItems": 1},
        "count": {"type": "integer", "minimum": 0},
        "level": {"enum": ["low", "high"]},
    },
    "required": ["files", "count"],
    "additionalProperties": False,
}
WITH_SCHEMA = {"role": {"lead": {"result_schema": SCHEMA}}}


def deliver(result: Any, summary: str = "Done.") -> Any:
    return reply(tool_use("done", {"summary": summary, "result": result}))


def test_the_validator_reports_every_mismatch() -> None:
    assert validate({"files": ["a.py"], "count": 2, "level": "low"}, SCHEMA) == []
    errors = validate({"files": ["a.txt"], "count": -1, "level": "mid", "extra": 1}, SCHEMA)
    assert errors == [
        "result.files[0]: does not match the pattern '\\\\.py$'",
        "result.count: -1 is less than 0",
        'result.level: must be one of "low", "high"',
        "result: the property 'extra' is not allowed",
    ]
    assert validate([], SCHEMA) == ["result: expected object, got array"]
    assert validate({}, SCHEMA) == [
        "result: missing the required property 'files'",
        "result: missing the required property 'count'",
    ]
    assert validate(True, {"type": "integer"}) == ["result: expected integer, got boolean"]
    assert validate(3.0, {"type": "integer"}) == []
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate([1, 1], {"uniqueItems": True}) == ["result: items are not unique"]
    assert validate("x", {"oneOf": [{"type": "string"}, {"minLength": 1}]})[0].endswith("not exactly one")
    assert validate(5, {"anyOf": [{"maximum": 1}, {"minimum": 9}]}) == ["result: matches none of the anyOf schemas"]
    assert validate("a", {"not": {"const": "a"}}) == ["result: matches the schema under not"]
    assert len(validate(list(range(50)), {"items": {"type": "string"}})) == 21


def test_unsupported_keywords_are_policy_problems() -> None:
    assert problems(SCHEMA) == []
    found = problems({"type": "object", "properties": {"a": {"$ref": "#/x"}}, "if": {}, "format": "date"})
    assert found == [
        "result_schema.properties.a.$ref: the keyword '$ref' is not supported (see synthic/schema.py)",
        "result_schema.if: the keyword 'if' is not supported (see synthic/schema.py)",
    ]
    assert problems({"type": "date"}) and problems({"pattern": "("}) and problems("x")


def test_policy_checks_the_schema_and_replaces_it_whole(tmp_path: Path) -> None:
    with pytest.raises(PolicyError) as error:
        make_policy(tmp_path, {"role": {"lead": {"result_schema": {"type": "thing"}}}})
    assert error.value.problems[0].startswith("role.lead.result_schema.type")
    user = {
        "initial_role": "r",
        "models": {"m": MODEL},
        "role": {"r": {"model": "m", "result_schema": {"type": "object", "required": ["a"]}}},
    }
    task = {"role": {"r": {"result_schema": {"type": "string"}}}}
    merged = merge_levels([Level("defaults", factory_defaults()), Level("user", user), Level("task", task)])
    assert merged.policy.roles["r"].result_schema == {"type": "string"}
    assert make_policy(tmp_path).policy.roles["lead"].result_schema is None


def test_done_requires_a_matching_result_and_the_agent_fixes_it(tmp_path: Path) -> None:
    runtime, result, provider = go(
        tmp_path,
        [
            reply(tool_use("done", {"summary": "Done."})),
            deliver({"files": ["a.txt"], "count": 1}),
            deliver({"files": ["a.py"], "count": 1}),
        ],
        WITH_SCHEMA,
    )
    assert result.status == "delivered" and result.result == {"files": ["a.py"], "count": 1}
    missing, wrong = results(provider.requests[1])[0], results(provider.requests[2])[0]
    assert missing["is_error"] and "`result` is required" in missing["content"]
    assert wrong["is_error"] and "- result.files[0]: does not match the pattern" in wrong["content"]
    definition = next(t for t in provider.requests[0].tools if t["name"] == "done")
    assert definition["input_schema"]["properties"]["result"] == SCHEMA
    assert "result" not in definition["input_schema"]["required"]
    end = [r for r in records(runtime) if r["kind"] == "done"][-1]
    assert end["data"]["result"] == {"files": ["a.py"], "count": 1}


def test_a_failed_delivery_needs_no_result_and_roles_without_schema_pay_nothing(tmp_path: Path) -> None:
    _, result, _ = go(tmp_path, [reply(tool_use("done", {"summary": "No.", "status": "failed"}))], WITH_SCHEMA)
    assert result.status == "failed" and result.result is None
    (tmp_path / "plain").mkdir()
    _, result, provider = go(tmp_path / "plain", [done()])
    definition = next(t for t in provider.requests[0].tools if t["name"] == "done")
    assert "result" not in definition["input_schema"]["properties"] and result.result is None


def test_a_child_result_reaches_the_parent(tmp_path: Path) -> None:
    task = {"role": {"worker": {"result_schema": {"type": "object", "properties": {"n": {"type": "integer"}}}}}}
    provider = RoutedProvider(
        {
            "PARENT": [
                reply(tool_use("spawn", {"role": "worker", "task": "CHILD"})),
                reply(tool_use("agent", {"action": "wait", "id": "a1"})),
                done(),
            ],
            "CHILD": [deliver({"n": 7})],
        }
    )
    async def scenario() -> Any:
        runtime = Runtime(make_policy(tmp_path, task), project(tmp_path), provider)
        try:
            return await runtime.run("PARENT")
        finally:
            await runtime.close()

    assert asyncio.run(scenario()).status == "delivered"
    notice = results(provider.requests["PARENT"][2])[0]["content"]
    assert 'Result: {"n": 7}' in notice


def test_resume_of_an_ended_run_returns_the_result(tmp_path: Path) -> None:
    runtime, _, _ = go(tmp_path, [deliver({"files": ["a.py"], "count": 0})], WITH_SCHEMA)
    result, provider = resume_run(tmp_path, runtime.run_id, [], task=WITH_SCHEMA)
    assert result.result == {"files": ["a.py"], "count": 0} and provider.requests == []


def test_cli_result_schema_flag_and_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYNTHIC_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("SYNTHIC_ORG_POLICY", raising=False)
    folder = tmp_path / "proj"
    folder.mkdir()
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object", "required": ["ok"]}), encoding="utf-8")
    policy = tmp_path / "policy.toml"
    policy.write_text(
        f'[journal]\ndir = "{(tmp_path / "runs").as_posix()}"\n[memory]\nenabled = false\n[verify]\nreread = false\n', encoding="utf-8"
    )
    provider = ScriptedProvider([deliver({"ok": True})])
    args = ["run", "Do it.", "--json", "--cwd", str(folder), "--policy", str(policy), "--result-schema", str(schema)]
    assert main(args, provider_factory=lambda: provider) == 0
    final = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert final["kind"] == "result" and final["result"] == {"ok": True}
    definition = next(t for t in provider.requests[0].tools if t["name"] == "done")
    assert definition["input_schema"]["properties"]["result"] == {"type": "object", "required": ["ok"]}
    schema.write_text("{not json", encoding="utf-8")
    assert main(args, provider_factory=lambda: provider) == 2
