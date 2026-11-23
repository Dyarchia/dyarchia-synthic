"""A small JSON Schema validator for structured delivery (``role.<name>.result_schema``).

Standard library only, so it covers the keywords a delivery schema needs and
refuses, when the policy loads, any keyword it would not check: a schema that
silently passes everything would be worse than none.

Checked: ``type`` (a name or a list of names: ``object``, ``array``,
``string``, ``integer``, ``number``, ``boolean``, ``null``), ``enum``,
``const``, ``properties``, ``required``, ``additionalProperties`` (true, false
or a schema), ``items`` (one schema for every item), ``minItems``,
``maxItems``, ``uniqueItems``, ``minLength``, ``maxLength``, ``pattern``
(Python ``re``, searched anywhere in the string, as JSON Schema does),
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum`` (numbers),
``allOf``, ``anyOf``, ``oneOf``, ``not``. Read and ignored, being annotations:
``title``, ``description``, ``default``, ``examples``, ``format``,
``$schema``, ``$id``, ``$comment``. Anything else (``$ref``, ``patternProperties``,
``if``...) is a policy problem.
"""

from __future__ import annotations

import json
import re
from typing import Any

CHECKED = frozenset({
    "type", "enum", "const", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "uniqueItems", "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum",
    "exclusiveMaximum", "allOf", "anyOf", "oneOf", "not",
})
ANNOTATIONS = frozenset({"title", "description", "default", "examples", "format", "$schema", "$id", "$comment"})
TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
MAX_ERRORS = 20


def problems(schema: Any, where: str = "result_schema") -> list[str]:
    """What makes ``schema`` unusable: not an object, an unsupported keyword, a malformed value."""
    if not isinstance(schema, dict):
        return [f"{where} must be a table (a JSON Schema object)"]
    found: list[str] = []
    for key, value in schema.items():
        here = f"{where}.{key}"
        if key in ANNOTATIONS:
            continue
        if key not in CHECKED:
            found.append(f"{here}: the keyword {key!r} is not supported (see synthic/schema.py)")
            continue
        match key:
            case "type":
                names = value if isinstance(value, list) else [value]
                if not names or any(n not in TYPES for n in names):
                    found.append(f"{here} must name one or more of {', '.join(TYPES)}")
            case "properties":
                if not isinstance(value, dict):
                    found.append(f"{here} must be a table of schemas")
                else:
                    for name, sub in value.items():
                        found += problems(sub, f"{here}.{name}")
            case "required":
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    found.append(f"{here} must be a list of property names")
            case "additionalProperties":
                if not isinstance(value, bool):
                    found += problems(value, here)
            case "items" | "not":
                found += problems(value, here)
            case "allOf" | "anyOf" | "oneOf":
                if not isinstance(value, list) or not value:
                    found.append(f"{here} must be a non-empty list of schemas")
                else:
                    for index, sub in enumerate(value):
                        found += problems(sub, f"{here}[{index}]")
            case "enum":
                if not isinstance(value, list) or not value:
                    found.append(f"{here} must be a non-empty list")
            case "pattern":
                try:
                    re.compile(value)
                except (re.error, TypeError) as error:
                    found.append(f"{here} is not a valid regular expression: {error}")
            case "minItems" | "maxItems" | "minLength" | "maxLength":
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    found.append(f"{here} must be a non-negative integer")
            case "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum":
                if isinstance(value, bool) or not isinstance(value, int | float):
                    found.append(f"{here} must be a number")
            case "uniqueItems":
                if not isinstance(value, bool):
                    found.append(f"{here} must be true or false")
    return found


def _is(value: Any, kind: str) -> bool:
    match kind:
        case "object":
            return isinstance(value, dict)
        case "array":
            return isinstance(value, list)
        case "string":
            return isinstance(value, str)
        case "boolean":
            return isinstance(value, bool)
        case "null":
            return value is None
        case "integer":
            return not isinstance(value, bool) and (
                isinstance(value, int) or (isinstance(value, float) and value.is_integer())
            )
        case "number":
            return not isinstance(value, bool) and isinstance(value, int | float)
    return False


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def validate(value: Any, schema: dict[str, Any], where: str = "result") -> list[str]:
    """Every way ``value`` breaks ``schema``, one line each naming the place (at most ``MAX_ERRORS``)."""
    errors: list[str] = []
    _check(value, schema, where, errors)
    if len(errors) > MAX_ERRORS:
        errors = [*errors[:MAX_ERRORS], f"... and {len(errors) - MAX_ERRORS} more"]
    return errors


def _check(value: Any, schema: dict[str, Any], where: str, errors: list[str]) -> None:
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is(value, n) for n in names):
            errors.append(f"{where}: expected {' or '.join(names)}, got {_kind(value)}")
            return
    if "const" in schema and _canonical(value) != _canonical(schema["const"]):
        errors.append(f"{where}: must be {json.dumps(schema['const'])}")
    if "enum" in schema and _canonical(value) not in {_canonical(v) for v in schema["enum"]}:
        errors.append(f"{where}: must be one of {', '.join(json.dumps(v) for v in schema['enum'])}")
    if isinstance(value, dict):
        _object(value, schema, where, errors)
    elif isinstance(value, list):
        _array(value, schema, where, errors)
    elif isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{where}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{where}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{where}: does not match the pattern {schema['pattern']!r}")
    elif _is(value, "number"):
        for key, fails, says in (
            ("minimum", lambda v, b: v < b, "less than"),
            ("maximum", lambda v, b: v > b, "more than"),
            ("exclusiveMinimum", lambda v, b: v <= b, "at most"),
            ("exclusiveMaximum", lambda v, b: v >= b, "at least"),
        ):
            if key in schema and fails(value, schema[key]):
                errors.append(f"{where}: {value} is {says} {schema[key]}")
    for sub in schema.get("allOf", []):
        _check(value, sub, where, errors)
    if "anyOf" in schema and not any(not validate(value, sub, where) for sub in schema["anyOf"]):
        errors.append(f"{where}: matches none of the anyOf schemas")
    if "oneOf" in schema:
        matches = sum(not validate(value, sub, where) for sub in schema["oneOf"])
        if matches != 1:
            errors.append(f"{where}: matches {matches} of the oneOf schemas, not exactly one")
    if "not" in schema and not validate(value, schema["not"], where):
        errors.append(f"{where}: matches the schema under not")


def _object(value: dict[str, Any], schema: dict[str, Any], where: str, errors: list[str]) -> None:
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in value:
            errors.append(f"{where}: missing the required property {name!r}")
    extra = schema.get("additionalProperties", True)
    for name, item in value.items():
        here = f"{where}.{name}"
        if name in properties:
            _check(item, properties[name], here, errors)
        elif extra is False:
            errors.append(f"{where}: the property {name!r} is not allowed")
        elif isinstance(extra, dict):
            _check(item, extra, here, errors)


def _array(value: list[Any], schema: dict[str, Any], where: str, errors: list[str]) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{where}: fewer than {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{where}: more than {schema['maxItems']} items")
    if schema.get("uniqueItems") and len({_canonical(v) for v in value}) != len(value):
        errors.append(f"{where}: items are not unique")
    if isinstance(items := schema.get("items"), dict):
        for index, item in enumerate(value):
            _check(item, items, f"{where}[{index}]", errors)


def _kind(value: Any) -> str:
    """The JSON type name of ``value``, the narrowest one for numbers."""
    for name in TYPES:
        if _is(value, name):
            return name
    return type(value).__name__
