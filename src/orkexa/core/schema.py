"""Conversion of tool and output JSON schemas into provider dialects.

Tool input schemas come from Pydantic (:class:`orkexa.Tool`): optional parameters
have a ``default`` and are not ``required``; nested models are ``$defs`` + ``$ref``.

* Anthropic accepts them unchanged.
* OpenAI strict function calling (and strict ``json_schema`` output) requires every
  property to be required and ``additionalProperties: false``; optional parameters
  therefore become *required and nullable*. A ``null`` the model sends for such a
  parameter is removed again (:func:`drop_null_optionals`) before local validation, so
  the Python default applies. A parameter that is nullable *and* has a non-``None``
  default is ambiguous (``null`` could mean "``None``" or "default") and raises
  :class:`~orkexa.errors.ToolSchemaError`.
* Gemini receives ``$ref``-free schemas (:func:`inline_refs`); recursive models raise
  :class:`~orkexa.errors.ToolSchemaError`.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..errors import ToolSchemaError

__all__ = ["drop_null_optionals", "inline_refs", "strict_schema"]

_NULL = {"type": "null"}


def _is_nullable(schema: dict[str, Any]) -> bool:
    if schema.get("type") == "null" or (isinstance(schema.get("type"), list) and "null" in schema["type"]):
        return True
    return any(isinstance(option, dict) and _is_nullable(option) for option in schema.get("anyOf", ()))


def strict_schema(schema: dict[str, Any], *, name: str) -> dict[str, Any]:
    """``schema`` in OpenAI's strict structured-output dialect (a new dict).

    Raises:
        ToolSchemaError: an optional nullable property has a non-``None`` default.
    """
    result = deepcopy(schema)
    _strict(result, name, "")
    return result


def _strict(node: Any, name: str, path: str) -> None:
    if isinstance(node, list):
        for item in node:
            _strict(item, name, path)
        return
    if not isinstance(node, dict):
        return
    node.pop("default", None)
    node.pop("title", None)
    properties = node.get("properties")
    if node.get("type") == "object" or isinstance(properties, dict):
        node["additionalProperties"] = False
        properties = properties if isinstance(properties, dict) else {}
        node["properties"] = properties
        required = set(node.get("required", ()))
        for key, prop in list(properties.items()):
            if not isinstance(prop, dict):
                continue
            if key not in required:
                if _is_nullable(prop) and prop.get("default") is not None:
                    raise ToolSchemaError(
                        f"{name}: parameter {path}{key!r} is nullable and has the default "
                        f"{prop['default']!r}; OpenAI strict mode cannot distinguish null from "
                        "'use the default'. Use a default of None or make the parameter non-nullable."
                    )
                description = prop.pop("description", None)
                prop.pop("default", None)
                if not _is_nullable(prop):
                    prop = {"anyOf": [prop, dict(_NULL)]}
                if description is not None:
                    prop["description"] = description
                properties[key] = prop
        node["required"] = list(properties)
        for key, prop in properties.items():
            _strict(prop, name, f"{path}{key}.")
    for key, value in node.items():
        if key != "properties":
            _strict(value, name, path)


def drop_null_optionals(arguments: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Remove ``null`` values of non-required properties (see the module docstring)."""
    defs = schema.get("$defs", {})
    return _drop(arguments, schema, defs, 0)  # type: ignore[no-any-return]


def _resolve(node: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        target = defs.get(ref.removeprefix("#/$defs/"))
        if isinstance(target, dict):
            return target
    return node


def _drop(value: Any, node: Any, defs: dict[str, Any], depth: int) -> Any:
    if depth > 64 or not isinstance(node, dict):
        return value
    node = _resolve(node, defs)
    if isinstance(value, dict) and isinstance(node.get("properties"), dict):
        required = set(node.get("required", ()))
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None and key not in required and key in node["properties"]:
                continue
            cleaned[key] = _drop(item, node["properties"].get(key), defs, depth + 1)
        return cleaned
    if isinstance(value, list) and isinstance(node.get("items"), dict):
        return [_drop(item, node["items"], defs, depth + 1) for item in value]
    for option in node.get("anyOf", ()):
        if isinstance(option, dict) and not _is_nullable(option) and value is not None:
            resolved = _resolve(option, defs)
            if (isinstance(value, dict) and "properties" in resolved) or (
                isinstance(value, list) and "items" in resolved
            ):
                return _drop(value, resolved, defs, depth + 1)
    return value


def inline_refs(schema: dict[str, Any], *, name: str) -> dict[str, Any]:
    """``schema`` with every local ``$ref`` replaced by its definition and titles removed.

    Raises:
        ToolSchemaError: the schema is recursive.
    """
    defs = schema.get("$defs", {})

    def walk(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [walk(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str):
            key = ref.removeprefix("#/$defs/")
            if key in seen:
                raise ToolSchemaError(
                    f"{name}: recursive schema ({' -> '.join((*seen, key))}) is not supported"
                )
            if key not in defs:
                raise ToolSchemaError(f"{name}: unresolvable $ref {ref!r}")
            merged = {**defs[key], **{k: v for k, v in node.items() if k != "$ref"}}
            return walk(merged, (*seen, key))
        return {k: walk(v, seen) for k, v in node.items() if k not in ("$defs", "title")}

    result: dict[str, Any] = walk(schema, ())
    return result
