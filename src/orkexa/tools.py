"""Strict, automatically documented Python tools for every provider.

A tool is declared with ``@Tool`` (or ``@Tool()``) and a Google-style docstring:
one or more description sentences, ``Args:`` for each parameter, ``Returns:``.
Parameter and return type annotations are mandatory. Unsupported signatures
and schemas fail immediately during decoration.

A tool that returns ``ToolContent`` answers with content blocks (images,
documents, search results) instead of text or JSON. A tool that raises
:class:`~orkexa.ToolResultError` sends its message back to the model as an
error result, so the model can react (for example to an unknown id).

:meth:`Tool.definition` is the provider-neutral definition; provider adapters
convert its JSON schema into their own dialect (see :mod:`orkexa.core.schema`).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from functools import update_wrapper
import inspect
import json
import re
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, create_model

from .content import ContentBlock, ToolContent, to_tool_content
from .core.types import ToolDefinition
from .errors import ToolDefinitionError, ToolExecutionError, ToolInputError, ToolResultError

__all__ = ["Tool", "ToolOutput"]

ToolOutput = str | list[ContentBlock]
"""What :meth:`Tool.execute` returns: text/JSON, or blocks for a ``-> ToolContent`` tool."""

_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _parse_tool_docstring(function: Callable[..., Any]) -> tuple[str, dict[str, str], str]:
    """Split a Google-style docstring into description, Args entries and Returns."""
    doc = inspect.getdoc(function)
    if not doc:
        raise ToolDefinitionError(f"{function.__name__}: missing docstring")

    sections: dict[str, list[str]] = {"description": []}
    current = "description"
    for line in doc.splitlines():
        heading = re.fullmatch(r"([A-Za-z][A-Za-z ]*):", line.strip())
        if heading and not line.startswith((" ", "\t")):
            current = heading.group(1).lower()
            sections.setdefault(current, [])
        else:
            sections.setdefault(current, []).append(line)

    description = " ".join(line.strip() for line in sections["description"] if line.strip())
    if not description:
        raise ToolDefinitionError(f"{function.__name__}: missing tool description")

    arg_lines = sections.get("args", sections.get("arguments", sections.get("parameters", [])))
    descriptions: dict[str, str] = {}
    last: str | None = None
    for line in arg_lines:
        if not line.strip():
            continue
        matched = re.fullmatch(r"\s+([A-Za-z_]\w*)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if matched:
            key, value = matched.groups()
            if key in descriptions:
                raise ToolDefinitionError(f"{function.__name__}: duplicate Args entry {key!r}")
            descriptions[key] = value.strip()
            last = key
        elif last is not None and line.startswith((" ", "\t")):
            descriptions[last] += " " + line.strip()
        else:
            raise ToolDefinitionError(f"{function.__name__}: cannot parse Args line: {line!r}")

    returns = " ".join(line.strip() for line in sections.get("returns", []) if line.strip())
    if not returns:
        raise ToolDefinitionError(f"{function.__name__}: missing Returns: description")
    return description, descriptions, returns


def _check_schema_supported(schema: dict[str, Any], name: str) -> None:
    """Reject JSON object maps/unspecified objects unsupported by strict tool use."""

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                # Pydantic BaseModel defaults to extra='ignore'; the API's strict
                # JSON Schema requires additionalProperties=false. Normalize
                # well-defined objects, but reject arbitrary key/value maps.
                extra = value.get("additionalProperties", False)
                if extra is not False:
                    raise ToolDefinitionError(
                        f"{name}: object schema must have additionalProperties=false. "
                        "Use a Pydantic BaseModel instead of an arbitrary dict/Any."
                    )
                value["additionalProperties"] = False
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


class Tool:
    """Decorator + tool adapter: ``@Tool``, ``@Tool()``, or ``@Tool(name='alias')``.

    Derives a strict JSON input schema from Python type annotations,
    descriptions from a Google-style docstring, and checks runtime I/O via
    Pydantic. Never manually specify a JSON schema.

    The decorated object stays callable like the original function; use
    :meth:`execute` to run it with validated, JSON-shaped tool input.
    """

    def __init__(
        self,
        function: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        strict: bool = True,
    ) -> None:
        self.name = name
        self.strict = strict
        self.function: Callable[..., Any] | None = None
        self.description: str | None = None
        self.return_description: str | None = None
        self.input_schema: dict[str, Any] | None = None
        self.output_schema: dict[str, Any] | None = None
        self._args_model: type[BaseModel] | None = None
        self._return_adapter: TypeAdapter[Any] | None = None
        self.returns_content = False
        self.is_async = False
        if function is not None:
            self._prepare(function)

    def _prepare(self, function: Callable[..., Any]) -> None:
        if not inspect.isfunction(function):
            raise ToolDefinitionError("@Tool requires a plain Python function")
        if inspect.isasyncgenfunction(function):
            raise ToolDefinitionError(f"{function.__name__}: async generators are not supported")

        name = self.name or function.__name__
        if not _TOOL_NAME.fullmatch(name):
            raise ToolDefinitionError(f"Invalid tool name: {name!r}")
        description, descriptions, returns = _parse_tool_docstring(function)
        signature = inspect.signature(function)
        try:
            hints = get_type_hints(function, include_extras=True)
        except Exception as exc:
            raise ToolDefinitionError(f"{name}: unresolved type annotations: {exc}") from exc

        fields: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                raise ToolDefinitionError(
                    f"{name}: parameter {parameter.name!r} must be keyword-compatible; "
                    "*args, **kwargs and positional-only parameters are unsupported"
                )
            if parameter.name not in hints or hints[parameter.name] in (Any, object):
                raise ToolDefinitionError(f"{name}: missing or vague type hint for {parameter.name!r}")
            if not descriptions.get(parameter.name, "").strip():
                raise ToolDefinitionError(f"{name}: missing Args: documentation for {parameter.name!r}")
            default = ... if parameter.default is inspect.Parameter.empty else parameter.default
            fields[parameter.name] = (
                hints[parameter.name],
                Field(default=default, description=descriptions[parameter.name]),
            )

        unexpected = descriptions.keys() - signature.parameters.keys()
        if unexpected:
            raise ToolDefinitionError(f"{name}: Args documents nonexistent parameters: {sorted(unexpected)}")
        if "return" not in hints or hints["return"] in (Any, object):
            raise ToolDefinitionError(f"{name}: explicit return type annotation is required")

        try:
            args_model = create_model(
                f"_ToolArgs_{name}",
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
            schema = args_model.model_json_schema()
            _check_schema_supported(schema, name)
            returns_content = hints["return"] == ToolContent
            return_adapter: TypeAdapter[Any] = TypeAdapter(hints["return"])
            output_schema = return_adapter.json_schema()  # Fail if unrepresentable.
        except ToolDefinitionError:
            raise
        except Exception as exc:
            raise ToolDefinitionError(f"{name}: cannot derive Pydantic schemas: {exc}") from exc

        self.name = name
        self.function = function
        self.description = f"{description}\n\nReturns: {returns}"
        self.return_description = returns
        self.input_schema = schema
        self.output_schema = output_schema
        self._args_model = args_model
        self._return_adapter = return_adapter
        self.returns_content = returns_content
        self.is_async = inspect.iscoroutinefunction(function)
        update_wrapper(self, function)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Decorating with @Tool() or @Tool(name='...')
        if self.function is None:
            if len(args) != 1 or kwargs or not callable(args[0]):
                raise ToolDefinitionError("Decorator expects exactly one function")
            return type(self)(args[0], name=self.name, strict=self.strict)
        # Normal direct Python call; use .execute() for API calls with validation.
        return self.function(*args, **kwargs)

    def __repr__(self) -> str:
        if self.function is None:
            return f"{type(self).__name__}(name={self.name!r}, strict={self.strict!r})"
        return f"<{type(self).__name__} {self.name!r}>"

    def schema(self) -> dict[str, Any]:
        """Tool definition in the Anthropic Messages API format (as in ``claude-agent``)."""
        if self.function is None:
            raise ToolDefinitionError("Uninitialized Tool decorator")
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": deepcopy(self.input_schema),
            "strict": self.strict,
        }

    def definition(self) -> ToolDefinition:
        """The provider-neutral tool definition the agent runtime sends to its provider."""
        if self.function is None or self.name is None or self.input_schema is None:
            raise ToolDefinitionError("Uninitialized Tool decorator")
        return ToolDefinition(
            name=self.name,
            description=self.description or "",
            input_schema=deepcopy(self.input_schema),
            strict=self.strict,
        )

    def _validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.function is None or self._args_model is None or self._return_adapter is None:
            raise ToolDefinitionError("Uninitialized tool")
        try:
            args = self._args_model.model_validate_json(json.dumps(payload), strict=True)
        except (ValidationError, TypeError, ValueError) as exc:
            raise ToolInputError(f"{self.name}: invalid tool inputs: {exc}") from exc
        return dict(args)  # Preserve nested BaseModel parameters.

    def _serialize_output(self, result: Any) -> ToolOutput:
        assert self._return_adapter is not None
        if self.returns_content:
            try:
                return to_tool_content(result)
            except ValueError as exc:
                raise ToolExecutionError(f"{self.name}: invalid return value: {exc}") from exc
        try:
            validated = self._return_adapter.validate_python(result, strict=True)
            if isinstance(validated, str):
                return validated
            return self._return_adapter.dump_json(validated).decode("utf-8")
        except (ValidationError, ValueError, TypeError) as exc:
            raise ToolExecutionError(f"{self.name}: invalid return value: {exc}") from exc

    def execute(self, payload: dict[str, Any]) -> ToolOutput:
        """Validate tool input/output; serialize the output for the tool result.

        ``payload`` is the JSON object the model produced (``tool_use.input``).
        It is validated in Pydantic's strict *JSON* mode, so values such as
        enums, dates, UUIDs and nested models are accepted in their JSON form.
        ``async def`` tools cannot run here; use :meth:`aexecute`. A
        :class:`~orkexa.ToolResultError` from the handler propagates unchanged.
        """
        if self.is_async:
            raise ToolExecutionError(f"{self.name}: async tool; use an async agent or aexecute()")
        args = self._validate_input(payload)
        assert self.function is not None
        try:
            result = self.function(**args)
        except ToolResultError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{self.name}: handler failed: {exc}") from exc
        return self._serialize_output(result)

    async def aexecute(self, payload: dict[str, Any]) -> ToolOutput:
        """Like :meth:`execute`; awaits ``async def`` tools and runs others in a worker thread."""
        if not self.is_async:
            return await asyncio.to_thread(self.execute, payload)
        args = self._validate_input(payload)
        assert self.function is not None
        try:
            result = await self.function(**args)
        except ToolResultError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"{self.name}: handler failed: {exc}") from exc
        return self._serialize_output(result)
