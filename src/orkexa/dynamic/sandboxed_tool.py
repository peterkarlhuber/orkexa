"""A Tool whose code runs in a sandbox instead of the host process."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from ..errors import ToolDefinitionError, ToolExecutionError
from ..tools import _TOOL_NAME, Tool, _check_schema_supported
from .sandbox import Sandbox

__all__ = ["SandboxedTool"]

_MAX_PAYLOAD_CHARS = 100_000


class SandboxedTool(Tool):
    """Tool adapter for generated code.

    The schema was derived inside the sandbox (by ``@Tool``, from the code's
    type hints and docstring) and is passed in as JSON. Every call, both
    :meth:`execute` and a direct Python call, starts a new sandbox run; the
    code is never imported into the host process. It can be added to any
    agent with :meth:`ClaudeAgent.add_tool <claude_agent.ClaudeAgent.add_tool>`.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None,
        code: str,
        function_name: str,
        sandbox: Sandbox,
        strict: bool = True,
    ) -> None:
        super().__init__(name=name, strict=strict)
        if not _TOOL_NAME.fullmatch(name):
            raise ToolDefinitionError(f"Invalid tool name: {name!r}")
        if not description.strip():
            raise ToolDefinitionError(f"{name}: missing description")
        if not isinstance(input_schema, dict) or input_schema.get("type") != "object":
            raise ToolDefinitionError(f"{name}: input schema must be a JSON object schema")
        schema = deepcopy(input_schema)
        _check_schema_supported(schema, name)
        self.description = description
        self.input_schema = schema
        self.output_schema = deepcopy(output_schema)
        self.code = code
        self.function_name = function_name
        self.sandbox = sandbox
        self.function = self._call_in_sandbox

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} ({type(self.sandbox).__name__})>"

    def _call_in_sandbox(self, **kwargs: Any) -> str:
        return self.execute(kwargs)

    def _check_payload(self, payload: dict[str, Any]) -> None:
        # Cheap structural check in the host; the sandbox validates strictly
        # with the same Pydantic model a hand-written tool would use.
        if not isinstance(payload, dict):
            raise ToolExecutionError(f"{self.name}: tool input must be a JSON object")
        assert self.input_schema is not None
        properties = self.input_schema.get("properties", {})
        unknown = sorted(set(payload) - set(properties))
        missing = sorted(set(self.input_schema.get("required", [])) - set(payload))
        if unknown or missing:
            raise ToolExecutionError(
                f"{self.name}: invalid tool inputs (unknown: {unknown}, missing: {missing})"
            )
        try:
            size = len(json.dumps(payload))
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError(f"{self.name}: tool input is not JSON-serializable: {exc}") from exc
        if size > _MAX_PAYLOAD_CHARS:
            raise ToolExecutionError(f"{self.name}: tool input too large ({size} characters)")

    def execute(self, payload: dict[str, Any]) -> str:
        self._check_payload(payload)
        return self.sandbox.call(self.code, self.function_name, payload)
