"""Runs *inside* the sandbox. Do not import this module in the host process.

Protocol: one JSON request on stdin, one JSON response on stdout.

Request keys: ``mode`` ("validate" or "call"), ``function_name``, ``payload``
(call only), ``limits`` (optional: ``cpu_seconds``, ``memory_mb``,
``max_output_chars``). The working directory holds ``tool.py`` and, for
"validate", ``test_tool.py``.

This file only uses the standard library until the limits are applied; it then
imports ``orkexa.tools.Tool`` (without the package's ``__init__``) to
derive the tool schema exactly like the host would for a hand-written tool.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
import types
from typing import Any

_MAX_DIAGNOSTIC_CHARS = 4000


def _apply_limits(limits: dict[str, Any]) -> None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows: only the host's timeout applies
        return

    def limit(name: str, value: int) -> None:
        kind = getattr(resource, name, None)
        if kind is None:  # pragma: no cover - platform without this limit
            return
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(kind, (value, value))

    cpu = limits.get("cpu_seconds")
    if cpu:
        limit("RLIMIT_CPU", int(cpu))
    memory = limits.get("memory_mb")
    if memory and sys.platform != "darwin":  # macOS does not support RLIMIT_AS reliably
        limit("RLIMIT_AS", int(memory) * 1024 * 1024)
    limit("RLIMIT_FSIZE", 1024 * 1024)
    limit("RLIMIT_NPROC", 0)
    limit("RLIMIT_CORE", 0)


def _short(text: str) -> str:
    return text if len(text) <= _MAX_DIAGNOSTIC_CHARS else text[-_MAX_DIAGNOSTIC_CHARS:]


def _error(exc: BaseException) -> str:
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    cause = exc.__cause__
    if cause is not None:
        message += f" (caused by {type(cause).__name__}: {cause})"
    return _short(message)


def _import_tool_class() -> Any:
    # Register orkexa as a bare namespace so that importing orkexa.tools does
    # not run orkexa/__init__.py, which would import every subsystem. The
    # directory is this file's package, so the runner works for an installed
    # orkexa, a source checkout and the read-only Docker mount alike.
    package_dir = Path(__file__).resolve().parents[1]
    package = types.ModuleType("orkexa")
    package.__path__ = [str(package_dir)]
    sys.modules.setdefault("orkexa", package)
    from orkexa.tools import Tool

    return Tool


def _load_tool(function_name: str) -> Any:
    Tool = _import_tool_class()

    import tool as module  # type: ignore[import-not-found]

    function = getattr(module, function_name, None)
    if function is None:
        raise LookupError(f"tool.py does not define {function_name!r}")
    return Tool(function)


def _validate(function_name: str) -> dict[str, Any]:
    tool = _load_tool(function_name)
    response: dict[str, Any] = {
        "ok": True,
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "output_schema": tool.output_schema,
    }
    import test_tool  # type: ignore[import-not-found]

    results = []
    for name in sorted(vars(test_tool)):
        test = getattr(test_tool, name)
        if not (name.startswith("test_") and callable(test)):
            continue
        try:
            test()
        except BaseException as exc:
            results.append({"name": name, "ok": False, "error": _error(exc)})
        else:
            results.append({"name": name, "ok": True, "error": None})
    response["tests"] = results
    response["ok"] = bool(results) and all(r["ok"] for r in results)
    return response


def _call(function_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    tool = _load_tool(function_name)
    return {"ok": True, "result": tool.execute(payload)}


def main() -> None:
    real_stdout = sys.stdout
    request = json.loads(sys.stdin.read())
    _apply_limits(request.get("limits") or {})
    # -I removes the working directory and the user's site-packages from sys.path.
    sys.path.insert(0, os.getcwd())

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            if request["mode"] == "validate":
                response = _validate(request["function_name"])
            elif request["mode"] == "call":
                response = _call(request["function_name"], request["payload"])
            else:
                raise ValueError(f"unknown mode {request['mode']!r}")
    except BaseException as exc:
        response = {"ok": False, "error": _error(exc)}
    response["output"] = _short(captured.getvalue())
    real_stdout.write(json.dumps(response))
    real_stdout.flush()


if __name__ == "__main__":
    main()
