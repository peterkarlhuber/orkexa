"""Versioned registry of trusted (hand-written) and generated tools."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Literal

from .._storage import make_private_dirs, write_private_text
from ..tools import _TOOL_NAME, Tool
from ._search import keyword_search
from .errors import RegistryError
from .sandbox import DockerSandbox, Sandbox
from .sandboxed_tool import SandboxedTool

__all__ = ["ToolRecord", "ToolRegistry", "ToolStatus"]

logger = logging.getLogger(__name__)

ToolStatus = Literal["pending", "approved", "rejected", "revoked"]


def _sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _valid_name(name: object) -> bool:
    return isinstance(name, str) and _TOOL_NAME.fullmatch(name) is not None


def _check_name(name: object) -> None:
    if not _valid_name(name):
        raise RegistryError(f"Invalid tool name {name!r}: use 1-128 characters of A-Z, a-z, 0-9, '_', '-'")


def _write_json_atomic(file: Path, data: Any) -> None:
    """Write ``data`` as JSON to ``file`` so readers see the old or the new file, never a torn one.

    Missing directories are created with mode ``0700``, the file with ``0600``: records
    hold generated code, specs and requirements that may contain confidential data.
    """
    make_private_dirs(file.parent)
    write_private_text(file, json.dumps(data, indent=2, ensure_ascii=False))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ToolRecord:
    """Metadata of one tool version.

    ``kind="trusted"`` tools are hand-written ``@Tool`` functions that run in the
    host process and are kept in memory only. ``kind="generated"`` tools carry
    their source code and always run in the sandbox.
    """

    name: str
    version: int
    kind: Literal["trusted", "generated"]
    status: ToolStatus
    description: str
    created_at: str
    function_name: str | None = None
    code: str | None = None
    tests: str | None = None
    sha256: str | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    requirement: str | None = None


class ToolRegistry:
    """Tool versions, metadata, approval status and persistence.

    Args:
        path: Directory for generated tools (one JSON file per version). ``None``
            keeps everything in memory.
        sandbox: Sandbox in which approved generated tools are executed.
            Defaults to a :class:`DockerSandbox`, which fails closed without
            Docker. Pass ``SubprocessSandbox(acknowledge_unsafe=True)`` only for
            trusted or reviewed code.
    """

    def __init__(self, path: str | Path | None = None, *, sandbox: Sandbox | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.sandbox = sandbox if sandbox is not None else DockerSandbox()
        self._records: dict[str, list[ToolRecord]] = {}
        self._trusted: dict[str, Tool] = {}
        if self.path is not None:
            self._load()

    # ---- persistence --------------------------------------------------------

    def _file(self, record: ToolRecord) -> Path:
        assert self.path is not None
        _check_name(record.name)
        root = self.path.resolve()
        file = (root / record.name / f"v{int(record.version)}.json").resolve()
        if file.parent.parent != root:
            raise RegistryError(f"Tool record path escapes the registry directory: {record.name!r}")
        return file

    def _save(self, record: ToolRecord) -> None:
        if self.path is None or record.kind != "generated":
            return
        _write_json_atomic(self._file(record), asdict(record))

    def _load(self) -> None:
        assert self.path is not None
        for file in sorted(self.path.glob("*/v*.json")):
            try:
                record = ToolRecord(**json.loads(file.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Skipping unreadable tool record %s: %s", file, exc)
                continue
            if not _valid_name(record.name) or record.name != file.parent.name:
                logger.warning("Skipping tool record %s: name does not match its directory", file)
                continue
            if record.kind != "generated" or record.code is None or record.sha256 != _sha256(record.code):
                logger.warning("Skipping tool record %s: code does not match its stored hash", file)
                continue
            self._records.setdefault(record.name, []).append(record)
        for versions in self._records.values():
            versions.sort(key=lambda r: r.version)

    # ---- registration -------------------------------------------------------

    def register_trusted(self, tool: Tool) -> ToolRecord:
        """Register a hand-written ``@Tool``; it runs in the host process."""
        if not isinstance(tool, Tool) or tool.function is None or tool.name is None:
            raise TypeError("register_trusted() expects an initialized @Tool")
        if tool.name in self._records:
            raise RegistryError(f"Tool name already registered: {tool.name}")
        record = ToolRecord(
            name=tool.name,
            version=1,
            kind="trusted",
            status="approved",
            description=tool.description or "",
            created_at=_now(),
            input_schema=tool.input_schema,
            output_schema=tool.output_schema,
        )
        self._trusted[tool.name] = tool
        self._records[tool.name] = [record]
        return record

    def add_generated(
        self,
        *,
        name: str,
        function_name: str,
        description: str,
        code: str,
        tests: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None,
        requirement: str | None = None,
    ) -> ToolRecord:
        """Store a validated, not yet approved generated tool as a new version.

        ``name`` is untrusted input: it must match ``[A-Za-z0-9_-]{1,128}`` (the
        tool name rule of every provider), so it can never leave the registry
        directory.
        """
        _check_name(name)
        versions = self._records.get(name, [])
        if any(r.kind == "trusted" for r in versions):
            raise RegistryError(f"Tool name is taken by a trusted tool: {name}")
        record = ToolRecord(
            name=name,
            version=(versions[-1].version + 1) if versions else 1,
            kind="generated",
            status="pending",
            description=description,
            created_at=_now(),
            function_name=function_name,
            code=code,
            tests=tests,
            sha256=_sha256(code),
            input_schema=input_schema,
            output_schema=output_schema,
            requirement=requirement,
        )
        self._save(record)  # persist first: a failed write leaves the registry unchanged
        self._records.setdefault(name, []).append(record)
        return record

    def set_status(self, name: str, version: int, status: ToolStatus) -> ToolRecord:
        versions = self._records.get(name, [])
        for index, record in enumerate(versions):
            if record.version == version:
                if record.kind == "trusted":
                    raise RegistryError(f"Status of trusted tool {name!r} cannot be changed")
                updated = replace(record, status=status)
                self._save(updated)
                versions[index] = updated
                return updated
        raise RegistryError(f"Unknown tool version: {name} v{version}")

    def approve(self, name: str, version: int) -> ToolRecord:
        return self.set_status(name, version, "approved")

    def reject(self, name: str, version: int) -> ToolRecord:
        return self.set_status(name, version, "rejected")

    def revoke(self, name: str) -> None:
        """Revoke every approved version of a tool (trusted tools are removed).

        Versions are revoked oldest first, so the version :meth:`record` returns
        is revoked last: if a write fails, ``record`` still returns the version it
        returned before (in memory and after a restart), the error is raised, and
        calling ``revoke`` again completes the revocation.
        """
        if name in self._trusted:
            del self._trusted[name]
            del self._records[name]
            return
        found = False
        for record in list(self._records.get(name, [])):
            if record.status == "approved":
                self.set_status(name, record.version, "revoked")
                found = True
        if not found:
            raise RegistryError(f"No approved tool named {name!r}")

    # ---- lookup -------------------------------------------------------------

    def record(self, name: str) -> ToolRecord:
        """The latest approved version of a tool."""
        for record in reversed(self._records.get(name, [])):
            if record.status == "approved":
                return record
        raise RegistryError(f"No approved tool named {name!r}")

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and any(r.status == "approved" for r in self._records.get(name, []))

    def __iter__(self) -> Iterator[ToolRecord]:
        """Latest approved version of every tool."""
        for name in self.names():
            yield self.record(name)

    def __len__(self) -> int:
        return len(self.names())

    def names(self) -> list[str]:
        return [name for name in self._records if name in self]

    def versions(self, name: str) -> tuple[ToolRecord, ...]:
        """All versions of a tool in any status (audit trail)."""
        return tuple(self._records.get(name, []))

    def get_tool(self, name: str) -> Tool:
        """A ready-to-use Tool for the latest approved version."""
        record = self.record(name)
        if record.kind == "trusted":
            return self._trusted[name]
        if record.code is None or record.sha256 != _sha256(record.code):
            raise RegistryError(f"Stored code of tool {name!r} does not match its hash")
        assert record.function_name is not None and record.input_schema is not None
        return SandboxedTool(
            name=record.name,
            description=record.description,
            input_schema=record.input_schema,
            output_schema=record.output_schema,
            code=record.code,
            function_name=record.function_name,
            sandbox=self.sandbox,
        )

    def search(self, query: str, limit: int = 10) -> list[ToolRecord]:
        """Approved tools whose name or description share words with ``query``."""
        return keyword_search(((r, f"{r.name} {r.description}") for r in self), query, limit)

    def catalog(self) -> list[dict[str, str]]:
        """Name and description of every approved tool, e.g. for prompts."""
        return [{"name": r.name, "description": r.description} for r in self]
