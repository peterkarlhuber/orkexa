"""Versioned registry of approved agent specifications."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ._search import keyword_search
from .agent_spec import AGENT_NAME, AgentSpec
from .errors import RegistryError
from .tool_registry import _write_json_atomic

__all__ = ["AgentRecord", "AgentRegistry"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRecord:
    spec: AgentSpec
    version: int
    status: Literal["approved", "revoked"]
    created_at: str
    requirement: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name


def _check_name(name: object) -> None:
    if not isinstance(name, str) or not AGENT_NAME.fullmatch(name):
        raise RegistryError(
            f"Invalid agent name {name!r}: a letter, then up to 63 of A-Z, a-z, 0-9, '_', '-'"
        )


class AgentRegistry:
    """Stores approved :class:`AgentSpec` objects, not agent instances.

    Agents are instantiated by the :class:`AgentFactory` for each task, so they
    always start with a clean history and the current policy.

    Args:
        path: Directory for persistence (one JSON file per version), or ``None``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._records: dict[str, list[AgentRecord]] = {}
        if self.path is not None:
            self._load()

    def _save(self, record: AgentRecord) -> None:
        if self.path is None:
            return
        _check_name(record.name)
        root = self.path.resolve()
        file = (root / record.name / f"v{int(record.version)}.json").resolve()
        if file.parent.parent != root:
            raise RegistryError(f"Agent record path escapes the registry directory: {record.name!r}")
        data = {
            "spec": record.spec.model_dump(mode="json"),
            "version": record.version,
            "status": record.status,
            "created_at": record.created_at,
            "requirement": record.requirement,
        }
        _write_json_atomic(file, data)

    def _load(self) -> None:
        assert self.path is not None
        for file in sorted(self.path.glob("*/v*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                record = AgentRecord(
                    spec=AgentSpec.model_validate(data["spec"]),
                    version=int(data["version"]),
                    status=data["status"],
                    created_at=str(data["created_at"]),
                    requirement=data.get("requirement"),
                )
            except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
                logger.warning("Skipping unreadable agent record %s: %s", file, exc)
                continue
            if not AGENT_NAME.fullmatch(record.name) or record.name != file.parent.name:
                logger.warning("Skipping agent record %s: name does not match its directory", file)
                continue
            self._records.setdefault(record.name, []).append(record)
        for versions in self._records.values():
            versions.sort(key=lambda r: r.version)

    def register(self, spec: AgentSpec, requirement: str | None = None) -> AgentRecord:
        """Store an approved spec as the newest version under its name.

        The name must match ``[A-Za-z][A-Za-z0-9_-]{0,63}`` (it is a directory name).
        """
        _check_name(spec.name)
        versions = self._records.get(spec.name, [])
        record = AgentRecord(
            spec=spec,
            version=(versions[-1].version + 1) if versions else 1,
            status="approved",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            requirement=requirement,
        )
        self._save(record)  # persist first: a failed write leaves the registry unchanged
        self._records.setdefault(spec.name, []).append(record)
        return record

    def revoke(self, name: str) -> None:
        versions = self._records.get(name, [])
        if not any(r.status == "approved" for r in versions):
            raise RegistryError(f"No approved agent named {name!r}")
        for index, record in enumerate(versions):
            if record.status == "approved":
                revoked = replace(record, status="revoked")
                self._save(revoked)
                versions[index] = revoked

    def get(self, name: str) -> AgentRecord:
        for record in reversed(self._records.get(name, [])):
            if record.status == "approved":
                return record
        raise RegistryError(f"No approved agent named {name!r}")

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and any(r.status == "approved" for r in self._records.get(name, []))

    def __iter__(self) -> Iterator[AgentRecord]:
        for name in self.names():
            yield self.get(name)

    def __len__(self) -> int:
        return len(self.names())

    def names(self) -> list[str]:
        return [name for name in self._records if name in self]

    def versions(self, name: str) -> tuple[AgentRecord, ...]:
        return tuple(self._records.get(name, []))

    def search(self, query: str, limit: int = 10) -> list[AgentRecord]:
        return keyword_search(((r, f"{r.name} {r.spec.description}") for r in self), query, limit)

    def catalog(self) -> list[dict[str, str]]:
        return [{"name": r.name, "description": r.spec.description} for r in self]
