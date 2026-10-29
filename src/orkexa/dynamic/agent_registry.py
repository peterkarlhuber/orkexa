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
from .agent_spec import AgentSpec
from .errors import RegistryError

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
        file = self.path / record.name / f"v{record.version}.json"
        file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "spec": record.spec.model_dump(mode="json"),
            "version": record.version,
            "status": record.status,
            "created_at": record.created_at,
            "requirement": record.requirement,
        }
        file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

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
            self._records.setdefault(record.name, []).append(record)
        for versions in self._records.values():
            versions.sort(key=lambda r: r.version)

    def register(self, spec: AgentSpec, requirement: str | None = None) -> AgentRecord:
        """Store an approved spec as the newest version under its name."""
        versions = self._records.setdefault(spec.name, [])
        record = AgentRecord(
            spec=spec,
            version=(versions[-1].version + 1) if versions else 1,
            status="approved",
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            requirement=requirement,
        )
        versions.append(record)
        self._save(record)
        return record

    def revoke(self, name: str) -> None:
        versions = self._records.get(name, [])
        if not any(r.status == "approved" for r in versions):
            raise RegistryError(f"No approved agent named {name!r}")
        for index, record in enumerate(versions):
            if record.status == "approved":
                versions[index] = replace(record, status="revoked")
                self._save(versions[index])

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
