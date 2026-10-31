"""Immutable, versioned, serializable price snapshots and deterministic lookups."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any, Literal

from ..errors import PriceNotFoundError, PricingError
from ._model import ModelPrice, _aware, _check_name

__all__ = ["SNAPSHOT_FORMAT", "MatchKind", "PriceLookup", "PriceSnapshot", "dated_base"]

SNAPSHOT_FORMAT = "orkexa.price-snapshot/1"
"""Value of the ``"format"`` key of a serialized snapshot."""

MatchKind = Literal["exact", "alias", "dated", "requested", "override"]

_DATED = (re.compile(r"(.+)-\d{8}"), re.compile(r"(.+)-\d{4}-\d{2}-\d{2}"), re.compile(r"(.+)@\d{8}"))


def dated_base(model: str) -> str | None:
    """The undated id of a dated model id (``claude-opus-4-8-20260101`` -> ``claude-opus-4-8``)."""
    for pattern in _DATED:
        match = pattern.fullmatch(model)
        if match:
            return match.group(1)
    return None


@dataclass(frozen=True)
class PriceLookup:
    """The result of resolving a provider and model: a price, or why there is none."""

    provider: str
    model: str
    price: ModelPrice | None
    matched_by: MatchKind | None
    snapshot_version: str | None
    reason: str | None = None
    """Why no price was found (``None`` when one was)."""

    @property
    def found(self) -> bool:
        return self.price is not None

    def require(self) -> ModelPrice:
        """The price, or raise :class:`~orkexa.errors.PriceNotFoundError`."""
        if self.price is None:
            raise PriceNotFoundError(self.reason or f"No price known for {self.provider}:{self.model}")
        return self.price


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class PriceSnapshot:
    """An immutable, versioned set of model prices: one consistent price basis.

    A snapshot never changes after construction. Updating prices means building a
    new snapshot with a new ``version`` (see :meth:`with_prices`), so every cost
    calculated with a snapshot can be reproduced from it later.

    Construction validates the snapshot: per provider, a model id or alias may be
    used by several prices only if their validity periods do not overlap, so a
    lookup has at most one answer.

    Attributes:
        version: Unique version id, e.g. ``"acme-contract-2026-11"``.
        prices: The model prices.
        created_at: When the snapshot was built (timezone-aware).
        source: Who or what produced it (a catalog service, a file, ``"orkexa"``).
        description: Free text.
    """

    version: str
    prices: tuple[ModelPrice, ...] = ()
    created_at: datetime | None = None
    source: str | None = None
    description: str | None = None
    _index: Mapping[tuple[str, str], tuple[tuple[ModelPrice, MatchKind], ...]] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        _check_name(self.version, "snapshot version")
        prices = tuple(self.prices)
        if not all(isinstance(price, ModelPrice) for price in prices):
            raise PricingError("PriceSnapshot.prices must contain ModelPrice objects")
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "created_at", _aware(self.created_at, "created_at"))
        index: dict[tuple[str, str], list[tuple[ModelPrice, MatchKind]]] = {}
        for price in prices:
            for name in price.names:
                kind: MatchKind = "exact" if name == price.model else "alias"
                entries = index.setdefault((price.provider, name), [])
                for other, _ in entries:
                    if other.overlaps(price):
                        raise PricingError(
                            f"Snapshot {self.version!r}: {price.provider}:{name} is priced twice "
                            "for overlapping validity periods"
                        )
                entries.append((price, kind))
        object.__setattr__(self, "_index", {key: tuple(value) for key, value in index.items()})

    # ---- lookups -------------------------------------------------------------------------

    def lookup(self, provider: str, model: str, *, at: datetime | None = None) -> PriceLookup:
        """Resolve ``provider``'s ``model`` at time ``at`` (default: now).

        Order: exact model id, then explicit alias, then the undated id of a dated
        model id (only for prices with ``match_dated_versions``). Prices are never
        matched by prefix, never taken from another provider and never invented.
        """
        when = _aware(at, "at") if at is not None else _now()
        assert when is not None
        expired = False
        for name, dated in ((model, False), (dated_base(model), True)):
            if name is None:
                continue
            entries = self._index.get((provider, name), ())
            for price, kind in entries:
                if dated and (kind != "exact" or not price.match_dated_versions):
                    continue
                if price.valid_at(when):
                    return PriceLookup(provider, model, price, "dated" if dated else kind, self.version)
                expired = True
        reason = (
            f"No price for {provider}:{model} is valid at {when.isoformat()} in snapshot {self.version!r}"
            if expired
            else f"No known pricing for {provider}:{model} in snapshot {self.version!r}"
        )
        return PriceLookup(provider, model, None, None, self.version, reason)

    def get(self, provider: str, model: str, *, at: datetime | None = None) -> ModelPrice | None:
        return self.lookup(provider, model, at=at).price

    def __contains__(self, key: object) -> bool:
        return isinstance(key, tuple) and len(key) == 2 and self.lookup(*key).found

    def __len__(self) -> int:
        return len(self.prices)

    def __iter__(self) -> Iterator[ModelPrice]:
        return iter(self.prices)

    # ---- derived snapshots ---------------------------------------------------------------

    def with_prices(
        self,
        prices: Iterable[ModelPrice],
        *,
        version: str | None = None,
        replace: bool = True,
        description: str | None = None,
    ) -> PriceSnapshot:
        """A new snapshot with ``prices`` added.

        With ``replace=True`` (default) existing prices of the same provider and model
        id are dropped first, otherwise they must not overlap. Without ``version`` the
        new version is ``"<version>+<content hash>"``, so the same change always yields
        the same version.
        """
        added = tuple(prices)
        keys = {(price.provider, price.model) for price in added}
        kept = tuple(p for p in self.prices if not (replace and (p.provider, p.model) in keys))
        draft = PriceSnapshot("draft", (*kept, *added))
        return PriceSnapshot(
            version or f"{self.version}+{draft.content_hash()[:12]}",
            draft.prices,
            created_at=_now(),
            source=self.source,
            description=description if description is not None else self.description,
        )

    # ---- serialization -------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form (format :data:`SNAPSHOT_FORMAT`)."""
        data: dict[str, Any] = {"format": SNAPSHOT_FORMAT, "version": self.version}
        if self.created_at is not None:
            data["created_at"] = self.created_at.isoformat()
        if self.source is not None:
            data["source"] = self.source
        if self.description is not None:
            data["description"] = self.description
        data["prices"] = [price.to_dict() for price in self.prices]
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PriceSnapshot:
        """Load a snapshot written by :meth:`to_dict` (or by any external system)."""
        if not isinstance(data, Mapping):
            raise PricingError("A price snapshot must be a JSON object")
        if data.get("format") != SNAPSHOT_FORMAT:
            raise PricingError(
                f"Unsupported price snapshot format {data.get('format')!r}; expected {SNAPSHOT_FORMAT!r}"
            )
        unknown = set(data) - {"format", "version", "created_at", "source", "description", "prices"}
        if unknown:
            raise PricingError(f"Unknown keys in price snapshot: {sorted(unknown)}")
        prices = data.get("prices", [])
        if not isinstance(prices, list):
            raise PricingError("snapshot prices must be a list")
        return cls(
            version=data.get("version"),  # type: ignore[arg-type]
            prices=tuple(ModelPrice.from_dict(price) for price in prices),
            created_at=data.get("created_at"),
            source=data.get("source"),
            description=data.get("description"),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str | bytes) -> PriceSnapshot:
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise PricingError(f"Price snapshot is not valid JSON: {exc}") from None
        return cls.from_dict(data)

    def content_hash(self) -> str:
        """SHA-256 of the prices (canonical JSON), independent of version and metadata.

        Store it next to a snapshot to detect modified copies.
        """
        canonical = json.dumps(
            sorted(
                (price.to_dict() for price in self.prices),
                key=lambda p: (p["provider"], p["model"], str(p.get("valid_from"))),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode()).hexdigest()
