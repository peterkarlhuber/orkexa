"""A deterministic embedding model for the conformance cases (no network, no randomness)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math

from ...embeddings import EmbeddingIdentity, EmbeddingModel
from ...retrieval import tokenize

#: One axis per topic; a text gets 1.0 on the axis of every topic word (or synonym) it contains.
AXES: tuple[tuple[str, ...], ...] = (
    ("alpha", "first"),
    ("beta", "second"),
    ("gamma", "third"),
    ("delta", "fourth"),
    ("epsilon", "fifth"),
    ("zeta", "sixth"),
    ("eta", "seventh"),
)
DIMENSIONS = len(AXES) + 1  # the last axis: texts without a topic word

#: Marker words that replace a text's vector (to feed a backend invalid or unusual vectors).
OVERRIDES: dict[str, list[float]] = {
    "nanvector": [math.nan] + [0.0] * (DIMENSIONS - 1),
    "infvector": [math.inf] + [0.0] * (DIMENSIONS - 1),
    "neginfvector": [-math.inf] + [1.0] * (DIMENSIONS - 1),
    "zerovector": [0.0] * DIMENSIONS,
    "shortvector": [1.0, 0.0, 0.0],
}
#: A marker word that multiplies a text's vector by 10 (the direction, and so the cosine, stays the same).
LOUD = "loudvector"


class ModelFailure(RuntimeError):
    """Raised by :class:`KeywordEmbeddings` when it is told to fail (an interrupted reindex)."""


class KeywordEmbeddings(EmbeddingModel):
    """Vectors from topic words (:data:`AXES`), with a controllable identity.

    ``identity()`` caches the identity of the model "on the server" (``served_digest``) until
    :meth:`refresh_identity`, as :class:`~orkexa.embeddings.OllamaEmbeddings` does; changing ``served_digest``
    simulates a model updated on its server. ``fail_after`` lets :meth:`embed_documents` fail after that many
    calls; ``on_embed`` is called with the texts of every document call (before they are embedded).
    """

    def __init__(
        self,
        name: str = "conformance-keywords",
        *,
        digest: str = "v1",
        normalized: bool = False,
        fail_after: int | None = None,
        on_embed: Callable[[Sequence[str]], None] | None = None,
    ) -> None:
        self.name = name
        self.served_digest = digest
        self.normalized = normalized
        self.fail_after = fail_after
        self.on_embed = on_embed
        self.document_calls = 0
        self._cached: EmbeddingIdentity | None = None

    @staticmethod
    def vector(text: str) -> list[float]:
        """The vector of ``text`` (documents and queries alike)."""
        words = {word.lower() for word in tokenize(text)}
        for marker, values in OVERRIDES.items():
            if marker in words:
                return list(values)
        values = [1.0 if words & set(axis) else 0.0 for axis in AXES]
        values.append(0.0 if any(values) else 1.0)
        if LOUD in words:
            values = [value * 10.0 for value in values]
        return values

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.document_calls += 1
        if self.on_embed is not None:
            self.on_embed(texts)
        if self.fail_after is not None and self.document_calls > self.fail_after:
            raise ModelFailure("the embedding model failed (conformance case)")
        return [self.vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.vector(text)

    def identity(self) -> EmbeddingIdentity:
        if self._cached is None:
            self._cached = EmbeddingIdentity(
                provider="conformance",
                model=self.name,
                dimensions=DIMENSIONS,
                digest=self.served_digest,
                normalized=self.normalized,
            )
        return self._cached

    def refresh_identity(self) -> None:
        self._cached = None


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """The cosine similarity of two vectors (0 if one has no direction)."""
    norm = math.hypot(*left) * math.hypot(*right)
    return 0.0 if norm == 0 else sum(a * b for a, b in zip(left, right, strict=True)) / norm
