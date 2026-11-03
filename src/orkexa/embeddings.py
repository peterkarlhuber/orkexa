"""Embedding models: text in, vectors out, with an identity that binds a vector index to one model.

:class:`EmbeddingModel` is the provider-independent interface; :class:`OllamaEmbeddings`
implements it for a local Ollama server (``/api/embed``). A vector index stores the
:class:`EmbeddingIdentity` of the model that filled it. Vectors of different models are
never mixed: the same number of dimensions does not make two vector spaces compatible.

The older ``embedder=`` callables of :class:`~orkexa.SQLiteIndex` keep working; an
:class:`EmbeddingModel` is also such a callable (it embeds documents).

Nothing here installs software or downloads models. :class:`OllamaEmbeddings` contacts
only the server it was given, and only when it embeds or resolves its identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
from typing import Any, Literal

from .local._http import DEFAULT_OLLAMA_URL, OllamaError
from .local.client import OllamaClient

__all__ = [
    "KNOWN_PREFIXES",
    "QWEN3_QUERY_INSTRUCTION",
    "CallableEmbeddings",
    "EmbeddingError",
    "EmbeddingIdentity",
    "EmbeddingModel",
    "EmbeddingUsage",
    "OllamaEmbeddings",
]

QWEN3_QUERY_INSTRUCTION = "Given a search query, retrieve relevant passages that answer the query"
"""The task instruction Qwen3-Embedding expects in front of queries (documents get none)."""

KNOWN_PREFIXES: dict[str, tuple[str, str]] = {
    "mdenseon": ("document: ", "query: "),
}
"""``(document prefix, query prefix)`` that models were trained with, by model name without
namespace and tag (``mdenseon:307m`` -> ``mdenseon``)."""


class EmbeddingError(OllamaError):
    """Embedding failed; ``code`` as in :class:`~orkexa.local.OllamaError`, plus
    ``dimension_mismatch``, ``identity_mismatch``, ``empty_input`` and ``too_many_inputs``."""


@dataclass(frozen=True)
class EmbeddingIdentity:
    """What makes vectors comparable: the model and how texts were prepared.

    ``digest`` pins the exact model build (a ``latest`` update changes it). Two indexes
    may share vectors only if their :attr:`key` is equal.
    """

    provider: str
    model: str
    dimensions: int
    digest: str = ""
    normalized: bool = True
    document_instruction: str = ""
    query_instruction: str = ""
    #: Plain text put in front of every document/query (e.g. ``"query: "``).
    document_prefix: str = ""
    query_prefix: str = ""

    @property
    def key(self) -> str:
        """A short stable id of this vector space (hex)."""
        fields: list[Any] = [
            self.provider,
            self.model,
            self.digest,
            self.dimensions,
            self.normalized,
            self.document_instruction,
            self.query_instruction,
        ]
        if self.document_prefix or self.query_prefix:  # keeps the keys of older spaces unchanged
            fields += [self.document_prefix, self.query_prefix]
        data = json.dumps(fields)
        return hashlib.sha256(data.encode()).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "digest": self.digest,
            "normalized": self.normalized,
            "document_instruction": self.document_instruction,
            "query_instruction": self.query_instruction,
            "document_prefix": self.document_prefix,
            "query_prefix": self.query_prefix,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingIdentity:
        return cls(
            provider=str(data["provider"]),
            model=str(data["model"]),
            dimensions=int(data["dimensions"]),
            digest=str(data.get("digest") or ""),
            normalized=bool(data.get("normalized", True)),
            document_instruction=str(data.get("document_instruction") or ""),
            query_instruction=str(data.get("query_instruction") or ""),
            document_prefix=str(data.get("document_prefix") or ""),
            query_prefix=str(data.get("query_prefix") or ""),
        )


@dataclass
class EmbeddingUsage:
    """Measured use of an embedding model; ``None`` means the server did not report it."""

    kind: Literal["document", "query"]
    model: str
    inputs: int
    prompt_tokens: int | None = None
    duration_ms: float | None = None
    load_duration_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class EmbeddingModel(ABC):
    """Turns texts into vectors.

    Subclasses implement :meth:`embed_documents`, :meth:`embed_query` and
    :meth:`identity`. Calling the object embeds documents, so it can be passed where
    an ``embedder=`` callable is expected.
    """

    #: Called with every measured use (e.g. to record usage); ``None``: not recorded.
    on_usage: Callable[[EmbeddingUsage], None] | None = None

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """The vector of a search query (may differ from a document's)."""

    @abstractmethod
    def identity(self) -> EmbeddingIdentity:
        """The identity of the vectors this model produces (may contact the server once)."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    def query_embedder(self) -> Callable[[list[str]], list[list[float]]]:
        """An ``embedder``-style callable for queries."""
        return lambda texts: [self.embed_query(text) for text in texts]


class CallableEmbeddings(EmbeddingModel):
    """Wraps ``embedder``/``query_embedder`` callables; their identity is only what you declare."""

    def __init__(
        self,
        embedder: Callable[[list[str]], Sequence[Sequence[float]]],
        *,
        name: str,
        dimensions: int,
        query_embedder: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
        provider: str = "custom",
        normalized: bool = False,
    ) -> None:
        self._embedder = embedder
        self._query = query_embedder or embedder
        self._identity = EmbeddingIdentity(
            provider=provider, model=name, dimensions=dimensions, normalized=normalized
        )

    def _check(self, vectors: Sequence[Sequence[float]], count: int) -> list[list[float]]:
        result = [[float(x) for x in vector] for vector in vectors]
        if len(result) != count:
            raise EmbeddingError("invalid_response", "The embedder must return one vector per text")
        for vector in result:
            if len(vector) != self._identity.dimensions:
                raise EmbeddingError(
                    "dimension_mismatch",
                    f"The embedder gave {len(vector)} dimensions, declared are {self._identity.dimensions}",
                )
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._check(self._embedder(list(texts)), len(texts)) if texts else []

    def embed_query(self, text: str) -> list[float]:
        return self._check(self._query([text]), 1)[0]

    def identity(self) -> EmbeddingIdentity:
        return self._identity


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 0 else vector


class OllamaEmbeddings(EmbeddingModel):
    """Embeddings of a model on an Ollama server, e.g. ``qwen3-embedding:0.6b``.

    Texts are never truncated silently: Ollama is called with ``truncate=false``, and a
    too long text raises :class:`EmbeddingError` (``input_too_long``). Split documents
    first (:func:`~orkexa.split_text`).

    Args:
        model: The installed embedding model.
        base_url: The Ollama server (loopback unless ``allow_remote``).
        query_instruction: Instruction put in front of queries; defaults to
            :data:`QWEN3_QUERY_INSTRUCTION` for ``qwen3-embedding`` models and none otherwise.
        document_prefix: Text put in front of every document; defaults to the model's entry
            in :data:`KNOWN_PREFIXES` (e.g. ``"document: "`` for ``mdenseon``), else none.
        query_prefix: Text put in front of every query; same default. A model uses either
            a query instruction or a query prefix, not both.
        dimensions: Ask for shorter vectors (Matryoshka models); ``None``: the model's size.
        batch_size: Texts per request.
        max_inputs: Most texts one :meth:`embed_documents` call accepts.
        timeout: Seconds per request (the first one also loads the model).
        keep_alive: How long Ollama keeps the model loaded, e.g. ``"5m"``.
        cancel: An event that stops a running :meth:`embed_documents` between batches.
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:0.6b",
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        query_instruction: str | None = None,
        document_prefix: str | None = None,
        query_prefix: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 16,
        max_inputs: int = 100_000,
        timeout: float = 300.0,
        keep_alive: str | None = None,
        allow_remote: bool = False,
        client: OllamaClient | None = None,
        cancel: threading.Event | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.client = client or OllamaClient(base_url, allow_remote=allow_remote)
        self.model = model
        if query_instruction is None:
            query_instruction = QWEN3_QUERY_INSTRUCTION if model.startswith("qwen3-embedding") else ""
        known = KNOWN_PREFIXES.get(model.split("/")[-1].split(":")[0].lower(), ("", ""))
        self.document_prefix = known[0] if document_prefix is None else document_prefix
        self.query_prefix = known[1] if query_prefix is None else query_prefix
        if query_instruction and self.query_prefix:
            raise ValueError("Pass either query_instruction or query_prefix, not both")
        self.query_instruction = query_instruction
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.max_inputs = max_inputs
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.cancel = cancel
        self._identity: EmbeddingIdentity | None = None
        self._lock = threading.Lock()

    def _query_text(self, text: str) -> str:
        if self.query_instruction:
            return f"Instruct: {self.query_instruction}\nQuery: {text}"
        return self.query_prefix + text

    def _embed(self, texts: Sequence[str], kind: Literal["document", "query"]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise EmbeddingError("empty_input", "Empty texts cannot be embedded")
        try:
            result = self.client.embed(
                self.model,
                texts,
                dimensions=self.dimensions,
                keep_alive=self.keep_alive,
                timeout=self.timeout,
            )
        except EmbeddingError:
            raise
        except OllamaError as exc:
            raise EmbeddingError(exc.code, str(exc), status=exc.status) from exc
        vectors = [_normalize(vector) for vector in result.embeddings]
        expected = self.identity().dimensions
        for vector in vectors:
            if len(vector) != expected:
                raise EmbeddingError(
                    "dimension_mismatch",
                    f"{self.model} returned {len(vector)} dimensions, expected {expected}",
                )
        if self.on_usage is not None:
            self.on_usage(
                EmbeddingUsage(
                    kind=kind,
                    model=self.model,
                    inputs=len(texts),
                    prompt_tokens=result.prompt_tokens,
                    duration_ms=result.total_duration_ms,
                    load_duration_ms=result.load_duration_ms,
                )
            )
        return vectors

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if len(texts) > self.max_inputs:
            raise EmbeddingError("too_many_inputs", f"At most {self.max_inputs} texts per call")
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            if self.cancel is not None and self.cancel.is_set():
                raise EmbeddingError("cancelled", "Embedding was cancelled")
            batch = texts[start : start + self.batch_size]
            if self.document_prefix:
                if any(not text.strip() for text in batch):
                    raise EmbeddingError("empty_input", "Empty texts cannot be embedded")
                batch = [self.document_prefix + text for text in batch]
            vectors += self._embed(batch, "document")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        if not text.strip():
            raise EmbeddingError("empty_input", "Empty texts cannot be embedded")
        return self._embed([self._query_text(text)], "query")[0]

    def identity(self) -> EmbeddingIdentity:
        """Model, digest and dimensions, from the server (``model_missing`` if not installed)."""
        with self._lock:
            if self._identity is None:
                try:
                    installed = self.client.find(self.model)
                    if installed is None:
                        raise EmbeddingError(
                            "model_missing", f"The embedding model {self.model} is not installed"
                        )
                    details = self.client.show(self.model)
                except EmbeddingError:
                    raise
                except OllamaError as exc:
                    raise EmbeddingError(exc.code, str(exc), status=exc.status) from exc
                if not details.is_embedding:
                    raise EmbeddingError("not_embedding_model", f"{self.model} is not an embedding model")
                dimensions = self.dimensions or details.embedding_length
                if not dimensions:
                    raise EmbeddingError(
                        "invalid_response", f"Ollama reports no vector size for {self.model}"
                    )
                self._identity = EmbeddingIdentity(
                    provider="ollama",
                    model=installed.name,
                    dimensions=int(dimensions),
                    digest=installed.digest,
                    normalized=True,
                    query_instruction=self.query_instruction,
                    document_prefix=self.document_prefix,
                    query_prefix=self.query_prefix,
                )
            return self._identity
