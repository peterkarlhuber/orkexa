"""ORK3-07: the embedding identity follows the model's current digest on the server.

Internal backend audit round 3 (2026-11-04). No network: a stub stands in for OllamaClient.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from orkexa import SQLiteIndex
from orkexa.embeddings import OllamaEmbeddings
from orkexa.local import EmbedResult
from orkexa.retrieval import RetrievalError
from tests.audit3._vec import needs_vec


class Server:
    """The OllamaClient methods OllamaEmbeddings uses; ``digest`` changes when the model is re-pulled."""

    def __init__(self) -> None:
        self.digest = "a" * 64
        self.embeds = 0
        self.on_embed: Any = None

    def find(self, model: str) -> Any:
        return SimpleNamespace(name=model, digest=self.digest)

    def show(self, model: str) -> Any:
        return SimpleNamespace(is_embedding=True, embedding_length=3)

    def embed(self, model: str, inputs: list[str], **_: Any) -> EmbedResult:
        self.embeds += 1
        if self.on_embed is not None:
            self.on_embed(self.embeds)
        return EmbedResult(model=model, embeddings=[[1.0, float(len(text)), 0.0] for text in inputs])


def _index(tmp_path: Path) -> tuple[SQLiteIndex, Server]:
    server = Server()
    embeddings = OllamaEmbeddings("emb:1", client=server)  # type: ignore[arg-type]
    index = SQLiteIndex(tmp_path / "i.db", embeddings=embeddings)
    index.add("first document", source="a")
    return index, server


@needs_vec
def test_an_updated_model_is_not_filed_under_its_old_identity(tmp_path: Path) -> None:
    index, server = _index(tmp_path)
    server.digest = "b" * 64  # e.g. `ollama pull emb:1` brought a new build
    with pytest.raises(RetrievalError) as error:
        index.add("second document", source="b")
    assert error.value.code == "reindex_required"
    status = index.reindex()
    assert status.active is not None and status.active["digest"] == "b" * 64 and status.embedded == 1
    index.add("second document", source="b")
    index.close()


@needs_vec
def test_a_model_update_during_a_reindex_discards_the_mixed_vectors(tmp_path: Path) -> None:
    index, server = _index(tmp_path)
    for number in range(4):
        index.add(f"document {number}", source=f"s{number}")
    old = index.vector_status().active
    server.digest = "b" * 64

    def update_midway(count: int) -> None:
        if count == 1:  # while the first reindex batch is embedded
            server.digest = "c" * 64

    server.on_embed, server.embeds = update_midway, 0
    with pytest.raises(RetrievalError) as error:
        index.reindex(batch_size=2)
    assert error.value.code == "reindex_required"
    status = index.vector_status()
    assert status.active == old and status.building is None  # old space intact, mixed space gone
    server.on_embed = None
    assert index.reindex().active["digest"] == "c" * 64  # type: ignore[index]
    index.close()


def test_searches_see_a_new_digest_after_the_ttl(tmp_path: Path) -> None:
    server = Server()
    embeddings = OllamaEmbeddings("emb:1", client=server, identity_ttl=0.0)  # type: ignore[arg-type]
    assert embeddings.identity().digest == "a" * 64
    server.digest = "b" * 64
    assert embeddings.identity().digest == "b" * 64
    cached = OllamaEmbeddings("emb:1", client=server)  # type: ignore[arg-type]
    assert cached.identity().digest == "b" * 64
    server.digest = "c" * 64
    assert cached.identity().digest == "b" * 64  # within the TTL
    cached.refresh_identity()
    assert cached.identity().digest == "c" * 64
