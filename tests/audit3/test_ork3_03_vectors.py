"""ORK3-03: non-finite vectors are refused; a zero vector does not break vector search.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orkexa import SQLiteIndex
from orkexa.embeddings import CallableEmbeddings
from orkexa.local import OllamaClient, OllamaError
from tests.audit3._vec import needs_vec

# ---- ORK3-03: non-finite and zero vectors ----------------------------------------------


@needs_vec
def test_non_finite_vectors_are_refused(tmp_path: Path) -> None:
    def embed(texts: list[str]) -> list[list[float]]:
        return [[float("nan"), 1.0, 0.0] for _ in texts]

    index = SQLiteIndex(tmp_path / "n.db", embeddings=CallableEmbeddings(embed, name="n", dimensions=3))
    with pytest.raises(ValueError, match="finite"):
        index.add("text", source="a")
    assert len(index) == 0
    index.close()


@needs_vec
@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_a_zero_vector_does_not_break_vector_search(tmp_path: Path, mode: str) -> None:
    def embed(texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.0, 0.0] if "zero" in text else [1.0, 0.0, 0.0] for text in texts]

    index = SQLiteIndex(tmp_path / "z.db", embeddings=CallableEmbeddings(embed, name="z", dimensions=3))
    index.add("normal text", source="a")
    index.add("zero text", source="b")
    hits = index.retrieve("normal", mode=mode)  # type: ignore[arg-type]
    assert hits[0].source == "a"
    index.close()


def test_ollama_client_refuses_non_finite_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    client = OllamaClient()
    answer = json.loads('{"embeddings": [[NaN, Infinity, 1.0]]}')
    monkeypatch.setattr(client.http, "request", lambda *args, **kwargs: answer)
    with pytest.raises(OllamaError) as error:
        client.embed("emb:1", ["hello"])
    assert error.value.code == "invalid_response"
    monkeypatch.setattr(client.http, "request", lambda *args, **kwargs: {"embeddings": [[None, 1.0]]})
    with pytest.raises(OllamaError):
        client.embed("emb:1", ["hello"])
