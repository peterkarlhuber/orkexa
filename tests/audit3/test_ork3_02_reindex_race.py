"""ORK3-02: a chunk deleted while reindex() embeds its batch leaves no vector behind.

Internal backend audit round 3 (2026-11-04). No network.
"""

from __future__ import annotations

from pathlib import Path
import struct

from orkexa import SQLiteIndex
from orkexa.embeddings import CallableEmbeddings

# ---- ORK3-02: a chunk deleted while reindex() embeds its batch -------------------------


def _embeddings(name: str, hook=None) -> CallableEmbeddings:  # type: ignore[no-untyped-def]
    def embed(texts: list[str]) -> list[list[float]]:
        if hook is not None:
            hook(texts)
        return [[float(len(text)), float(ord(text[0])), 1.0] for text in texts]

    return CallableEmbeddings(embed, name=name, dimensions=3)


def _filled(path: Path) -> SQLiteIndex:
    index = SQLiteIndex(path, embeddings=_embeddings("A"))
    for number in range(4):
        index.add(f"document number {number} alpha", source=f"s{number}")
    return index


def test_reindex_leaves_no_vector_of_a_chunk_deleted_meanwhile(tmp_path: Path) -> None:
    index = _filled(tmp_path / "a.db")
    done: list[int] = []

    def delete_last(texts: list[str]) -> None:
        if any("number 3" in text for text in texts) and not done:
            done.append(index.delete("s3"))

    index.reindex(_embeddings("B", delete_last), batch_size=2)
    assert done == [1]
    orphans = index._connection.execute(
        "SELECT count(*) FROM index_vectors v LEFT JOIN index_chunks c ON c.id = v.chunk_id"
        " WHERE c.id IS NULL"
    ).fetchone()[0]
    assert orphans == 0
    # the next chunk reuses the deleted chunk's id: adding it must still work
    index.add("a new document beta", source="new")
    assert index.vector_status().embedded == 4
    index.close()


def test_reindex_does_not_give_a_new_chunk_the_vector_of_a_deleted_one(tmp_path: Path) -> None:
    index = _filled(tmp_path / "b.db")
    done: list[int] = []

    def replace_last(texts: list[str]) -> None:
        if any("number 3" in text for text in texts) and not done:
            index.delete("s3")
            done.extend(index.add("Zebra", source="new"))

    index.reindex(_embeddings("B", replace_last), batch_size=2)
    blob = index._connection.execute(
        "SELECT embedding FROM index_vectors WHERE chunk_id = ?", (done[0],)
    ).fetchone()[0]
    assert struct.unpack("3f", blob) == (5.0, float(ord("Z")), 1.0)
    index.close()


def test_add_replaces_a_stale_vector_left_by_an_earlier_version(tmp_path: Path) -> None:
    """Databases written before the fix may hold such a vector: adding the next chunk heals it."""
    index = _filled(tmp_path / "c.db")
    space = index._connection.execute("SELECT space FROM index_spaces").fetchone()[0]
    index.delete("s3")
    index._connection.execute(
        "INSERT INTO index_vectors (space, chunk_id, embedding) VALUES (?, 4, ?)",
        (space, struct.pack("3f", 9.0, 9.0, 9.0)),
    )
    assert index.add("Zebra", source="new") == [4]
    blob = index._connection.execute("SELECT embedding FROM index_vectors WHERE chunk_id = 4").fetchone()[0]
    assert struct.unpack("3f", blob) == (5.0, float(ord("Z")), 1.0)
    index.close()
