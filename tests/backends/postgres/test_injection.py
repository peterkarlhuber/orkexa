"""SQL injection: hostile values are data (MUP-ONA B5: T-PG-5, T-PG-INJ).

Values with ``'``, ``;``, ``\\0``, ``$$``, statement text and 1 MB of text go through every store as sources,
titles, metadata, chunk texts, queries, namespaces, tenants, file names and bytes, memory paths and contents.
They are stored as data and found again; the tables are intact afterwards (``verify_schema``) and another
tenant sees nothing. PostgreSQL text cannot hold U+0000: the index keeps U+FFFD in its place, the file store
keeps bytes as they are, and the memory refuses it (``MemoryToolError``) without writing.
"""

from __future__ import annotations

import pytest

from orkexa import MemoryToolError
from orkexa.backends.postgres import PostgresFileStore, PostgresIndex, PostgresMemory, verify_schema

from . import support

pytestmark = pytest.mark.postgres

HOSTILE = [
    "it's",
    "a; DROP TABLE t_chunks; --",
    "x' OR '1'='1",
    "$$ SELECT 1 $$",
    "$tag$ body $tag$",
    "%(tenant_id)s %s {chunks}",
    '"; DELETE FROM t_files; --',
    "back\\slash",
]
BIG = ("word " * 210_000)[:1_000_000] + "needle"


# MUP-ONA T-PG-5
@pytest.mark.parametrize("value", HOSTILE)
def test_hostile_values_are_stored_and_found_again(db: support.Database, value: str) -> None:
    tenant = f"tenant {value}"[:128]
    index = PostgresIndex(db.source, tenant, layout=db.layout, namespace=value)
    index.add(f"Zebrafish note {value}", source=value, title=value, metadata={"key": value})
    hits = index.retrieve("zebrafish", 5, mode="fts")
    assert [(hit.source, hit.title, hit.metadata) for hit in hits] == [(value, value, {"key": value})]
    assert value in hits[0].text
    assert index.sources() == [value]
    assert index.retrieve(value, 5, mode="fts") is not None  # a hostile query is only data
    assert index.retrieve("zebrafish", 5, mode="fts", sources=[value])[0].source == value
    files = PostgresFileStore(db.source, tenant, layout=db.layout, namespace=value)
    stored = files.add(
        value.encode() + b"\x00\xff",
        name=f"{value}.bin",
        media_type="image/png",
        description=value,
        metadata={"k": value},
    )
    assert files.get_bytes(stored.id) == value.encode() + b"\x00\xff"
    assert files.info(stored.id).description == value
    memory = PostgresMemory(db.source, tenant, layout=db.layout, namespace=value)
    path = "/memories/" + "".join(ch if ch not in "\\" else "_" for ch in value).replace("/", "_") + ".md"
    memory.create(path, value)
    assert value in memory.view(path)
    other = PostgresIndex(db.source, "someone-else", layout=db.layout, namespace=value)
    assert other.retrieve("zebrafish", 5, mode="fts") == [] and len(other) == 0
    assert index.delete(value) == 1
    assert verify_schema(db.source, db.layout).ok
    assert db.query("SELECT count(*) FROM t_chunks") == [(0,)]


# MUP-ONA T-PG-5
def test_nul_characters(db: support.Database) -> None:
    index = PostgresIndex(db.source, "tenant-a", layout=db.layout)
    index.add("before\x00after zebrafish", source="nul\x00.md", title="t\x00", metadata={"k\x00": "v\x00"})
    hit = index.retrieve("zebrafish", 5, mode="fts")[0]
    assert (hit.text, hit.source, hit.title) == ("before�after zebrafish", "nul�.md", "t�")
    assert hit.metadata == {"k�": "v�"}
    memory = PostgresMemory(db.source, "tenant-a", layout=db.layout)
    with pytest.raises(MemoryToolError, match="U\\+0000"):
        memory.create("/memories/a.md", "x\x00y")
    assert memory._files("") == {}
    files = PostgresFileStore(db.source, "tenant-a", layout=db.layout)
    stored = files.add(b"\x00" * 10, name="zeros.bin", media_type="image/png", description="d\x00")
    assert files.get_bytes(stored.id) == b"\x00" * 10


# MUP-ONA T-PG-5
def test_one_megabyte_of_text(db: support.Database) -> None:
    index = PostgresIndex(db.source, "tenant-a", layout=db.layout, max_chars=2_000_000, overlap=0)
    index.add(BIG, source="big.md")
    hits = index.retrieve("needle", 5, mode="fts")
    assert [hit.source for hit in hits] == ["big.md"] and hits[0].text == BIG
    memory = PostgresMemory(db.source, "tenant-a", layout=db.layout, max_file_chars=2_000_000)
    memory.create("/memories/big.md", BIG)
    assert memory._read("big.md") == BIG
    files = PostgresFileStore(db.source, "tenant-a", layout=db.layout, max_document_bytes=3_000_000)
    stored = files.add(BIG.encode(), name="big.txt")
    assert files.get_bytes(stored.id) == BIG.encode()
    assert verify_schema(db.source, db.layout).ok
