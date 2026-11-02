"""Tests for the RAG building blocks: chunking, SQLiteIndex and the search tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from orkexa import (
    ClaudeAgent,
    SQLiteFileStore,
    SQLiteIndex,
    citations,
    split_text,
)
from orkexa.retrieval import _fts_query

from .conftest import FakeClient, make_message

# sqlite-vec is a loadable extension, which some Python builds (such as macOS runners') cannot load.
HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None and hasattr(
    sqlite3.connect(":memory:"), "enable_load_extension"
)
HAS_PYPDF = importlib.util.find_spec("pypdf") is not None

HANDBOOK = """# Warranty

The warranty lasts two years from the date of purchase.

# Returns

Returns are accepted within 30 days. Refunds take five business days.

# Shipping

Orders ship within three days. Express shipping costs extra.
"""


def test_split_text_respects_paragraphs_and_limits() -> None:
    assert split_text("") == []
    assert split_text("One.\n\nTwo.") == ["One.\n\nTwo."]
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = split_text(paragraphs, max_chars=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 500 + 100 for chunk in chunks)
    assert chunks[0].startswith("Paragraph 0")
    assert "Paragraph 9" in chunks[-1]


def test_split_text_cuts_long_paragraphs_and_tokens() -> None:
    sentence = "This is a sentence. " * 50
    chunks = split_text(sentence, max_chars=300, overlap=0)
    assert all(len(chunk) <= 300 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == sentence.replace(" ", "")
    token = "x" * 1000
    assert all(len(chunk) <= 300 for chunk in split_text(token, max_chars=300, overlap=0))
    with pytest.raises(ValueError):
        split_text("x", max_chars=50)
    with pytest.raises(ValueError):
        split_text("x", max_chars=200, overlap=150)


def test_split_text_overlap_repeats_the_previous_tail() -> None:
    text = "\n\n".join(["alpha " * 30, "beta " * 30, "gamma " * 30])
    _first, second, *_ = split_text(text, max_chars=250, overlap=60)
    assert second.split()[0] == "alpha"
    assert "beta" in second


def test_fts_query_quotes_words() -> None:
    assert _fts_query('Is the "warranty" the issue?') == '"is"* OR "the"* OR "warranty"* OR "issue"*'
    assert _fts_query("!!!") == ""


def test_index_adds_searches_and_deletes(tmp_path: Path) -> None:
    index = SQLiteIndex(tmp_path / "rag.db", namespace="handbook", max_chars=200, overlap=0)
    ids = index.add(HANDBOOK, source="handbook.md", title="Handbook", metadata={"lang": "en"})
    assert len(ids) == len(index) >= 2
    hits = index.search("How long is the warranty?", k=2)
    assert hits[0].source == "handbook.md"
    assert "two years" in hits[0].text
    assert hits[0].metadata == {"lang": "en"}
    assert hits[0].score > 0
    assert index.search("refund", k=1)[0].text.count("Refunds") == 1
    assert index.search("???") == []
    assert index.sources() == ["handbook.md"]
    with pytest.raises(ValueError):
        index.search("x", k=0)

    other = SQLiteIndex(tmp_path / "rag.db", namespace="other")
    assert other.search("warranty") == []
    assert len(other) == 0

    assert index.delete("handbook.md") == len(ids)
    assert index.delete("handbook.md") == 0
    assert index.search("warranty") == []
    index.close()


def test_umlauts_and_diacritics_match() -> None:
    index = SQLiteIndex(":memory:")
    index.add("Die Rücksendung ist 30 Tage möglich.", source="de.md")
    assert index.search("rucksendung")[0].source == "de.md"
    assert index.search("Rücksendung")[0].source == "de.md"


def test_index_rejects_bad_settings() -> None:
    with pytest.raises(ValueError, match="namespace"):
        SQLiteIndex(":memory:", namespace="")
    with pytest.raises(ValueError, match="tokenizer"):
        SQLiteIndex(":memory:", tokenizer="porter'); DROP TABLE x; --")
    with pytest.raises(ValueError):
        SQLiteIndex(":memory:", max_chars=10)


def test_add_file_with_store_links_the_original(tmp_path: Path) -> None:
    path = tmp_path / "handbook.md"
    path.write_text(HANDBOOK, encoding="utf-8")
    store = SQLiteFileStore(":memory:")
    with SQLiteIndex(":memory:") as index:
        index.add_file(path, store=store, description="Product handbook")
        (hit,) = index.search("warranty", k=1)
        assert hit.file_id is not None
        assert store.info(hit.file_id).description == "Product handbook"
        block = hit.block()
        assert block["source"] == "handbook.md"
        assert block["content"][-1]["text"].startswith(f"(Original: file {hit.file_id}")
        count = len(index)
        index.add_file(path)  # indexing the same source again replaces it
        assert len(index) == count
        binary = tmp_path / "blob.txt"
        binary.write_bytes(b"\xff\xfe\x00")
        with pytest.raises(ValueError, match="UTF-8"):
            index.add_file(binary)


@pytest.mark.skipif(not HAS_PYPDF, reason="needs pypdf")
def test_add_file_indexes_pdfs_page_by_page(tmp_path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    index = SQLiteIndex(":memory:")
    assert index.add_file(path) == []  # blank pages have no text

    import orkexa.retrieval as retrieval

    pages = ["Page one talks about warranty.", "", "Page three is about shipping."]
    original = retrieval._pdf_pages
    retrieval._pdf_pages = lambda data: pages  # type: ignore[assignment]
    try:
        index.add_file(path, title="Manual")
    finally:
        retrieval._pdf_pages = original  # type: ignore[assignment]
    (hit,) = index.search("shipping", k=1)
    assert hit.title == "Manual, p. 3"
    assert hit.metadata == {"page": "3"}


def test_blocks_and_tool_return_search_results() -> None:
    index = SQLiteIndex(":memory:", max_chars=200, overlap=0)
    index.add(HANDBOOK, source="handbook.md", title="Handbook")
    (block,) = index.blocks("warranty", k=1)
    assert block["type"] == "search_result"
    assert block["title"] == "Handbook"
    assert block["citations"] == {"enabled": True}
    tool = index.tool()
    assert tool.name == "search"
    results = tool.execute({"query": "warranty", "k": 100})
    assert isinstance(results, list)
    assert all(result["type"] == "search_result" for result in results)
    assert tool.execute({"query": "zebra"}) == [{"type": "text", "text": "No results found."}]
    assert index.tool(citations=False).execute({"query": "warranty", "k": 1})[0]["citations"] == {
        "enabled": False
    }


def test_agent_with_index_and_citations() -> None:
    index = SQLiteIndex(":memory:")
    index.add(HANDBOOK, source="handbook.md", title="Handbook")
    answer = make_message(
        [
            {
                "type": "text",
                "text": "Two years.",
                "citations": [
                    {
                        "type": "search_result_location",
                        "source": "handbook.md",
                        "title": "Handbook",
                        "cited_text": "The warranty lasts two years from the date of purchase.",
                        "search_result_index": 0,
                        "start_block_index": 0,
                        "end_block_index": 1,
                    }
                ],
            }
        ]
    )
    client = FakeClient(
        [
            make_message(
                [{"type": "tool_use", "id": "toolu_0", "name": "search", "input": {"query": "warranty"}}],
                stop_reason="tool_use",
            ),
            answer,
        ]
    )
    agent: ClaudeAgent[str] = ClaudeAgent(client=client, index=index)  # type: ignore[arg-type]
    assert agent.ask("How long is the warranty?") == "Two years."
    result = client.messages.create_calls[1]["messages"][-1]["content"][0]
    assert result["content"][0]["type"] == "search_result"
    (cite,) = citations(agent.last_response)
    assert cite.source == "handbook.md"
    assert cite.cited_text.startswith("The warranty")
    assert citations(None) == []
    with pytest.raises(TypeError, match="Index"):
        ClaudeAgent(client=client, index="rag.db")  # type: ignore[arg-type]


def _embed(texts: list[str]) -> list[list[float]]:
    vocabulary = ("warranty", "return", "shipping")
    vectors = []
    for text in texts:
        lowered = text.lower()
        vector = [float(word in lowered) for word in vocabulary]
        if "guarantee" in lowered:
            vector[0] += 1.0
        vectors.append([*vector, 0.01])
    return vectors


@pytest.mark.skipif(not HAS_VEC, reason="needs sqlite-vec")
def test_hybrid_search_finds_synonyms_via_vectors(tmp_path: Path) -> None:
    index = SQLiteIndex(tmp_path / "rag.db", embedder=_embed)
    index.add("The warranty lasts two years.", source="warranty.md")
    index.add("Shipping takes three days.", source="shipping.md")
    assert index.search("guarantee period", k=1)[0].source == "warranty.md"  # no shared word
    assert index.delete("warranty.md") == 1
    assert [hit.source for hit in index.search("warranty")] == ["shipping.md"]
    with pytest.raises(ValueError, match="dimensional"):
        SQLiteIndex(tmp_path / "rag.db", embedder=lambda texts: [[1.0, 2.0] for _ in texts]).add(
            "x", source="x"
        )

    def wrong_count(texts: list[str]) -> Any:
        return []

    with pytest.raises(ValueError, match="one vector per text"):
        SQLiteIndex(":memory:", embedder=wrong_count).add("text", source="s")


def test_missing_sqlite_vec_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sqlite_vec":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"orkexa\[vector\]"):
        SQLiteIndex(":memory:", embedder=_embed)
