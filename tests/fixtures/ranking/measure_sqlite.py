"""Measure FX-RANK-0 on :class:`orkexa.SQLiteIndex` and store the results in ``FX-RANK-0.json``.

Two backends are recorded:

* ``sqlite``: ``SQLiteIndex(embeddings=...)`` with ``sqlite_vec`` (the ``vector`` extra), every mode.
* ``sqlite-novec``: the same without ``sqlite_vec``. ``SQLiteIndex(embeddings=...)`` then raises
  ``ImportError`` (recorded under ``meta.notes.<identity>.embeddings_error``), so the index is opened
  without an embedding model: full text only, ``vector`` and ``hybrid`` give ``RetrievalError``.

For each identity (``norm``, ``raw``) a fresh database holds the 200 chunks (added in corpus order in
one ``add_chunks`` call, so chunk ids are 1..200), and every query runs in every mode with ``k`` hits.
``default`` passes no mode; ``mode_used`` names the explicit modes whose result equals it.

Usage (from the repository root)::

    python tests/fixtures/ranking/measure_sqlite.py                 # measure, update FX-RANK-0.json
    python tests/fixtures/ranking/measure_sqlite.py --print sqlite  # print one backend, write nothing
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
from typing import Any

from orkexa import Chunk, Hit, SQLiteIndex
from orkexa.embeddings import CallableEmbeddings

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_rank_corpus as corpus  # noqa: E402

del sys.path[0]


def embeddings(data: dict[str, Any], identity: str) -> CallableEmbeddings:
    """The fixture's embedding model for ``identity``: looks the vector of each text up in the corpus."""
    field = f"vector_{identity}"
    documents = {chunk["text"]: chunk[field] for chunk in data["chunks"]}
    queries = {query["text"]: query[field] for query in data["queries"]}
    spec = data["identities"][identity]
    return CallableEmbeddings(
        lambda texts: [documents[text] for text in texts],
        query_embedder=lambda texts: [queries[text] for text in texts],
        name=spec["name"],
        dimensions=spec["dimensions"],
        provider=spec["provider"],
        normalized=spec["normalized"],
    )


def chunks(data: dict[str, Any]) -> list[Chunk]:
    return [
        Chunk(text=c["text"], source=c["source"], title=c["title"], metadata={"key": c["key"]})
        for c in data["chunks"]
    ]


def hit_row(hit: Hit) -> list[Any]:
    """``[chunk key, id, score as returned, score rounded to 6 places, ranks]``.

    Scores are ``repr`` strings, so NaN and every bit of the float survive JSON."""
    ranks = ",".join(f"{name}={rank}" for name, rank in sorted(hit.ranks.items()))
    return [hit.metadata.get("key"), hit.id, repr(hit.score), repr(round(hit.score, 6)), ranks]


def measure_queries(retrieve: Callable[..., list[Hit]], data: dict[str, Any]) -> dict[str, Any]:
    """Every query in every mode.

    ``{query id: {mode: {"hits": [...], "mode_used": ...} or {"error": {...}}}}``."""
    results: dict[str, Any] = {}
    for query in data["queries"]:
        by_mode: dict[str, Any] = {}
        for mode in data["modes"]:
            try:
                hits = retrieve(
                    query["text"],
                    data["k"],
                    mode=None if mode == "default" else mode,
                    sources=query["sources"],
                )
            except Exception as exc:
                by_mode[mode] = {
                    "error": {
                        "code": getattr(exc, "code", None),
                        "message": str(exc),
                        "type": type(exc).__name__,
                    }
                }
            else:
                by_mode[mode] = {"hits": [hit_row(hit) for hit in hits], "mode_used": mode}
        default = by_mode["default"]
        if "hits" in default:
            same = [
                m for m in ("fts", "vector", "hybrid") if by_mode.get(m, {}).get("hits") == default["hits"]
            ]
            default["mode_used"] = "|".join(same) or "unknown"
        results[query["id"]] = by_mode
    return results


@contextmanager
def without_sqlite_vec() -> Iterator[None]:
    """``import sqlite_vec`` fails inside, as when the ``vector`` extra is not installed."""
    saved = sys.modules.get("sqlite_vec")
    sys.modules["sqlite_vec"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        if saved is None:
            del sys.modules["sqlite_vec"]
        else:
            sys.modules["sqlite_vec"] = saved


def measure_identity(
    data: dict[str, Any], identity: str, directory: Path, *, vec: bool
) -> tuple[Any, dict[str, Any]]:
    """``(results, notes)`` of one identity on a fresh database in ``directory``."""
    model = embeddings(data, identity)
    notes: dict[str, Any] = {}
    database = directory / f"{identity}-{'vec' if vec else 'novec'}.db"
    if vec:
        index = SQLiteIndex(database, embeddings=model)
    else:
        with without_sqlite_vec():
            try:
                SQLiteIndex(directory / "probe.db", embeddings=model).close()
            except ImportError as exc:
                notes["embeddings_error"] = f"{type(exc).__name__}: {exc}"
            index = SQLiteIndex(database)
    with index:
        ids = index.add_chunks(chunks(data))
        assert ids == list(range(1, len(data["chunks"]) + 1)), "chunk ids must follow the corpus order"
        notes["index_mode"] = index.mode
        notes["vector_status"] = index.vector_status().model_dump(mode="json")
        if vec:
            results = measure_queries(index.retrieve, data)
        else:
            with without_sqlite_vec():
                results = measure_queries(index.retrieve, data)
    return results, notes


def measure(data: dict[str, Any], directory: Path, *, vec: bool) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    notes: dict[str, Any] = {}
    for identity in data["identities"]:
        runs[identity], notes[identity] = measure_identity(data, identity, directory, vec=vec)
    return {"runs": runs, "notes": notes}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()


def meta(*, vec: bool) -> dict[str, Any]:
    import orkexa

    found: dict[str, Any] = {
        "backend": "sqlite" if vec else "sqlite-novec",
        "class": "orkexa.SQLiteIndex",
        "image": None,
        "orkexa_sha": _git("rev-parse", "HEAD"),
        "orkexa_src_dirty": bool(_git("status", "--porcelain", "--", str(Path(orkexa.__file__).parent))),
        "runtime_sha": None,
        "python": platform.python_version(),
        "sqlite_version": sqlite3.sqlite_version,
        "sqlite_vec": None,
    }
    if vec:
        import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]

        connection = sqlite3.connect(":memory:")
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        found["sqlite_vec"] = connection.execute("SELECT vec_version()").fetchone()[0]
        connection.close()
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", type=Path, default=corpus.FIXTURE)
    parser.add_argument(
        "--print", dest="show", choices=["sqlite", "sqlite-novec"], help="print, do not write"
    )
    args = parser.parse_args(argv)
    data = corpus.load_fixture(args.fixture)
    if corpus.corpus_digest(data) != corpus.corpus_digest(corpus.header()):
        raise SystemExit("FX-RANK-0.json does not hold the generated corpus: run make_rank_corpus.py write")
    with tempfile.TemporaryDirectory() as tmp:
        if args.show:
            variants = [args.show == "sqlite"]
        elif importlib.util.find_spec("sqlite_vec") is not None:
            variants = [True, False]
        else:
            raise SystemExit('Measuring needs sqlite-vec: pip install -e ".[dev]"')
        for vec in variants:
            measured = measure(data, Path(tmp), vec=vec)
            result = {"meta": meta(vec=vec) | {"notes": measured["notes"]}, "runs": measured["runs"]}
            if args.show:
                print(json.dumps(result, indent=1, sort_keys=True))
                return
            data["results"][result["meta"]["backend"]] = result
    corpus.write_fixture(data, args.fixture)


if __name__ == "__main__":
    main()
