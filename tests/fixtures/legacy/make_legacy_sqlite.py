"""Write the FX-SQ-0 legacy SQLite fixtures with the installed Orkexa (MUP-ONA A0).

Usage (from the repository root, with the ``dev`` extra installed)::

    python tests/fixtures/legacy/make_legacy_sqlite.py [--out tests/fixtures/legacy/FX-SQ-0]

The files record what databases written by Orkexa *before* the backend move (MUP-ONA B1/B2) look like, so
later packages can prove that such files stay readable and writable (MUP-ONA T-SQL-10, upgrade scenario U6).
Everything is written through the public API; all texts, names and vectors are synthetic and deterministic (no
randomness, no network, no embedding server). The generator refuses to overwrite existing files: delete them
first on purpose.

Files (each with a ``<name>.sha256`` next to it, in ``sha256sum`` format):

* ``index_fts.db``: ``SQLiteIndex`` without vectors; namespaces ``handbook`` and ``tickets``.
* ``index_vec.db``: ``SQLiteIndex`` with sqlite-vec; namespace ``spaces`` (an ``EmbeddingModel`` vector space,
  including one zero vector) and namespace ``legacy`` (the older ``embedder=`` callable, a ``vec0`` table).
* ``files.db``: ``SQLiteFileStore`` with a text, a Markdown and a PDF-typed entry in namespaces
  ``alpha``/``beta``.
* ``memory.db``: ``SQLiteMemory`` with nested memory files in namespaces ``alpha``/``beta``.

``manifest.json`` records the Orkexa commit, the versions used and the expected contents that MUP-ONA T-FX-1
and T-SQL-10 check. SQLite pages and the ``created_at`` timestamps of stored files differ between runs, so a
regenerated file has another checksum; the checksums pin the committed files, not the generator's output.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys

DIMENSIONS = 4

HANDBOOK = {
    "warranty.md": "The warranty covers two years from the date of purchase.\n\n"
    "Refunds are paid within fourteen days after the returned item arrives.",
    "shipping.md": "Parcels ship from the central warehouse every weekday.\n\n"
    "Express delivery arrives on the next business day.",
    "privacy.md": "Customer records are deleted on request. Backups expire after thirty days.",
}
TICKETS = {
    "ticket-1001": "The invoice for order 1001 shows the wrong delivery address.",
    "ticket-1002": "A refund for order 1002 was requested because the parcel arrived damaged.",
}
SPACE_TEXTS = {
    "alpha.txt": "Alpha station reports a stable temperature.",
    "beta.txt": "Beta station reports rising humidity in the north wing.",
    "gamma.txt": "Gamma station is offline for maintenance.",
    "zero.txt": "Zero vector chunk: it has no cosine distance and never ranks.",
}
LEGACY_TEXTS = {
    "legacy-a.txt": "Legacy embedder chunk about solar panels.",
    "legacy-b.txt": "Legacy embedder chunk about wind turbines.",
}
FILES = {
    "alpha": [
        ("notes.txt", b"Plain text note for the alpha project.\n", "Alpha notes"),
        ("readme.md", b"# Alpha\n\nMarkdown readme of the alpha project.\n", "Alpha readme"),
    ],
    "beta": [
        # Stored as application/pdf by name; the bytes are a minimal synthetic PDF header, never parsed.
        ("scan.pdf", b"%PDF-1.4\n% synthetic placeholder for MUP-ONA FX-SQ-0\n%%EOF\n", "Beta scan"),
    ],
}
MEMORIES = {
    "alpha": {
        "/memories/profile.md": "Prefers short answers.\nTime zone: UTC.\n",
        "/memories/projects/alpha/status.md": "Phase 2 started.\nNext review in March.\n",
    },
    "beta": {
        "/memories/profile.md": "Prefers detailed answers.\n",
    },
}

_WORDS = ("station", "temperature", "humidity", "maintenance", "solar", "wind")


def vectors_for(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic 4-dimensional vectors; a text that starts with "Zero vector" gets the zero vector."""
    result: list[list[float]] = []
    for text in texts:
        if text.startswith("Zero vector"):
            result.append([0.0] * DIMENSIONS)
            continue
        lowered = text.lower()
        counts = [float(lowered.count(word)) for word in _WORDS]
        vector = [counts[0] + 0.25, counts[1] + counts[4], counts[2] + counts[5], counts[3] + 0.5]
        result.append([round(value, 6) for value in vector])
    return result


def _git_head(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path) -> dict[str, object]:
    from orkexa import SQLiteFileStore, SQLiteIndex, SQLiteMemory
    from orkexa.embeddings import CallableEmbeddings

    out.mkdir(parents=True, exist_ok=True)
    names = ("index_fts.db", "index_vec.db", "files.db", "memory.db")
    existing = [name for name in names if (out / name).exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite {', '.join(existing)} in {out}; delete them first")

    with SQLiteIndex(out / "index_fts.db", namespace="handbook") as index:
        for source, text in HANDBOOK.items():
            index.add(text, source=source, title=source.removesuffix(".md").title(), metadata={"kind": "doc"})
    with SQLiteIndex(out / "index_fts.db", namespace="tickets") as index:
        for source, text in TICKETS.items():
            index.add(text, source=source, metadata={"kind": "ticket"})

    model = CallableEmbeddings(vectors_for, name="fx-sq-0-synthetic", dimensions=DIMENSIONS)
    with SQLiteIndex(out / "index_vec.db", namespace="spaces", embeddings=model) as index:
        for source, text in SPACE_TEXTS.items():
            index.add(text, source=source)
    with SQLiteIndex(out / "index_vec.db", namespace="legacy", embedder=vectors_for) as index:
        for source, text in LEGACY_TEXTS.items():
            index.add(text, source=source)

    file_ids: dict[str, list[str]] = {}
    for namespace, entries in FILES.items():
        with SQLiteFileStore(out / "files.db", namespace=namespace) as store:
            file_ids[namespace] = [
                store.add(data, name=name, description=description).id for name, data, description in entries
            ]

    for namespace, files in MEMORIES.items():
        memory = SQLiteMemory(out / "memory.db", namespace=namespace)
        try:
            for path, text in files.items():
                memory.create(path, text)
        finally:
            memory.close()

    checksums: dict[str, str] = {}
    for name in names:
        path = out / name
        # The last connection closed: the WAL was checkpointed into the database file and removed.
        for suffix in ("-wal", "-shm"):
            if Path(f"{path}{suffix}").exists():
                raise SystemExit(f"{name}{suffix} was left behind; the database is not self-contained")
        digest = _sha256(path)
        checksums[name] = digest
        (out / f"{name}.sha256").write_text(f"{digest}  {name}\n", encoding="utf-8")

    import orkexa

    manifest: dict[str, object] = {
        "fixture": "FX-SQ-0",
        "plan": "MUP-ONA 5.2 (package MP-A0)",
        "orkexa_commit": _git_head(Path(__file__).resolve().parents[3]),
        "orkexa_version": getattr(orkexa, "__version__", "unknown"),
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "sqlite_vec": _sqlite_vec_version(),
        "checksums": checksums,
        "contents": {
            "index_fts.db": {
                "handbook": sorted(HANDBOOK),
                "tickets": sorted(TICKETS),
            },
            "index_vec.db": {
                "spaces": sorted(SPACE_TEXTS),
                "legacy": sorted(LEGACY_TEXTS),
                "embedding_model": "fx-sq-0-synthetic",
                "dimensions": DIMENSIONS,
            },
            "files.db": file_ids,
            "memory.db": {namespace: sorted(files) for namespace, files in MEMORIES.items()},
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _sqlite_vec_version() -> str:
    try:
        import sqlite_vec  # type: ignore[import-not-found,import-untyped,unused-ignore]
    except ImportError:
        return "missing"
    return str(getattr(sqlite_vec, "__version__", "unknown"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "FX-SQ-0")
    args = parser.parse_args(argv)
    manifest = build(args.out)
    print(json.dumps(manifest["checksums"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
