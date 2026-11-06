"""FX-RANK-0: the fixed ranking corpus of MUP-ONA (plan chapter 5.2), generated from a seed.

200 synthetic chunks in several sources with fixed 16-dimensional vectors and 30 queries with query
text and query vectors. Each chunk and query has two vectors: ``vector_norm`` (unit length, embedded
under an identity with ``normalized=True``) and ``vector_raw`` (the same direction with another length,
identity ``normalized=False``). Every component is a float32 value, so SQLite (float32 blobs),
PostgreSQL ``real[]`` and pgvector store it exactly.

The corpus exercises the PostgreSQL drift points of the plan (annex A): normalization (scaled copies of
one direction), equal scores (identical vectors and identical word bags), a zero vector, tokenization
(underscores, umlauts, diacritics, case), prefix matching, near-duplicate texts and several sources per
namespace.

Usage (from the repository root)::

    python tests/fixtures/ranking/make_rank_corpus.py write   # corpus and queries into FX-RANK-0.json
    python tests/fixtures/ranking/make_rank_corpus.py merge pg-a.json pg-b.json  # add measured results
    python tests/fixtures/ranking/make_rank_corpus.py check   # checksum and regeneration

``write`` keeps the measured results when the corpus is unchanged and drops them otherwise. Only the standard
library is used, so the generator runs anywhere.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import sys
from typing import Any

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "FX-RANK-0.json"
SEED = 20261106
DIMENSIONS = 16
K = 10
MODES = ("default", "fts", "vector", "hybrid")
FORMAT = 1

#: The two embedding identities the corpus is measured with (``CallableEmbeddings`` arguments).
IDENTITIES: dict[str, dict[str, Any]] = {
    "norm": {"provider": "fixture", "name": "fx-rank-0-norm", "dimensions": DIMENSIONS, "normalized": True},
    "raw": {"provider": "fixture", "name": "fx-rank-0-raw", "dimensions": DIMENSIONS, "normalized": False},
}

TOPICS: list[tuple[str, list[str]]] = [
    (
        "billing",
        [
            "invoice",
            "invoices",
            "invoicing",
            "refund",
            "refunds",
            "payment",
            "credit",
            "Rechnung",
            "Zahlung",
            "Rückerstattung",
            "ledger",
            "balance",
        ],
    ),
    (
        "shipping",
        [
            "parcel",
            "parcels",
            "delivery",
            "warehouse",
            "courier",
            "tracking",
            "Lieferung",
            "Paket",
            "Versand",
            "Straße",
            "route",
            "dispatch",
        ],
    ),
    (
        "people",
        [
            "vacation",
            "holiday",
            "payroll",
            "employee",
            "overtime",
            "Urlaub",
            "Mitarbeiter",
            "Gehalt",
            "Übergabe",
            "leave",
            "schedule",
            "onboarding",
        ],
    ),
    (
        "product",
        [
            "size",
            "Größe",
            "colour",
            "catalogue",
            "variant",
            "SKU",
            "fabric",
            "Farbe",
            "Modell",
            "café",
            "façade",
            "design",
        ],
    ),
    (
        "access",
        [
            "password",
            "token",
            "access_token",
            "login",
            "session",
            "Zugang",
            "Kennwort",
            "role",
            "permission",
            "audit",
            "rotation",
            "expiry",
        ],
    ),
    (
        "stock",
        [
            "inventory",
            "stock",
            "shelf",
            "reorder",
            "Lager",
            "Bestand",
            "Inventur",
            "pallet",
            "supplier",
            "Lieferant",
            "count",
            "batch",
        ],
    ),
    (
        "support",
        [
            "ticket",
            "escalation",
            "hotline",
            "customer_number",
            "Prüfung",
            "Maßnahme",
            "Anfrage",
            "Kunde",
            "response",
            "priority",
            "résumé",
            "naïve",
        ],
    ),
    (
        "travel",
        [
            "Zürich",
            "München",
            "Köln",
            "hotel",
            "flight",
            "itinerary",
            "Reisekosten",
            "Reise",
            "booking",
            "visa",
            "receipt",
            "mileage",
        ],
    ),
]
FILLER = [
    "the",
    "a",
    "of",
    "for",
    "and",
    "with",
    "per",
    "each",
    "der",
    "die",
    "das",
    "und",
    "mit",
    "für",
    "nach",
]
SOURCES_PER_TOPIC = 3

#: Drift points of annex A (B3) that a query exercises.
DRIFT = {
    1: "identity refresh",
    2: "NaN/Inf",
    3: "zero vector",
    4: "normalization (dot product vs cosine)",
    5: "score scale and rounding",
    6: "default mode",
    7: "source limit",
    8: "tokenization",
    9: "reindex race",
    10: "VectorStatus.legacy",
}


def f32(value: float) -> float:
    """``value`` rounded to the nearest float32 (exact in SQLite blobs, ``real[]`` and pgvector)."""
    return float(struct.unpack("<f", struct.pack("<f", value))[0])


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector]


class _Builder:
    def __init__(self) -> None:
        self.rng = random.Random(SEED)
        self.bases = [_unit([self.rng.gauss(0.0, 1.0) for _ in range(DIMENSIONS)]) for _ in TOPICS]
        self.chunks: list[dict[str, Any]] = []

    def direction(self, topic: int, noise: float) -> list[float]:
        base = self.bases[topic]
        return _unit([b + self.rng.gauss(0.0, noise) for b in base])

    def vectors(self, direction: list[float], scale: float | None = None) -> tuple[list[float], list[float]]:
        """``(vector_norm, vector_raw)``: the unit vector and a scaled copy (``scale`` default random)."""
        if scale is None:
            scale = round(self.rng.uniform(0.2, 5.0), 3)
        return [f32(x) for x in direction], [f32(x * scale) for x in direction]

    def add(
        self,
        text: str,
        source: str,
        title: str,
        group: str,
        vectors: tuple[list[float], list[float]],
    ) -> str:
        key = f"c{len(self.chunks) + 1:03d}"
        self.chunks.append(
            {
                "key": key,
                "source": source,
                "title": title,
                "text": text,
                "group": group,
                "vector_norm": vectors[0],
                "vector_raw": vectors[1],
            }
        )
        return key

    def sentence(self, topic: int, words: int) -> str:
        vocabulary = TOPICS[topic][1]
        picked = []
        for _ in range(words):
            roll = self.rng.random()
            if roll < 0.6:
                picked.append(self.rng.choice(vocabulary))
            elif roll < 0.9:
                picked.append(self.rng.choice(FILLER))
            else:
                picked.append(self.rng.choice(TOPICS[self.rng.randrange(len(TOPICS))][1]))
        picked[0] = picked[0][:1].upper() + picked[0][1:]
        return " ".join(picked) + "."


def _source(topic: int, number: int) -> tuple[str, str]:
    name = TOPICS[topic][0]
    return f"kb/{name}-{number}.md", f"{name.capitalize()} handbook {number}"


def build_corpus() -> dict[str, Any]:
    """The corpus and the queries (no results); the same on every run and platform."""
    b = _Builder()

    # Special chunks first, so their keys stay fixed when the regular part changes.
    # Equal scores: four identical vectors and four texts with the same words (order and case differ).
    tie_direction = b.direction(0, 0.5)
    tie_vectors = b.vectors(tie_direction, 2.0)
    for number, text in enumerate(
        [
            "Ledger tie marker quarterly.",
            "LEDGER TIE MARKER QUARTERLY.",
            "quarterly marker tie ledger.",
            "Tie Ledger Quarterly Marker.",
        ],
        start=1,
    ):
        b.add(text, f"special/ties-{number}.md", "Tie group", "tie", tie_vectors)

    # Normalization: one direction at lengths 1/4, 1 and 4 (powers of two keep the copies proportional).
    scale_direction = b.direction(1, 0.5)
    for scale, text in [
        (0.25, "Courier dispatch scale probe short."),
        (1.0, "Courier dispatch scale probe medium."),
        (4.0, "Courier dispatch scale probe long."),
    ]:
        b.add(text, "special/scale.md", "Scale probe", "scale", b.vectors(scale_direction, scale))

    # A zero vector (SQLite leaves it out of vector rankings; PostgreSQL stores and ranks it).
    zero = [0.0] * DIMENSIONS
    b.add("Invoice refund zero vector placeholder.", "special/zero.md", "Zero vector", "zero", (zero, zero))

    # Near-duplicates: one word, punctuation or case changed; vectors within 1e-3 of each other.
    near_texts = [
        ("The refund for invoice 4711 was approved by the ledger team.", 0),
        ("Parcels leave the warehouse within three days of the order.", 1),
        ("Employees request vacation two weeks before the holiday.", 2),
        ("Rotate the access token after every session expiry.", 4),
    ]
    for number, (text, topic) in enumerate(near_texts, start=1):
        direction = b.direction(topic, 0.3)
        variants = [
            text,
            text.replace("the", "a", 1),
            text.rstrip(".").upper() + "!",
        ]
        for variant in variants:
            nudged = _unit([x + b.rng.gauss(0.0, 1e-3) for x in direction])
            b.add(variant, f"special/near-{number}.md", f"Near duplicate {number}", "near", b.vectors(nudged))

    # Tokenization: underscores, umlauts and sharp s, diacritics, case.
    tokens = [
        ("The order_id field links each invoice to its order.", 0),
        ("Order id and customer number are printed on the invoice.", 0),
        ("Search by customer_number or by order_id in the ticket form.", 6),
        ("A snake_case_identifier is one word for some tokenizers.", 4),
        ("Die Größe der Lieferung wird an der Straße geprüft.", 1),
        ("Die Groesse der Lieferung wird an der Strasse geprueft.", 1),
        ("Grosse Pakete gehen über die Hauptstrasse.", 1),
        ("Das Café an der Façade in Zürich.", 7),
        ("Das Cafe an der Facade in Zurich.", 7),
        ("CAFÉ RÉSUMÉ NAÏVE IN CAPITALS.", 6),
        ("Ein naiver Lebenslauf: resume and naive spelled plainly.", 6),
        ("REFUND POLICY FOR ALL INVOICES.", 0),
        ("ReFuNd PoLiCy in mixed case.", 0),
        ("Übergabe der Mitarbeiter in München und Köln.", 2),
        ("Uebergabe der Mitarbeiter in Muenchen und Koeln.", 2),
    ]
    for number, (text, topic) in enumerate(tokens, start=1):
        b.add(
            text,
            f"special/tokens-{(number - 1) // 5 + 1}.md",
            "Tokenization",
            "tokens",
            b.vectors(b.direction(topic, 0.4)),
        )

    # Prefixes: "inv" finds invoice, invoices, invoicing, inventory and invent; "pay" payment and payroll.
    for text, topic in [
        ("Invoice", 0),
        ("Invoices are numbered per year.", 0),
        ("Invoicing runs every night.", 0),
        ("Inventory is counted at year end.", 5),
        ("They invent new shelf labels.", 5),
        ("Payment and payroll share the prefix pay.", 2),
    ]:
        b.add(text, "special/prefix.md", "Prefix probe", "prefix", b.vectors(b.direction(topic, 0.4)))

    # Words only in the title, numbers, and one long chunk (BM25 and ts_rank length normalization).
    b.add(
        "Figures for the second half are attached.",
        "special/title.md",
        "Quartalsbericht Lieferant",
        "title",
        b.vectors(b.direction(5, 0.4)),
    )
    b.add(
        "Summary of open items.",
        "special/title.md",
        "Quartalsbericht Reisekosten",
        "title",
        b.vectors(b.direction(7, 0.4)),
    )
    b.add(
        "SKU-4711 version v2 released in 2024.",
        "special/numbers.md",
        "Numbers",
        "numbers",
        b.vectors(b.direction(3, 0.4)),
    )
    b.add(
        "SKU 4711 v2 2024 recall notice.",
        "special/numbers.md",
        "Numbers",
        "numbers",
        b.vectors(b.direction(3, 0.4)),
    )
    b.add(
        " ".join(["refund"] * 40 + ["invoice"] * 40 + ["ledger"] * 40) + ".",
        "special/long.md",
        "Long chunk",
        "long",
        b.vectors(b.direction(0, 0.4)),
    )

    # Regular chunks: topic sentences spread over three sources per topic.
    special = len(b.chunks)
    regular = 200 - special
    for number in range(regular):
        topic = number % len(TOPICS)
        source, title = _source(topic, (number // len(TOPICS)) % SOURCES_PER_TOPIC + 1)
        text = b.sentence(topic, b.rng.randint(8, 28))
        b.add(text, source, title, "regular", b.vectors(b.direction(topic, 0.35)))

    texts = [chunk["text"] for chunk in b.chunks]
    duplicates = {text for text in texts if texts.count(text) > 1}
    for text in duplicates:  # equal texts must carry equal vectors: the fixture embedder looks texts up
        found = {json.dumps([c["vector_norm"], c["vector_raw"]]) for c in b.chunks if c["text"] == text}
        assert len(found) == 1, text

    queries = _queries(b, tie_direction, scale_direction)
    return {"chunks": b.chunks, "queries": queries}


def _queries(b: _Builder, tie_direction: list[float], scale_direction: list[float]) -> list[dict[str, Any]]:
    entries: list[tuple[str, list[float] | None, list[int], str, list[str] | None]] = [
        # (text, direction or None for zero, drift points, purpose, sources)
        ("refund for an invoice", b.direction(0, 0.25), [4, 5, 6], "topic query, billing", None),
        ("parcel delivery tracking", b.direction(1, 0.25), [4, 5, 6], "topic query, shipping", None),
        ("vacation and overtime of employees", b.direction(2, 0.25), [4, 5], "topic query, people", None),
        ("catalogue variant colour", b.direction(3, 0.25), [4, 5], "topic query, product", None),
        ("password rotation audit", b.direction(4, 0.25), [4, 5], "topic query, access", None),
        ("inventory reorder supplier", b.direction(5, 0.25), [4, 5], "topic query, stock", None),
        ("ticket escalation priority", b.direction(6, 0.25), [4, 5], "topic query, support", None),
        ("hotel booking itinerary", b.direction(7, 0.25), [4, 5], "topic query, travel", None),
        (
            "ledger tie marker",
            _unit([x + b.rng.gauss(0.0, 0.05) for x in tie_direction]),
            [4, 5],
            "equal scores: identical vectors and word bags; tie-break by chunk id",
            None,
        ),
        (
            "courier dispatch scale probe",
            _unit([x + b.rng.gauss(0.0, 0.05) for x in scale_direction]),
            [4],
            "normalization: one direction at three lengths (cosine equal, dot product not)",
            None,
        ),
        ("invoice refund placeholder", None, [3], "zero query vector; zero chunk vector in the corpus", None),
        (
            "zero vector placeholder invoice",
            b.direction(0, 0.3),
            [3],
            "a chunk with a zero vector matches the text",
            None,
        ),
        ("order_id", b.direction(0, 0.4), [8], "underscore: one word or two", None),
        ("customer_number lookup", b.direction(6, 0.4), [8], "underscore in a longer query", None),
        ("Größe Straße", b.direction(1, 0.4), [8], "umlaut and sharp s", None),
        ("groesse strasse", b.direction(1, 0.4), [8], "transliterated umlauts (no folding to ö/ß)", None),
        ("cafe facade zurich", b.direction(7, 0.4), [8], "diacritics removed in the query", None),
        ("CAFÉ Résumé naïve", b.direction(6, 0.4), [8], "diacritics and case in the query", None),
        ("REFUND POLICY", b.direction(0, 0.4), [8], "upper case query", None),
        ("inv", b.direction(5, 0.5), [8], "prefix matching (short prefix)", None),
        ("pay", b.direction(2, 0.5), [8], "prefix matching (payment, payroll)", None),
        ("SKU-4711 v2", b.direction(3, 0.4), [8], "numbers and hyphen", None),
        ("Quartalsbericht", b.direction(5, 0.4), [5], "word only in titles", None),
        (
            "refund for invoice 4711 approved by the ledger team",
            b.direction(0, 0.3),
            [5],
            "near-duplicate texts",
            None,
        ),
        ("Übergabe Mitarbeiter München", b.direction(2, 0.4), [8], "German query with umlauts", None),
        ("zzqx wvvk", b.direction(4, 0.5), [5, 6], "no word matches: fts empty, vector still ranks", None),
        ("?!", b.direction(1, 0.5), [5, 6], "no words at all", None),
        (
            "refund invoice ledger",
            b.direction(0, 0.3),
            [5],
            "long chunk with repeated words (length norm)",
            None,
        ),
        (
            "parcel warehouse delivery",
            b.direction(1, 0.25),
            [7],
            "limited to two sources before ranking",
            ["kb/shipping-1.md", "special/near-2.md"],
        ),
        (
            "refund invoice",
            b.direction(0, 0.25),
            [7],
            "limited to sources that hold ties and the zero vector",
            ["special/ties-1.md", "special/ties-2.md", "special/zero.md", "kb/billing-2.md"],
        ),
    ]
    assert len(entries) == 30
    queries = []
    for number, (text, direction, drift, purpose, sources) in enumerate(entries, start=1):
        zero = [0.0] * DIMENSIONS
        vectors = (zero, zero) if direction is None else b.vectors(direction)
        queries.append(
            {
                "id": f"q{number:02d}",
                "text": text,
                "purpose": purpose,
                "drift": drift,
                "sources": sources,
                "vector_norm": vectors[0],
                "vector_raw": vectors[1],
            }
        )
    texts = [query["text"] for query in queries]
    assert len(set(texts)) == len(texts), "query texts must be unique (the fixture embedder looks them up)"
    return queries


# ---- the fixture file ----------------------------------------------------------------------------------


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def dumps(value: Any, indent: int = 0) -> str:
    """Stable JSON: sorted keys, one entry per line, lists of scalars on one line (vectors, hits)."""
    pad = " " * indent
    inner = " " * (indent + 1)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [f"{inner}{json.dumps(key)}: {dumps(value[key], indent + 1)}" for key in sorted(value)]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(value, list):
        if all(_scalar(item) for item in value):
            return json.dumps(value, ensure_ascii=False, allow_nan=False)
        items = [f"{inner}{dumps(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_fixture(data: dict[str, Any], path: Path = FIXTURE) -> None:
    """Write the fixture and its ``.sha256`` (``sha256sum`` format)."""
    path.write_text(dumps(data) + "\n", encoding="utf-8")
    Path(f"{path}.sha256").write_text(f"{checksum(path)}  {path.name}\n", encoding="utf-8")


def load_fixture(path: Path = FIXTURE) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def header() -> dict[str, Any]:
    corpus = build_corpus()
    return {
        "fixture": "FX-RANK-0",
        "format": FORMAT,
        "description": (
            "MUP-ONA ranking corpus (plan 5.2): synthetic chunks and queries with fixed float32 vectors, "
            "and the top-k results of the code before MP-B1/B3 per backend. Hits: [chunk key, backend "
            "chunk id, score as "
            "returned (repr), score rounded to 6 places (repr), ranks]. Generated by make_rank_corpus.py."
        ),
        "generator": {
            "dimensions": DIMENSIONS,
            "seed": SEED,
            "script": "tests/fixtures/ranking/make_rank_corpus.py",
        },
        "identities": IDENTITIES,
        "k": K,
        "modes": list(MODES),
        "drift_points": {str(number): name for number, name in DRIFT.items()},
        "chunks": corpus["chunks"],
        "queries": corpus["queries"],
    }


def _write(path: Path) -> None:
    data = header()
    if path.exists():
        old = load_fixture(path)
        if corpus_digest(old) == corpus_digest(data):
            data["results"] = old.get("results", {})
        else:
            print("corpus changed: measured results dropped, measure again", file=sys.stderr)
    data.setdefault("results", {})
    write_fixture(data, path)


def _merge(path: Path, files: list[Path]) -> None:
    data = load_fixture(path)
    for file in files:
        measured = json.loads(file.read_text(encoding="utf-8"))
        if measured.get("corpus_sha256") != corpus_digest(data):
            raise SystemExit(f"{file}: measured on another corpus")
        for backend, result in measured["results"].items():
            data["results"][backend] = result
    write_fixture(data, path)


def corpus_digest(data: dict[str, Any]) -> str:
    """SHA-256 of the corpus part (chunks, queries, identities, k, modes): ties measurements to a corpus."""
    part = {key: data[key] for key in ("chunks", "queries", "identities", "k", "modes")}
    return hashlib.sha256(json.dumps(part, sort_keys=True).encode()).hexdigest()


def _check(path: Path) -> None:
    recorded = Path(f"{path}.sha256").read_text(encoding="utf-8").split()[0]
    if recorded != checksum(path):
        raise SystemExit("checksum mismatch")
    data = load_fixture(path)
    if any(data.get(key) != value for key, value in header().items()):
        raise SystemExit("the generator no longer produces the corpus of the fixture")
    print(f"ok: {path.name}, backends: {', '.join(sorted(data['results']))}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["write", "merge", "check"])
    parser.add_argument(
        "files", nargs="*", type=Path, help="merge: measured result files (measure_pg.py --out)"
    )
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    args = parser.parse_args(argv)
    if args.command == "write":
        _write(args.fixture)
    elif args.command == "merge":
        _merge(args.fixture, args.files)
    else:
        _check(args.fixture)


if __name__ == "__main__":
    main()
