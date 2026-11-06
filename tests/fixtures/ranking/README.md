# FX-RANK-0: fixed ranking corpus (MUP-ONA)

FX-RANK-0 is the ranking baseline of MUP-ONA (plan chapter 5.2, package MP-A0 step 5). It records how the code
**before** MP-B1/B3 ranks a fixed corpus on every backend, so that later packages (MP-B3 report G1, plan 7.2 and
equivalence criteria 10.6) can compare before and after. All texts and vectors are synthetic.

| File | Purpose |
|---|---|
| `make_rank_corpus.py` | Deterministic generator (standard library only): corpus, queries, the stable JSON writer, `merge` and `check`. |
| `measure_sqlite.py` | Measures `orkexa.SQLiteIndex` with and without `sqlite_vec` and writes the results into the fixture. |
| `FX-RANK-0.json` | Corpus, queries and the results per backend. |
| `FX-RANK-0.json.sha256` | Checksum (`sha256sum` format). |

The PostgreSQL measurement lives in nodestro-runtime (`tests/fixtures/ranking/measure_pg.py`), which keeps a
byte-identical copy of `FX-RANK-0.json` and its checksum. The tests are `tests/test_rank_fixture.py` in both
repositories (T-RANK-0).

## Corpus

* 200 chunks (`c001`..`c200`) in 41 sources of one namespace, each with a title; chunk ids are 1..200 on every
  backend (one `add_chunks` call in corpus order on a fresh database or schema). Hits also carry the chunk key in
  `metadata["key"]`.
* 16-dimensional vectors from seed `20261106`, every component a float32 value (stored exactly by SQLite blobs,
  PostgreSQL `real[]` and pgvector). Each chunk and query has `vector_norm` (unit length, identity
  `fx-rank-0-norm` with `normalized=True`) and `vector_raw` (same direction, length 0.2..5, identity
  `fx-rank-0-raw` with `normalized=False`, the `CallableEmbeddings` default).
* Groups (`group` field) for the drift points of plan annex A:
  * `tie`: four identical vectors and four texts with the same words in other order and case (equal BM25,
    `ts_rank` and cosine scores; tie-break by chunk id).
  * `scale`: one direction at lengths 1/4, 1 and 4 (cosine equal; dot product not).
  * `zero`: a zero vector (`c008`).
  * `near`: four groups of three near-duplicate texts (one word, punctuation, case), vectors within 1e-3.
  * `tokens`: underscores (`order_id`, `customer_number`), umlauts and sharp s, transliterations (`Groesse`),
    diacritics (`Café`, `naïve`, `résumé`), upper and mixed case.
  * `prefix`: `invoice`/`invoices`/`invoicing`/`inventory`/`invent`, `payment`/`payroll`.
  * `title`, `numbers`, `long` (one chunk repeating three words 40 times: length normalization).
  * `regular`: topic sentences (eight topics, English and German-like words).
* 30 queries (`q01`..`q30`) with text, vectors, `purpose`, the annex A drift points they exercise (`drift`) and an
  optional `sources` list (passed as `retrieve(..., sources=...)`). `q11` has a zero query vector; `q29` and `q30`
  limit the sources before ranking.

## Results

`results.<backend>.runs.<identity>.<query id>.<mode>` holds either `{"hits": [...], "mode_used": ...}` or
`{"error": {"type", "code", "message"}}`. Modes: `default` (no `mode` argument), `fts`, `vector`, `hybrid`, each
with `k = 10`. For `default`, `mode_used` names the explicit modes whose hits are identical (`fts`, `hybrid`, or
several joined by `|` when they coincide). A hit is

```
[chunk key, backend chunk id, score as returned (repr), score rounded to 6 places (repr), ranks]
```

Scores are `repr` strings, so every bit and `nan` survive JSON. `ranks` is the `Hit.ranks` mapping as
`name=rank` pairs.

| Backend | Code | Server |
|---|---|---|
| `sqlite` | `orkexa.SQLiteIndex(embeddings=...)` with `sqlite_vec` | - |
| `sqlite-novec` | `orkexa.SQLiteIndex` without `sqlite_vec` (with `embeddings=` it raises `ImportError`, recorded in `meta.notes`; measured without an embedding model) | - |
| `pg-a` | `nodestro_runtime.postgres_tables.PostgresIndex`, Python ranking | `postgres:17` |
| `pg-b` | the same, pgvector ranking (extension created by migration 0010) | `pgvector/pgvector:pg17` |

`meta` records the backend, class, image digest, orkexa and runtime SHAs, versions and per identity the index's
own `mode` attribute, `VectorStatus` and (PostgreSQL) the detected `pgvector`.

## Regenerate

From the orkexa repository root (dev extras installed, so `sqlite-vec` is present):

```sh
python tests/fixtures/ranking/make_rank_corpus.py write     # corpus and queries; keeps results if unchanged
python tests/fixtures/ranking/measure_sqlite.py             # sqlite and sqlite-novec
```

From the nodestro-runtime root, after copying the fixture there, against a server each (a role that may create
schemas; each identity gets its own schema, dropped afterwards):

```sh
python tests/fixtures/ranking/measure_pg.py --dsn postgresql://postgres:PW@localhost:PORT/postgres \
    --backend pg-a --image postgres@sha256:748a18a5... --out /tmp/pg-a.json
python tests/fixtures/ranking/measure_pg.py --dsn ... --backend pg-b \
    --image pgvector/pgvector:pg17@sha256:7d0a4146... --out /tmp/pg-b.json
```

Back in orkexa:

```sh
python tests/fixtures/ranking/make_rank_corpus.py merge /tmp/pg-a.json /tmp/pg-b.json
python tests/fixtures/ranking/make_rank_corpus.py check
cp tests/fixtures/ranking/FX-RANK-0.json* ../nodestro-runtime/tests/fixtures/ranking/
```

The baseline is a record of the state before MP-B1/B3: re-measure it only to replace that baseline on purpose.
After a deliberate ranking change, compare against it (plan 7.2) instead of overwriting it.
