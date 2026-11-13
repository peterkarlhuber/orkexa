# Legacy fixtures (FX-SQ-0)

SQLite databases written by Orkexa **before** the MUP-ONA backend work (packages B1/B2 move the SQLite classes
to `orkexa.backends.sqlite`). Later changes must keep such files readable and writable: MUP-ONA T-SQL-10 and the
upgrade scenario U6 open them with the current code. Never edit or regenerate the committed files to make a test
pass; a regenerated file is a new fixture and needs a reason in the pull request.

| File | Written with | Contents |
|---|---|---|
| `FX-SQ-0/index_fts.db` | `SQLiteIndex` | full text only; namespaces `handbook` (3 sources) and `tickets` (2 sources) |
| `FX-SQ-0/index_vec.db` | `SQLiteIndex` + sqlite-vec | namespace `spaces`: a vector space of a 4-dimensional `CallableEmbeddings` model, one chunk with a zero vector; namespace `legacy`: an `embedder=` callable (`vec0` table) |
| `FX-SQ-0/files.db` | `SQLiteFileStore` | namespaces `alpha` (text, Markdown) and `beta` (a PDF-typed placeholder) |
| `FX-SQ-0/memory.db` | `SQLiteMemory` | namespaces `alpha` (nested files) and `beta` |
| `FX-SQ-0/manifest.json` | the generator | Orkexa commit and versions used, checksums, expected sources, file ids and memory paths |

Every database has a `<name>.sha256` file in `sha256sum` format. The test
`tests/test_legacy_fixtures.py` (MUP-ONA T-FX-1) checks the checksums and opens copies of the files.

## Generating

```bash
pip install -r requirements/tabrion.txt -e ".[dev]"
rm tests/fixtures/legacy/FX-SQ-0/*.db   # the generator never overwrites
python tests/fixtures/legacy/make_legacy_sqlite.py
```

`make_legacy_sqlite.py` writes only through the public API, with synthetic texts and deterministic vectors (no
embedding server, no network, no randomness). SQLite page layout and the `created_at` timestamps of stored files
differ between runs, so a regenerated database gets a new checksum: the checksums pin the committed files.
The committed files were written at orkexa `80b95e5` (Python 3.13, SQLite 3.53.1, sqlite-vec 0.1.9;
see `manifest.json`).

The databases are excluded from the source distribution (`pyproject.toml`, `[tool.hatch.build.targets.sdist]`),
whose packaging test refuses `*.db` files.
