# Memory, files and RAG

The storage layer is provider-neutral; only its rendering differs.

| Feature | Anthropic | OpenAI, Gemini, compatible |
| --- | --- | --- |
| `memory=` (`EphemeralMemory`, `FileMemory`, `SQLiteMemory`) | native memory tool | function tool `memory` with the same commands |
| `files=` (`LocalFileStore`, `SQLiteFileStore`) | tool `files`; loads images/documents as blocks | same tool; images/PDFs as media input where supported, text documents as attributed text |
| `index=` (`SQLiteIndex`) | tool `search`; hits are `search_result` blocks with API-native citations | same tool; hits are attributed text `[Search result | source: ... | title: ...]` |

## Three kinds of sources

```python
from orkexa import citations, retrieval_sources

answer = agent.ask("How long is the warranty?")
retrieval_sources(agent)  # passages the search tool really delivered (kind="retrieval")
citations(agent.last_response)  # citations attached by the provider API (kind="api_native")
```

Source claims the model writes into its text are neither and are never parsed as
sources. `retrieval_sources` works for every provider; API-native citations exist
where the provider returns them (Anthropic search-result/document citations, OpenAI
URL/file annotations).

`SQLiteIndex` uses FTS5/BM25 by default and offers vector or hybrid search with an embedding model
(`embeddings=`, `orkexa[vector]`; see [Local models, embeddings and retrieval modes](local-models.md))
and PDF extraction (`orkexa[pdf]`). The older `embedder=` callables keep working.

## PDF limits

`Index.add_file()` extracts PDF text with `orkexa.retrieval.pdf_pages()` (needs `orkexa[pdf]`, pypdf 6.21.4 or
later), which stops between pages at a page and a character budget. For PDFs from untrusted sources, extract the
text with `pdf_pages_isolated()` and index it yourself: it runs pypdf in a separate interpreter that is stopped at
the time budget, whatever pypdf does.

```python
from orkexa.retrieval import RetrievalError, pdf_pages_isolated

try:
    pages = pdf_pages_isolated(data, timeout=30, max_memory_mb=512, max_pages=2000, max_chars=5_000_000)
except RetrievalError as exc:  # exc.code == "document.too_complex"
    print(exc.reason)  # "timeout", "memory", "pages", "chars", "limit" (pypdf's own limit) or "crash"
for number, text in enumerate(pages, start=1):
    index.add(text, source="report.pdf", title=f"Report, p. {number}", metadata={"page": str(number)})
```

| Budget (`PdfLimits`) | Default | `pdf_pages` | `pdf_pages_isolated` |
| --- | --- | --- | --- |
| `max_pages` | 2000 | yes (checked before the page tree is read, and while reading it) | yes |
| `max_chars` | 5,000,000 | yes, between pages | yes, between pages; the parent also reads at most `max_chars * 4 + 64 KiB` bytes |
| `timeout` (seconds, wall clock) | 30 | no | yes: `SIGTERM` to the worker's process group, `SIGKILL` 2 seconds later (Windows: terminate, then kill) |
| `max_memory_mb` | 512 | no | Linux only (`RLIMIT_AS`) |

The worker (`python -I -m orkexa._pdf_worker`) gets a minimal environment (`LC_ALL`, `PYTHONIOENCODING`,
`PYTHONHASHSEED`; on Windows `SYSTEMROOT` instead of `LC_ALL`), so no keys, database URLs, credentials or proxy
settings reach it; it inherits no file descriptors, runs in an empty private directory that is deleted afterwards,
and lowers its own limits before it imports pypdf (Linux: memory, CPU time, 16 open files, no file writes; macOS:
the same without the memory limit; Windows: none). It is not a sandbox: it runs as the same operating-system user
without namespaces, seccomp or network isolation, so a code-execution flaw in pypdf could reach that user's files
and the network. Starting the interpreter costs about 0.1 to 0.3 seconds per document.

## SQLite limits and error codes

`SQLiteIndex`, `SQLiteFileStore` and `SQLiteMemory` can share one database file between threads, instances and
processes: every read-then-write step runs in one write transaction (`BEGIN IMMEDIATE`), so concurrent deletes or
adds of the same source or file neither damage the full-text index nor lose entries.

Searches of `SQLiteIndex` are limited, because full-text and hybrid search run one or two full-text queries per
query word while they hold the index:

| Limit | Default | Constructor argument |
| --- | --- | --- |
| characters per query | `orkexa.retrieval.MAX_QUERY_CHARS` = 1000 | `max_query_chars=` |
| distinct query words (`query_words()`) | `orkexa.retrieval.MAX_QUERY_WORDS` = 64 | `max_query_words=` |

A query over a limit raises `RetrievalError` before it is embedded or the index is locked, so it cannot hold up
other searches or writes. The search tool reports the error to the model like any other tool error.

`RetrievalError.code` values (the full list is in the `RetrievalError` docstring):

| Code | Raised by | Meaning |
| --- | --- | --- |
| `query.too_long` | `SQLiteIndex.retrieve`, `search`, the search tool | the query exceeds `max_query_chars` or `max_query_words` |
| `vector.dimension` | `SQLiteIndex.retrieve` (`vector`, `hybrid`) | the query vector's length differs from the stored vectors' |
| `storage.symlink` | the SQLite classes' constructors | the database path, or its `-wal`, `-shm` or `-journal` file, is a symbolic link (`reason` `symlink`), belongs to another user (`other_owner`) or is writable by group or others (`writable_by_others`); it is refused, not followed or changed |
| `storage.corrupt` | the SQLite classes | the file is damaged or not a SQLite database; the `sqlite3.DatabaseError` is `__cause__` |
| `document.too_complex` | `pdf_pages`, `pdf_pages_isolated`, `Index.add_file` | a PDF exceeds a budget; `exc.reason` says which (see [PDF limits](#pdf-limits)) |
| `mode_unsupported`, `vector_unavailable`, `reindex_required`, `vector_incomplete`, `cancelled` | `SQLiteIndex` vector search and `reindex()` | see [Local models](local-models.md) |

Error messages never contain file paths.

## Deleting data

Every store can delete what it holds, e.g. to answer an erasure request:

| Store | Delete | What remains |
| --- | --- | --- |
| `EphemeralMemory` | `delete(path)`, `clear()` | nothing; it lives in the process only |
| `FileMemory`, `LocalFileStore` | `delete(path)`, `delete(file_id)` | the file is unlinked; whether the disk overwrites the blocks depends on the file system and storage (backups, snapshots and SSD wear leveling are out of Orkexa's reach) |
| `SQLiteMemory`, `SQLiteFileStore` | `delete(path)`, `delete(file_id)` | nothing in the database file: the connections use `PRAGMA secure_delete=ON`, so deleted rows are overwritten |
| `SQLiteIndex` | `delete(source)`, then `optimize()` | after `optimize()`, nothing in the full-text index; before it, FTS5 keeps the terms of deleted chunks in its index segments (they no longer match searches) |
| Agent history | `agent.reset()` | nothing in the agent; histories you saved with `save_history()` are your files |
| Anthropic Files API copy (`files_api=client`) | `client.files.delete(store.info(file_id).api_file_id)`, before `store.delete(file_id)` | `store.delete()` removes only the local copy; the uploaded copy stays until you delete it or the provider's retention ends |

SQLite removes the write-ahead log (`*-wal`) when the last connection closes; close
the stores (or use them as context managers) after deleting data that must be erased.
Data you sent to a model provider is subject to that provider's retention policy, not
to these deletions (see [Data protection and AI regulation](../compliance/index.md)).
