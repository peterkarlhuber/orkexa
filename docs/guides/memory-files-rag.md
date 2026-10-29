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

`SQLiteIndex` uses FTS5/BM25, optional vector or hybrid search with `embedder=`
(`orkexa[vector]`) and PDF extraction (`orkexa[pdf]`).

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
