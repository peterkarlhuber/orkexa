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
