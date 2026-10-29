"""8. Memory, a file store, multimodal input, RAG and verified sources on any provider.

On Anthropic the memory backend is the native memory tool and search hits are
``search_result`` blocks with API-native citations. On OpenAI, Gemini and
compatible endpoints the same backends are ordinary function tools and search
hits arrive as attributed text; :func:`orkexa.retrieval_sources` lists the passages
the search tool really delivered, independent of the provider.
"""

from __future__ import annotations

import base64
from pathlib import Path
import tempfile

from _setup import model

from orkexa import Agent, LocalFileStore, SQLiteIndex, SQLiteMemory, citations, image, retrieval_sources

# A 1x1 PNG, so the example needs no image file.
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def main() -> None:
    name = model()
    directory = Path(tempfile.mkdtemp(prefix="orkexa-rag-"))
    database = directory / "agent.db"

    index = SQLiteIndex(database, namespace="handbook")
    index.add(
        "Devices have a warranty of 24 months from the date of purchase.",
        source="handbook.md",
        title="Warranty",
    )
    index.add("Returns are accepted within 30 days with the receipt.", source="handbook.md", title="Returns")
    store = LocalFileStore(directory / "files")
    store.add(b"Q3 revenue grew 12% to EUR 4.2m.", name="q3.txt", description="Q3 results")
    memory = SQLiteMemory(database, namespace="user-42")

    agent = Agent(model=name, index=index, files=store, memory=memory)
    print(agent.ask("How long is the warranty? Search the handbook."))
    for source in retrieval_sources(agent):
        print(f"  verified source: {source.source} / {source.title}: {source.cited_text[:50]}")
    for cite in citations(agent.last_response):
        print(f"  API-native citation: {cite.source}: {cite.cited_text[:50]}")

    print(agent.ask("Load the Q3 results file and remember the revenue in your memory."))
    print(agent.ask([image(PIXEL, media_type="image/png"), "What color is this pixel, roughly?"]))
    index.close()


if __name__ == "__main__":
    main()
