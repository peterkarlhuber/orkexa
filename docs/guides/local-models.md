# Local models, embeddings and retrieval modes

Orkexa runs agents and embeddings on local models with [Ollama](https://ollama.com), without a cloud
API key. Nothing local happens on its own: `pip install orkexa`, `import orkexa` and `Agent(...)` never
install software, start processes or download models. Every such step is an explicit call (or a
command of `orkexa-local`).

Three steps are kept apart:

| Step | What | Network |
|---|---|---|
| 1. Runtime | Use an Ollama that already runs, or install the pinned runtime and let `ManagedOllama` run it | Only for `install_runtime(download=True)` |
| 2. Models | `OllamaClient.pull()`, or offline: a GGUF file, an offline model package, an existing model folder | Only for `pull()` |
| 3. Inference | Agents and embeddings against the local server | None |

After steps 1 and 2, step 3 works without internet.

```bash
pip install "orkexa[openai,vector,local]"   # openai: the OpenAI-compatible protocol Ollama speaks
```

## Running a local agent

```python
from orkexa import Agent
from orkexa.local import OllamaClient

client = OllamaClient()  # http://127.0.0.1:11434, loopback only by default
details = client.show("qwen3.5:4b")  # capabilities from the installed model
agent = Agent(
    model="ollama:qwen3.5:4b",
    base_url=client.openai_base_url,
    capabilities=details.capability_overrides(),
)
print(agent.ask("Summarize this in one sentence: ..."))
```

Capabilities of OpenAI-compatible endpoints are `unknown` unless declared, and Orkexa never treats
`unknown` as supported. `ModelDetails.capability_overrides()` declares what Ollama reports for the
installed model in `/api/show` (`tools`, `vision`, `completion`, `embedding`); everything Ollama does not
report stays `unknown`. A model name says nothing reliable about its capabilities, so Orkexa never
derives them from it.

## The `orkexa-local` command

The same steps from a terminal (`orkexa-local --help` lists every command; `python -m orkexa.local`
works the same way):

```bash
orkexa-local status                      # installed runtime, a running Ollama, its models
orkexa-local install-runtime --download  # network; offline: --archive ollama-linux-amd64.tar.zst
orkexa-local serve                       # runs the managed server in the foreground (Ctrl+C stops it)
orkexa-local pull qwen3.5:4b             # in a second terminal; network
```

`serve` picks a free loopback port (or `--port`) and prints the address; the other commands find that
server through the home folder, and `--url http://127.0.0.1:11434` points them at an Ollama you run
yourself. The home is `~/.orkexa/local` unless `--home` or `ORKEXA_LOCAL_HOME` names another folder.
Errors print their code, for example `error (unreachable): Ollama is not reachable at
http://127.0.0.1:11434: ...` (no server answers: start one with `serve`) or `error (runtime_missing)` (run
`install-runtime` first).

## Provisioning Ollama without a manual installation

```python
from orkexa.local import ManagedOllama, install_runtime

install_runtime("~/.orkexa/local", archive="ollama-linux-amd64.tar.zst")  # offline: a copied archive
# install_runtime("~/.orkexa/local", download=True)                       # or download it (network)

with ManagedOllama("~/.orkexa/local", models_dir="/Volumes/SSD/models") as server:
    client = server.client()
```

* `install_runtime` knows the archives of Ollama **0.40.14** by name and SHA-256 (from the release's
  `sha256sum.txt`). The checksum must match before anything is unpacked. Unpacking refuses absolute
  paths, `..`, links that leave the target, devices and oversized archives; no installer script runs.
  The runtime lives in `<home>/runtime/0.40.14`, nothing is installed system-wide. On Linux the NVIDIA
  libraries (about 2 GB) are unpacked only when an NVIDIA driver is present (`gpu=True` forces them).
  `.tar.zst` archives need the `local` extra (`zstandard`). `artifact=artifact_for("windows", "x86_64")`
  prepares the runtime of another platform (for example for an application package built for another
  machine); its executable is then found by that platform's name (`ollama.exe` on Windows).
* `ManagedOllama` starts `ollama serve` as a child process on a **free loopback port** with its own model
  folder, its own home (keys and caches stay there), `OLLAMA_NO_CLOUD=1` and a minimal environment (no
  secrets of the parent process). `offline=True` removes proxy variables. It stops only the process it
  started. An Ollama that already runs (for example on port 11434) is only detected with
  `detect_ollama()`, never stopped or reconfigured.
* Ollama has **no authentication**. `OllamaClient` refuses non-loopback hosts unless
  `allow_remote=True`, never follows redirects and never sends credentials.

## Getting models offline

| Way | Call | Notes |
|---|---|---|
| Download | `client.pull("qwen3.5:4b")` | Progress per layer; an interrupted download continues |
| GGUF file | `client.create_from_gguf("mymodel:q4", "model.gguf", projector="mmproj.gguf")` | Uploaded to the server's blob store; the chat template comes from the GGUF unless given |
| Offline package | `export_package(...)` on an online machine, `import_package(...)` on the target | Every file checked by SHA-256 |
| Existing folder | `ManagedOllama(home, models_dir=folder)` or `scan_store(folder)` | Uses the folder in place, no copy |

An **offline model package** is a folder with the content-addressed files of the chosen models and an
index (`orkexa-models.json`). A model's digest is the SHA-256 of its manifest, the value `ollama list`
and the Ollama registry show, so it can be compared with a catalog. `import_package` verifies the
structure, names, sizes and the SHA-256 of every file, writes files under a temporary name and makes a
model visible only when all its files are in place; running it again continues an interrupted import.
Packages are **not signed**: checksums prove integrity, not origin.

Transformers/safetensors folders are not imported: convert them to GGUF first (for example with
llama.cpp's conversion script) and import the GGUF file.

### Building a model from pinned upstream weights

Some open-weight models are not published in a form Ollama can pull, although Ollama's engine runs
their architecture from a GGUF file. `build_model(client, recipe)` builds such a model reproducibly
from a `GGUFRecipe`: the upstream `safetensors` weights and a *template* GGUF header (tokenizer and
hyperparameters of the same architecture) are both pinned by URL, size and SHA-256; the recipe's
metadata overrides are applied; the GGUF file is created in Ollama offline; and the model is kept only
if it reproduces the recipe's reference vectors (cosine >= 0.9999). It is built and checked under a
temporary name; a failed build is removed and an installed model of the same name stays as it was.
Inputs come from local files or, only with `download=True`, from the pinned URLs (the template is
fetched with an HTTP range request: only its 6 MB header). Needs the `convert` extra (`gguf`).

`MDENSEON` is the recipe for [LightOn mDenseOn](https://huggingface.co/lightonai/mDenseOn)
(Apache-2.0, 307M parameters, 768 dimensions, 8192 tokens, trained for en, fr, de, it, es, pt, sv, no
and ar, including cross-lingual pairs). It becomes `mdenseon:307m` (F16, 620 MB). Its tokenizer metadata
comes from the llama.cpp conversion of mmBERT-base (MIT), the model mDenseOn was fine-tuned from, with
two corrections found by comparing with the reference implementation: CLS pooling and
`add_space_prefix=true`. Without the second one every vector drifts to cosine ~0.98.

```bash
orkexa-local build mdenseon --download                      # network: 1.2 GB weights + 6 MB header
orkexa-local build mdenseon --weights model.safetensors --template mmBERT-Base-307M-F16.gguf  # offline
```

The built model is an ordinary Ollama model: export it in an offline package like any other.

```bash
orkexa-local --url http://127.0.0.1:11434 export /media/usb/models qwen3.5:4b qwen3-embedding:0.6b --models-dir ~/.ollama/models
orkexa-local import /media/usb/models --models-dir /Volumes/SSD/models
orkexa-local verify /media/usb/models
```

## Embeddings

```python
from orkexa.embeddings import OllamaEmbeddings

embeddings = OllamaEmbeddings("qwen3-embedding:0.6b")
embeddings.identity()  # provider, model, digest, dimensions, instructions
```

`EmbeddingModel` is provider-independent (`embed_documents`, `embed_query`, `identity`).
`OllamaEmbeddings` calls `/api/embed` in batches with `truncate=false`: a text longer than the model's
context raises `EmbeddingError("input_too_long")` instead of being cut silently. Qwen3-Embedding
models get their query instruction in front of queries, documents get none. Models trained with plain
prefixes get them from `KNOWN_PREFIXES` (`mdenseon`: `"document: "` and `"query: "`), or pass
`document_prefix=`/`query_prefix=`; the prefixes are part of the identity, so changing them needs a
reindex. `on_usage` receives the
measured prompt tokens and durations (`None` when Ollama does not report them). The older `embedder=`
callables keep working; `CallableEmbeddings` gives them an explicit identity.

## Retrieval modes

`SQLiteIndex` ranks in one of three modes:

| Mode | Ranking | Needs |
|---|---|---|
| `fts` | SQLite FTS5 full-text search (BM25) | Nothing; the default without embeddings |
| `vector` | Embeddings only, exact cosine similarity | An embedding model and `sqlite-vec` |
| `hybrid` | Both, fused by evidence-weighted score fusion (see below), ties by chunk id | Both; the default with embeddings |

```python
from orkexa import Agent
from orkexa.embeddings import OllamaEmbeddings
from orkexa.retrieval import SQLiteIndex

index = SQLiteIndex("kb.db", embeddings=OllamaEmbeddings("qwen3-embedding:0.6b"))
index.add_file("handbook.pdf")
index.retrieve("money back?", 5, mode="vector", sources=["handbook.pdf"])
agent = Agent(model="ollama:qwen3.5:9b", tools=[index.tool(mode="hybrid")])
```

* `vector` and `hybrid` never fall back to full-text search: missing vectors raise `RetrievalError`
  (`vector_unavailable`, `reindex_required`). `vector` also finds passages that share no word with
  the query.
* `sources=` limits the candidates **before** ranking (an SQL condition), so a scoped search never
  ranks or returns other sources. `Hit.ranks` shows each hit's rank in every ranking that found it.
* Each namespace keeps its vectors in a *space* bound to the embedding model's identity (model, digest,
  dimensions, instructions). The same number of dimensions does not make two models compatible.
  Another model raises `reindex_required`; `index.reindex(new_model)` embeds all chunks into a new
  space in batches (resumable after interruption or `cancel`) and switches in one transaction, deleting
  the old vectors. Searches keep using the old, complete space until then. Chunk texts and the
  full-text index stay unchanged. `vector_status()` reports the state; `drop_vectors()` returns to
  full text only.
* `hybrid` takes `max(4 * k, 20)` candidates from each ranking and scores every candidate
  `(1 - w) * vector + w * lexical` with `w = lexical_weight` (default 0.5). `vector` is the cosine
  similarity, min-max normalized over the vector candidates (so the scale of a model does not matter).
  `lexical` is the squared IDF-weighted share of the query's words the chunk contains (a word no chunk
  contains has the highest weight). A chunk that contains the whole query, such as an exact product
  code, name or statute section, scores 1 and rises above vector neighbors that lack it; a chunk that
  shares only a common or accidental word, as queries in another language than the documents usually
  do (`de`, `la`, `days`), scores almost 0 and keeps the vector order. A query with no lexical
  evidence ranks exactly as `vector`. `Hit.score` is the fused value in [0, 1]. The earlier reciprocal
  rank fusion (k=60) is still available as `SQLiteIndex(..., fusion="rrf")`: it ignores how strong a
  full-text match is, so in cross-lingual searches a weak full-text hit could outrank the right vector
  hit. `fts` and `vector` are not affected by either setting. The building blocks (`query_words`,
  `word_weight`, `lexical_evidence`, `fuse_hybrid`) are public in `orkexa.retrieval` for other index
  backends.
* Images and scanned PDFs have no text to embed: extract text first (OCR, or a vision model) and keep
  the origin in the chunk metadata.

## Usage and costs of local models

A request to provider `ollama` is not billed by a model API. Orkexa prices it with an explicit zero
**API** price whose source is `"local"` (`UsageRecord.local_inference`, `UsageSummary.local_requests`),
so budgets and strict price checks treat it as known rather than as an unpriced cloud request. That is
not a statement that local inference is free: hardware, energy, storage and administration are costs
Orkexa does not measure. Present such usage as "local, no API billing; operating costs not recorded",
never as a total of 0. A price you register for the model yourself takes precedence. Token counts are
the ones Ollama reports; reasoning tokens are part of the output tokens. `vllm` and `compatible`
endpoints stay unpriced, because they may be billed gateways.

## Models

The Nodestro catalog uses these Ollama tags (verified against the registry on 2026-11-03; sizes are
download sizes of the default Q4_K_M/Q8_0 builds, memory needs are higher). Capabilities of `qwen3.5:4b`
and `qwen3-embedding:0.6b` were read from `/api/show` of the installed models; for the larger models they
are the ones the Ollama library lists, and a client must read them from `/api/show` once installed:

| Model | Download | Capabilities (Ollama) | Min. Ollama |
|---|---|---|---|
| `qwen3.5:4b` | 3.3 GB | completion, vision, tools, thinking | 0.30.0 |
| `qwen3.5:9b` | 6.6 GB | completion, vision, tools, thinking | 0.30.0 |
| `qwen3.6:35b-a3b` | 22.6 GB | completion, vision, tools, thinking (MoE, 3B active) | 0.30.0 |
| `qwen3.8:27b` | 17.7 GB | completion, vision, tools, thinking | 0.32.12 |
| `qwen3-embedding:0.6b` | 0.64 GB | embedding (1024 dimensions, 32k context) | |
| `qwen3-embedding:4b` | 2.5 GB | embedding | |
| `qwen3-embedding:8b` | 4.7 GB | embedding | |
| `ministral-3:3b` | 2.95 GB | completion, vision, tools (Mistral AI, Apache-2.0) | 0.13.1 |
| `ministral-3:8b` | 6.02 GB | completion, vision, tools | 0.13.1 |
| `ministral-3:14b` | 9.08 GB | completion, vision, tools | 0.13.1 |
| `devstral-small-2:24b` | 15.18 GB | completion, vision, tools (coding agents) | 0.13.3 |
| `mdenseon:307m` | 0.62 GB | embedding (768 dimensions, 8k context), built with `build_model(MDENSEON)` | |

The Qwen and Ministral language models ship an Apache-2.0 license layer; the Qwen3-Embedding and
`devstral-small-2` manifests have none (the upstream models are Apache-2.0 per their model cards).
Ministral 3 needs an explicit system prompt for reliable classification (measured on CPU: routing 2/6
without, 6/6 with). `devstral-small-2` reports 393,216 tokens of context (YaRN 8,192 x 48); Mistral
documents 256k. Any other installed Ollama model works the same way.
