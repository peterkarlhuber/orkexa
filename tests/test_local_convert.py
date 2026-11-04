"""Building models from pinned weights (orkexa.local.convert) and embedding prefixes.

The build is tested on a tiny synthetic ModernBERT (template header + safetensors made
here) against a stub Ollama client; the real mDenseOn build runs only when
ORKEXA_TEST_OLLAMA_URL and ORKEXA_TEST_MDENSEON_WEIGHTS / _TEMPLATE are set.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any

import pytest

from orkexa.embeddings import KNOWN_PREFIXES, EmbeddingError, EmbeddingIdentity, OllamaEmbeddings
from orkexa.local import EmbedResult, OllamaError, PullProgress
from orkexa.local.convert import MDENSEON, GGUFRecipe, SourceFile, _read_header, build_model

gguf = pytest.importorskip("gguf")
np = pytest.importorskip("numpy")


# ---- fixtures -----------------------------------------------------------------------------


def _template(path: Path) -> int:
    """A tiny modern-bert GGUF (2 layers, 8 dims) with tokenizer metadata; returns header length."""
    writer = gguf.GGUFWriter(str(path), "modern-bert")
    writer.add_block_count(2)
    writer.add_embedding_length(8)
    writer.add_uint32("modern-bert.pooling_type", 1)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(["<pad>", "<eos>", "<bos>", "▁a", "▁b"])
    writer.add_token_scores([-1000.0, -1000.0, -1000.0, -1.0, -2.0])
    writer.add_bool("tokenizer.ggml.add_space_prefix", False)
    writer.add_string("general.name", "Base")
    writer.add_tensor("token_embd.weight", np.zeros((5, 8), dtype=np.float16))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    reader = gguf.GGUFReader(str(path))
    return int(reader.data_offset)


def _safetensors(path: Path, tensors: dict[str, Any]) -> None:
    header: dict[str, Any] = {}
    data = b""
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array, dtype=np.float32).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [len(data), len(data) + len(raw)],
        }
        data += raw
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _weights() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    tensors = {
        "embeddings.tok_embeddings.weight": rng.standard_normal((5, 8)),
        "embeddings.norm.weight": np.ones(8),
        "final_norm.weight": np.ones(8),
    }
    for n in range(2):
        tensors[f"layers.{n}.attn.Wqkv.weight"] = rng.standard_normal((24, 8))
        tensors[f"layers.{n}.attn.Wo.weight"] = rng.standard_normal((8, 8))
        tensors[f"layers.{n}.mlp.Wi.weight"] = rng.standard_normal((16, 8))
        tensors[f"layers.{n}.mlp.Wo.weight"] = rng.standard_normal((8, 8))
        tensors[f"layers.{n}.mlp_norm.weight"] = np.ones(8)
    return tensors


def _sha(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data if limit is None else data[:limit]).hexdigest()


REFERENCE = ((("query: x"), (1.0, 0.0, 0.0)), (("document: y"), (0.0, 1.0, 0.0)))


@pytest.fixture
def sources(tmp_path: Path) -> tuple[GGUFRecipe, Path, Path]:
    template = tmp_path / "template.gguf"
    header = _template(template)
    weights = tmp_path / "model.safetensors"
    _safetensors(weights, _weights())
    recipe = GGUFRecipe(
        name="tiny:1m",
        architecture="modern-bert",
        weights=SourceFile("https://example.invalid/w", _sha(weights), weights.stat().st_size),
        template=SourceFile("https://example.invalid/t", _sha(template, header), header),
        tensor_names=dict(MDENSEON.tensor_names),
        overrides={
            "modern-bert.pooling_type": 2,
            "tokenizer.ggml.add_space_prefix": True,
            "general.name": "Tiny",
        },
        license="Apache-2.0",
        source="https://example.invalid",
        reference=REFERENCE,
        extra={"tensors": 13},
    )
    return recipe, weights, template


class StubClient:
    """The few OllamaClient methods build_model and OllamaEmbeddings use."""

    def __init__(self, vectors: list[list[float]] | None = None, embedding_length: int = 3) -> None:
        self.vectors = vectors
        self.created: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.inputs: list[list[str]] = []
        self.embedding_length = embedding_length
        self.copied: list[tuple[str, str]] = []
        self.embedded_models: list[str] = []

    def create_from_gguf(self, name: str, path: str | os.PathLike[str], **_: Any) -> Iterator[PullProgress]:
        self.created[name] = Path(path).read_bytes()
        yield PullProgress(status="success")

    def embed(self, model: str, inputs: list[str], **_: Any) -> EmbedResult:
        self.inputs.append(list(inputs))
        self.embedded_models.append(model)
        vectors = self.vectors or [[1.0, 0.0, 0.0] for _ in inputs]
        return EmbedResult(model=model, embeddings=vectors[: len(inputs)])

    def delete(self, name: str) -> None:
        self.deleted.append(name)
        self.created.pop(name, None)

    def copy(self, source: str, destination: str) -> None:
        self.copied.append((source, destination))
        self.created[destination] = self.created[source]

    def find(self, name: str) -> Any:
        return SimpleNamespace(name=name, digest="d" * 64)

    def show(self, name: str) -> Any:
        return SimpleNamespace(is_embedding=True, embedding_length=self.embedding_length)


# ---- build ----------------------------------------------------------------------------------


def test_build_writes_verified_gguf_with_overrides(
    sources: tuple[GGUFRecipe, Path, Path], tmp_path: Path
) -> None:
    recipe, weights, template = sources
    client = StubClient(vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    report = build_model(client, recipe, weights=weights, template=template, work_dir=tmp_path)  # type: ignore[arg-type]
    assert report.min_cosine == pytest.approx(1.0) and report.digest == "d" * 64
    # built and verified under a temporary name, then installed under the recipe's name
    (staging,) = client.deleted
    assert staging.startswith("tiny-orkexa-build-") and staging.endswith(":1m")
    assert client.embedded_models == [staging] and client.copied == [(staging, "tiny:1m")]
    built = tmp_path / "built.gguf"
    built.write_bytes(client.created["tiny:1m"])
    assert hashlib.sha256(built.read_bytes()).hexdigest() == report.gguf_sha256
    reader = gguf.GGUFReader(str(built))
    fields = {key: field.contents() for key, field in reader.fields.items()}
    assert fields["modern-bert.pooling_type"] == 2
    assert fields["tokenizer.ggml.add_space_prefix"] is True
    assert fields["general.name"] == "Tiny"
    assert fields["tokenizer.ggml.tokens"] == ["<pad>", "<eos>", "<bos>", "▁a", "▁b"]
    types = {t.name: t.tensor_type.name for t in reader.tensors}
    assert len(types) == 13
    assert types["blk.1.attn_qkv.weight"] == "F16" and types["blk.0.ffn_norm.weight"] == "F32"
    # The same inputs give the same file (reproducible build).
    again = StubClient(vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert build_model(again, recipe, weights=weights, template=template).gguf_sha256 == report.gguf_sha256  # type: ignore[arg-type]


def test_build_removes_model_that_misses_the_reference(sources: tuple[GGUFRecipe, Path, Path]) -> None:
    recipe, weights, template = sources
    client = StubClient(vectors=[[0.9, 0.44, 0.0], [0.0, 1.0, 0.0]])
    with pytest.raises(OllamaError) as error:
        build_model(client, recipe, weights=weights, template=template)  # type: ignore[arg-type]
    assert error.value.code == "reference_mismatch"
    assert not client.copied and "tiny:1m" not in client.deleted and list(client.created) == []


def test_build_refuses_wrong_or_missing_inputs(
    sources: tuple[GGUFRecipe, Path, Path], tmp_path: Path
) -> None:
    recipe, weights, template = sources
    client = StubClient()
    with pytest.raises(OllamaError) as error:
        build_model(client, recipe, weights=weights, template=None)  # type: ignore[arg-type]
    assert error.value.code == "source_missing"  # no silent download
    tampered = tmp_path / "tampered.safetensors"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    tampered.write_bytes(bytes(data))
    with pytest.raises(OllamaError) as error:
        build_model(client, recipe, weights=tampered, template=template)  # type: ignore[arg-type]
    assert error.value.code == "checksum_mismatch"
    extra = tmp_path / "extra.safetensors"
    tensors = _weights() | {"pooler.dense.weight": np.ones((8, 8))}
    _safetensors(extra, tensors)
    odd = GGUFRecipe(
        **{
            **recipe.__dict__,
            "weights": SourceFile("https://example.invalid/w", _sha(extra), extra.stat().st_size),
        }
    )
    with pytest.raises(OllamaError) as error:
        build_model(client, odd, weights=extra, template=template)  # type: ignore[arg-type]
    assert error.value.code == "invalid_weights" and not client.created


def test_template_header_parser_rejects_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(OllamaError) as error:
        _read_header(bad, 68)
    assert error.value.code == "invalid_template"


def test_mdenseon_recipe_is_pinned() -> None:
    assert MDENSEON.weights.url.startswith("https://huggingface.co/lightonai/mDenseOn/resolve/a5fdb000")
    assert len(MDENSEON.weights.sha256) == 64 and len(MDENSEON.template.sha256) == 64
    assert MDENSEON.overrides["modern-bert.pooling_type"] == 2  # CLS, as 1_Pooling/config.json
    assert MDENSEON.overrides["tokenizer.ggml.add_space_prefix"] is True
    assert len(MDENSEON.reference) == 3 and all(len(v) == 768 for _, v in MDENSEON.reference)
    assert [text.split(":")[0] for text, _ in MDENSEON.reference] == ["query", "document", "document"]


# ---- embedding prefixes ---------------------------------------------------------------------


def test_known_prefixes_are_applied_and_part_of_the_identity() -> None:
    client = StubClient()
    model = OllamaEmbeddings("mdenseon:307m", client=client)  # type: ignore[arg-type]
    assert (model.document_prefix, model.query_prefix) == KNOWN_PREFIXES["mdenseon"]
    model.embed_documents(["Gewährleistung"])
    model.embed_query("garantie ?")
    assert client.inputs == [["document: Gewährleistung"], ["query: garantie ?"]]
    identity = model.identity()
    assert identity.document_prefix == "document: " and identity.query_prefix == "query: "
    assert EmbeddingIdentity.from_dict(identity.to_dict()) == identity
    plain = OllamaEmbeddings("mdenseon:307m", client=client, document_prefix="", query_prefix="")  # type: ignore[arg-type]
    assert plain.identity().key != identity.key  # other preparation, other vector space


def test_prefixes_keep_old_identity_keys_and_reject_conflicts() -> None:
    old = EmbeddingIdentity(
        provider="ollama", model="qwen3-embedding:0.6b", dimensions=1024, digest="x", query_instruction="i"
    )
    expected = hashlib.sha256(
        json.dumps(["ollama", "qwen3-embedding:0.6b", "x", 1024, True, "", "i"]).encode()
    ).hexdigest()[:20]
    assert old.key == expected
    with pytest.raises(ValueError):
        OllamaEmbeddings("mdenseon:307m", client=StubClient(), query_instruction="Find")  # type: ignore[arg-type]
    model = OllamaEmbeddings("mdenseon:307m", client=StubClient())  # type: ignore[arg-type]
    with pytest.raises(EmbeddingError) as error:
        model.embed_query("  ")
    assert error.value.code == "empty_input"
    with pytest.raises(EmbeddingError):
        model.embed_documents(["ok", " "])


# ---- live build (real Ollama and the real 1.2 GB weights) -----------------------------------

LIVE = os.environ.get("ORKEXA_TEST_OLLAMA_URL")
WEIGHTS = os.environ.get("ORKEXA_TEST_MDENSEON_WEIGHTS")
TEMPLATE = os.environ.get("ORKEXA_TEST_MDENSEON_TEMPLATE")


@pytest.mark.skipif(not (LIVE and WEIGHTS and TEMPLATE), reason="needs a real Ollama and the mDenseOn files")
def test_live_mdenseon_build_matches_reference_and_ranks_across_languages() -> None:
    from orkexa.local import OllamaClient

    assert LIVE and WEIGHTS and TEMPLATE
    client = OllamaClient(LIVE)
    report = build_model(client, MDENSEON, weights=WEIGHTS, template=TEMPLATE)
    assert report.min_cosine >= 0.9999
    model = OllamaEmbeddings(MDENSEON.name, base_url=LIVE)
    docs = model.embed_documents(
        [
            "Bei Verbrauchsgüterkäufen beträgt die gesetzliche Gewährleistungsfrist zwei Jahre.",
            "Passwörter müssen mindestens zwölf Zeichen lang sein.",
        ]
    )
    query = model.embed_query("Combien de temps dure la garantie légale ?")
    scores = [sum(a * b for a, b in zip(query, doc, strict=True)) for doc in docs]
    assert scores[0] > scores[1]
