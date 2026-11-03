"""orkexa.local: Ollama client, managed server, runtime install, offline packages; embeddings; pricing.

A fake Ollama server (standard library HTTP server) stands in for the real one. Live tests
against a real Ollama run only when ORKEXA_TEST_OLLAMA_URL is set (see tests at the end).
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal
import hashlib
import http.server
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import textwrap
import threading
from typing import ClassVar
import zipfile

import pytest

import orkexa
from orkexa import Agent
from orkexa.embeddings import EmbeddingError, OllamaEmbeddings
from orkexa.errors import ConfigurationError
from orkexa.local import (
    ManagedOllama,
    OllamaClient,
    OllamaError,
    check_model_name,
    check_ollama_url,
    export_package,
    import_package,
    install_runtime,
    is_cloud_model,
    scan_store,
    verify_package,
)
from orkexa.local.install import RuntimeArtifact
from orkexa.pricing import PricingBasis
from orkexa.usage import record_token_usage, summarize_usage

# ---- a fake Ollama ----------------------------------------------------------------------

MODELS = {
    "qwen3.5:4b": {"capabilities": ["completion", "vision", "tools", "thinking"], "ctx": 262144, "emb": 2560},
    "qwen3-embedding:0.6b": {"capabilities": ["embedding"], "ctx": 32768, "emb": 1024},
}


class FakeOllama(http.server.BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, str, object]]] = []
    blobs: ClassVar[dict[str, bytes]] = {}

    def log_message(self, *args: object) -> None:
        pass

    def _json(self, status: int, body: object) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> object:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if self.path.startswith("/api/blobs/"):
            return raw
        return json.loads(raw) if raw else None

    def do_HEAD(self) -> None:
        digest = self.path.rsplit("/", 1)[-1]
        self.send_response(200 if digest in self.blobs else 404)
        self.end_headers()

    def do_GET(self) -> None:
        self.requests.append(("GET", self.path, None))
        if self.path == "/api/version":
            self._json(200, {"version": "0.40.14"})
        elif self.path == "/api/tags":
            self._json(
                200,
                {
                    "models": [
                        {
                            "name": name,
                            "digest": hashlib.sha256(name.encode()).hexdigest(),
                            "size": 1000,
                            "details": {"family": "qwen", "quantization_level": "Q4_K_M", "format": "gguf"},
                            "capabilities": info["capabilities"],
                        }
                        for name, info in MODELS.items()
                    ]
                    + [{"name": "big:cloud", "digest": "x", "size": 0, "remote_host": "https://ollama.com"}]
                },
            )
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.com/")
            self.end_headers()
        else:
            self._json(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        body = self._body()
        self.requests.append(("DELETE", self.path, body))
        self._json(200, {})

    def do_POST(self) -> None:
        body = self._body()
        self.requests.append(("POST", self.path, body if not isinstance(body, bytes) else len(body)))
        if self.path == "/api/show":
            assert isinstance(body, dict)
            info = MODELS.get(body["model"])
            if info is None:
                self._json(404, {"error": f"model '{body['model']}' not found"})
                return
            self._json(
                200,
                {
                    "capabilities": info["capabilities"],
                    "details": {"family": "qwen"},
                    "model_info": {"q.context_length": info["ctx"], "q.embedding_length": info["emb"]},
                },
            )
        elif self.path == "/api/embed":
            assert isinstance(body, dict)
            if body["model"] not in MODELS:
                self._json(404, {"error": f"model '{body['model']}' not found"})
                return
            if any(len(text) > 2000 for text in body["input"]):
                self._json(400, {"error": "the input length exceeds the context length"})
                return
            dims = body.get("dimensions") or 1024
            vectors = [[float(len(text) % 7 + 1)] + [0.5] * (dims - 1) for text in body["input"]]
            self._json(
                200,
                {
                    "model": body["model"],
                    "embeddings": vectors,
                    "prompt_eval_count": 7,
                    "total_duration": 2_000_000,
                },
            )
        elif self.path == "/api/pull":
            self.send_response(200)
            self.end_headers()
            for line in (
                {"status": "pulling manifest"},
                {"status": "pulling a1", "digest": "sha256:a1", "total": 100, "completed": 50},
                {"status": "pulling a1", "digest": "sha256:a1", "total": 100, "completed": 100},
                {"status": "success"},
            ):
                self.wfile.write(json.dumps(line).encode() + b"\n")
        elif self.path.startswith("/api/blobs/"):
            assert isinstance(body, bytes)
            digest = self.path.rsplit("/", 1)[-1]
            if "sha256:" + hashlib.sha256(body).hexdigest() != digest:
                self._json(400, {"error": "digest mismatch"})
                return
            self.blobs[digest] = body
            self._json(201, {})
        elif self.path == "/api/create":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"parsing GGUF"}\n{"status":"success"}\n')
        else:
            self._json(404, {"error": "not found"})


@pytest.fixture
def ollama() -> Iterator[str]:
    FakeOllama.requests = []
    FakeOllama.blobs = {}
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


# ---- import never contacts anything --------------------------------------------------------


def test_import_and_agent_construction_do_not_load_or_contact_ollama() -> None:
    code = textwrap.dedent(
        """
        import socket, sys
        def refuse(*a, **k):
            raise AssertionError("network used")
        socket.socket.connect = refuse
        import orkexa
        from orkexa import Agent
        Agent(model="ollama:qwen3.5:4b")
        assert "orkexa.local.server" not in sys.modules
        print("ok")
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.stdout.strip() == "ok", result.stderr
    assert hasattr(orkexa, "Agent")


# ---- URLs and names -----------------------------------------------------------------------


def test_urls_are_loopback_only_unless_allowed() -> None:
    assert check_ollama_url("http://localhost:11434/v1") == "http://localhost:11434"
    assert check_ollama_url("http://[::1]:11434") == "http://[::1]:11434"
    for bad in (
        "ftp://127.0.0.1",
        "http://user:pw@127.0.0.1",
        "http://127.0.0.1/api",
        "http://10.0.0.5:11434",
    ):
        with pytest.raises(ConfigurationError):
            check_ollama_url(bad)
    assert check_ollama_url("http://10.0.0.5:11434", allow_remote=True) == "http://10.0.0.5:11434"


def test_model_names_and_cloud_tags() -> None:
    assert check_model_name("qwen3.5:4b") == "qwen3.5:4b"
    assert check_model_name("hf.co/org/model:Q4_K_M") == "hf.co/org/model:Q4_K_M"
    assert (
        is_cloud_model("gpt-oss:120b-cloud")
        and is_cloud_model("x:cloud")
        and not is_cloud_model("qwen3.5:4b")
    )
    for bad in ("../x", "/abs", "a b", "x:cloud"):
        with pytest.raises(OllamaError):
            check_model_name(bad)


# ---- client -------------------------------------------------------------------------------


def test_client_lists_shows_and_maps_capabilities(ollama: str) -> None:
    client = OllamaClient(ollama)
    assert client.version() == "0.40.14"
    models = {m.name: m for m in client.list_models()}
    assert models["big:cloud"].remote and not models["qwen3.5:4b"].remote
    details = client.show("qwen3.5:4b")
    assert details.context_length == 262144
    caps = details.model_capabilities()
    assert caps.tools == "supported" and caps.image_input == "supported"
    assert caps.parallel_tool_calls == "unknown"  # never guessed
    embedding = client.show("qwen3-embedding:0.6b").model_capabilities()
    assert embedding.tools == "unsupported" and embedding.structured_output == "unsupported"
    with pytest.raises(OllamaError) as error:
        client.show("missing:1b")
    assert error.value.code == "model_missing"


def test_client_refuses_redirects_and_reports_unreachable(ollama: str) -> None:
    client = OllamaClient(ollama)
    with pytest.raises(OllamaError) as error:
        client.http.request("GET", "/redirect")
    assert error.value.code == "redirect"
    with pytest.raises(OllamaError) as error:
        OllamaClient("http://127.0.0.1:9").version()
    assert error.value.code == "unreachable"


def test_pull_streams_progress_and_can_be_cancelled(ollama: str) -> None:
    client = OllamaClient(ollama)
    steps = list(client.pull("qwen3.5:4b"))
    assert steps[-1].status == "success" and steps[1].total == 100
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(OllamaError) as error:
        list(client.pull("qwen3.5:4b", cancel=cancel))
    assert error.value.code == "cancelled"


def test_gguf_import_uploads_a_verified_blob(ollama: str, tmp_path: Path) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 100)
    statuses = [p.status for p in OllamaClient(ollama).create_from_gguf("mine:1", gguf)]
    assert statuses[-1] == "success"
    create = next(body for method, path, body in FakeOllama.requests if path == "/api/create")
    assert isinstance(create, dict)
    assert create["files"] == {"model.gguf": "sha256:" + hashlib.sha256(gguf.read_bytes()).hexdigest()}


# ---- embeddings ---------------------------------------------------------------------------


def test_ollama_embeddings_identity_batches_and_usage(ollama: str) -> None:
    seen = []
    model = OllamaEmbeddings("qwen3-embedding:0.6b", base_url=ollama, batch_size=2)
    model.on_usage = seen.append
    identity = model.identity()
    assert identity.dimensions == 1024 and identity.provider == "ollama" and identity.digest
    vectors = model.embed_documents(["a", "bb", "ccc"])
    assert len(vectors) == 3 and len(vectors[0]) == 1024
    assert abs(sum(x * x for x in vectors[0]) - 1.0) < 1e-6  # normalized
    assert [u.inputs for u in seen] == [2, 1] and seen[0].prompt_tokens == 7 and seen[0].duration_ms == 2.0
    model.embed_query("refund?")
    body = [b for _, path, b in FakeOllama.requests if path == "/api/embed"][-1]
    assert isinstance(body, dict)
    assert body["truncate"] is False  # never truncated silently
    assert body["input"][0].startswith("Instruct: Given a search query")


def test_embedding_errors_are_explicit(ollama: str) -> None:
    with pytest.raises(EmbeddingError) as error:
        OllamaEmbeddings("qwen3-embedding:0.6b", base_url=ollama).embed_documents(["x" * 5000])
    assert error.value.code == "input_too_long"
    with pytest.raises(EmbeddingError) as error:
        OllamaEmbeddings("missing:1b", base_url=ollama).identity()
    assert error.value.code == "model_missing"
    with pytest.raises(EmbeddingError) as error:
        OllamaEmbeddings("qwen3.5:4b", base_url=ollama).identity()
    assert error.value.code == "not_embedding_model"
    with pytest.raises(EmbeddingError) as error:
        OllamaEmbeddings("qwen3-embedding:0.6b", base_url=ollama).embed_documents([" "])
    assert error.value.code == "empty_input"


# ---- local pricing ------------------------------------------------------------------------


def test_local_inference_has_no_api_price_but_is_not_unpriced() -> None:
    from orkexa.core.types import TokenUsage

    usage = TokenUsage(input_tokens=33, output_tokens=227)
    local = record_token_usage(
        usage, agent_name="a", model="qwen3.5:4b", provider="ollama", custom_pricing=PricingBasis()
    )
    assert (
        local.estimated_cost_usd == Decimal(0) and local.pricing_source == "local" and local.local_inference
    )
    remote = record_token_usage(
        usage, agent_name="a", model="my-model", provider="vllm", custom_pricing=PricingBasis()
    )
    assert remote.estimated_cost_usd is None  # a self-hosted OpenAI-compatible server stays unpriced
    summary = summarize_usage([local, remote])
    assert (
        summary.local_requests == 1 and summary.unpriced_requests == 1 and summary.estimated_cost_usd is None
    )


# ---- runtime install ----------------------------------------------------------------------


def _artifact(path: Path, name: str) -> RuntimeArtifact:
    return RuntimeArtifact("linux", "x86_64", name, hashlib.sha256(path.read_bytes()).hexdigest())


def _tar(path: Path, members: dict[str, bytes], links: dict[str, str] | None = None) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            bundle.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            bundle.addfile(info)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX runtime layout with symlinks")
def test_install_from_a_verified_archive(tmp_path: Path) -> None:
    archive = tmp_path / "ollama-darwin.tgz"
    _tar(
        archive,
        {"bin/ollama": b"#!/bin/sh\n", "lib/ollama/libx.so.1": b"x"},
        {"lib/ollama/libx.so": "libx.so.1"},
    )
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "home", artifact=_artifact(archive, archive.name))
    assert error.value.code == "runtime_missing"  # neither archive nor download: nothing happens
    binary = install_runtime(tmp_path / "home", archive=archive, artifact=_artifact(archive, archive.name))
    assert binary.name == "ollama" and os.access(binary, os.X_OK)
    assert (binary.parent.parent / "lib/ollama/libx.so").is_symlink()


def test_install_refuses_wrong_checksums_and_unsafe_archives(tmp_path: Path) -> None:
    archive = tmp_path / "ollama-darwin.tgz"
    _tar(archive, {"bin/ollama": b"x"})
    wrong = RuntimeArtifact("linux", "x86_64", archive.name, "0" * 64)
    with pytest.raises(OllamaError) as error:
        install_runtime(tmp_path / "h1", archive=archive, artifact=wrong)
    assert error.value.code == "checksum_mismatch"
    for members, links in (
        ({"../evil": b"x", "bin/ollama": b"x"}, None),
        ({"bin/ollama": b"x"}, {"lib/l": "/etc/passwd"}),
    ):
        bad = tmp_path / "ollama-bad.tgz"
        _tar(bad, members, links)
        with pytest.raises(OllamaError) as error:
            install_runtime(tmp_path / "h2", archive=bad, artifact=_artifact(bad, "x.tgz"))
        assert error.value.code == "unsafe_archive"
    assert not (tmp_path / "evil").exists()
    zipped = tmp_path / "ollama.zip"
    with zipfile.ZipFile(zipped, "w") as bundle:
        bundle.writestr("ollama.exe", b"MZ")
        bundle.writestr("../../escape.txt", b"x")
    with pytest.raises(OllamaError):
        install_runtime(tmp_path / "h3", archive=zipped, artifact=_artifact(zipped, "x.zip"))


# ---- managed server -----------------------------------------------------------------------

FAKE_BINARY = """#!{python}
import http.server, json, os, sys
host, port = os.environ["OLLAMA_HOST"].rsplit(":", 1)
names = ("OLLAMA_NO_CLOUD", "OLLAMA_MODELS", "HOME", "HTTPS_PROXY", "SECRET_TOKEN")
seen = {{k: os.environ.get(k) for k in names}}
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({{"version": "0.40.14", "env": seen}}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)
http.server.HTTPServer((host, int(port)), H).serve_forever()
"""


def test_managed_server_starts_on_loopback_with_a_minimal_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if sys.platform == "win32":
        pytest.skip("the fake binary is a shebang script")
    binary = tmp_path / "ollama"
    binary.write_text(FAKE_BINARY.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("SECRET_TOKEN", "do-not-pass")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    server = ManagedOllama(tmp_path / "home", binary=binary, offline=True)
    with server:
        state = server.status()
        assert state.running and state.base_url and state.base_url.startswith("http://127.0.0.1:")
        env = server.client().http.request("GET", "/api/version")["env"]
        assert env["OLLAMA_NO_CLOUD"] == "1" and env["SECRET_TOKEN"] is None and env["HTTPS_PROXY"] is None
        assert env["OLLAMA_MODELS"] == str(tmp_path / "home" / "models")
        assert env["HOME"] == str(tmp_path / "home" / "ollama-home")
    assert not server.status().running
    with pytest.raises(ValueError):
        ManagedOllama(tmp_path, host="0.0.0.0")
    with pytest.raises(ValueError):
        ManagedOllama(tmp_path, environment={"OLLAMA_HOST": "0.0.0.0:1"})


def test_managed_server_needs_a_runtime(tmp_path: Path) -> None:
    with pytest.raises(OllamaError) as error:
        ManagedOllama(tmp_path).start()
    assert error.value.code == "runtime_missing"


# ---- offline packages ---------------------------------------------------------------------


def _blob(store: Path, data: bytes) -> tuple[str, int]:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    (store / "blobs").mkdir(parents=True, exist_ok=True)
    (store / "blobs" / digest.replace(":", "-")).write_bytes(data)
    return digest, len(data)


def _store(root: Path) -> Path:
    """A model folder like Ollama 0.40 writes it: a manifest list (v2) and a plain v1 model."""
    weights, wsize = _blob(root, b"GGUF weights" * 100)
    config, csize = _blob(root, b'{"model_format":"gguf"}')
    manifest = json.dumps(
        {
            "schemaVersion": 2,
            "config": {"digest": config, "size": csize},
            "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": weights, "size": wsize}],
        }
    ).encode()
    child, _ = _blob(root, manifest)
    listing = json.dumps(
        {
            "mediaType": "application/vnd.ollama.manifest.list.v2+json",
            "manifests": [
                {"digest": child, "runner": "llamacpp", "format": "gguf"},
                {"digest": "sha256:" + "f" * 64, "runner": "mlx", "format": "safetensors"},
            ],
        }
    ).encode()
    entry, _ = _blob(root, listing)
    link = root / "manifests-v2/ollama.com/library/tiny/1b"
    link.parent.mkdir(parents=True)
    link.symlink_to(Path("../../../../blobs", entry.replace(":", "-")))
    old = root / "manifests/registry.ollama.ai/library/old/latest"
    old.parent.mkdir(parents=True)
    old.write_bytes(manifest)
    return root


def test_scan_export_verify_import_roundtrip(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    models = {m.name: m for m in scan_store(store)}
    assert set(models) == {"tiny:1b", "old:latest"}
    child_digest = hashlib.sha256(
        (store / "manifests/registry.ollama.ai/library/old/latest").read_bytes()
    ).hexdigest()
    assert models["tiny:1b"].digest == child_digest  # the GGUF manifest, as `ollama list` shows it
    report = export_package(store, ["tiny:1b", "old"], tmp_path / "usb")
    assert report.models == ["tiny:1b", "old:latest"] and not report.signed
    assert verify_package(tmp_path / "usb").files == report.files
    target = tmp_path / "ssd/models"
    first = import_package(tmp_path / "usb", target)
    assert set(first.imported) == {"tiny:1b", "old:latest"}
    assert {m.name: m.digest for m in scan_store(target)} == {n: m.digest for n, m in models.items()}
    again = import_package(tmp_path / "usb", target)
    assert again.skipped_blobs == again.files  # resumable / idempotent


def test_damaged_or_malicious_packages_are_refused(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    export_package(store, ["tiny:1b"], tmp_path / "usb")
    biggest = max((tmp_path / "usb/blobs").iterdir(), key=lambda p: p.stat().st_size)
    data = bytearray(biggest.read_bytes())
    data[0] ^= 1
    biggest.write_bytes(bytes(data))
    with pytest.raises(OllamaError) as error:
        import_package(tmp_path / "usb", tmp_path / "target")
    assert error.value.code == "checksum_mismatch"
    assert not any((tmp_path / "target").glob("manifests-v2/**/1b"))  # no half model is visible
    index = json.loads((tmp_path / "usb/orkexa-models.json").read_text())
    index["models"][0]["name"] = "../../etc:passwd"
    (tmp_path / "usb/orkexa-models.json").write_text(json.dumps(index))
    with pytest.raises(OllamaError):
        verify_package(tmp_path / "usb")
    (tmp_path / "usb2").mkdir()
    (tmp_path / "usb2/orkexa-models.json").symlink_to(tmp_path / "usb/orkexa-models.json")
    with pytest.raises(OllamaError):
        verify_package(tmp_path / "usb2")
    with pytest.raises(OllamaError) as error:
        scan_store(tmp_path / "nothing")
    assert error.value.code == "not_a_store"


# ---- live tests (real Ollama) ---------------------------------------------------------------

LIVE = os.environ.get("ORKEXA_TEST_OLLAMA_URL")
live = pytest.mark.skipif(not LIVE, reason="set ORKEXA_TEST_OLLAMA_URL to a real Ollama with qwen3.5:4b")


@live
def test_live_agent_answer_and_embeddings() -> None:
    assert LIVE
    client = OllamaClient(LIVE)
    caps = client.show("qwen3.5:4b").capability_overrides()
    agent = Agent(model="ollama:qwen3.5:4b", base_url=client.openai_base_url, capabilities=caps)
    answer = agent.ask("Reply with exactly one word: the capital of France?")
    assert "paris" in answer.lower()
    usage = agent.usage
    assert usage.local_requests == 1 and usage.estimated_cost_usd == 0 and usage.input_tokens > 0
    vectors = OllamaEmbeddings("qwen3-embedding:0.6b", base_url=LIVE).embed_documents(
        ["Garantie", "warranty"]
    )
    assert len(vectors[0]) == 1024
