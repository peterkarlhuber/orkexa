"""The management API of an Ollama server: models, capabilities, downloads, imports, embeddings.

:class:`OllamaClient` talks to a server that already runs (your own, or one started by
:class:`~orkexa.local.ManagedOllama`). It never starts processes and never downloads
anything unless :meth:`OllamaClient.pull` is called explicitly.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
import os
from pathlib import Path
import re
import threading
from typing import Any

from ..core.capabilities import CAPABILITY_NAMES, ModelCapabilities, Support
from ._http import DEFAULT_OLLAMA_URL, OllamaError, OllamaHTTP

__all__ = [
    "EmbedResult",
    "LocalModel",
    "ModelDetails",
    "OllamaClient",
    "PullProgress",
    "check_model_name",
    "is_cloud_model",
    "split_model_name",
]

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-/]{0,190}(:[A-Za-z0-9][A-Za-z0-9._\-]{0,100})?")


def split_model_name(name: str) -> tuple[str, str]:
    """``("qwen3.5", "4b")`` for ``"qwen3.5:4b"``; the tag defaults to ``latest``.

    Only the last colon separates the tag, so a registry host with a port
    (``host:5000/ns/model:tag``) is not supported as a model name here.
    """
    base, _, tag = name.partition(":")
    return base, tag or "latest"


def is_cloud_model(name: str) -> bool:
    """Whether an Ollama model name points to Ollama's cloud service (``:cloud``, ``-cloud`` tags).

    Such models run remotely; Orkexa's local paths refuse them."""
    _, tag = split_model_name(name)
    return tag == "cloud" or tag.endswith("-cloud")


def check_model_name(name: str, *, allow_cloud: bool = False) -> str:
    """``name`` if it is a plausible local Ollama model name, else :class:`OllamaError` (``invalid_name``)."""
    text = name.strip()
    if not _NAME.fullmatch(text) or ".." in text or text.startswith("/"):
        raise OllamaError("invalid_name", f"Not a valid Ollama model name: {name!r}")
    if not allow_cloud and is_cloud_model(text):
        raise OllamaError("cloud_model", f"{text!r} is an Ollama cloud model; it does not run locally")
    return text


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")[:26] + value[-6:] if "." in value else value
        )
    except ValueError:
        try:
            return datetime.fromisoformat(value[:19])
        except ValueError:
            return None


@dataclass(frozen=True)
class LocalModel:
    """A model installed on the server (``/api/tags``)."""

    name: str
    digest: str
    size: int
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    modified_at: datetime | None = None
    capabilities: tuple[str, ...] = ()
    context_length: int | None = None
    embedding_length: int | None = None
    remote: bool = False


@dataclass(frozen=True)
class ModelDetails:
    """What ``/api/show`` reports about one installed model."""

    name: str
    capabilities: tuple[str, ...]
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    format: str = ""
    context_length: int | None = None
    embedding_length: int | None = None
    license: str = ""
    parameters: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def model_capabilities(self) -> ModelCapabilities:
        """Orkexa's capability values from the server's metadata.

        ``tools`` and ``image_input`` follow the reported capabilities. Streaming,
        system messages and structured output are features of the Ollama server for
        every model. Everything Ollama does not report stays ``unknown`` (never guessed
        from the model name)."""
        caps = set(self.capabilities)
        values: dict[str, Support] = {
            "streaming": "supported",
            "system_messages": "supported",
            "structured_output": "supported" if "completion" in caps else "unsupported",
            "tools": "supported" if "tools" in caps else "unsupported",
            "image_input": "supported" if "vision" in caps else "unsupported",
            "document_input": "unsupported",
            "native_memory_tool": "unsupported",
            "batch": "unsupported",
            "token_counting": "unsupported",
            "server_tools": "unsupported",
            "caching": "unsupported",
        }
        reasons = {
            "tools": f"Ollama reports no tool calling for {self.name}",
            "image_input": f"Ollama reports no vision for {self.name}",
            "document_input": "Ollama takes images, not PDF documents; extract the text first",
        }
        return ModelCapabilities.all("unknown", reasons=reasons, **values)

    def capability_overrides(self) -> dict[str, Support]:
        """The known values of :meth:`model_capabilities`, for ``Agent(capabilities=...)``::

        details = OllamaClient().show("qwen3.5:9b")
        Agent(model="ollama:qwen3.5:9b", capabilities=details.capability_overrides())
        """
        caps = self.model_capabilities()
        return {name: getattr(caps, name) for name in CAPABILITY_NAMES if getattr(caps, name) != "unknown"}

    @property
    def is_embedding(self) -> bool:
        return "embedding" in self.capabilities

    @property
    def is_chat(self) -> bool:
        return "completion" in self.capabilities


@dataclass(frozen=True)
class PullProgress:
    """One progress line of a download: ``status`` with byte counts of the current layer."""

    status: str
    digest: str = ""
    total: int | None = None
    completed: int | None = None


@dataclass(frozen=True)
class EmbedResult:
    """The vectors of ``/api/embed`` with the metrics Ollama reported (``None`` when missing)."""

    model: str
    embeddings: list[list[float]]
    prompt_tokens: int | None = None
    total_duration_ms: float | None = None
    load_duration_ms: float | None = None


def _ms(value: Any) -> float | None:
    return round(value / 1e6, 3) if isinstance(value, (int, float)) and value >= 0 else None


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and value >= 0 else None


class OllamaClient:
    """A client of one Ollama server.

    Args:
        base_url: The server, e.g. ``http://127.0.0.1:11434`` (``/v1`` is accepted and removed).
        timeout: Seconds for ordinary requests (downloads and embeddings take their own).
        allow_remote: Allow a non-loopback host (Ollama has no authentication).
        allow_cloud: Allow Ollama cloud model names (``:cloud``); off by default.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        *,
        timeout: float = 30.0,
        allow_remote: bool = False,
        allow_cloud: bool = False,
    ) -> None:
        self.http = OllamaHTTP(base_url, timeout=timeout, allow_remote=allow_remote)
        self.allow_cloud = allow_cloud

    @property
    def base_url(self) -> str:
        return self.http.base_url

    @property
    def openai_base_url(self) -> str:
        """The OpenAI-compatible endpoint for :class:`~orkexa.providers.compatible.OllamaProvider`."""
        return f"{self.base_url}/v1"

    def _name(self, model: str) -> str:
        return check_model_name(model, allow_cloud=self.allow_cloud)

    # ---- server ---------------------------------------------------------------------

    def version(self, *, timeout: float = 3.0) -> str:
        """The server's version, e.g. ``"0.40.14"`` (raises :class:`OllamaError` if unreachable)."""
        data = self.http.request("GET", "/api/version", timeout=timeout) or {}
        version = data.get("version") if isinstance(data, dict) else None
        if not isinstance(version, str):
            raise OllamaError("invalid_response", "Ollama sent no version")
        return version

    def is_running(self) -> bool:
        try:
            self.version(timeout=1.5)
        except OllamaError:
            return False
        return True

    # ---- models ---------------------------------------------------------------------

    def list_models(self) -> list[LocalModel]:
        """The installed models, sorted by name. Cloud entries are marked ``remote``."""
        data = self.http.request("GET", "/api/tags") or {}
        models: list[LocalModel] = []
        for item in data.get("models") or []:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            details: dict[str, Any] = item["details"] if isinstance(item.get("details"), dict) else {}
            caps = item.get("capabilities")
            models.append(
                LocalModel(
                    name=item["name"],
                    digest=str(item.get("digest") or ""),
                    size=int(item.get("size") or 0),
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                    format=str(details.get("format") or ""),
                    modified_at=_time(item.get("modified_at")),
                    capabilities=tuple(str(c) for c in caps) if isinstance(caps, list) else (),
                    context_length=_int(details.get("context_length")),
                    embedding_length=_int(details.get("embedding_length")),
                    remote=bool(item.get("remote_host")) or is_cloud_model(item["name"]),
                )
            )
        return sorted(models, key=lambda model: model.name)

    def find(self, model: str) -> LocalModel | None:
        """The installed model named ``model`` (``name`` and ``name:latest`` are the same)."""
        base, tag = split_model_name(self._name(model))
        for item in self.list_models():
            if split_model_name(item.name) == (base, tag):
                return item
        return None

    def show(self, model: str) -> ModelDetails:
        """Capabilities and metadata of an installed model (``model_missing`` if not installed)."""
        name = self._name(model)
        data = self.http.request("POST", "/api/show", {"model": name}, model=name) or {}
        details: dict[str, Any] = data["details"] if isinstance(data.get("details"), dict) else {}
        info: dict[str, Any] = data["model_info"] if isinstance(data.get("model_info"), dict) else {}
        context = next((v for k, v in info.items() if k.endswith(".context_length")), None)
        length = next((v for k, v in info.items() if k.endswith(".embedding_length")), None)
        caps = data.get("capabilities")
        return ModelDetails(
            name=name,
            capabilities=tuple(str(c) for c in caps) if isinstance(caps, list) else (),
            family=str(details.get("family") or ""),
            parameter_size=str(details.get("parameter_size") or ""),
            quantization=str(details.get("quantization_level") or ""),
            format=str(details.get("format") or ""),
            context_length=_int(context),
            embedding_length=_int(length),
            license=str(data.get("license") or "")[:20000],
            parameters=str(data.get("parameters") or ""),
            raw={k: v for k, v in data.items() if k not in ("tensors", "license", "modelfile", "template")},
        )

    def running(self) -> list[dict[str, Any]]:
        """The models loaded in memory now (``/api/ps``)."""
        data = self.http.request("GET", "/api/ps") or {}
        return [item for item in data.get("models") or [] if isinstance(item, dict)]

    def pull(
        self, model: str, *, cancel: threading.Event | None = None, timeout: float = 60.0
    ) -> Iterator[PullProgress]:
        """Download ``model`` from the Ollama registry, yielding progress (uses the network!).

        Ollama keeps completed layers and partial downloads, so pulling again after an
        interruption continues. Set ``cancel`` to stop; the generator then raises
        :class:`OllamaError` (``cancelled``)."""
        name = self._name(model)
        for item in self.http.stream(
            "POST", "/api/pull", {"model": name, "stream": True}, timeout=timeout, model=name
        ):
            if cancel is not None and cancel.is_set():
                raise OllamaError("cancelled", f"The download of {name} was cancelled")
            yield PullProgress(
                status=str(item.get("status") or ""),
                digest=str(item.get("digest") or ""),
                total=_int(item.get("total")),
                completed=_int(item.get("completed")),
            )

    def delete(self, model: str) -> None:
        """Remove an installed model (blobs other models share stay)."""
        name = self._name(model)
        self.http.request("DELETE", "/api/delete", {"model": name}, model=name)

    def copy(self, source: str, destination: str) -> None:
        self.http.request(
            "POST", "/api/copy", {"source": self._name(source), "destination": self._name(destination)}
        )

    # ---- imports --------------------------------------------------------------------

    def has_blob(self, digest: str) -> bool:
        try:
            self.http.request("HEAD", f"/api/blobs/{digest}")
        except OllamaError as exc:
            if exc.status == 404 or exc.code == "model_missing":
                return False
            raise
        return True

    def push_blob(
        self, path: str | os.PathLike[str], *, digest: str | None = None, timeout: float = 3600.0
    ) -> str:
        """Upload a file as a blob (``sha256:<hex>``); the server checks the digest."""
        file = Path(path)
        digest = digest or "sha256:" + file_sha256(file)
        if not self.has_blob(digest):
            with file.open("rb") as handle:
                self.http.request(
                    "POST",
                    f"/api/blobs/{digest}",
                    handle,
                    timeout=timeout,
                    headers={"Content-Length": str(file.stat().st_size)},
                )
        return digest

    def create_from_gguf(
        self,
        name: str,
        gguf: str | os.PathLike[str],
        *,
        projector: str | os.PathLike[str] | None = None,
        template: str | None = None,
        system: str | None = None,
        parameters: dict[str, Any] | None = None,
        timeout: float = 3600.0,
    ) -> Iterator[PullProgress]:
        """Create model ``name`` from a local GGUF file (and an optional vision projector GGUF).

        Works offline: the file is uploaded to the server's blob store, nothing is downloaded.
        Without ``template`` Ollama uses the chat template embedded in the GGUF file."""
        target = self._name(name)
        files = {Path(gguf).name: self.push_blob(gguf, timeout=timeout)}
        if projector is not None:
            files[Path(projector).name] = self.push_blob(projector, timeout=timeout)
        body: dict[str, Any] = {"model": target, "files": files, "stream": True}
        if template is not None:
            body["template"] = template
        if system is not None:
            body["system"] = system
        if parameters:
            body["parameters"] = parameters
        for item in self.http.stream("POST", "/api/create", body, timeout=timeout, model=target):
            yield PullProgress(status=str(item.get("status") or ""))

    # ---- embeddings -------------------------------------------------------------------

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        dimensions: int | None = None,
        keep_alive: str | None = None,
        timeout: float = 120.0,
    ) -> EmbedResult:
        """Vectors for ``inputs`` (never truncated: too long inputs raise ``input_too_long``)."""
        name = self._name(model)
        body: dict[str, Any] = {"model": name, "input": list(inputs), "truncate": False}
        if dimensions is not None:
            body["dimensions"] = dimensions
        if keep_alive is not None:
            body["keep_alive"] = keep_alive
        data = self.http.request("POST", "/api/embed", body, timeout=timeout, model=name) or {}
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise OllamaError("invalid_response", "Ollama returned a different number of embeddings")
        try:
            embeddings = [[float(x) for x in vector] for vector in vectors]
        except (TypeError, ValueError) as exc:
            raise OllamaError("invalid_response", "Ollama returned an embedding that is not numbers") from exc
        if not all(math.isfinite(x) for vector in embeddings for x in vector):
            raise OllamaError("invalid_response", "Ollama returned an embedding with NaN or infinity")
        return EmbedResult(
            model=str(data.get("model") or name),
            embeddings=embeddings,
            prompt_tokens=_int(data.get("prompt_eval_count")),
            total_duration_ms=_ms(data.get("total_duration")),
            load_duration_ms=_ms(data.get("load_duration")),
        )


def file_sha256(path: str | os.PathLike[str], *, chunk: int = 4 * 1024 * 1024) -> str:
    """The hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
