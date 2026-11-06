"""Building an Ollama model from pinned upstream weights (explicit calls only).

Some open-weight models are not published in a form Ollama can pull, but Ollama's
engine runs their architecture from a GGUF file. A :class:`GGUFRecipe` describes how to
build that file reproducibly: the upstream ``safetensors`` weights (URL, size, SHA-256),
a *template* GGUF header that holds the tokenizer and hyperparameters of the same
architecture (URL, byte length, SHA-256), the metadata to override, and reference
vectors that the finished model must reproduce.

:func:`build_model` verifies both inputs, writes the GGUF file, creates the model in
Ollama with :meth:`~orkexa.local.OllamaClient.create_from_gguf` (offline) and embeds the
reference texts: the model is kept only if every vector matches the reference
(cosine >= ``min_cosine``). Inputs come from local files (air-gapped machines) or, only
with ``download=True``, from the recipe's URLs.

:data:`MDENSEON` is the recipe of LightOn's mDenseOn (Apache-2.0), a multilingual
retrieval model on mmBERT-base (ModernBERT). Its tokenizer metadata comes from the
llama.cpp conversion of mmBERT-base (MIT), with two corrections: CLS pooling (as
mDenseOn's ``1_Pooling`` configuration) and ``add_space_prefix`` (the Hugging Face
tokenizer prepends ``▁``; without it every vector drifts to cosine ~0.98).

Needs the ``convert`` extra (``gguf``, which brings ``numpy``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import struct
import tempfile
import threading
from typing import Any
import urllib.request

from ._http import OllamaError
from .client import OllamaClient, check_model_name

__all__ = [
    "MDENSEON",
    "BuildReport",
    "GGUFRecipe",
    "SourceFile",
    "build_model",
]

_MAX_HEADER = 64 * 1024 * 1024
_MAX_WEIGHTS = 8 * 1024**3


@dataclass(frozen=True)
class SourceFile:
    """A pinned input file: where it is published and what it must hash to.

    ``size`` is the exact byte length; for a template header it is the number of bytes
    taken from the start of the published file (everything before the tensor data).
    """

    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class GGUFRecipe:
    """How to build one model; see the module documentation."""

    name: str
    architecture: str
    weights: SourceFile
    template: SourceFile
    #: Upstream tensor name -> GGUF tensor name; ``{n}`` stands for the layer number.
    tensor_names: dict[str, str]
    #: Metadata values set on top of the template's (string, bool, int or float).
    overrides: dict[str, Any]
    license: str
    source: str
    document_prefix: str = ""
    query_prefix: str = ""
    #: ``(text, vector)`` pairs the built model must reproduce, texts with their prefix.
    reference: tuple[tuple[str, tuple[float, ...]], ...] = ()
    min_cosine: float = 0.9999
    #: Tensors written as float16 (by GGUF name suffix); all others stay float32.
    half_suffixes: tuple[str, ...] = (
        "token_embd.weight",
        "attn_qkv.weight",
        "attn_output.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    )
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildReport:
    """The result of :func:`build_model`."""

    name: str
    gguf_sha256: str
    gguf_bytes: int
    min_cosine: float
    digest: str
    steps: list[str] = field(default_factory=list)


# ---- inputs ---------------------------------------------------------------------------


def _sha256_file(path: Path, limit: int | None = None) -> str:
    digest = hashlib.sha256()
    left = limit
    with path.open("rb") as handle:
        while True:
            size = 4 * 1024 * 1024 if left is None else min(4 * 1024 * 1024, left)
            if size == 0:
                break
            block = handle.read(size)
            if not block:
                break
            digest.update(block)
            if left is not None:
                left -= len(block)
    return digest.hexdigest()


def _download(
    source: SourceFile,
    target: Path,
    *,
    header: bool,
    cancel: threading.Event | None,
    transferred: Callable[[int, int], None] | None = None,
) -> None:
    if not source.url.startswith("https://"):
        raise OllamaError("invalid_source", "Recipe sources must be https URLs")
    headers = {"User-Agent": "orkexa-local"}
    if header:
        headers["Range"] = f"bytes=0-{source.size - 1}"
    request = urllib.request.Request(source.url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as out:
        if not response.geturl().startswith("https://"):
            raise OllamaError("redirect", "The download was redirected to a non-https URL")
        done = 0
        while block := response.read(1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise OllamaError("cancelled", "The download was cancelled")
            block = block[: source.size - done]
            done += len(block)
            out.write(block)
            if transferred is not None:
                transferred(done, source.size)
            if done >= source.size:
                break
    if done != source.size:
        raise OllamaError("download_incomplete", f"{source.url}: got {done} of {source.size} bytes")


def _input(
    source: SourceFile,
    local: str | os.PathLike[str] | None,
    work: Path,
    *,
    header: bool,
    download: bool,
    cancel: threading.Event | None,
    transferred: Callable[[int, int], None] | None = None,
) -> Path:
    """A verified copy (or the verified local file) of one recipe input."""
    if source.size > (_MAX_HEADER if header else _MAX_WEIGHTS):
        raise OllamaError("too_large", "A recipe input is larger than allowed")
    if local is not None:
        path = Path(local)
        if not path.is_file():
            raise OllamaError("source_missing", f"{path} does not exist; check the --weights/--template path")
        if path.stat().st_size < source.size or (not header and path.stat().st_size != source.size):
            raise OllamaError("checksum_mismatch", f"{path.name} has the wrong size")
        if _sha256_file(path, source.size if header else None) != source.sha256:
            raise OllamaError("checksum_mismatch", f"{path.name} does not match the pinned SHA-256")
        return path
    if not download:
        raise OllamaError(
            "source_missing",
            "Pass the local files (weights=/template=, or --weights FILE --template FILE) "
            "or allow the download (download=True, or --download)",
        )
    target = work / ("template.gguf" if header else "weights.safetensors")
    _download(source, target, header=header, cancel=cancel, transferred=transferred)
    if _sha256_file(target) != source.sha256:
        raise OllamaError("checksum_mismatch", f"{source.url} does not match the pinned SHA-256")
    return target


# ---- GGUF template header ----------------------------------------------------------------

_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_header(path: Path, size: int) -> list[tuple[str, int, Any]]:
    """The metadata of a GGUF header as ``(key, type, value)``; arrays as ``(item_type, list)``."""
    if size > _MAX_HEADER:
        raise OllamaError("too_large", "The template header is larger than allowed")
    with path.open("rb") as handle:
        data = handle.read(size)
    pos = 0

    def take(fmt: str) -> Any:
        nonlocal pos
        width = struct.calcsize(fmt)
        if pos + width > len(data):
            raise OllamaError("invalid_template", "The template header is truncated")
        (value,) = struct.unpack_from(fmt, data, pos)
        pos += width
        return value

    def text() -> str:
        nonlocal pos
        length = take("<Q")
        if pos + length > len(data):
            raise OllamaError("invalid_template", "The template header is truncated")
        value = data[pos : pos + length].decode("utf-8")
        pos += length
        return value

    def value(kind: int) -> Any:
        if kind == 8:
            return text()
        if kind == 9:
            item = take("<I")
            count = take("<Q")
            if count > 10_000_000 or item == 9:
                raise OllamaError("invalid_template", "Unsupported array in the template header")
            return (item, [value(item) for _ in range(count)])
        if kind not in _SCALARS:
            raise OllamaError("invalid_template", f"Unknown GGUF value type {kind}")
        return take(_SCALARS[kind])

    if data[:4] != b"GGUF" or take("<4s") != b"GGUF" or take("<I") != 3:
        raise OllamaError("invalid_template", "The template is not a GGUF v3 file")
    take("<Q")  # tensor count
    entries = []
    for _ in range(take("<Q")):
        key = text()
        kind = take("<I")
        entries.append((key, kind, value(kind)))
    return entries


# ---- safetensors ---------------------------------------------------------------------


def _safetensors(path: Path) -> Iterator[tuple[str, Any]]:
    import numpy as np

    try:
        with path.open("rb") as handle:
            length = struct.unpack("<Q", handle.read(8))[0]
            if length > 100 * 1024 * 1024:
                raise OllamaError("invalid_weights", "The safetensors header is too large")
            meta = json.loads(handle.read(length))
        if not isinstance(meta, dict):
            raise ValueError("the header is not an object")
        meta.pop("__metadata__", None)
    except (struct.error, ValueError) as exc:
        raise OllamaError("invalid_weights", f"The safetensors header is invalid: {exc}") from None
    base = 8 + length
    data = np.memmap(path, dtype=np.uint8, mode="r")
    for name in sorted(meta):
        info = meta[name]
        try:
            if not isinstance(info, dict) or info.get("dtype") != "F32":
                raise OllamaError("invalid_weights", f"{name}: only float32 weights are supported")
            start, end = (int(x) for x in info["data_offsets"])
            shape = tuple(int(x) for x in info["shape"])
        except (KeyError, TypeError, ValueError):
            raise OllamaError("invalid_weights", f"{name}: invalid tensor entry") from None
        if (
            start < 0
            or any(x < 0 for x in shape)
            or end - start != 4 * math.prod(shape)
            or base + end > data.size
        ):
            raise OllamaError("invalid_weights", f"{name}: inconsistent size")
        yield name, data[base + start : base + end].view(np.float32).reshape(shape)


def _gguf_name(recipe: GGUFRecipe, name: str) -> str:
    if name in recipe.tensor_names:
        return recipe.tensor_names[name]
    parts = name.split(".")
    if len(parts) > 2 and parts[1].isdigit():
        pattern = ".".join([parts[0], "{n}", *parts[2:]])
        if pattern in recipe.tensor_names:
            return recipe.tensor_names[pattern].format(n=parts[1])
    raise OllamaError("invalid_weights", f"No GGUF name for tensor {name}")


def _write_gguf(recipe: GGUFRecipe, weights: Path, template: Path, out: Path) -> None:
    try:
        import gguf
        import numpy as np
    except ImportError:  # pragma: no cover - depends on the environment
        raise ImportError('Building models needs gguf: pip install "orkexa[convert]"') from None
    entries = _read_header(template, recipe.template.size)
    values = {key: (kind, value) for key, kind, value in entries}
    if values.get("general.architecture", (0, ""))[1] != recipe.architecture:
        raise OllamaError("invalid_template", f"The template is not a {recipe.architecture} model")
    writer = gguf.GGUFWriter(str(out), recipe.architecture)
    T = gguf.GGUFValueType
    for key, kind, value in entries:
        if key in ("general.architecture",) or key in recipe.overrides:
            continue
        if kind == 9:
            item, items = value
            writer.add_key_value(key, items, T.ARRAY, sub_type=T(item))
        else:
            writer.add_key_value(key, value, T(kind))
    for key, value in recipe.overrides.items():
        known: int | None = values[key][0] if key in values else None
        if known is not None and known != 9:
            writer.add_key_value(key, value, T(known))
        elif isinstance(value, bool):
            writer.add_bool(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)
        else:
            writer.add_string(key, str(value))
    count = 0
    for name, array in _safetensors(weights):
        target = _gguf_name(recipe, name)
        half = target.endswith(recipe.half_suffixes)
        writer.add_tensor(target, np.ascontiguousarray(array, dtype=np.float16 if half else np.float32))
        count += 1
    expected = int(recipe.extra.get("tensors", 0))
    if expected and count != expected:
        raise OllamaError("invalid_weights", f"Expected {expected} tensors, found {count}")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---- build ---------------------------------------------------------------------------


def _cosine(a: list[float], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _staging_name(name: str) -> str:
    """A unique temporary model name next to ``name`` (same namespace and tag)."""
    base, _, tag = name.partition(":")
    return check_model_name(f"{base}-orkexa-build-{secrets.token_hex(4)}:{tag or 'latest'}")


def build_model(
    client: OllamaClient,
    recipe: GGUFRecipe,
    *,
    weights: str | os.PathLike[str] | None = None,
    template: str | os.PathLike[str] | None = None,
    download: bool = False,
    work_dir: str | os.PathLike[str] | None = None,
    progress: Callable[[str], None] | None = None,
    cancel: threading.Event | None = None,
    transferred: Callable[[int, int], None] | None = None,
) -> BuildReport:
    """Build ``recipe`` into a model of the Ollama server behind ``client`` and verify it.

    Args:
        client: The Ollama server that gets the model (usually the managed one).
        recipe: What to build, e.g. :data:`MDENSEON`.
        weights: The upstream ``safetensors`` file on this computer (offline build).
        template: The template GGUF file (whole file or just its header) on this computer.
        download: Download the missing inputs from the recipe's URLs (network; only with
            this explicit flag).
        work_dir: Where the temporary GGUF file is written (default: system temp dir).
            It is deleted afterwards; Ollama keeps its own copy.
        progress: Called with a short status text per step.
        cancel: Stops a download.
        transferred: ``(bytes, total)`` of the file being downloaded in the step that runs.

    Raises:
        OllamaError: ``source_missing``, ``checksum_mismatch``, ``invalid_template``,
            ``invalid_weights``, ``too_large``, ``reference_mismatch`` (the build is removed; a
            model already installed under the recipe's name stays as it was) or a server error.
    """
    check_model_name(recipe.name)
    report = BuildReport(name=recipe.name, gguf_sha256="", gguf_bytes=0, min_cosine=0.0, digest="")

    def step(text: str) -> None:
        report.steps.append(text)
        if progress is not None:
            progress(text)

    with tempfile.TemporaryDirectory(dir=work_dir, prefix=".orkexa-build-") as scratch:
        work = Path(scratch)
        step("verify template")

        def fetch(source: SourceFile, local: str | os.PathLike[str] | None, *, header: bool) -> Path:
            return _input(
                source, local, work, header=header, download=download, cancel=cancel, transferred=transferred
            )

        template_path = fetch(recipe.template, template, header=True)
        step("verify weights")
        weights_path = fetch(recipe.weights, weights, header=False)
        out = work / f"{recipe.name.replace(':', '-').replace('/', '-')}.gguf"
        step("write gguf")
        _write_gguf(recipe, weights_path, template_path, out)
        report.gguf_sha256 = _sha256_file(out)
        report.gguf_bytes = out.stat().st_size
        # Built and checked under a temporary name: an installed model of the recipe's name is
        # replaced only by a verified build and stays as it is when the build fails.
        staging = _staging_name(recipe.name)
        try:
            step("create model")
            for _ in client.create_from_gguf(staging, out):
                pass
            step("check reference vectors")
            texts = [text for text, _ in recipe.reference]
            if texts:
                result = client.embed(staging, texts)
                report.min_cosine = min(
                    _cosine(list(vector), reference)
                    for vector, (_, reference) in zip(result.embeddings, recipe.reference, strict=True)
                )
                if report.min_cosine < recipe.min_cosine:
                    raise OllamaError(
                        "reference_mismatch",
                        f"{recipe.name} does not reproduce the reference vectors "
                        f"(cosine {report.min_cosine:.5f} < {recipe.min_cosine}); it was not installed",
                    )
            step("install model")
            client.copy(staging, recipe.name)
        finally:
            with contextlib.suppress(OllamaError):
                client.delete(staging)
    installed = client.find(recipe.name)
    report.digest = installed.digest if installed is not None else ""
    step("done")
    return report


def _load_reference(name: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    path = Path(__file__).with_name("data") / name
    data = json.loads(path.read_text(encoding="utf-8"))
    return tuple((item["text"], tuple(float(x) for x in item["vector"])) for item in data["vectors"])


MDENSEON = GGUFRecipe(
    name="mdenseon:307m",
    architecture="modern-bert",
    weights=SourceFile(
        url="https://huggingface.co/lightonai/mDenseOn/resolve/a5fdb000f7a21da96c3bddde3a782ef777316df3/model.safetensors",
        sha256="a336c49fc679aeb23969114b12fb7317804da3cc7d8bc214818009dea39d545c",
        size=1_227_771_776,
    ),
    template=SourceFile(
        url="https://huggingface.co/mykor/mmBERT-base-GGUF/resolve/66e31bd2787f6b5fe9b762a664f4899af38b0605/mmBERT-Base-307M-F16.gguf",
        sha256="a224ccdee4310594e73c9907934a1520405bfc4467a3577d3dcf418e7c017901",
        size=6_036_928,
    ),
    tensor_names={
        "embeddings.tok_embeddings.weight": "token_embd.weight",
        "embeddings.norm.weight": "token_embd_norm.weight",
        "final_norm.weight": "output_norm.weight",
        "layers.{n}.attn.Wqkv.weight": "blk.{n}.attn_qkv.weight",
        "layers.{n}.attn.Wo.weight": "blk.{n}.attn_output.weight",
        "layers.{n}.attn_norm.weight": "blk.{n}.attn_norm.weight",
        "layers.{n}.mlp.Wi.weight": "blk.{n}.ffn_up.weight",
        "layers.{n}.mlp.Wo.weight": "blk.{n}.ffn_down.weight",
        "layers.{n}.mlp_norm.weight": "blk.{n}.ffn_norm.weight",
    },
    overrides={
        "general.name": "mDenseOn",
        "general.organization": "LightOn",
        "general.license": "apache-2.0",
        "modern-bert.pooling_type": 2,  # CLS
        "tokenizer.ggml.add_space_prefix": True,
    },
    license="Apache-2.0",
    source="https://huggingface.co/lightonai/mDenseOn",
    document_prefix="document: ",
    query_prefix="query: ",
    reference=_load_reference("mdenseon-reference.json"),
    extra={"tensors": 134},
)
"""LightOn mDenseOn (307M, 768 dimensions, 8192 tokens, Apache-2.0) as ``mdenseon:307m``."""
