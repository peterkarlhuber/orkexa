"""Offline model packages and existing Ollama model folders.

An *offline model package* is a folder (for a USB drive or an external SSD) with the
content-addressed files of the chosen models and an index, ``orkexa-models.json``, that
lists every model with its digest and every file with its size::

    package/
        orkexa-models.json
        blobs/sha256-<hex>        (model weights, projector, config, the manifest itself)

A model's digest is the SHA-256 of its manifest, the same value ``ollama list`` and the
Ollama registry show, so a package can be checked against the catalog. Every file name
is its SHA-256, and :func:`import_package` recomputes it while copying.

:func:`export_package` writes a package from a model folder of a machine with internet
access; :func:`import_package` verifies and copies it into another model folder without
any network access. Files are written under a temporary name and renamed when verified;
the entries that make a model visible to Ollama are written last, so an interrupted
import never leaves half a model behind, and running it again continues.

A package has no signature: the checksums prove the files are intact and match the
listed digests, not who made them. :class:`PackageReport` says so (``signed=False``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import threading

from ._http import OllamaError
from .client import check_model_name, split_model_name

__all__ = [
    "PACKAGE_INDEX",
    "PackageReport",
    "StoredModel",
    "export_package",
    "import_package",
    "scan_store",
    "verify_package",
]

PACKAGE_INDEX = "orkexa-models.json"
_FORMAT = "orkexa-ollama-models"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_HOSTS_V2 = "ollama.com"
_HOSTS_V1 = "registry.ollama.ai"
_MAX_INDEX = 1024 * 1024
_MAX_MANIFEST = 256 * 1024
_MAX_BLOB = 512 * 1024**3
_MAX_MODELS = 200


@dataclass(frozen=True)
class StoredModel:
    """A model found in an Ollama model folder."""

    name: str
    digest: str  # hex SHA-256 of the model's manifest (what ``ollama list`` shows)
    blobs: tuple[tuple[str, int], ...]  # (sha256:<hex>, size) of every file except the entry
    entry: bytes = field(
        repr=False, default=b""
    )  # what the folder entry points to (manifest or manifest list)

    @property
    def entry_digest(self) -> str:
        return hashlib.sha256(self.entry).hexdigest()

    @property
    def size(self) -> int:
        return sum(size for _, size in self.blobs)


@dataclass
class PackageReport:
    """The result of :func:`export_package`, :func:`verify_package` and :func:`import_package`."""

    models: list[str] = field(default_factory=list)
    files: int = 0
    bytes: int = 0
    imported: list[str] = field(default_factory=list)
    skipped_blobs: int = 0
    created: str = ""
    signed: bool = False
    """Always ``False``: packages carry checksums, not signatures."""
    integrity: str = "sha256-verified"


def _parts(name: str) -> tuple[str, str, str]:
    base, tag = split_model_name(check_model_name(name))
    pieces = base.split("/")
    if len(pieces) == 1:
        pieces = ["library", *pieces]
    if len(pieces) != 2 or any(not p or p in (".", "..") for p in [*pieces, tag]):
        raise OllamaError("invalid_name", f"Cannot store model {name!r} in a model folder")
    return pieces[0], pieces[1], tag


def _display(namespace: str, model: str, tag: str) -> str:
    return f"{model if namespace == 'library' else f'{namespace}/{model}'}:{tag}"


def _blob(root: Path, digest: str) -> Path:
    return root / "blobs" / digest.replace(":", "-")


def _parse_manifest(raw: bytes) -> list[tuple[str, int]]:
    if len(raw) > _MAX_MANIFEST:
        raise OllamaError("invalid_package", "A manifest is larger than allowed")
    try:
        manifest = json.loads(raw)
    except ValueError as exc:
        raise OllamaError("invalid_package", "A manifest is not JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("layers"), list):
        raise OllamaError("invalid_package", "A manifest has no layers")
    entries = [manifest["config"]] if isinstance(manifest.get("config"), dict) else []
    entries += [layer for layer in manifest["layers"] if isinstance(layer, dict)]
    blobs = []
    for entry in entries:
        digest, size = entry.get("digest"), entry.get("size")
        if (
            not isinstance(digest, str)
            or not _DIGEST.fullmatch(digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise OllamaError("invalid_package", "A manifest layer has an invalid digest or size")
        if size > _MAX_BLOB:
            raise OllamaError("invalid_package", "A model file is larger than allowed")
        blobs.append((digest, size))
    return blobs


def _regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise OllamaError("invalid_package", f"{path.name} is not a regular file (links are refused)")
    return info


def _resolve(read: Callable[[str], bytes | None], raw: bytes) -> tuple[str, list[tuple[str, int]]] | None:
    """(model digest, files) of a manifest or of a manifest list (Ollama 0.40: one entry per runner).

    From a list the GGUF (llama.cpp) manifest is used; the other runners' files are not needed."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if isinstance(data, dict) and str(data.get("mediaType", "")).endswith("manifest.list.v2+json"):
        children = [
            c
            for c in data.get("manifests") or []
            if isinstance(c, dict) and _DIGEST.fullmatch(str(c.get("digest")))
        ]
        children.sort(key=lambda c: (c.get("format") != "gguf", c.get("runner") != "llamacpp"))
        for child in children:
            child_raw = read(child["digest"])
            if child_raw is None or "sha256:" + hashlib.sha256(child_raw).hexdigest() != child["digest"]:
                continue
            try:
                layers = _parse_manifest(child_raw)
            except OllamaError:
                continue
            digest = child["digest"].removeprefix("sha256:")
            return digest, [(d, s) for d, s in layers if d != child["digest"]] + [
                (child["digest"], len(child_raw))
            ]
        return None
    try:
        layers = _parse_manifest(raw)
    except OllamaError:
        return None
    digest = hashlib.sha256(raw).hexdigest()
    return digest, [(d, s) for d, s in layers if d != f"sha256:{digest}"]


def _reader(root: Path) -> Callable[[str], bytes | None]:
    def read(digest: str) -> bytes | None:
        path = _blob(root, digest)
        try:
            if _regular(path).st_size > _MAX_MANIFEST:
                return None
        except (FileNotFoundError, OllamaError):
            return None
        return path.read_bytes()

    return read


def _stored(root: Path, name: str, raw: bytes) -> StoredModel | None:
    resolved = _resolve(_reader(root), raw)
    if resolved is None:
        return None
    digest, blobs = resolved
    if any(not _blob(root, d).is_file() for d, _ in blobs):
        return None
    return StoredModel(name=name, digest=digest, blobs=tuple(blobs), entry=raw)


def scan_store(models_dir: str | os.PathLike[str]) -> list[StoredModel]:
    """The complete models of an Ollama model folder (``OLLAMA_MODELS``), e.g. on an external drive.

    Reads both layouts (``manifests-v2`` of Ollama 0.40+ and the older ``manifests``).
    Raises :class:`OllamaError` (``not_a_store``) if the folder has no ``blobs``.
    Models with missing files and links that leave the folder are skipped."""
    root = Path(models_dir)
    if not (root / "blobs").is_dir():
        raise OllamaError("not_a_store", f"{root} is not an Ollama model folder (no blobs/)")
    found: dict[str, StoredModel] = {}
    blobs_dir = (root / "blobs").resolve()
    for layout, host in (("manifests-v2", _HOSTS_V2), ("manifests", _HOSTS_V1)):
        registry = root / layout / host
        if not registry.is_dir():
            continue
        for namespace in sorted(p for p in registry.iterdir() if p.is_dir() and not p.is_symlink()):
            for model in sorted(p for p in namespace.iterdir() if p.is_dir() and not p.is_symlink()):
                for tag in sorted(model.iterdir()):
                    name = _display(namespace.name, model.name, tag.name)
                    if name in found:
                        continue
                    target = tag.resolve()
                    if tag.is_symlink() and target.parent != blobs_dir:
                        continue
                    if not target.is_file() or target.stat().st_size > _MAX_MANIFEST:
                        continue
                    item = _stored(root, name, target.read_bytes())
                    if item is not None:
                        found[name] = item
    return sorted(found.values(), key=lambda item: item.name)


def _copy_verified(
    source: Path,
    target: Path,
    digest: str,
    size: int,
    cancel: threading.Event | None,
    done: Callable[[int], None],
) -> None:
    if _regular(source).st_size != size:
        raise OllamaError("invalid_package", f"{source.name} has another size than listed")
    partial = target.with_name(target.name + ".partial")
    hasher = hashlib.sha256()
    with source.open("rb") as reader, partial.open("wb") as writer:
        while block := reader.read(8 * 1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise OllamaError("cancelled", "Copying was cancelled; run it again to continue")
            hasher.update(block)
            writer.write(block)
            done(len(block))
    if "sha256:" + hasher.hexdigest() != digest:
        partial.unlink(missing_ok=True)
        raise OllamaError("checksum_mismatch", f"{source.name} is damaged (SHA-256 differs)")
    os.replace(partial, target)


def _progress(total: int, progress: Callable[[int, int], None] | None) -> Callable[[int], None]:
    copied = 0

    def done(count: int) -> None:
        nonlocal copied
        copied += count
        if progress is not None:
            progress(copied, total)

    return done


def export_package(
    models_dir: str | os.PathLike[str],
    names: Iterable[str],
    target: str | os.PathLike[str],
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> PackageReport:
    """Write an offline package with ``names`` from ``models_dir`` into the folder ``target``."""
    available = {model.name: model for model in scan_store(models_dir)}
    wanted: list[StoredModel] = []
    for name in names:
        key = _display(*_parts(name))
        if key not in available:
            raise OllamaError("model_missing", f"{key} is not in {models_dir}")
        wanted.append(available[key])
    if not wanted:
        raise OllamaError("invalid_package", "Choose at least one model")
    root, out = Path(models_dir), Path(target)
    (out / "blobs").mkdir(parents=True, exist_ok=True)
    blobs = {digest: size for model in wanted for digest, size in model.blobs}
    done = _progress(sum(blobs.values()), progress)
    for digest, size in sorted(blobs.items()):
        destination = _blob(out, digest)
        if destination.is_file() and destination.stat().st_size == size:
            done(size)
            continue
        _copy_verified(_blob(root, digest), destination, digest, size, cancel, done)
    for model in wanted:  # the entry (manifest or manifest list) is a blob named by its digest
        _blob(out, f"sha256:{model.entry_digest}").write_bytes(model.entry)
        blobs[f"sha256:{model.entry_digest}"] = len(model.entry)
    index = {
        "format": _FORMAT,
        "version": 1,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": [
            {"name": m.name, "digest": m.digest, "entry": m.entry_digest, "size": m.size} for m in wanted
        ],
        "blobs": [{"digest": digest, "size": size} for digest, size in sorted(blobs.items())],
    }
    (out / PACKAGE_INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return PackageReport(
        models=[m.name for m in wanted],
        files=len(blobs),
        bytes=sum(blobs.values()),
        created=str(index["created"]),
    )


def _plan(package: Path) -> tuple[PackageReport, list[tuple[str, str, bytes]], dict[str, int]]:
    """(report, [(name, manifest digest, manifest bytes)], {blob digest: size}) after structural checks."""
    if package.is_symlink() or not package.is_dir():
        raise OllamaError("invalid_package", f"{package} is not a folder")
    path = package / PACKAGE_INDEX
    try:
        info = _regular(path)
    except FileNotFoundError:
        raise OllamaError(
            "invalid_package", f"{package} has no {PACKAGE_INDEX}; is it an offline model package?"
        ) from None
    if info.st_size > _MAX_INDEX:
        raise OllamaError("invalid_package", f"{PACKAGE_INDEX} is larger than allowed")
    try:
        index = json.loads(path.read_bytes())
    except ValueError as exc:
        raise OllamaError("invalid_package", f"{PACKAGE_INDEX} is not JSON") from exc
    if not isinstance(index, dict) or index.get("format") != _FORMAT or index.get("version") != 1:
        raise OllamaError("invalid_package", "Unknown package format or version")
    listed: dict[str, object] = {}
    declared = index.get("blobs")
    for item in declared if isinstance(declared, list) else []:
        if (
            isinstance(item, dict)
            and isinstance(item.get("digest"), str)
            and _DIGEST.fullmatch(item["digest"])
        ):
            listed[item["digest"]] = item.get("size")
    models = index.get("models")
    if not isinstance(models, list) or not models or len(models) > _MAX_MODELS:
        raise OllamaError("invalid_package", "The package lists no models (or too many)")
    entries: list[tuple[str, str, bytes]] = []
    needed: dict[str, int] = {}
    for item in models:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("digest"), str)
        ):
            raise OllamaError("invalid_package", "A model entry has no name or digest")
        name = _display(*_parts(item["name"]))
        digest, entry = item["digest"], item.get("entry", item["digest"])
        if not isinstance(entry, str) or not all(re.fullmatch(r"[0-9a-f]{64}", d) for d in (digest, entry)):
            raise OllamaError("invalid_package", f"{name}: invalid digest")
        raw = _reader(package)(f"sha256:{entry}")
        if raw is None or hashlib.sha256(raw).hexdigest() != entry:
            raise OllamaError(
                "checksum_mismatch", f"{name}: the manifest is missing or does not match its digest"
            )
        resolved = _resolve(_reader(package), raw)
        if resolved is None or resolved[0] != digest:
            raise OllamaError(
                "checksum_mismatch", f"{name}: the manifest does not describe model {digest[:12]}"
            )
        needed[f"sha256:{entry}"] = len(raw)
        for blob, size in resolved[1]:
            if needed.setdefault(blob, size) != size:
                raise OllamaError("invalid_package", f"{blob}: two different sizes")
        entries.append((name, entry, raw))
    for blob, size in needed.items():
        if listed.get(blob) != size:
            raise OllamaError("invalid_package", f"{blob} is not listed with its size in the index")
        try:
            if _regular(_blob(package, blob)).st_size != size:
                raise OllamaError("invalid_package", f"{blob}: wrong size")
        except FileNotFoundError:
            raise OllamaError("invalid_package", f"{blob.replace(':', '-')} is missing") from None
    report = PackageReport(
        models=[name for name, _, _ in entries],
        files=len(needed),
        bytes=sum(needed.values()),
        created=str(index.get("created") or ""),
    )
    return report, entries, needed


def verify_package(
    package: str | os.PathLike[str], *, hashes: bool = True, cancel: threading.Event | None = None
) -> PackageReport:
    """Check a package completely without changing anything (``hashes=False``: structure and sizes only)."""
    root = Path(package)
    report, _, blobs = _plan(root)
    if hashes:
        for digest in blobs:
            hasher = hashlib.sha256()
            with _blob(root, digest).open("rb") as reader:
                while block := reader.read(8 * 1024 * 1024):
                    if cancel is not None and cancel.is_set():
                        raise OllamaError("cancelled", "The check was cancelled")
                    hasher.update(block)
            if "sha256:" + hasher.hexdigest() != digest:
                raise OllamaError(
                    "checksum_mismatch", f"{digest.replace(':', '-')} is damaged (SHA-256 differs)"
                )
    return report


def _link(target: Path, name: str, digest: str) -> None:
    namespace, model, tag = _parts(name)
    entry = target / "manifests-v2" / _HOSTS_V2 / namespace / model / tag
    entry.parent.mkdir(parents=True, exist_ok=True)
    partial = entry.with_name(f".{tag}.partial")
    partial.unlink(missing_ok=True)
    try:
        partial.symlink_to(Path("..", "..", "..", "..", "blobs", f"sha256-{digest}"))
    except OSError:  # e.g. Windows without the symlink privilege: a copy of the manifest works as well
        shutil.copyfile(_blob(target, f"sha256:{digest}"), partial)
    os.replace(partial, entry)


def import_package(
    package: str | os.PathLike[str],
    models_dir: str | os.PathLike[str],
    *,
    names: Iterable[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> PackageReport:
    """Verify and copy the models of an offline package into ``models_dir`` (no network).

    Files already in the target with the right size are kept. The target's other models
    stay untouched; an existing model of the same name is replaced by the package's."""
    source = Path(package)
    report, entries, _ = _plan(source)
    if names is not None:
        chosen = {_display(*_parts(n)) for n in names}
        entries = [entry for entry in entries if entry[0] in chosen]
        missing = chosen - {entry[0] for entry in entries}
        if missing:
            raise OllamaError("model_missing", f"Not in the package: {', '.join(sorted(missing))}")
    target = Path(models_dir)
    try:
        (target / "blobs").mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OllamaError("models_dir", f"The model folder {target} cannot be written: {exc}") from exc
    if not os.access(target / "blobs", os.W_OK):
        raise OllamaError("models_dir", f"The model folder {target} is read-only")
    needed: dict[str, int] = {}
    for _, entry, raw in entries:
        needed[f"sha256:{entry}"] = len(raw)
        resolved = _resolve(_reader(source), raw)
        assert resolved is not None  # checked by _plan
        for blob, size in resolved[1]:
            needed.setdefault(blob, size)
    present = {
        blob
        for blob, size in needed.items()
        if _blob(target, blob).is_file() and _blob(target, blob).stat().st_size == size
    }
    missing_bytes = sum(size for blob, size in needed.items() if blob not in present)
    free = shutil.disk_usage(target).free
    if missing_bytes > free:
        raise OllamaError(
            "disk_full", f"The import needs {missing_bytes} bytes, only {free} are free in {target}"
        )
    done = _progress(sum(needed.values()), progress)
    for blob, size in sorted(needed.items()):
        if blob in present:
            report.skipped_blobs += 1
            done(size)
            continue
        _copy_verified(_blob(source, blob), _blob(target, blob), blob, size, cancel, done)
    for name, entry, _ in entries:
        _link(target, name, entry)
        report.imported.append(name)
    report.models = [name for name, _, _ in entries]
    report.files = len(needed)
    report.bytes = sum(needed.values())
    return report
