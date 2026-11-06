"""Installing the Ollama runtime from a verified release archive (explicit calls only).

The archives of one pinned Ollama release are known by name and SHA-256 (from the
release's ``sha256sum.txt``). :func:`install_runtime` installs from a local archive
(air-gapped machines: copy the archive over on a USB drive) or, only with
``download=True``, downloads it from the release page first. Either way the SHA-256
must match before anything is unpacked, and unpacking refuses absolute paths, ``..``,
links that leave the target, device files and oversized archives. Nothing is installed
system-wide and no installer script is run: the runtime lives in ``<home>/runtime/<version>``.

A download that breaks (network, cancel, a full disk) keeps its partial archive in
``<home>/runtime/.downloads`` and continues from there the next time (an HTTP range request);
the SHA-256 check decides as before. One installation runs at a time per ``home`` (a lock file),
also across processes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import errno
import hashlib
import http.client
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import tarfile
import tempfile
import threading
import urllib.error
import urllib.request
import zipfile

from ._http import OllamaError

__all__ = [
    "INSTALL_PHASES",
    "OLLAMA_VERSION",
    "RUNTIME_ARTIFACTS",
    "RuntimeArtifact",
    "artifact_for",
    "find_binary",
    "install_runtime",
    "installed_runtime",
    "remove_runtime",
]

OLLAMA_VERSION = "0.40.14"
_RELEASE = f"https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}"
_MAX_ARCHIVE = 4 * 1024**3
_MAX_UNPACKED = 8 * 1024**3
_MAX_MEMBERS = 20_000
_DOWNLOADS = ".downloads"
_LOCK = ".install.lock"

#: The phases :func:`install_runtime` reports through ``phase`` (``download`` only without an archive).
INSTALL_PHASES = ("download", "verify", "unpack")


@dataclass(frozen=True)
class RuntimeArtifact:
    system: str  # linux, darwin, windows
    machine: str  # x86_64, arm64 ("any" on macOS: one universal archive)
    name: str
    sha256: str
    gpu_dirs: tuple[str, ...] = ()

    @property
    def url(self) -> str:
        return f"{_RELEASE}/{self.name}"


RUNTIME_ARTIFACTS: tuple[RuntimeArtifact, ...] = (
    RuntimeArtifact(
        "linux",
        "x86_64",
        "ollama-linux-amd64.tar.zst",
        "993e83d08c3d13d17c55009a3e2664c8a25445e644aec3a68bef5abb6abce67d",
        gpu_dirs=("lib/ollama/cuda_v12", "lib/ollama/cuda_v13"),
    ),
    RuntimeArtifact(
        "linux",
        "arm64",
        "ollama-linux-arm64.tar.zst",
        "39bb333b7b4c0bafcdd7fad35eab1ae9aaf4518ea0f2c731de4376e3a29c94d0",
    ),
    RuntimeArtifact(
        "darwin",
        "any",
        "ollama-darwin.tgz",
        "f3bf9a528e908da0da5a2b3a4ad3252a369d62186501dc4880261e8d9023bff5",
    ),
    RuntimeArtifact(
        "windows",
        "x86_64",
        "ollama-windows-amd64.zip",
        "65d68cade6cf6375f398b518224f8c8a15e062387f38cf477abfc848de7e7e29",
    ),
    RuntimeArtifact(
        "windows",
        "arm64",
        "ollama-windows-arm64.zip",
        "df8a4e954587ac65ba52ec26b453490a21cd58f80b55f75cf0d706f1506764a8",
    ),
)


def _machine() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)


def artifact_for(system: str | None = None, machine: str | None = None) -> RuntimeArtifact:
    """The pinned archive for this (or the given) platform.

    Raises :class:`OllamaError` (``unsupported_platform``) if there is none."""
    system = (system or platform.system()).lower()
    machine = machine or _machine()
    for artifact in RUNTIME_ARTIFACTS:
        if artifact.system == system and artifact.machine in (machine, "any"):
            return artifact
    raise OllamaError("unsupported_platform", f"No Ollama {OLLAMA_VERSION} archive for {system}/{machine}")


def _binary_name(system: str | None = None) -> str:
    return "ollama.exe" if (system or platform.system()).lower() == "windows" else "ollama"


def find_binary(root: Path, system: str | None = None) -> Path | None:
    """The ``ollama`` executable inside an unpacked runtime (layouts differ per platform).

    ``system`` (``linux``, ``darwin``, ``windows``) names the runtime's platform when it is not this
    computer's, e.g. a runtime prepared for another machine (an exported application)."""
    name = _binary_name(system)
    for candidate in (
        root / "bin" / name,
        root / name,
        root / "Ollama.app/Contents/Resources/ollama",
    ):
        if candidate.is_file():
            return candidate
    return None


def installed_runtime(home: Path, version: str = OLLAMA_VERSION, system: str | None = None) -> Path | None:
    """The binary of an installed runtime under ``home``, or ``None``.

    ``system``: the runtime's platform, see :func:`find_binary`."""
    root = home / "runtime" / version
    return find_binary(root, system) if (root / ".complete").is_file() else None


def _sha256(
    path: Path,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: threading.Event | None = None,
) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    done = 0
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            if cancel is not None and cancel.is_set():
                raise OllamaError("cancelled", "The installation was cancelled")
            digest.update(block)
            done += len(block)
            if progress is not None:
                progress(done, total)
    return digest.hexdigest()


def _free_bytes(folder: Path) -> int | None:
    try:
        return shutil.disk_usage(folder).free
    except OSError:
        return None


def _disk_full(exc: OSError, where: Path) -> OllamaError | None:
    if exc.errno in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
        return OllamaError("disk_full", f"No space left on the drive of {where}")
    return None


def _download(
    artifact: RuntimeArtifact,
    target: Path,
    progress: Callable[[int, int | None], None] | None,
    cancel: threading.Event | None,
) -> None:
    """Download the archive to ``target``, continuing a partial file there (range request).

    Raises ``network`` when the release page cannot be reached or the connection breaks, ``disk_full``
    when the drive has less free space than the rest of the archive (checked before, and when writing
    fails), ``cancelled`` and ``too_large``. The partial file stays for the next attempt."""
    have = target.stat().st_size if target.is_file() else 0
    headers = {"User-Agent": "orkexa-local"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(artifact.url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and have:  # the partial file is complete (or longer): the checksum decides
            if progress is not None:
                progress(have, have)
            return
        raise OllamaError("network", f"The download of {artifact.name} failed: HTTP {exc.code}") from None
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        reason = getattr(exc, "reason", exc)
        raise OllamaError("network", f"{artifact.url} cannot be reached: {reason}") from None
    with response:
        final = response.geturl()
        if not final.startswith("https://"):
            raise OllamaError("redirect", "The download was redirected to a non-https URL")
        resumed = response.status == 206 and have > 0
        if not resumed:
            have = 0  # the server sends the whole file: start over
        length = response.length
        total = have + length if length is not None else None
        if total is not None and total > _MAX_ARCHIVE:
            raise OllamaError("too_large", "The runtime archive is larger than allowed")
        free = _free_bytes(target.parent)
        if length is not None and free is not None and free < length:
            raise OllamaError(
                "disk_full",
                f"{target.parent} has {free} bytes free; the rest of the download needs {length}",
            )
        done = have
        if progress is not None:
            progress(done, total)
        try:
            with target.open("ab" if resumed else "wb") as out:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise OllamaError("cancelled", "The runtime download was cancelled")
                    try:
                        block = response.read(1024 * 1024)
                    except (OSError, http.client.HTTPException) as exc:
                        raise OllamaError("network", f"The download broke: {exc}") from None
                    if not block:
                        break
                    done += len(block)
                    if done > _MAX_ARCHIVE:
                        raise OllamaError("too_large", "The runtime archive is larger than allowed")
                    out.write(block)
                    if progress is not None:
                        progress(done, total)
        except OSError as exc:
            raise (_disk_full(exc, target.parent) or exc) from None
        if total is not None and done < total:
            raise OllamaError("network", f"The download ended after {done} of {total} bytes")


@contextlib.contextmanager
def _install_lock(runtime: Path) -> Iterator[None]:
    """One installation at a time per runtime folder, also across processes (an advisory lock)."""
    path = runtime / _LOCK
    handle = path.open("a+b")
    try:
        try:
            if os.name == "nt":  # pragma: no cover - Windows
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise OllamaError(
                "install_running", "Another installation of the Ollama runtime is running"
            ) from None
        yield
    finally:
        handle.close()  # releases the lock


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts or ":" in path.parts[0]:
        raise OllamaError("unsafe_archive", f"Refusing archive member {name!r}")
    return path


def _inside(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _tar_stream(archive: Path, name: str) -> tarfile.TarFile:
    if name.endswith(".tar.zst"):
        try:
            import zstandard  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError:
            raise ImportError('Unpacking .tar.zst needs zstandard: pip install "orkexa[local]"') from None
        reader = zstandard.ZstdDecompressor().stream_reader(archive.open("rb"))
        return tarfile.open(fileobj=reader, mode="r|")
    return tarfile.open(archive, mode="r:gz")


def _unpack(
    archive: Path, name: str, target: Path, skip: tuple[str, ...], cancel: threading.Event | None = None
) -> None:
    try:
        _unpack_members(archive, name, target, skip, cancel)
    except OSError as exc:
        raise (_disk_full(exc, target) or exc) from None


def _unpack_members(
    archive: Path, name: str, target: Path, skip: tuple[str, ...], cancel: threading.Event | None
) -> None:
    unpacked = 0
    members = 0
    links: list[tuple[Path, str]] = []
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                if cancel is not None and cancel.is_set():
                    raise OllamaError("cancelled", "The installation was cancelled")
                members += 1
                path = _safe_member(info.filename)
                if members > _MAX_MEMBERS:
                    raise OllamaError("unsafe_archive", "The archive has too many files")
                if any(str(path).startswith(prefix) for prefix in skip):
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise OllamaError("unsafe_archive", f"Refusing link {info.filename!r} in a zip archive")
                destination = target / path
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                unpacked += info.file_size
                if unpacked > _MAX_UNPACKED:
                    raise OllamaError("unsafe_archive", "The archive unpacks to more than allowed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)
        return
    with _tar_stream(archive, name) as bundle:
        for member in bundle:
            if cancel is not None and cancel.is_set():
                raise OllamaError("cancelled", "The installation was cancelled")
            members += 1
            if members > _MAX_MEMBERS:
                raise OllamaError("unsafe_archive", "The archive has too many files")
            path = _safe_member(member.name)
            if any(str(path).startswith(prefix) for prefix in skip):
                continue
            destination = target / path
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or not _inside(target, destination.parent / link):
                    raise OllamaError(
                        "unsafe_archive", f"Refusing link {member.name!r} -> {member.linkname!r}"
                    )
                links.append((destination, member.linkname))
            elif member.isfile():
                unpacked += member.size
                if unpacked > _MAX_UNPACKED:
                    raise OllamaError("unsafe_archive", "The archive unpacks to more than allowed")
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise OllamaError("unsafe_archive", f"Unreadable member {member.name!r}")
                with destination.open("wb") as out:
                    shutil.copyfileobj(extracted, out)
                os.chmod(destination, 0o755 if member.mode & 0o111 else 0o644)
            else:  # hard links, devices, fifos
                raise OllamaError("unsafe_archive", f"Refusing special file {member.name!r}")
    # The check above sees no other link of the archive yet, so a chain (``x -> .`` and then
    # ``y -> x/../..``) passes it. Every link is checked again against the links that exist:
    # its folder before it is created, and all of them once the last one exists.
    for destination, linkname in links:
        if not _inside(target, destination.parent):
            raise OllamaError(
                "unsafe_archive", f"Refusing link {linkname!r} through a link that leaves the target"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(linkname)
    for destination, linkname in links:
        if not _inside(target, destination):
            raise OllamaError("unsafe_archive", f"Refusing link {linkname!r} that leaves the target")


def _has_nvidia() -> bool:
    return Path("/proc/driver/nvidia").exists() or shutil.which("nvidia-smi") is not None


def install_runtime(
    home: str | os.PathLike[str],
    *,
    archive: str | os.PathLike[str] | None = None,
    download: bool = False,
    gpu: bool | None = None,
    artifact: RuntimeArtifact | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    cancel: threading.Event | None = None,
    phase: Callable[[str], None] | None = None,
) -> Path:
    """Install the pinned Ollama runtime under ``home`` and return its binary.

    Args:
        home: Orkexa's local directory (the runtime goes to ``<home>/runtime/<version>``).
        archive: A release archive already on this computer (offline installation).
        download: Download the archive from the Ollama release page (needs the network;
            only with this explicit flag). A partial download of an earlier attempt continues.
        gpu: Unpack the NVIDIA GPU libraries (about 2 GB on Linux); ``None``: only if an
            NVIDIA driver is present. The CPU and Vulkan backends are always installed.
        artifact: The archive to expect; defaults to the one for this platform. Another platform's
            archive installs a runtime for that platform (its binary is found by that platform's name).
        progress: ``(bytes, total)`` callback of the download; with ``phase`` also of the checksum
            (bytes read of the archive).
        cancel: Stops the download, the checksum and the unpacking.
        phase: Called with each phase as it starts: ``download`` (only without ``archive``),
            ``verify`` and ``unpack`` (:data:`INSTALL_PHASES`).

    Raises:
        OllamaError: ``runtime_missing``, ``network``, ``disk_full``, ``cancelled``,
            ``checksum_mismatch``, ``unsafe_archive``, ``too_large``, ``redirect`` or
            ``install_running`` (another installation holds the lock).
    """
    base = Path(home)
    artifact = artifact or artifact_for()
    existing = installed_runtime(base, system=artifact.system)
    if existing is not None:
        return existing
    if archive is None and not download:
        raise OllamaError(
            "runtime_missing",
            f"Ollama {OLLAMA_VERSION} is not installed; pass archive= (offline) or download=True",
        )
    runtime = base / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)

    def enter(name: str) -> None:
        if phase is not None:
            phase(name)

    with _install_lock(runtime):
        # Another process may have finished meanwhile.
        existing = installed_runtime(base, system=artifact.system)
        if existing is not None:
            return existing
        partial = runtime / _DOWNLOADS / f"{artifact.name}.partial"
        if archive is None:
            partial.parent.mkdir(exist_ok=True)
            enter("download")
            _download(artifact, partial, progress, cancel)
        source = Path(archive) if archive is not None else partial
        if not source.is_file():
            raise OllamaError("runtime_missing", f"{source} is not a file")
        if source.stat().st_size > _MAX_ARCHIVE:
            raise OllamaError("too_large", "The runtime archive is larger than allowed")
        enter("verify")
        actual = _sha256(source, progress if phase is not None else None, cancel)
        if actual != artifact.sha256:
            if archive is None:
                partial.unlink(missing_ok=True)  # a broken download is not continued
            raise OllamaError(
                "checksum_mismatch",
                f"{source.name}: SHA-256 {actual[:16]}… does not match "
                f"Ollama {OLLAMA_VERSION}'s {artifact.name}",
            )
        enter("unpack")
        with tempfile.TemporaryDirectory(dir=runtime, prefix=".install-") as scratch:
            target = Path(scratch) / "unpacked"
            target.mkdir()
            skip = () if (gpu if gpu is not None else _has_nvidia()) else artifact.gpu_dirs
            _unpack(source, artifact.name, target, skip, cancel)
            binary = find_binary(target, artifact.system)
            if binary is None:
                raise OllamaError("unsafe_archive", "The archive contains no ollama executable")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            marker = f"version={OLLAMA_VERSION}\narchive={artifact.name}\nsha256={artifact.sha256}\n"
            (target / ".complete").write_text(marker, encoding="utf-8")
            final = runtime / OLLAMA_VERSION
            if final.exists():
                shutil.rmtree(final)
            target.rename(final)
        if archive is None:
            partial.unlink(missing_ok=True)
    result = installed_runtime(base, system=artifact.system)
    assert result is not None
    return result


def remove_runtime(home: str | os.PathLike[str], version: str = OLLAMA_VERSION) -> bool:
    """Delete an installed runtime (models are kept); ``False`` if none was installed."""
    root = Path(home) / "runtime" / version
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True
