"""``orkexa-local`` (``python -m orkexa.local``): manage local Ollama models from the command line.

Every networked step is an explicit command (``install-runtime --download``, ``pull``);
the others work offline. ``--url`` talks to a running server (default: the one this
tool started, else ``http://127.0.0.1:11434``).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import signal
import sys
import threading

from ..errors import OrkexaError
from ._http import DEFAULT_OLLAMA_URL, OllamaError
from .client import OllamaClient
from .install import OLLAMA_VERSION, artifact_for, install_runtime, installed_runtime, remove_runtime
from .packages import export_package, import_package, scan_store, verify_package
from .server import ManagedOllama, detect_ollama

_DEFAULT_HOME = Path(os.environ.get("ORKEXA_LOCAL_HOME") or Path.home() / ".orkexa" / "local")


def _size(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if amount < 1000 or unit == "TB":
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1000
    return f"{value} B"


def _progress(label: str):  # type: ignore[no-untyped-def]
    def show(done: int, total: int | None) -> None:
        share = f" {done * 100 // total}%" if total else ""
        print(
            f"\r{label}: {_size(done)}{f' / {_size(total)}' if total else ''}{share}   ",
            end="",
            file=sys.stderr,
        )

    return show


def _client(args: argparse.Namespace) -> OllamaClient:
    state = args.home / "ollama.state.json"
    url = args.url
    if url is None and state.is_file():
        port = json.loads(state.read_text()).get("port")
        url = f"http://127.0.0.1:{port}" if port else None
    return OllamaClient(url or DEFAULT_OLLAMA_URL, allow_remote=args.allow_remote)


def _cmd_status(args: argparse.Namespace) -> int:
    binary = installed_runtime(args.home)
    print(f"runtime: {binary or 'not installed'} (pinned Ollama {OLLAMA_VERSION})")
    if binary is None:
        print("  next: orkexa-local install-runtime --download (or --archive FILE offline)")
    external = detect_ollama()
    print(f"Ollama on 127.0.0.1:11434: {external or 'none'}")
    try:
        client = _client(args)
        print(f"server {client.base_url}: {client.version()}")
        models = client.list_models()
        for model in models:
            caps = ",".join(model.capabilities)
            print(f"  {model.name:32} {_size(model.size):>10}  {model.digest[:12]}  {caps}")
        if not models:
            print("  no models; next: orkexa-local pull MODEL (or import-gguf, import)")
    except OllamaError as exc:
        print(f"server: {exc}")
        if binary is not None and exc.code == "unreachable":
            print("  next: orkexa-local serve")
    return 0


def _cmd_install(args: argparse.Namespace) -> int:
    artifact = artifact_for()
    if args.archive is None and not args.download:
        print(f"Pass --archive FILE ({artifact.name}) or --download (network).", file=sys.stderr)
        return 2
    binary = install_runtime(
        args.home, archive=args.archive, download=args.download, gpu=args.gpu, progress=_progress("download")
    )
    print(f"\ninstalled: {binary}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    server = ManagedOllama(args.home, models_dir=args.models_dir, port=args.port, offline=args.offline)

    stop = threading.Event()

    def stop_on_signal(number: int, frame: object) -> None:
        # Only set a flag: an exception raised here can land in a finalizer (for example an
        # HTTP response closed during ``status()``), where Python ignores it and serve runs on.
        stop.set()

    signal.signal(signal.SIGTERM, stop_on_signal)  # e.g. `kill`, a service manager
    try:
        url = server.start()
        print(
            f"Ollama {OLLAMA_VERSION} runs at {url} (models: {server.models_dir}); Ctrl+C stops it.",
            flush=True,
        )
        # A SIGTERM during the start is seen on the first check, before any wait.
        while not stop.is_set() and server.status().running and not stop.wait(1):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    for item in _client(args).pull(args.model):
        if item.total:
            _progress(item.status)(item.completed or 0, item.total)
        else:
            print(f"\n{item.status}", end="", file=sys.stderr)
    print()
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    _client(args).delete(args.model)
    return 0


def _cmd_import_gguf(args: argparse.Namespace) -> int:
    for item in _client(args).create_from_gguf(args.name, args.gguf, projector=args.projector):
        print(item.status, file=sys.stderr)
    return 0


_RECIPES = {"mdenseon": "MDENSEON"}


def _cmd_build(args: argparse.Namespace) -> int:
    from . import convert

    recipe = getattr(convert, _RECIPES[args.recipe])
    report = convert.build_model(
        _client(args),
        recipe,
        weights=args.weights,
        template=args.template,
        download=args.download,
        progress=lambda text: print(text, file=sys.stderr),
    )
    size = _size(report.gguf_bytes)
    print(f"{report.name}: {size}, digest {report.digest[:12]}, cosine {report.min_cosine:.6f}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    report = export_package(args.models_dir, args.models, args.target, progress=_progress("export"))
    print(f"\n{len(report.models)} models, {report.files} files, {_size(report.bytes)} -> {args.target}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    report = import_package(
        args.package, args.models_dir, names=args.models or None, progress=_progress("import")
    )
    print(f"\nimported: {', '.join(report.imported)} ({report.skipped_blobs} files were already there)")
    print("integrity: SHA-256 of every file checked; the package is not signed")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    report = verify_package(args.package, hashes=not args.quick)
    print(f"ok: {', '.join(report.models)} ({report.files} files, {_size(report.bytes)}; not signed)")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    for model in scan_store(args.folder):
        print(f"{model.name:32} {_size(model.size):>10}  {model.digest[:12]}")
    return 0


def _cmd_remove_runtime(args: argparse.Namespace) -> int:
    print("removed" if remove_runtime(args.home) else "not installed")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="orkexa-local", description="Manage local Ollama models from the command line."
    )
    root.add_argument(
        "--home",
        type=Path,
        default=_DEFAULT_HOME,
        help="runtime, state and default model folder (default: $ORKEXA_LOCAL_HOME or ~/.orkexa/local)",
    )
    root.add_argument(
        "--url", help="a running Ollama server (default: the one `serve` started, else 127.0.0.1:11434)"
    )
    root.add_argument(
        "--allow-remote", action="store_true", help="allow a non-loopback --url (no authentication!)"
    )
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("status", help="show the runtime, the server and its models").set_defaults(
        run=_cmd_status
    )
    install = commands.add_parser("install-runtime", help="install the pinned Ollama runtime")
    install.add_argument("--archive", type=Path, help="release archive on this computer (offline)")
    install.add_argument("--download", action="store_true", help="download it from the Ollama release page")
    install.add_argument(
        "--gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="unpack the NVIDIA GPU libraries (default: only if an NVIDIA driver is present)",
    )
    install.set_defaults(run=_cmd_install)
    commands.add_parser("remove-runtime", help="delete the installed runtime (models are kept)").set_defaults(
        run=_cmd_remove_runtime
    )
    serve = commands.add_parser("serve", help="run the managed server in the foreground")
    serve.add_argument("--models-dir", type=Path, help="model folder (default: <home>/models)")
    serve.add_argument("--port", type=int, help="fixed port (default: a free one)")
    serve.add_argument("--offline", action="store_true", help="no proxy route for the server")
    serve.set_defaults(run=_cmd_serve)
    pull = commands.add_parser("pull", help="download a model (network)")
    pull.add_argument("model", help="model name, e.g. qwen3.5:4b")
    pull.set_defaults(run=_cmd_pull)
    remove = commands.add_parser("rm", help="remove a model")
    remove.add_argument("model", help="model name")
    remove.set_defaults(run=_cmd_rm)
    gguf = commands.add_parser("import-gguf", help="create a model from a local GGUF file (offline)")
    gguf.add_argument("name", help="name of the new model")
    gguf.add_argument("gguf", type=Path, help="the GGUF file")
    gguf.add_argument("--projector", type=Path, help="vision projector GGUF")
    gguf.set_defaults(run=_cmd_import_gguf)
    build = commands.add_parser("build", help="build a model from pinned upstream weights and verify it")
    build.add_argument("recipe", choices=sorted(_RECIPES), help="the recipe to build")
    build.add_argument("--weights", type=Path, help="the upstream safetensors file (offline)")
    build.add_argument("--template", type=Path, help="the template GGUF file or header (offline)")
    build.add_argument("--download", action="store_true", help="download missing inputs (network)")
    build.set_defaults(run=_cmd_build)
    export = commands.add_parser("export", help="write an offline model package")
    export.add_argument("target", type=Path, help="the package folder to write")
    export.add_argument("models", nargs="+", help="names of the models to export")
    export.add_argument("--models-dir", type=Path, required=True, help="the Ollama model folder to read")
    export.set_defaults(run=_cmd_export)
    imp = commands.add_parser("import", help="import an offline model package (offline)")
    imp.add_argument("package", type=Path, help="the package folder")
    imp.add_argument("models", nargs="*", help="names to import (default: all)")
    imp.add_argument("--models-dir", type=Path, required=True, help="the Ollama model folder to write")
    imp.set_defaults(run=_cmd_import)
    verify = commands.add_parser("verify", help="check an offline model package")
    verify.add_argument("package", type=Path, help="the package folder")
    verify.add_argument("--quick", action="store_true", help="sizes only, no SHA-256")
    verify.set_defaults(run=_cmd_verify)
    scan = commands.add_parser("scan", help="list the models of an Ollama model folder")
    scan.add_argument("folder", type=Path, help="the Ollama model folder")
    scan.set_defaults(run=_cmd_scan)
    return root


def _code(exc: BaseException) -> str:
    """A short machine-readable name of an error (``OllamaError.code``, else the class name)."""
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and code else type(exc).__name__


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (OrkexaError, ImportError) as exc:
        # Every Orkexa error (and a missing optional package) is one line, never a traceback.
        print(f"error ({_code(exc)}): {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
