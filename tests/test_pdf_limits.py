"""PDF extraction budgets and the isolated extraction process (MUP-ONA A1, PS-1).

The fixtures come from ``tests/fixtures/pdf/make_pdf_fixtures.py``; the large ones are generated at test time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("pypdf")

from orkexa import SQLiteIndex, _pdf_launch
from orkexa._pdf_launch import child_environment, run_worker
import orkexa.retrieval as retrieval
from orkexa.retrieval import PdfLimits, RetrievalError, pdf_pages, pdf_pages_isolated

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pdf"
POSIX = os.name == "posix"
LINUX = sys.platform.startswith("linux")


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_pdf_fixtures", FIXTURES / "make_pdf_fixtures.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAKE = _generator()


@pytest.fixture(scope="module")
def pages_100k() -> bytes:
    data: bytes = MAKE.pages_100k()
    return data


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def launched(monkeypatch: pytest.MonkeyPatch) -> list[subprocess.Popen[bytes]]:
    """Every worker process the launcher starts during the test."""
    started: list[subprocess.Popen[bytes]] = []
    real = subprocess.Popen

    def spy(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        process: subprocess.Popen[bytes] = real(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(_pdf_launch.subprocess, "Popen", spy)
    return started


def _assert_gone(process: subprocess.Popen[bytes]) -> None:
    assert process.returncode is not None  # reaped
    if POSIX:
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        with pytest.raises(ProcessLookupError):  # nothing is left in its process group either
            os.killpg(process.pid, 0)


def _too_complex(call: Callable[[], object]) -> RetrievalError:
    with pytest.raises(RetrievalError) as error:
        call()
    assert error.value.code == "document.too_complex"
    return error.value


def _fastest(call: Callable[[], object], runs: int, below: float = 0.0) -> float:
    """The fastest of up to ``runs`` wall-clock runs of ``call`` in seconds; stops at a run under ``below``.

    A busy or shared machine only ever adds time to a run (the process waits for a CPU), so the fastest run is
    the best estimate of what the code itself costs. A refusal that does too much work is slow in every run.
    """
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
        if best < below:
            break
    return best


# MUP-ONA T-PDF-1
@pytest.mark.parametrize("extract", [pdf_pages, pdf_pages_isolated], ids=["pdf_pages", "isolated"])
def test_a_pdf_with_100k_pages_is_refused_quickly(pages_100k: bytes, extract: Callable[..., Any]) -> None:
    """The refusal costs about as much as opening the file, not as much as reading its 100,000 pages.

    pypdf has to read the cross-reference table and the page tree root (with its 100,000 /Kids) before /Count
    can be checked, so the refusal takes time in proportion to the file (0.5 seconds for these 12 MB on a fast
    machine), not a fixed time. A fixed limit in seconds failed on slow shared CI machines (macOS Intel) even
    though nothing was wrong. The refusal is therefore compared with two costs measured on the same machine:
    opening the file with pypdf, and one extraction of a small PDF (for ``pdf_pages_isolated`` the start of
    its worker process). Measured on one machine, in multiples of opening the file: the refusal 1.4 (1.8 with
    the worker process); without the /Count check, the refusal after reading the whole page tree 24.
    """
    from pypdf import PdfReader

    small = _fixture("normal_10p.pdf")
    opened = _fastest(lambda: PdfReader(io.BytesIO(pages_100k)), runs=2)
    started = _fastest(lambda: extract(small), runs=2)
    allowed = 5 * opened + started

    def refuse() -> None:
        assert _too_complex(lambda: extract(pages_100k)).reason == "pages"

    refused = _fastest(refuse, runs=3, below=allowed)
    assert refused < allowed, f"refused in {refused:.2f} s; opening {opened:.2f} s, starting {started:.2f} s"


# MUP-ONA T-PDF-1
def test_the_page_budget_holds_when_the_page_count_lies() -> None:
    data = MAKE.pages_100k(3000).replace(b"/Count 3000", b"/Count 1")
    assert _too_complex(lambda: pdf_pages(data)).reason == "pages"
    assert _too_complex(lambda: pdf_pages_isolated(data)).reason == "pages"
    assert len(pdf_pages(data, max_pages=3000)) == 3000


# MUP-ONA T-PDF-1
def test_add_file_applies_the_budgets(tmp_path: Path, pages_100k: bytes) -> None:
    path = tmp_path / "huge.pdf"
    path.write_bytes(pages_100k)
    with SQLiteIndex(tmp_path / "index.db") as index:
        index.add("kept", source="huge.pdf")
        assert _too_complex(lambda: index.add_file(path)).reason == "pages"
        assert index.sources() == ["huge.pdf"]  # the old chunks stay


# MUP-ONA T-PDF-2
@pytest.mark.parametrize("extract", [pdf_pages, pdf_pages_isolated], ids=["pdf_pages", "isolated"])
def test_the_character_budget_stops_the_extraction(extract: Callable[..., Any]) -> None:
    data = _fixture("normal_10p.pdf")
    error = _too_complex(lambda: extract(data, max_chars=150))
    assert error.reason == "chars"
    assert len(extract(data, max_chars=1000)) == 10


# MUP-ONA T-PDF-2
def test_an_oversized_worker_output_is_cut_off() -> None:
    run = run_worker(
        b"x" * 200_000,
        timeout=10,
        max_memory_mb=512,
        max_pages=10,
        max_chars=1,  # the parent reads at most 4 bytes + 64 KiB; the probe echoes more than that
        worker_module="orkexa._pdf_probe",
        extra_header={"probe": "echo"},
    )
    assert run.output_exceeded and run.result is None


# MUP-ONA T-PDF-3
def test_a_slow_pdf_is_stopped_at_the_timeout_and_its_process_is_gone(
    launched: list[subprocess.Popen[bytes]],
) -> None:
    data = _fixture("slow_content_stream.pdf")
    started = time.monotonic()
    error = _too_complex(lambda: pdf_pages_isolated(data, timeout=2))
    elapsed = time.monotonic() - started
    assert error.reason == "timeout"
    assert elapsed <= 2 + 5
    (process,) = launched
    _assert_gone(process)


# MUP-ONA T-PDF-4
@pytest.mark.skipif(not LINUX, reason="the memory limit (RLIMIT_AS) is enforced on Linux only")
def test_a_flate_bomb_hits_the_memory_limit_and_the_parent_is_unaffected() -> None:
    data = MAKE.flate_bomb()
    error = _too_complex(lambda: pdf_pages_isolated(data, max_memory_mb=256))
    assert error.reason == "memory"
    assert pdf_pages_isolated(_fixture("normal_10p.pdf")) == MAKE.NORMAL_TEXTS  # the parent works on
    block = bytearray(64 << 20)  # and can still allocate
    assert len(block) == 64 << 20


# MUP-ONA T-PDF-5
def test_a_normal_pdf_gives_the_same_text_as_before() -> None:
    from pypdf import PdfReader

    data = _fixture("normal_10p.pdf")
    before = [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]  # the code before A1
    assert before == MAKE.NORMAL_TEXTS
    assert pdf_pages(data) == before
    assert retrieval._pdf_pages(data) == before
    assert pdf_pages_isolated(data) == before


# MUP-ONA T-PDF-5
def test_the_private_alias_is_kept() -> None:
    assert retrieval._pdf_pages is pdf_pages
    assert PdfLimits() == PdfLimits(max_pages=2000, max_chars=5_000_000, timeout=30.0, max_memory_mb=512)


# MUP-ONA T-PDF-6
def test_parallel_extractions_leave_no_process_behind(launched: list[subprocess.Popen[bytes]]) -> None:
    data = _fixture("normal_10p.pdf")
    results: list[list[str]] = []
    errors: list[BaseException] = []

    def work() -> None:
        try:
            results.append(pdf_pages_isolated(data))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert errors == []
    assert results == [MAKE.NORMAL_TEXTS] * 20
    assert len(launched) == 20
    for process in launched:
        _assert_gone(process)


# MUP-ONA T-PDF-7
def test_a_crashing_worker_gives_a_defined_error(launched: list[subprocess.Popen[bytes]]) -> None:
    run = run_worker(
        b"%PDF",
        timeout=10,
        max_memory_mb=512,
        max_pages=10,
        max_chars=100,
        worker_module="orkexa._pdf_probe",
        extra_header={"probe": "crash"},
    )
    assert run.result is None and run.returncode == 7 and not run.timed_out
    _assert_gone(launched[0])


# MUP-ONA T-PDF-7
def test_a_worker_without_result_is_reported_as_a_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def crashed(data: bytes, **limits: Any) -> _pdf_launch.WorkerRun:
        return _pdf_launch.WorkerRun(result=None, returncode=-11)

    monkeypatch.setattr(_pdf_launch, "run_worker", crashed)
    assert _too_complex(lambda: pdf_pages_isolated(b"%PDF")).reason == "crash"


# MUP-ONA T-PDF-7
def test_an_unreadable_pdf_raises_value_error() -> None:
    with pytest.raises(ValueError, match="cannot be read"):
        pdf_pages_isolated(_fixture("nested_objects.pdf"))
    with pytest.raises(ValueError, match="cannot be read"):
        pdf_pages_isolated(b"not a pdf at all")


# ---- isolation of the child (T-PDF-11..14) ------------------------------------------------------------

SECRETS = {
    "ANTHROPIC_API_KEY": "sk-ant-test-value",
    "NODESTRO_SECRET_KEY": "nodestro-secret-test-value",
    "NODESTRO_TEST_POSTGRES": "postgresql://user:pw@localhost:5432/db",
    "PGPASSWORD": "pg-password-test-value",
    "AWS_SECRET_ACCESS_KEY": "aws-secret-test-value",
    "HTTPS_PROXY": "http://proxy.invalid:3128",
    "PYTHONPATH": "/somewhere/else",
}


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., dict[str, Any]]]:
    """Runs the diagnostic worker through the production launcher; parent secrets are set meanwhile."""
    marker = f"orkexa-test-{os.urandom(8).hex()}"
    monkeypatch.setenv("ORKEXA_TEST_SECRET", marker)
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)

    def run(**header: Any) -> dict[str, Any]:
        result = run_worker(
            b"%PDF-diagnostic",
            timeout=20,
            max_memory_mb=300,
            max_pages=10,
            max_chars=1_000_000,
            worker_module="orkexa._pdf_probe",
            extra_header=header,
        )
        assert result.result is not None, result
        report: dict[str, Any] = result.result
        report["marker"] = marker
        return report

    yield run


# MUP-ONA T-PDF-11
def test_the_child_environment_is_exactly_the_allowlist(probe: Callable[..., dict[str, Any]]) -> None:
    report = probe()
    environ: dict[str, str] = report["environ"]
    names = set(environ)
    if sys.platform == "darwin":  # set by macOS itself for every process, not inherited
        names.discard("__CF_USER_TEXT_ENCODING")
    if sys.platform == "win32":  # set by Windows on Arm itself for every process, not inherited
        names.discard("PROCESSOR_ARCHITECTURE")
    expected = {"PYTHONIOENCODING", "PYTHONHASHSEED"} | (
        {"SYSTEMROOT"} if sys.platform == "win32" else {"LC_ALL"}
    )
    assert names == expected == set(child_environment())
    text = json.dumps(environ)
    for value in [report["marker"], *SECRETS.values()]:
        assert value not in text
    for name in ["ORKEXA_TEST_SECRET", *SECRETS]:
        assert name not in names


# MUP-ONA T-PDF-12
@pytest.mark.skipif(sys.platform == "win32", reason="descriptor listing of Linux and macOS")
def test_the_child_inherits_no_file_descriptors(probe: Callable[..., dict[str, Any]], tmp_path: Path) -> None:
    held = (tmp_path / "held.txt").open("w")
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    for handle in (held, listener, client, server):  # even inheritable descriptors are closed
        os.set_inheritable(handle.fileno(), True)
    try:
        descriptors = set(probe()["descriptors"])
    finally:
        for closable in (held, listener, client, server):
            closable.close()
    extra = descriptors - {0, 1, 2}
    if LINUX:
        assert len(extra) <= 1  # the descriptor that lists /proc/self/fd
    else:
        assert extra == set()


# MUP-ONA T-PDF-13
@pytest.mark.skipif(not LINUX, reason="RLIMIT_AS is enforced on Linux only")
def test_the_child_runs_with_lowered_resource_limits(probe: Callable[..., dict[str, Any]]) -> None:
    report = probe()
    limits = report["limits"]
    assert limits["RLIMIT_AS"] == [300 << 20, 300 << 20]
    assert limits["RLIMIT_NOFILE"] == [16, 16]
    assert limits["RLIMIT_FSIZE"] == [0, 0]
    assert limits["RLIMIT_CPU"][0] <= 20 + 2
    assert report["write"].startswith("refused")


# MUP-ONA T-PDF-14
def test_the_child_works_in_an_empty_private_directory_that_is_removed(
    probe: Callable[..., dict[str, Any]],
) -> None:
    report = probe()
    assert report["cwd_entries"] == []
    cwd = Path(report["cwd"])
    assert not cwd.exists()
    if POSIX:
        assert report["write"].startswith("refused")


# MUP-ONA T-PDF-14
def test_the_child_arguments_hold_no_environment_data(probe: Callable[..., dict[str, Any]]) -> None:
    report = probe()
    argv: list[str] = report["argv"]
    assert len(argv) == 2  # the module and the directory pypdf was found in
    assert Path(argv[1]).is_dir()
    text = json.dumps(argv)
    for value in [report["marker"], *SECRETS.values()]:
        assert value not in text
    assert report["stdin_bytes"] == len(b"%PDF-diagnostic")


# MUP-ONA T-PDF-14
def test_the_public_function_does_not_accept_another_worker() -> None:
    parameters = inspect.signature(pdf_pages_isolated).parameters
    assert set(parameters) == {"data", "timeout", "max_memory_mb", "max_pages", "max_chars"}
    with pytest.raises(TypeError):
        pdf_pages_isolated(b"%PDF", worker_module="orkexa._pdf_probe")  # type: ignore[call-arg]
    assert _pdf_launch.WORKER_MODULE == "orkexa._pdf_worker"


# MUP-ONA T-PDF-14
def test_the_launch_command_is_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    real = subprocess.Popen

    def spy(args: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        seen.update(kwargs, args=args)
        return real(args, **kwargs)

    monkeypatch.setattr(_pdf_launch.subprocess, "Popen", spy)
    pdf_pages_isolated(_fixture("normal_10p.pdf"))
    assert seen["args"][:4] == [sys.executable, "-I", "-m", "orkexa._pdf_worker"]
    assert seen["close_fds"] is True
    assert seen["start_new_session"] is POSIX
    assert seen["env"] == child_environment()
    assert "pass_fds" not in seen


# ---- fixtures ---------------------------------------------------------------------------------------


# MUP-ONA T-PDF-5
@pytest.mark.parametrize("name", MAKE.COMMITTED)
def test_the_committed_fixtures_are_reproduced_by_the_script(name: str) -> None:
    assert MAKE.BUILDERS[name]() == _fixture(name)
