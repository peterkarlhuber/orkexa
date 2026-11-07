"""Cases, the runner and the report of the conformance suite (see :mod:`orkexa.testing.conformance`)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import contextlib
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import traceback
from typing import Any, Literal
import uuid

from ...embeddings import EmbeddingModel
from ...files import FileStore
from ...memory import MemoryBackend
from ...retrieval import Index, RetrievalError

CAPABILITIES: frozenset[str] = frozenset({"vectors", "reindex", "partitioned", "concurrent_writers"})
"""The capability names a backend may report (``capabilities``) and a caller may expect."""

SECURITY_NEEDS: frozenset[str] = frozenset({"partitioned", "vectors", "reindex"})
"""The only capabilities a security case may need: partition cases need ``partitioned`` (their vector and
reindex variants also ``vectors``/``reindex``, without which a backend has nothing to isolate there), vector
checks ``vectors``; input limits need nothing. ``concurrent_writers`` never switches a security case off."""

DEFAULT_PARTITION = "conformance-a"
"""The partition (tenant, or SQLite namespace) of the instances a case opens unless it names another."""

Contract = Literal["index", "files", "memory"]
Status = Literal["passed", "failed", "skipped"]

_CONTRACTS: dict[str, type] = {"index": Index, "files": FileStore, "memory": MemoryBackend}


class CaseFailure(AssertionError):
    """A conformance case found a deviation from the contract."""


def expect(condition: object, message: str) -> None:
    """Fail the running case with ``message`` unless ``condition`` holds."""
    if not condition:
        raise CaseFailure(message)


def expect_equal(actual: object, expected: object, what: str) -> None:
    """Fail the running case unless ``actual == expected``."""
    if actual != expected:
        raise CaseFailure(f"{what}: expected {expected!r}, got {actual!r}")


def expect_error(
    call: Callable[[], object],
    errors: type[BaseException] | tuple[type[BaseException], ...],
    what: str,
    *,
    codes: Iterable[str] | None = None,
) -> BaseException:
    """Run ``call`` and fail the case unless it raises one of ``errors`` (with a ``code`` of ``codes``)."""
    try:
        result = call()
    except errors as exc:
        if codes is not None:
            wanted = set(codes)
            code = getattr(exc, "code", None)
            if code not in wanted:
                raise CaseFailure(
                    f"{what}: expected code {' or '.join(sorted(wanted))}, got {type(exc).__name__} "
                    f"with code {code!r}: {exc}"
                ) from exc
        return exc
    except Exception as exc:
        names = errors.__name__ if isinstance(errors, type) else " or ".join(e.__name__ for e in errors)
        raise CaseFailure(f"{what}: expected {names}, got {type(exc).__name__}: {exc}") from exc
    names = errors.__name__ if isinstance(errors, type) else " or ".join(e.__name__ for e in errors)
    raise CaseFailure(f"{what}: expected {names}, but it returned {result!r:.200}")


def expect_code(call: Callable[[], object], what: str, *codes: str) -> RetrievalError:
    """Run ``call`` and fail the case unless it raises :class:`RetrievalError` with one of ``codes``."""
    error = expect_error(call, RetrievalError, what, codes=codes)
    assert isinstance(error, RetrievalError)
    return error


@dataclass(frozen=True)
class Request:
    """What a factory gets for every instance a case opens.

    Attributes:
        workdir: A fresh empty directory of this case; every instance of the case gets the same one (open the
            same database file or root in it, so that instances share their storage).
        scope: A fresh identifier of this case run (letters and digits). A backend that cannot use ``workdir``
            (a database server) keeps cases apart with it, e.g. as its namespace.
        partition: The partition value: a tenant id, or the namespace of a backend whose namespaces are its
            partitions (SQLite). Usually a string; the partition cases also pass hostile values (``None``,
            ``""``, ``"%"``, 10,000 characters, ...), which the factory hands to the backend unchanged.
        embeddings: The embedding model an index should use, or ``None`` (full text only). A backend without
            vectors ignores it.
    """

    workdir: Path
    scope: str
    partition: Any = DEFAULT_PARTITION
    embeddings: EmbeddingModel | None = None


Factory = Callable[[Request], Any]
"""Opens one instance (an ``Index``, ``FileStore`` or ``MemoryBackend``) for a :class:`Request`."""


class Context:
    """What a case's ``run`` gets: opens instances through the factory and closes them after the case."""

    def __init__(self, factory: Factory, workdir: Path, scope: str, capabilities: frozenset[str]) -> None:
        self.factory = factory
        self.workdir = workdir
        self.scope = scope
        self.capabilities = capabilities
        self._opened: list[Any] = []

    def make(self, partition: Any = DEFAULT_PARTITION, *, embeddings: EmbeddingModel | None = None) -> Any:
        """A new instance on this case's storage, in ``partition``, with ``embeddings`` (indexes)."""
        instance = self.factory(
            Request(workdir=self.workdir, scope=self.scope, partition=partition, embeddings=embeddings)
        )
        self._opened.append(instance)
        return instance

    def file(self, name: str, data: bytes) -> Path:
        """Write ``data`` to a file ``name`` in a subdirectory of the working directory; return its path."""
        folder = self.workdir / "inputs"
        folder.mkdir(exist_ok=True)
        path = folder / name
        path.write_bytes(data)
        return path

    def close(self) -> None:
        """Close every opened instance that has a ``close()`` (errors are ignored)."""
        for instance in reversed(self._opened):
            close = getattr(instance, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    close()
        self._opened.clear()


@dataclass(frozen=True)
class Case:
    """One conformance case.

    Attributes:
        name: Unique, stable name (``<area>.<what_it_checks>``); expectation files refer to it.
        needs: The capabilities the backend must report for the case to run; otherwise it is skipped.
        run: Performs the case; raises :class:`CaseFailure` (or any error) on a deviation.
        contract: Which contract the instances implement: ``index``, ``files`` or ``memory``.
        security: A security case (partition, input limits, vector checks): it may need only
            :data:`SECURITY_NEEDS`, and its skip is marked in the report.
    """

    name: str
    needs: frozenset[str]
    run: Callable[[Context], None]
    contract: Contract = "index"
    security: bool = False

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z]+(\.[a-z0-9_]+)+", self.name):
            raise ValueError(f"Invalid case name {self.name!r}")
        unknown = self.needs - CAPABILITIES
        if unknown:
            raise ValueError(f"{self.name}: unknown capabilities {sorted(unknown)}")
        if self.security and not self.needs <= SECURITY_NEEDS:
            raise ValueError(f"{self.name}: a security case may need only {sorted(SECURITY_NEEDS)}")
        if self.contract not in _CONTRACTS:
            raise ValueError(f"{self.name}: unknown contract {self.contract!r}")

    @property
    def description(self) -> str:
        """The first line of the run function's docstring."""
        return (inspect.getdoc(self.run) or "").split("\n", 1)[0]


@dataclass(frozen=True)
class Outcome:
    """The result of one case: ``passed``, ``failed`` or ``skipped``; ``detail`` says why."""

    name: str
    status: Status
    detail: str = ""
    security: bool = False
    seconds: float = 0.0


@dataclass
class Report:
    """The result of :func:`run_cases`.

    ``errors`` are problems of the run itself (fewer capabilities than expected, a ``tenant_id`` backend that
    does not report ``partitioned``, cases for another contract); any error or failed case makes ``ok`` false.
    """

    target: str
    capabilities: frozenset[str]
    expected: frozenset[str]
    outcomes: list[Outcome] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def _names(self, status: Status) -> list[str]:
        return [outcome.name for outcome in self.outcomes if outcome.status == status]

    @property
    def passed(self) -> list[str]:
        return self._names("passed")

    @property
    def failed(self) -> list[str]:
        return self._names("failed")

    @property
    def skipped(self) -> dict[str, str]:
        """The skipped cases and why."""
        return {o.name: o.detail for o in self.outcomes if o.status == "skipped"}

    @property
    def ok(self) -> bool:
        return not self.errors and not self.failed

    def skip_differences(self, expected: Mapping[str, str]) -> list[str]:
        """How the skipped cases differ from ``expected`` (case name -> reason); empty when they match."""
        actual = set(self.skipped)
        differences = [
            f"skipped but not expected: {name} ({self.skipped[name]})"
            for name in sorted(actual - set(expected))
        ]
        differences += [f"expected to be skipped but ran: {name}" for name in sorted(set(expected) - actual)]
        return differences

    def summary(self) -> str:
        """A readable report: counts, run errors, failures with their detail, skips with their reason."""
        lines = [
            f"Conformance report {self.target or '(unnamed)'}: {len(self.passed)} passed,"
            f" {len(self.failed)} failed, {len(self.skipped)} skipped",
            f"  capabilities reported: {', '.join(sorted(self.capabilities)) or '-'};"
            f" expected: {', '.join(sorted(self.expected)) or '-'}",
        ]
        lines += [f"  ERROR {error}" for error in self.errors]
        for outcome in self.outcomes:
            if outcome.status == "failed":
                lines.append(f"  FAILED {outcome.name}: {outcome.detail}")
        for outcome in self.outcomes:
            if outcome.status == "skipped":
                mark = " (security case)" if outcome.security else ""
                lines.append(f"  skipped {outcome.name}{mark}: {outcome.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The report as JSON-compatible data (e.g. a CI artifact)."""
        return {
            "target": self.target,
            "ok": self.ok,
            "capabilities": sorted(self.capabilities),
            "expected": sorted(self.expected),
            "errors": list(self.errors),
            "outcomes": [
                {
                    "name": o.name,
                    "status": o.status,
                    "detail": o.detail,
                    "security": o.security,
                    "seconds": round(o.seconds, 3),
                }
                for o in self.outcomes
            ],
        }

    def check(self, expected_skips: Mapping[str, str] | None = None) -> None:
        """Raise ``AssertionError`` with the summary if the run is not ok, or if the skipped cases differ from
        ``expected_skips`` (when given). A thin adapter for any test runner."""
        problems = [] if expected_skips is None else self.skip_differences(expected_skips)
        if not self.ok or problems:
            raise AssertionError("\n".join([self.summary(), *(f"  SKIPS {p}" for p in problems)]))


def _probe_model() -> EmbeddingModel:
    from ._models import KeywordEmbeddings

    return KeywordEmbeddings()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]


def run_cases(
    factory: Factory,
    cases: Sequence[Case],
    *,
    expect: Iterable[str],
    target: str = "",
    workdir: str | os.PathLike[str] | None = None,
) -> Report:
    """Run ``cases`` against the backend that ``factory`` opens, and report.

    First a probe instance is opened (partition :data:`DEFAULT_PARTITION`, with an embedding model when there
    are index cases) and its ``capabilities`` are read. The run fails (``Report.errors``) if they lack one of
    ``expect``, if ``expect`` names an unknown capability, if the backend's constructor takes ``tenant_id``
    but it does not report ``partitioned``, or if a case is for another contract than the instance implements.
    Each case then runs with a fresh working directory and scope; a case whose ``needs`` the backend does not
    report is skipped with the reason. Nothing is raised for failures: inspect the :class:`Report` (or call
    :meth:`Report.check`).

    Args:
        factory: Opens an instance for a :class:`Request`.
        cases: The cases to run (e.g. ``INDEX_CASES + REINDEX_CASES + PARTITION_INDEX_CASES``).
        expect: The capabilities the caller requires the backend to report.
        target: A name for the report.
        workdir: Where the per-case directories are created (default: a new temporary directory, removed
            afterwards).
    """
    expected = frozenset(expect)
    report = Report(target=target, capabilities=frozenset(), expected=expected)
    names = [case.name for case in cases]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        report.errors.append(f"duplicate case names: {', '.join(duplicates)}")
    unknown = expected - CAPABILITIES
    if unknown:
        report.errors.append(f"unknown expected capabilities: {', '.join(sorted(unknown))}")
    base = Path(tempfile.mkdtemp(prefix="orkexa-conformance-", dir=workdir))
    try:
        contracts = {case.contract for case in cases}
        probe_dir = base / "probe"
        probe_dir.mkdir()
        probe_request = Request(
            workdir=probe_dir,
            scope=uuid.uuid4().hex[:16],
            embeddings=_probe_model() if "index" in contracts else None,
        )
        context = Context(factory, probe_dir, probe_request.scope, frozenset())
        try:
            probe = factory(probe_request)
            context._opened.append(probe)
        except Exception as exc:
            report.errors.append(f"the factory could not open an instance: {type(exc).__name__}: {exc}")
            for case in cases:
                report.outcomes.append(Outcome(case.name, "failed", "no instance", case.security))
            return report
        try:
            reported = frozenset(getattr(probe, "capabilities", frozenset()))
            report.capabilities = reported
            missing = expected - reported
            if missing:
                report.errors.append(
                    "the backend reports fewer capabilities than expected: missing "
                    + ", ".join(sorted(missing))
                )
            try:
                parameters = inspect.signature(type(probe)).parameters
            except (TypeError, ValueError):  # pragma: no cover - a class without a signature
                parameters = {}  # type: ignore[assignment]
            if "tenant_id" in parameters and "partitioned" not in reported:
                report.errors.append(
                    f"{type(probe).__name__} takes tenant_id but does not report the capability 'partitioned'"
                )
            wrong = sorted(c for c in contracts if not isinstance(probe, _CONTRACTS[c]))
            if wrong:
                report.errors.append(
                    f"{type(probe).__name__} does not implement the contract of the {', '.join(wrong)} cases"
                )
        finally:
            context.close()
        for number, case in enumerate(cases, start=1):
            report.outcomes.append(
                _run_case(factory, case, base / f"{number:03d}-{_slug(case.name)}", reported, wrong)
            )
        return report
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _run_case(
    factory: Factory, case: Case, workdir: Path, reported: frozenset[str], wrong_contracts: Sequence[str]
) -> Outcome:
    if case.contract in wrong_contracts:
        return Outcome(case.name, "failed", f"the instance does not implement {case.contract}", case.security)
    lacking = case.needs - reported
    if lacking:
        return Outcome(case.name, "skipped", f"needs {', '.join(sorted(lacking))}", case.security)
    workdir.mkdir(parents=True)
    context = Context(factory, workdir, uuid.uuid4().hex[:16], reported)
    started = time.monotonic()
    try:
        case.run(context)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, CaseFailure):
            frames = traceback.extract_tb(exc.__traceback__)[-3:]
            detail += (
                " [" + " <- ".join(f"{Path(f.filename).name}:{f.lineno}" for f in reversed(frames)) + "]"
            )
        return Outcome(case.name, "failed", detail, case.security, time.monotonic() - started)
    finally:
        context.close()
    return Outcome(case.name, "passed", "", case.security, time.monotonic() - started)


def load_expected_skips(path: str | os.PathLike[str], job: str, target: str) -> dict[str, str]:
    """The expected skipped cases of ``target`` in CI job ``job`` from a JSON expectation file.

    The file maps ``"jobs"`` to an object per job: either ``{target: {case name: reason}}`` or the name of
    another job whose expectations it shares. Every reason must be a non-empty string. A job or target that is
    not listed raises ``KeyError``: a new job or backend needs an entry (an empty object when nothing skips).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    jobs: dict[str, Any] = data["jobs"]
    seen: list[str] = []
    entry: Any = jobs[job]
    while isinstance(entry, str):
        seen.append(job)
        if entry in seen:
            raise ValueError(f"{path}: job aliases form a loop: {' -> '.join([*seen, entry])}")
        job, entry = entry, jobs[entry]
    expected: dict[str, Any] = entry[target]
    for name, reason in expected.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{path}: {job}/{target}/{name} needs a reason")
    return dict(expected)
