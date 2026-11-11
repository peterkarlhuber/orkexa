"""QA-16 (AC-1): the import-linter contracts follow the edge list K-33, and a forbidden import turns them red.

Edge list K-33 of the Tabrion plan: Orkexa never imports Axentra, Nodestro or a Studio; ``orkexa[postgres]``
may use ``tabrion[postgres]`` (the PostgreSQL backend); the core may import ``tabrion`` only after MIG-ORK-3
(E-ORK-CORE).

* ``lint-imports`` on an unmodified copy of the package passes (positive control;
  ``tests/test_public_api.py`` runs it on the repository).
* Mutation probe (Tabrion plan 07 section 7): a copy with one forbidden import fails ``lint-imports`` and
  names the broken contract. Every contract has at least one probe.
* The allowed edge ``orkexa.backends.postgres -> tabrion`` stays green.
* The K-33 contract names every known platform package, and an AST scan catches any ``axentra*``,
  ``nodestro*``, ``levuvo*`` or ``*_studio`` package by prefix, which import-linter cannot express.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib

import pytest

pytest.importorskip("importlinter")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PYPROJECT = ROOT / "pyproject.toml"
FORBIDDEN_PREFIXES = ("axentra", "nodestro", "levuvo", "tabrion_studio")

NO_PLATFORM = "Orkexa depends on no Axentra, Nodestro or Studio package (K-33)"
CORE_NO_TABRION = "Core modules do not import tabrion until MIG-ORK-3 (E-ORK-CORE, K-33)"
CORE_NO_POSTGRES = "Core modules do not import orkexa.backends.postgres"
CORE_NO_TESTING = "Core modules do not import orkexa.testing"
CONTRACT_NO_BACKENDS = "Contract modules do not import orkexa.backends at runtime"
SQLCHECK_LEAF = "orkexa.testing.sqlcheck imports no other orkexa module"


def _lint_imports(cwd: Path, src: Path) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONPATH"] = str(src)
    # The console script's entry point, called through the running interpreter: ``python -m importlinter.cli``
    # does nothing (the module has no ``__main__`` block), and the script's location differs between POSIX and
    # Windows.
    entry = "import sys; from importlinter.cli import lint_imports_command; sys.exit(lint_imports_command())"
    return subprocess.run(
        [sys.executable, "-c", entry, "--config", str(cwd / "pyproject.toml"), "--no-cache"],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def _copy(tmp_path: Path) -> Path:
    shutil.copytree(SRC / "orkexa", tmp_path / "src" / "orkexa", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(PYPROJECT, tmp_path / "pyproject.toml")
    return tmp_path


def _inject(path: Path, line: str) -> None:
    text = path.read_text(encoding="utf-8")
    anchor = "from __future__ import annotations\n"
    assert anchor in text, path
    path.write_text(text.replace(anchor, anchor + "\n" + line + "\n", 1), encoding="utf-8")


def _report(result: subprocess.CompletedProcess[str]) -> str:
    # lint-imports wraps a long contract name over two lines, so compare the report with its whitespace
    # folded.
    return " ".join(result.stdout.split())


def test_unmodified_copy_is_kept(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    result = _lint_imports(root, root / "src")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 kept, 0 broken" in _report(result), "lint-imports produced no report"


@pytest.mark.parametrize(
    ("module", "line", "contract"),
    [
        ("agent.py", "import axentra", NO_PLATFORM),
        ("backends/postgres/connection.py", "import axentra.data", NO_PLATFORM),
        ("core/runtime.py", "import nodestro_runtime", NO_PLATFORM),
        ("testing/sqlcheck.py", "import nodestro", NO_PLATFORM),
        ("providers/__init__.py", "import nodestro_api", NO_PLATFORM),
        ("flow/__init__.py", "import axentra_studio", NO_PLATFORM),
        ("tools.py", "import tabrion_studio", NO_PLATFORM),
        ("_storage.py", "import tabrion", CORE_NO_TABRION),
        ("backends/sqlite/index.py", "import tabrion.sqlite", CORE_NO_TABRION),
        ("__init__.py", "import tabrion.errors", CORE_NO_TABRION),
        ("testing/conformance/_core.py", "import tabrion.testing", CORE_NO_TABRION),
        ("core/types.py", "import orkexa.backends.postgres.connection", CORE_NO_POSTGRES),
        ("usage.py", "import orkexa.testing.fakes", CORE_NO_TESTING),
        ("embeddings.py", "import orkexa.backends.sqlite", CONTRACT_NO_BACKENDS),
        ("testing/sqlcheck.py", "import orkexa.errors", SQLCHECK_LEAF),
    ],
)
def test_mutation_probe_turns_red(tmp_path: Path, module: str, line: str, contract: str) -> None:
    root = _copy(tmp_path)
    _inject(root / "src" / "orkexa" / module, line)
    result = _lint_imports(root, root / "src")
    assert result.returncode != 0, f"{line!r} in {module} went unnoticed:\n{result.stdout}"
    assert f"{contract} BROKEN" in _report(result), result.stdout


@pytest.mark.parametrize(
    ("module", "line"),
    [
        ("backends/postgres/connection.py", "import tabrion.postgres"),
        ("backends/postgres/__init__.py", "import tabrion"),
    ],
)
def test_the_postgres_backend_may_import_tabrion(tmp_path: Path, module: str, line: str) -> None:
    """K-33: orkexa[postgres] -> tabrion[postgres] is an allowed edge (MIG-ORK-2)."""
    root = _copy(tmp_path)
    _inject(root / "src" / "orkexa" / module, line)
    result = _lint_imports(root, root / "src")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 kept, 0 broken" in _report(result)


def _contracts() -> list[dict[str, object]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    contracts: list[dict[str, object]] = data["tool"]["importlinter"]["contracts"]
    return contracts


def test_platform_contract_names_every_known_package() -> None:
    (contract,) = [c for c in _contracts() if c["name"] == NO_PLATFORM]
    forbidden = set(contract["forbidden_modules"])  # type: ignore[call-overload]
    assert contract["source_modules"] == ["orkexa"]
    assert {
        "axentra",
        "axentra_studio",
        "nodestro",
        "nodestro_runtime",
        "nodestro_api",
        "nodestro_studio",
        "nodestro_examples",
        "nodestro_integrations",
        "nodestro_sdk",
        "nodestro_client",
        "tabrion_studio",
    } <= forbidden
    assert "tabrion" not in forbidden


def test_core_tabrion_contract_covers_the_same_modules_as_the_postgres_contract() -> None:
    """The tabrion core is the postgres contract's core plus orkexa.testing (plan 05 4.2, review C-1)."""
    by_name = {c["name"]: c for c in _contracts()}
    testing = ["orkexa.testing", "orkexa.testing.**"]
    postgres_core = by_name[CORE_NO_POSTGRES]["source_modules"]
    assert isinstance(postgres_core, list)
    assert by_name[CORE_NO_TABRION]["source_modules"] == [*postgres_core, *testing]
    assert by_name[CORE_NO_TABRION]["forbidden_modules"] == ["tabrion"]


def _imported_top_levels(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.partition(".")[0])
    return names


def _is_forbidden(name: str) -> bool:
    return name.startswith(FORBIDDEN_PREFIXES) or name.endswith("_studio")


def test_no_platform_import_by_prefix() -> None:
    """Covers any future Axentra, Nodestro, Levuvo or Studio package, also in ``TYPE_CHECKING`` blocks."""
    offenders = []
    for path in sorted((SRC / "orkexa").rglob("*.py")):
        for name in _imported_top_levels(path):
            if _is_forbidden(name):
                offenders.append(f"{path.relative_to(SRC)}: {name}")
    assert not offenders


def test_prefix_scan_detects_a_probe(tmp_path: Path) -> None:
    probe = tmp_path / "probe.py"
    source = "from nodestro_server import x\nimport orkexa_studio.y\nimport tabrion\n"
    probe.write_text(source, encoding="utf-8")
    names = _imported_top_levels(probe)
    assert names == {"nodestro_server", "orkexa_studio", "tabrion"}
    assert {name for name in names if _is_forbidden(name)} == {"nodestro_server", "orkexa_studio"}
