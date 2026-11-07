"""The SQLite backends after their move to ``orkexa.backends.sqlite`` (MUP-ONA MP-B2): import order, typing.

The contract modules re-export the classes lazily (PEP 562 ``__getattr__``) because the backends import them.
These tests prove that any module can be imported first (T-API-2), that the re-export stays lazy, and that the
old import paths still pass ``mypy --strict`` (T-API-3, ``mypy_probe.py``).
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src" / "orkexa"

FIRST_IMPORTS = [
    "orkexa",
    "orkexa.retrieval",
    "orkexa.files",
    "orkexa.memory",
    "orkexa.embeddings",
    "orkexa.backends.sqlite",
    "orkexa.backends.sqlite.index",
    "orkexa.backends.sqlite.files",
    "orkexa.backends.sqlite.memory",
]

CHECK_IDENTITY = """
import orkexa, orkexa.backends.sqlite as sqlite, orkexa.files, orkexa.memory, orkexa.retrieval
assert orkexa.SQLiteIndex is orkexa.retrieval.SQLiteIndex is sqlite.SQLiteIndex
assert orkexa.SQLiteFileStore is orkexa.files.SQLiteFileStore is sqlite.SQLiteFileStore
assert orkexa.SQLiteMemory is orkexa.memory.SQLiteMemory is sqlite.SQLiteMemory
"""


# MUP-ONA T-API-2
@pytest.mark.parametrize("module", FIRST_IMPORTS)
def test_any_module_can_be_imported_first(module: str) -> None:
    code = f"import {module}\n{CHECK_IDENTITY}"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr


# MUP-ONA T-API-2
@pytest.mark.parametrize(
    ("statement", "name"),
    [
        ("from orkexa.retrieval import SQLiteIndex", "SQLiteIndex"),
        ("from orkexa.files import SQLiteFileStore", "SQLiteFileStore"),
        ("from orkexa.memory import SQLiteMemory", "SQLiteMemory"),
        ("from orkexa.retrieval import *", "SQLiteIndex"),
        ("from orkexa.files import *", "SQLiteFileStore"),
        ("from orkexa.memory import *", "SQLiteMemory"),
    ],
)
def test_the_old_from_imports_work_in_a_fresh_interpreter(statement: str, name: str) -> None:
    code = f"{statement}\nassert {name}.__module__.startswith('orkexa.backends.sqlite.'), {name}.__module__"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr


def _imports_backends(node: ast.AST) -> bool:
    if isinstance(node, ast.ImportFrom):
        module = "." * node.level + (node.module or "")
        return module.lstrip(".").startswith("backends") or module.startswith("orkexa.backends")
    if isinstance(node, ast.Import):
        return any(alias.name.startswith("orkexa.backends") for alias in node.names)
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return isinstance(test, ast.Name) and test.id == "TYPE_CHECKING"


# MUP-ONA T-API-2
@pytest.mark.parametrize("module", ["retrieval", "files", "memory", "embeddings", "_storage"])
def test_contract_modules_import_backends_only_lazily(module: str) -> None:
    """Complements the import-linter contract, which ignores the three re-export edges: a contract module may
    import ``orkexa.backends`` only under ``if TYPE_CHECKING:`` or inside the ``__getattr__`` of an
    ``if not TYPE_CHECKING:`` block, never at module level."""
    tree = ast.parse((SRC / f"{module}.py").read_text(encoding="utf-8"))
    allowed: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            allowed.update(id(child) for child in ast.walk(node))
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and _is_type_checking(node.test.operand)
        ):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "__getattr__":
                    allowed.update(id(grandchild) for grandchild in ast.walk(child))
    offending = [
        node.lineno for node in ast.walk(tree) if _imports_backends(node) and id(node) not in allowed
    ]
    assert offending == [], f"orkexa/{module}.py imports orkexa.backends eagerly on lines {offending}"


# MUP-ONA T-API-3
def test_the_old_import_paths_pass_mypy_strict(tmp_path: Path) -> None:
    """``mypy_probe.py`` uses every old import path and passes; a misspelled name is still an error, so the
    lazy ``__getattr__`` does not hide unknown names from type checkers."""
    if importlib.util.find_spec("mypy") is None:  # pragma: no cover - mypy is part of the dev extra
        pytest.fail("mypy is not installed (pip install -e '.[dev]')")
    probe = Path(__file__).with_name("mypy_probe.py")
    typo = tmp_path / "typo_probe.py"
    typo.write_text(
        "from orkexa.retrieval import SQLiteIndx\n"
        "from orkexa.files import SQLiteFileStor\n"
        "from orkexa.memory import SQLiteMemry\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--no-incremental",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    result = subprocess.run(
        [*command, str(probe)], cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Success: no issues found in 1 source file" in result.stdout
    result = subprocess.run(
        [*command, str(typo)], cwd=ROOT, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout.count("[attr-defined]") == 3, result.stdout
