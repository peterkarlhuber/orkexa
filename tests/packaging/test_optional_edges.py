"""AC-3 (TABRION QA-18): one blocked-module test per optional edge of Orkexa (edge list K-33, document 13).

Orkexa's edges of the Tabrion plan (document 13, section 2.2; document 07, AC-3):

* ``tabrion``: the core imports without Tabrion until MIG-ORK-3 (E-ORK-CORE); today only the ``dev``
  extra names Tabrion (MIG-ORK-1, the test support), and this test keeps the core so until that step changes
  the edge on purpose;
* ``postgres``: ``orkexa[postgres]`` brings psycopg and ``tabrion[postgres]`` (MIG-ORK-2); the core and
  ``orkexa.backends.postgres`` import without them, and the backend names the extra only on use.

Each edge runs once in a fresh ``python -I`` with its modules blocked (``sys.modules[name] = None``). The
probe adds a top-level import of the blocked module to a copy of the package: the same check must fail. The
meta test keeps the edge list equal to K-33's Orkexa edges and to ``pyproject.toml``. The other extras
(provider SDKs, pdf, vector, sqlcheck, ...) are no Tabrion edge;
``tests/packaging/test_core_without_extras.py`` covers the core suite without them (MUP-ONA T-PKG-1/2).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

CORE = (
    "orkexa",
    "orkexa.retrieval",
    "orkexa.files",
    "orkexa.memory",
    "orkexa.embeddings",
    "orkexa.tools",
    "orkexa.agent",
    "orkexa.async_agent",
    "orkexa.flow",
    "orkexa.core",
    "orkexa.orchestration",
    "orkexa.content",
    "orkexa.evaluation",
    "orkexa.usage",
    "orkexa.errors",
    "orkexa.local",
    "orkexa.providers",
    "orkexa.dynamic",
    "orkexa.backends",
    "orkexa.backends.sqlite",
)
"""Orkexa's core: what Nodestro's desktop profile imports (no extra needed)."""


@dataclass(frozen=True)
class Edge:
    blocked: tuple[str, ...]
    """Modules made unimportable (``sys.modules[name] = None``)."""
    imports: tuple[str, ...]
    """Modules that import without the edge."""
    refuses: tuple[str, ...]
    """Statements that must raise ``ImportError`` carrying :attr:`hint` without the edge."""
    hint: str
    probe: tuple[str, str]
    """``(file below src/orkexa, line)`` the mutation probe adds to a copy of the package."""


EDGES: dict[str, Edge] = {
    "tabrion": Edge(
        blocked=("tabrion",),
        imports=(*CORE, "orkexa.backends.postgres"),
        refuses=(),
        hint="",
        probe=("backends/sqlite/__init__.py", "import tabrion"),
    ),
    "postgres": Edge(
        blocked=("psycopg", "psycopg_binary", "psycopg_c", "psycopg_pool", "tabrion.postgres"),
        imports=(*CORE, "orkexa.backends.postgres"),
        refuses=("from orkexa.backends.postgres import PsycopgSource; PsycopgSource('host=db')",),
        hint='pip install "orkexa[postgres]"',
        probe=("backends/postgres/__init__.py", "import psycopg"),
    ),
}
"""K-33 edge -> its blocked-module check (exactly one test each)."""

K33_EDGES = frozenset({"tabrion", "postgres"})
"""Orkexa's optional edges in document 13, section 2.2 (``orkexa → tabrion`` is optional until E-ORK-CORE)."""


def check_source(edge: Edge) -> str:
    return (
        "import importlib, sys\n"
        f"for _n in {edge.blocked!r}:\n"
        "    sys.modules[_n] = None\n"
        f"for _m in {edge.imports!r}:\n"
        "    importlib.import_module(_m)\n"
        f"for _s in {edge.refuses!r}:\n"
        "    try:\n"
        "        exec(_s, {})\n"
        "    except ImportError as _e:\n"
        f"        assert {edge.hint!r} in str(_e), (_s, str(_e))\n"
        "    else:\n"
        "        raise AssertionError(_s + ' worked without the edge')\n"
        f"_tops = {tuple(sorted({b.partition('.')[0] for b in edge.blocked if '.' not in b}))!r}\n"
        "_loaded = sorted(n for n in sys.modules if n.partition('.')[0] in _tops and sys.modules[n])\n"
        "assert not _loaded, _loaded\n"
        "print('edge ok')\n"
    )


def run_python(code: str, path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """``python -I -c code`` (no PYTHONPATH, no cwd on sys.path); ``path`` goes first on ``sys.path``."""
    if path is not None:
        code = f"import sys\nsys.path.insert(0, {str(path)!r})\n" + code
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    return subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, env=env, timeout=300, check=False
    )


@pytest.mark.parametrize("edge", sorted(EDGES))
def test_the_edge_imports_without_its_modules(edge: str) -> None:
    result = run_python(check_source(EDGES[edge]))
    assert result.returncode == 0, result.stderr
    assert "edge ok" in result.stdout


def test_every_k33_edge_has_exactly_one_test() -> None:
    assert set(EDGES) == K33_EDGES
    project = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["project"]
    names = {re.split(r"[\[<>=~!; ]", r, maxsplit=1)[0].lower() for r in project["dependencies"]}
    assert "tabrion" not in names, (
        "orkexa -> tabrion became mandatory: MIG-ORK-3 must change the tabrion edge"
    )
    postgres = {
        re.split(r"[\[<>=~!; ]", r, maxsplit=1)[0].lower()
        for r in project["optional-dependencies"]["postgres"]
    }
    blocked = set(EDGES["postgres"].blocked)
    assert all(n in blocked or f"{n}.postgres" in blocked for n in postgres), postgres
    for edge in EDGES.values():
        assert (SRC / "orkexa" / edge.probe[0]).is_file()
        assert not edge.refuses or edge.hint


def _copy_with(tmp_path: Path, file: str, line: str) -> Path:
    target = tmp_path / "src"
    shutil.copytree(SRC / "orkexa", target / "orkexa", ignore=shutil.ignore_patterns("__pycache__"))
    path = target / "orkexa" / file
    text = path.read_text(encoding="utf-8")
    anchor = "from __future__ import annotations\n"
    assert anchor in text, file
    path.write_text(text.replace(anchor, anchor + "\n" + line + "\n", 1), encoding="utf-8")
    return target


@pytest.mark.parametrize("edge", sorted(EDGES))
def test_a_top_level_import_turns_the_edge_red(tmp_path: Path, edge: str) -> None:
    """Mutation probe: a copy with the edge's module imported at the top of a core module fails the check."""
    spec = EDGES[edge]
    control = run_python(check_source(spec), _copy_with(tmp_path / "control", spec.probe[0], ""))
    assert control.returncode == 0, control.stderr  # the copy itself passes, and is the one imported
    result = run_python(check_source(spec), _copy_with(tmp_path / "probe", *spec.probe))
    assert result.returncode != 0, f"the probe {spec.probe} went unnoticed"
    assert "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr
