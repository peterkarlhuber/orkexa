"""Generate ``tenant_filter_cases.json``: Nodestro's checker cases as engine data (MUP-ONA T-SQLCHECK).

nodestro-runtime's ``tests/test_tenant_filters.py`` holds the 112 mandatory cases (leaks the check must catch)
and the 37 controls (code it must pass) of its tenant SQL check. Some are SQL texts, most are Python sources
from which the check rebuilds the statements before it calls the rule engine. This script runs Nodestro's own
check on every case, exactly as that test file does, and records per case the statements the engine was given
and the engine's findings (kind, position, message; with and without ``allow_any``), plus the kinds the whole
check reported. Orkexa's ``tests/test_sqlcheck.py`` runs :func:`orkexa.testing.sqlcheck.check_statement` on
these statements and compares.

Usage (any Python with pglast; nothing is installed or imported from the source tree's package ``__init__``)::

    git -C nodestro-runtime archive <commit> src tests/test_tenant_filters.py | tar -x -C tree
    python tests/fixtures/sqlcheck/make_tenant_filter_cases.py tree <commit> [--out -]

Only the standard library and pglast are used; ``sqlcheck_rules`` and ``tenant_sql_check`` are loaded from the
tree as a package of their own.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "tenant_filter_cases.json"

#: The case lists of tests/test_tenant_filters.py: name -> (group, shape, frag). ``sql``: a list of SQL texts;
#: ``source``: a dict of Python sources (``(source, kind)`` tuples or plain sources); ``frag``: the module
#: ``frag.py`` the test writes next to the sample (``hole``: HOLE_LED's literal, ``probe``: ``_scan_probe``'s
#: ``_BASE``).
LISTS: dict[str, tuple[str, str, str | None]] = {
    "REVIEW_SAMPLES": ("mandatory", "source", None),
    "LEAKING": ("mandatory", "sql", None),
    "BARE_NAME_LEAKS": ("mandatory", "source", None),
    "HOLE_LED": ("mandatory", "source", "hole"),
    "ROUND3_PROBES": ("mandatory", "source", "probe"),
    "ROUND3_LEAKING_SQL": ("mandatory", "sql", None),
    "ROUND4_PROBES": ("mandatory", "source", "probe"),
    "BOUND": ("control", "sql", None),
    "ROUND3_CLEAN": ("control", "source", "probe"),
    "ROUND3_BOUND_SQL": ("control", "sql", None),
    "ROUND4_CLEAN": ("control", "source", "probe"),
}
#: ANY_SAMPLES (MUP-ONA A11) are not among the 112: they are recorded as a group of their own.
EXTRA = {"ANY_SAMPLES": ("any", "sql", None)}


def _load_check(tree: Path) -> ModuleType:
    """``tenant_sql_check`` of the tree, with its ``sqlcheck_rules``, as the package ``nodestro_runtime``."""
    package = tree / "src" / "nodestro_runtime"
    stub = ModuleType("nodestro_runtime")
    stub.__path__ = [str(package)]
    sys.modules["nodestro_runtime"] = stub
    for name in ("sqlcheck_rules", "tenant_sql_check"):
        spec = importlib.util.spec_from_file_location(f"nodestro_runtime.{name}", package / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(stub, name, module)
    return sys.modules["nodestro_runtime.tenant_sql_check"]


def _lists(tree: Path) -> dict[str, Any]:
    """The case lists (and ``_BASE``) of the test file, evaluated without importing it."""
    module = ast.parse((tree / "tests" / "test_tenant_filters.py").read_text(encoding="utf-8"))
    wanted = {*LISTS, *EXTRA, "_BASE"}
    namespace: dict[str, Any] = {}
    for node in module.body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else None
        if isinstance(target, ast.Name) and target.id in wanted:
            code = compile(ast.Module(body=[node], type_ignores=[]), "test_tenant_filters.py", "exec")
            exec(code, {"__builtins__": {"range": range}}, namespace)
    missing = wanted - set(namespace)
    if missing:
        raise SystemExit(f"test_tenant_filters.py lacks {sorted(missing)}")
    return namespace


def _engine(check: ModuleType, text: str, tables: frozenset[str]) -> dict[str, list[list[Any]]]:
    rules = sys.modules["nodestro_runtime.sqlcheck_rules"]
    result = {}
    for key, allow_any in (("findings", False), ("findings_allow_any", True)):
        findings = rules.check_statement(text, tenant_tables=tables, allow_any=allow_any)
        result[key] = [[f.kind, f.position, f.message] for f in findings]
    return result


def _scan(check: ModuleType, source: str, frag: str | None, base: str, tables: frozenset[str]) -> Any:
    """``_scan`` / ``_scan_probe`` of the test file, in a temporary directory."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "sample"
        root.mkdir()
        if frag == "hole":
            (root / "frag.py").write_text(
                "BASE = 'SELECT data FROM nodestro_documents WHERE tenant_id = %s'\n"
            )
        elif frag == "probe":
            (root / "frag.py").write_text(f"BASE = {base}\n", encoding="utf-8")
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "sample.py").write_text(textwrap.dedent(source), encoding="utf-8")
        return check.scan_package(root, tables, {}, {})


def build(tree: Path, commit: str) -> dict[str, Any]:
    check = _load_check(tree)
    tables = check.tenant_tables(tree / "src" / "nodestro_runtime" / "migrations" / "postgres")
    lists = _lists(tree)
    cases = []
    for name, (group, shape, frag) in {**LISTS, **EXTRA}.items():
        value = lists[name]
        items = list(enumerate(value)) if shape == "sql" else sorted(value.items())
        for key, item in items:
            case: dict[str, Any] = {"id": f"{name}[{key}]", "group": group}
            if shape == "sql":
                case["statements"] = [{"text": item, **_engine(check, item, tables)}]
                case["checker"] = sorted({kind for kind, _, _ in case["statements"][0]["findings"]})
            else:
                source, kind = item if isinstance(item, tuple) else (item, None)
                if kind is not None:
                    case["expected"] = kind
                report = _scan(check, source, frag, lists["_BASE"], tables)
                case["statements"] = [
                    {"text": s.text, **_engine(check, s.text, tables)}
                    for s in report.checked
                    if not s.dynamic
                ]
                case["dynamic"] = [s.text for s in report.checked if s.dynamic]
                case["checker"] = sorted({f.kind for f in report.findings})
            cases.append(case)
    return {
        "description": (
            "Nodestro's tenant SQL check cases (tests/test_tenant_filters.py) as the rule engine sees them "
            "(tests/fixtures/sqlcheck/make_tenant_filter_cases.py)."
        ),
        "origin": {
            "repository": "nodestro-runtime",
            "commit": commit,
            "file": "tests/test_tenant_filters.py",
        },
        "column": "tenant_id",
        "placeholder": "psycopg",
        "tenant_tables": sorted(tables),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "tree", type=Path, help="a nodestro-runtime source tree (src/nodestro_runtime, tests)"
    )
    parser.add_argument("commit", help="the nodestro-runtime commit the tree comes from")
    parser.add_argument("--out", default=str(OUT), help="output file, or - for stdout")
    args = parser.parse_args(argv)
    text = json.dumps(build(args.tree, args.commit), indent=2, ensure_ascii=False) + "\n"
    if args.out == "-":
        sys.stdout.write(text)
    else:
        Path(args.out).write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
