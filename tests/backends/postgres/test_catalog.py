"""The statement catalog of :mod:`orkexa.backends.postgres` (MUP-ONA B5: T-CAT-1..4, T-PAR-2).

The backend's SQL is its catalog, whose templates name the tables only as ``{chunks}`` ...; it is checked
here, rendered with the test prefix ``t_``:

* **T-CAT-1** every entry rendered for ``PostgresLayout("t_")`` is bound to one tenant by the rule engine
  (:func:`orkexa.testing.sqlcheck.check_statement` with the five ``t_`` tables as tenant tables); no ``leak``,
  ``unparsed`` or ``unsupported``.
* **T-CAT-2** mutation probe: for every partitioned entry, each ``tenant_id`` predicate removed, each ``AND``
  next to it turned into ``OR``, and (``INSERT ... VALUES``) the ``tenant_id`` column dropped, is a ``leak``.
  With ``ORKEXA_CATALOG_REPORTS`` set, the outcome is written to ``<dir>/mutation-probe.json`` (a CI
  artifact).
* **T-CAT-3** completeness: the catalog's names are exactly :data:`EXPECTED` (with their partition flag); a
  new entry fails until it is listed here, and every partitioned entry gets the mutation probe and the guard
  cases of ``test_guard.py`` by parametrization.
* **T-CAT-4** AST checks: outside ``_sql.py`` the stores run SQL only through the executor with a catalog
  name; ``schema.py`` and ``connection.py`` run only their own constant statements; no ``sql.Literal``;
  psycopg only in ``connection.py`` (inside functions) or for type checking.
* **T-PAR-2** the catalog rendered with ``nodestro_`` is Nodestro's catalog of nodestro-runtime 76f9cb2
  (``tests/fixtures/postgres/nodestro_catalog.json``), text for text.

The tests need psycopg (rendering) and pglast (the rule engine), both in ``dev``; they need no server.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import pytest

from orkexa.backends import postgres
from orkexa.backends.postgres import PostgresLayout, schema_steps
from orkexa.backends.postgres._sql import CATALOG, PARTITION, TABLES, Statement, render

PACKAGE = Path(postgres.__file__).parent
FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "postgres"
LAYOUT = PostgresLayout("t_")
PREFIX = LAYOUT.prefix
TENANT_TABLES = frozenset(LAYOUT.tables.values())

needs_psycopg = pytest.mark.skipif(importlib.util.find_spec("psycopg") is None, reason="needs psycopg")
needs_pglast = pytest.mark.skipif(importlib.util.find_spec("pglast") is None, reason="needs pglast")

#: T-CAT-3: every statement of the catalog and whether it is partitioned by tenant. An unpartitioned entry
#: names no table (it says why here).
EXPECTED: dict[str, bool] = {
    "files_update_entry": True,
    "files_upsert": True,
    "files_entry": True,
    "files_data": True,
    "files_entries": True,
    "files_delete": True,
    "chunks_count": True,
    "chunks_sources": True,
    "chunks_insert": True,
    "chunks_delete_source": True,
    "chunks_max_id": True,
    "chunks_missing_up_to": True,
    "chunks_missing": True,
    "chunks_count_missing": True,
    "spaces_lock_shared": True,
    "spaces_lock_exclusive": True,
    "spaces_by_state": True,
    "spaces_insert_active": True,
    "spaces_insert_building": True,
    "spaces_activate": True,
    "spaces_delete": True,
    "spaces_delete_others": True,
    "spaces_delete_all": True,
    "vectors_insert": True,
    "vectors_store": True,
    "vectors_count": True,
    "vectors_count_excluded": True,
    "vectors_pgvector": False,  # whether pgvector is installed: to_regtype('vector'), no table
    "search_fts": True,
    "search_fts_sources": True,
    "search_pgvector": True,
    "search_pgvector_sources": True,
    "search_candidates": True,
    "search_candidates_sources": True,
    "search_hits": True,
    "search_word_counts": True,
    "search_word_hits": True,
    "memory_read": True,
    "memory_write": True,
    "memory_files": True,
    "memory_delete": True,
    "memory_rename": True,
    "memory_clear": True,
}
PARTITIONED = sorted(name for name, partitioned in EXPECTED.items() if partitioned)

#: A predicate that binds a reference to the tenant (``c.tenant_id = %(tenant_id)s``).
_PREDICATE = re.compile(r"(?:\b\w+\.)?tenant_id = %\(tenant_id\)s")


def _text(template: str, prefix: str = PREFIX) -> str:
    """A template as text, with the tables of ``prefix`` (as rendering names them, without quotes)."""
    return template.format(**{key: prefix + table for key, table in TABLES.items()})


def _findings(text: str) -> list[str]:
    from orkexa.testing.sqlcheck import check_statement

    return [finding.kind for finding in check_statement(text, tenant_tables=TENANT_TABLES)]


# ---- T-CAT-1
# ------------------------------------------------------------------------------------------------


# MUP-ONA T-CAT-1
@needs_psycopg
@needs_pglast
@pytest.mark.parametrize("name", sorted(CATALOG))
def test_every_rendered_entry_is_bound_to_one_tenant(name: str) -> None:
    from orkexa.testing.sqlcheck import check_statement

    rendered = render(name, PREFIX).as_string(None)
    assert check_statement(rendered, tenant_tables=TENANT_TABLES) == []
    # Rendering only quotes the table names: the text is the template with the test tables.
    assert rendered.replace('"', "") == _text(CATALOG[name].template).replace('"', "")
    if CATALOG[name].partitioned:
        assert {LAYOUT.table(table) for table in CATALOG[name].tables} <= TENANT_TABLES
    else:
        assert not CATALOG[name].tables


@needs_psycopg
def test_the_tenant_tables_are_the_tables_of_the_schema() -> None:
    """The rule engine's tenant tables are exactly the tables :func:`schema_steps` creates, each with a
    ``tenant_id`` column (else T-CAT-1 would pass for want of a tenant table)."""
    created = {}
    for step in schema_steps(LAYOUT):
        match = re.match(r'CREATE TABLE "(\w+)" \((.*)\)$', step, re.DOTALL)
        if match:
            created[match.group(1)] = match.group(2)
    assert set(created) == set(TENANT_TABLES)
    for table, body in created.items():
        assert 'tenant_id text COLLATE "C" NOT NULL' in body, table
        assert "CHECK (tenant_id <> '')" in body, table


# ---- T-CAT-2: mutation probe
# --------------------------------------------------------------------------------


def mutations(entry: Statement) -> list[tuple[str, str]]:
    """``(what, text)`` of each mutation of a partitioned entry that must leak."""
    text = _text(entry.template)
    found: list[tuple[str, str]] = []
    for match in _PREDICATE.finditer(text):
        found.append(
            (
                f"remove {match.group(0)!r} at {match.start()}",
                text[: match.start()] + "TRUE" + text[match.end() :],
            )
        )
        after = text.find(" AND ", match.end())
        if after == match.end():
            found.append(
                (
                    f"AND->OR after {match.group(0)!r} at {match.start()}",
                    text[:after] + " OR " + text[after + 5 :],
                )
            )
        elif text[: match.start()].endswith(" AND "):
            before = match.start() - 5
            found.append(
                (
                    f"AND->OR before {match.group(0)!r} at {match.start()}",
                    text[:before] + " OR " + text[match.start() :],
                )
            )
    values = re.match(r"INSERT INTO (\S+) \(tenant_id, (.*?)\) VALUES \(%\(tenant_id\)s, ", text)
    if values:
        dropped = text.replace("(tenant_id, ", "(", 1).replace("VALUES (%(tenant_id)s, ", "VALUES (", 1)
        found.append(("drop the tenant_id column", dropped))
    return found


_PROBE: dict[str, list[dict[str, Any]]] = {}


# MUP-ONA T-CAT-2
@needs_pglast
@pytest.mark.parametrize("name", PARTITIONED)
def test_every_mutation_of_a_partitioned_entry_leaks(name: str) -> None:
    probe = mutations(CATALOG[name])
    assert probe, f"{name}: no mutation to probe"
    outcome = []
    for what, text in probe:
        kinds = _findings(text)
        outcome.append({"mutation": what, "findings": sorted(set(kinds)), "text": text})
        assert "leak" in kinds and set(kinds) <= {"leak"}, (name, what, text, kinds)
    _PROBE[name] = outcome
    _write_probe()


def _write_probe() -> None:
    """The probe's outcome so far, as ``<ORKEXA_CATALOG_REPORTS>/mutation-probe.json`` (CI uploads it)."""
    target = os.environ.get("ORKEXA_CATALOG_REPORTS")
    if not target:
        return
    path = Path(target) / "mutation-probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "test": "T-CAT-2 (tests/backends/postgres/test_catalog.py)",
        "prefix": PREFIX,
        "entries": len(_PROBE),
        "mutations": sum(len(found) for found in _PROBE.values()),
        "all_leak": all("leak" in m["findings"] for found in _PROBE.values() for m in found),
        "probe": dict(sorted(_PROBE.items())),
    }
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def test_the_probe_mutates_every_tenant_predicate() -> None:
    """Each mutation set covers every ``tenant_id`` predicate of its entry (a predicate the regex misses would
    be left out of the probe silently)."""
    for name in PARTITIONED:
        text = _text(CATALOG[name].template)
        assert text.count("tenant_id = %(tenant_id)s") == len(_PREDICATE.findall(text)), name
        removals = [what for what, _ in mutations(CATALOG[name]) if what.startswith("remove")]
        assert len(removals) == len(_PREDICATE.findall(text)), name


# ---- T-CAT-3: completeness
# ----------------------------------------------------------------------------------


# MUP-ONA T-CAT-3
def test_the_catalog_is_the_checked_in_list() -> None:
    assert {name: entry.partitioned for name, entry in CATALOG.items()} == EXPECTED
    assert set(PARTITIONED) == {name for name, entry in CATALOG.items() if entry.partitioned}
    assert postgres.CATALOG is CATALOG


def test_a_new_entry_without_its_partition_case_is_caught() -> None:
    """What T-CAT-3 compares: an entry missing from :data:`EXPECTED` makes the comparison unequal."""
    extra = Statement(
        "files_extra", "SELECT 1 FROM {files} WHERE tenant_id = %(tenant_id)s", (PARTITION,), "read"
    )
    grown = {**CATALOG, extra.name: extra}
    assert {name: entry.partitioned for name, entry in grown.items()} != EXPECTED


@pytest.mark.parametrize(
    ("template", "params", "partitioned"),
    [
        ("SELECT 1 FROM {files} WHERE tenant_id = %(tenant_id)s AND id = %(id)s", (PARTITION,), True),
        ("SELECT 1 FROM {files} WHERE tenant_id = %(tenant_id)s", (PARTITION, "id"), True),
        ("SELECT 1 FROM {files} WHERE tenant_id = %s", (), False),
        ("SELECT 1 FROM {files} WHERE id = %(id)s", ("id",), True),
        ("SELECT 1 FROM {files} WHERE tenant_id = %(tenant_id)s", (PARTITION,), False),
        ("SELECT 1 FROM {other} WHERE tenant_id = %(tenant_id)s", (PARTITION,), True),
        ("SELECT 1 FROM {files} WHERE tenant_id = %(tenant_id)s", (PARTITION, PARTITION), True),
    ],
)
def test_an_inconsistent_entry_is_refused(template: str, params: tuple[str, ...], partitioned: bool) -> None:
    with pytest.raises(ValueError):
        Statement("x_y", template, params, "read", partitioned=partitioned)


def test_every_entry_binds_its_values() -> None:
    for name, entry in CATALOG.items():
        assert entry.kind in {"read", "write", "delete"}, name
        assert entry.rows == (entry.kind == "read" or "RETURNING" in entry.template), name
        words = entry.template.split()[0].upper()
        expected = {"read": {"SELECT"}, "write": {"INSERT", "UPDATE"}, "delete": {"DELETE"}}[entry.kind]
        assert words in expected, name


# ---- T-CAT-4: no free execute, no literal, psycopg late
# -----------------------------------------------------


def _modules() -> list[Path]:
    found = sorted(PACKAGE.glob("*.py"))
    assert {path.name for path in found} == {
        "__init__.py",
        "_sql.py",
        "_vector.py",
        "connection.py",
        "files.py",
        "index.py",
        "memory.py",
        "schema.py",
    }
    return found


def _constants(tree: ast.Module) -> set[str]:
    """Module-level names bound to a string constant (or a concatenation of string constants)."""
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if all(
                isinstance(n, ast.Constant)
                for n in ast.walk(node.value)
                if not isinstance(n, ast.BinOp | ast.Add)
            ):
                found |= {t.id for t in targets if isinstance(t, ast.Name)}
    return found


# MUP-ONA T-CAT-4
def test_no_module_outside_the_catalog_runs_free_sql() -> None:
    """The stores run SQL only through their executor with a catalog name as a constant (so every statement
    that runs is one T-CAT-1/2 check), and every catalog entry is used. ``schema.py`` (DDL and catalog reads
    of ``verify_schema``/``migrate``) and ``connection.py`` (``set_config``) run only their own constant
    statements, or a constant template rendered with identifiers (``_render``); never a text built at run
    time."""
    offenders = []
    used: set[str] = set()
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _constants(tree)
        # schema._rows(connection, query, params) forwards its query; its callers are checked instead.
        forwarding = {
            id(child)
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and path.name == "schema.py" and node.name == "_rows"
            for child in ast.walk(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"executemany", "copy", "cursor", "stream"}:
                offenders.append(f"{path.name}:{node.lineno} .{node.attr}")
            if path.name == "_sql.py" or not isinstance(node, ast.Call) or id(node) in forwarding:
                continue
            function = node.func
            called = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
            if called not in {"execute", "_rows"}:
                continue
            first = node.args[0] if node.args else None
            if path.name in {"schema.py", "connection.py"}:
                if called == "_rows" and isinstance(first, ast.Name) and first.id == "connection":
                    first = node.args[1]  # schema._rows(connection, query, params)
                constant = isinstance(first, ast.Constant) and isinstance(first.value, str)
                named = isinstance(first, ast.Name) and first.id in constants
                rendered = (
                    isinstance(first, ast.Call)
                    and isinstance(first.func, ast.Name)
                    and first.func.id == "_render"
                    and (isinstance(first.args[0], ast.Name) and first.args[0].id in constants | {"step"})
                )
                if not (constant or named or rendered):
                    offenders.append(f"{path.name}:{node.lineno} {called} of a text built at run time")
                continue
            if isinstance(first, ast.Name) and (path.name, first.id, called) == (
                "index.py",
                "name",
                "execute",
            ):
                continue  # PostgresIndex._rows(name, ...), whose callers pass constants (checked here)
            if not (isinstance(first, ast.Constant) and first.value in CATALOG):
                offenders.append(f"{path.name}:{node.lineno} {called} without a catalog name")
            elif called == "execute" and ast.unparse(function) != "self._sql.execute":
                offenders.append(f"{path.name}:{node.lineno} .execute outside the executor")
            else:
                used.add(first.value)
    assert offenders == []
    assert used == set(CATALOG)


def test_the_schema_steps_loop_runs_only_the_constant_steps() -> None:
    """The one ``_render(step, ...)`` that T-CAT-4 lets through iterates the constant ``_STEPS``."""
    tree = ast.parse((PACKAGE / "schema.py").read_text(encoding="utf-8"))
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For | ast.comprehension)]
    steps = [loop for loop in loops if isinstance(loop.target, ast.Name) and loop.target.id == "step"]
    assert steps and all(ast.unparse(loop.iter) == "_STEPS" for loop in steps)
    assert "_STEPS" in _constants(tree) or any(
        isinstance(node, ast.AnnAssign) and ast.unparse(node.target) == "_STEPS" for node in tree.body
    )


def test_values_are_bound_and_psycopg_is_imported_late() -> None:
    """No ``sql.Literal`` anywhere; psycopg is imported only in ``connection.py`` inside a function, elsewhere
    only for type checking (``import orkexa.backends.postgres`` needs no psycopg)."""
    offenders = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        checking: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
                checking |= {id(child) for child in ast.walk(node)}
        functions: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions |= {id(child) for child in ast.walk(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "Literal":
                offenders.append(f"{path.name}:{node.lineno} .Literal")
            if isinstance(node, ast.Import | ast.ImportFrom):
                modules = (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                )
                if not any(module.split(".")[0] == "psycopg" for module in modules):
                    continue
                names = {alias.name for alias in node.names}
                if "Literal" in names:
                    offenders.append(f"{path.name}:{node.lineno} imports Literal")
                if id(node) in checking:
                    continue
                if path.name != "connection.py" or id(node) not in functions:
                    offenders.append(f"{path.name}:{node.lineno} imports psycopg")
    assert offenders == []


def test_the_backend_imports_only_orkexa_and_the_standard_library() -> None:
    """import-linter does not see the third-party imports; this does (psycopg is checked above)."""
    allowed = set(sys.stdlib_module_names) | {"__future__", "orkexa", "psycopg"}
    for path in _modules():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                tops = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative: within the package, or orkexa's contract modules
                    assert node.level in {1, 3}, f"{path.name}:{node.lineno}"
                    continue
                tops = {(node.module or "").split(".")[0]}
            else:
                continue
            assert tops <= allowed, f"{path.name}:{node.lineno} imports {tops - allowed}"


# ---- T-PAR-2: Orkexa's catalog = Nodestro's (B4)
# ------------------------------------------------------------


def _nodestro() -> dict[str, Any]:
    path = FIXTURES / "nodestro_catalog.json"
    recorded = (FIXTURES / "nodestro_catalog.json.sha256").read_text(encoding="utf-8").split()
    assert recorded[1] == "nodestro_catalog.json"
    assert recorded[0] == hashlib.sha256(path.read_bytes()).hexdigest()
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    assert data["origin"].startswith("nodestro-runtime 76f9cb2")
    return data


# MUP-ONA T-PAR-2
@needs_psycopg
def test_the_catalog_rendered_for_nodestro_is_nodestros_catalog() -> None:
    nodestro = _nodestro()
    assert nodestro["prefix"] == "nodestro_"
    statements = nodestro["statements"]
    assert sorted(statements) == sorted(CATALOG)
    for name, entry in CATALOG.items():
        expected = statements[name]
        assert render(name, "nodestro_").as_string(None) == expected["text"], name
        assert render(name, PostgresLayout("nodestro_").prefix).as_string(None) == expected["text"], name
        assert list(entry.params) == expected["params"], name
        assert (entry.kind, entry.partitioned, entry.rows) == (
            expected["kind"],
            expected["partitioned"],
            expected["rows"],
        ), name


# MUP-ONA T-PAR-2
@needs_pglast
def test_the_catalog_equals_nodestros_after_normalization() -> None:
    """The same comparison as PostgreSQL's grammar prints the statements back (placeholders as ``$?``), as
    MUP-ONA defines T-PAR (immune to quoting and spacing)."""
    from pglast import parse_sql
    from pglast.stream import RawStream

    from orkexa.testing.sqlcheck import to_postgres_params

    def normalized(text: str) -> str:
        printed = "; ".join(RawStream()(raw) for raw in parse_sql(to_postgres_params(text)))
        return re.sub(r"\$\d+", "$?", printed)

    statements = _nodestro()["statements"]
    for name, entry in CATALOG.items():
        assert normalized(_text(entry.template, "nodestro_")) == normalized(statements[name]["text"]), name
