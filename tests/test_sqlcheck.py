"""The SQL rule engine :mod:`orkexa.testing.sqlcheck` (MUP-ONA B5: T-SQLC-1, T-SQLCHECK, T-PKG-1).

The engine was taken over without a change of behavior from nodestro-runtime's ``sqlcheck_rules`` (2ef5b3b,
MUP-ONA B4); the engine tests of ``tests/test_sqlcheck_rules.py`` there are carried over below with every
case. T-SQLC-1 proves that it is generic: with a foreign partition column (``org_id``) and foreign tables
(``t_docs``, ``t_notes``) the same rules hold, and Nodestro's column and tables mean nothing to it. T-SQLCHECK
runs Nodestro's 112 mandatory cases and 37 controls (``tests/fixtures/sqlcheck/tenant_filter_cases.json``)
against :func:`check_statement`.

The engine imports nothing but ``pglast`` (on first use) and the standard library and holds no Nodestro
constant. Tests that parse SQL need pglast (``orkexa[sqlcheck]``, part of ``dev``); without it they skip, and
the CI skip expectations (``tests/ci``) list no such skip, so a CI job without pglast fails.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pytest

from orkexa.testing import sqlcheck
from orkexa.testing.sqlcheck import KINDS, Finding, check_statement

ENGINE = Path(sqlcheck.__file__)
FIXTURES = Path(__file__).parent / "fixtures" / "sqlcheck"
BASELINE = FIXTURES / "sql_parity_before.json"
CASES = FIXTURES / "tenant_filter_cases.json"

#: The foreign set of T-SQLC-1.
COLUMN = "org_id"
TABLES = frozenset({"t_docs", "t_notes"})

needs_pglast = pytest.mark.skipif(
    importlib.util.find_spec("pglast") is None, reason="needs pglast (orkexa[sqlcheck])"
)


def _kinds(sql: str, **options: object) -> list[str]:
    options = {"tenant_tables": TABLES, "column": COLUMN, **options}
    return [f.kind for f in check_statement(sql, **options)]  # type: ignore[arg-type]


# --- T-SQLC-1: a foreign column name and table set ----------------------------------------------------------

GENERIC_BOUND = [
    "SELECT body FROM t_docs WHERE org_id = %s AND id = %s",
    "SELECT d.body FROM t_docs d WHERE d.org_id = %s::text",
    "SELECT d.body, n.text FROM t_docs d JOIN t_notes n ON n.doc = d.id AND n.org_id = d.org_id"
    " WHERE d.org_id = %s",
    "SELECT body FROM t_docs WHERE org_id = %s AND id IN (SELECT doc FROM t_notes WHERE org_id = %s)",
    "WITH x AS (SELECT id FROM t_docs WHERE org_id = %s) SELECT * FROM x",
    "UPDATE t_docs SET body = %s WHERE org_id = %s AND id = %s",
    "DELETE FROM t_notes WHERE org_id = %(org)s AND doc = %(doc)s",
    "INSERT INTO t_docs (org_id, id, body) VALUES (%s, %s, %s)",
    "INSERT INTO t_docs (org_id, id, body) VALUES (%s, %s, %s) ON CONFLICT (org_id, id) DO UPDATE SET body = "
    "excluded.body",
    "INSERT INTO t_notes (org_id, doc) SELECT d.org_id, d.id FROM t_docs d WHERE d.org_id = %s",
    # Nodestro's tables and column are nothing to the engine with this set.
    "SELECT data FROM nodestro_documents WHERE collection = %s",
    "SELECT body FROM other WHERE tenant_id = %s",
    "CREATE INDEX t_docs_org ON t_docs (org_id)",
    "ALTER TABLE t_docs ADD COLUMN extra text",
]

GENERIC_LEAKING = [
    "SELECT body FROM t_docs WHERE id = %s",
    "SELECT body FROM t_docs WHERE org_id = %s OR id = %s",
    "SELECT body FROM t_docs WHERE org_id = 'acme'",
    # Nodestro's column binds nothing here.
    "SELECT body FROM t_docs WHERE tenant_id = %s",
    "SELECT d.body FROM t_docs d JOIN t_notes n ON n.doc = d.id WHERE d.org_id = %s",
    "SELECT d.body FROM t_docs d LEFT JOIN t_notes n ON n.doc = d.id AND d.org_id = %s",
    "SELECT body FROM t_docs WHERE org_id = %s UNION SELECT text FROM t_notes",
    "SELECT body FROM t_docs WHERE org_id = %s AND id IN (SELECT doc FROM t_notes)",
    "UPDATE t_docs SET org_id = %s WHERE org_id = %s AND id = %s",
    "UPDATE t_docs SET body = %s",
    "DELETE FROM t_notes",
    "INSERT INTO t_docs (id, body) VALUES (%s, %s)",
    "INSERT INTO t_docs (org_id, id) VALUES ('acme', %s)",
    "INSERT INTO t_docs (org_id, id) VALUES (%s, %s) ON CONFLICT (id) DO UPDATE SET body = excluded.body",
    "INSERT INTO t_docs (org_id, id) VALUES (%s, %s) ON CONFLICT (org_id, id) DO UPDATE SET org_id = %s",
    "WITH t_docs AS (SELECT 1 AS id) SELECT * FROM t_docs",
    # Table names in any case are the same table.
    "SELECT body FROM T_Docs WHERE id = %s",
]


# MUP-ONA T-SQLC-1
@needs_pglast
@pytest.mark.parametrize("sql", GENERIC_BOUND)
def test_sqlc1_a_foreign_column_and_table_set_binds(sql: str) -> None:
    assert check_statement(sql, tenant_tables=TABLES, column=COLUMN) == []


# MUP-ONA T-SQLC-1
@needs_pglast
@pytest.mark.parametrize("sql", GENERIC_LEAKING)
def test_sqlc1_a_foreign_column_and_table_set_leaks(sql: str) -> None:
    findings = check_statement(sql, tenant_tables=TABLES, column=COLUMN)
    assert findings and {f.kind for f in findings} == {"leak"}


# MUP-ONA T-SQLC-1
@needs_pglast
def test_sqlc1_messages_name_the_column_and_table() -> None:
    [finding] = check_statement(
        "SELECT body FROM t_docs d WHERE id = %s", tenant_tables=TABLES, column=COLUMN
    )
    assert (
        finding.message
        == "t_docs (as d) is not bound to one tenant by a top-level `org_id = <param>` conjunct"
    )
    [insert] = check_statement("INSERT INTO t_docs (id) VALUES (%s)", tenant_tables=TABLES, column=COLUMN)
    assert insert.message == "the INSERT into t_docs lists no org_id column"


# MUP-ONA T-SQLC-1
@needs_pglast
def test_sqlc1_tenant_tables_are_matched_in_any_case() -> None:
    assert _kinds("SELECT body FROM t_docs WHERE id = %s", tenant_tables=["T_DOCS"]) == ["leak"]
    assert _kinds("SELECT body FROM t_docs WHERE org_id = %s", tenant_tables=("T_Docs",)) == []


# MUP-ONA T-SQLC-1
@needs_pglast
def test_sqlc1_the_real_statements_hold_with_another_column_and_prefix() -> None:
    """Every statement of Nodestro's parity baseline (postgres_tables.py before B4), with the tables renamed
    to ``t_`` and ``tenant_id`` to ``org_id``: bound with the foreign set, and a leak once its ``org_id = %s``
    conjuncts are gone."""
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    statements = [entry["text"] for entry in baseline["statements"]]
    tables = frozenset(table.replace("nodestro_", "t_") for table in baseline["tenant_tables"])
    assert len(statements) > 40
    mutated = 0
    for text in statements:
        sql = text.replace("nodestro_", "t_").replace("tenant_id", COLUMN)
        assert check_statement(sql, tenant_tables=tables, column=COLUMN) == [], sql
        # With Nodestro's own column the renamed statement binds nothing.
        assert check_statement(sql, tenant_tables=tables) != [], sql
        unbound = re.sub(r"\b(?:\w+\.)?org_id = %s", "true", sql)
        if unbound != sql:
            mutated += 1
            assert "leak" in [
                f.kind for f in check_statement(unbound, tenant_tables=tables, column=COLUMN)
            ], unbound
    assert mutated > 30


# --- ANY ----------------------------------------------------------------------------------------------------


@needs_pglast
def test_any_binds_only_with_allow_any() -> None:
    sql = "SELECT body FROM t_docs WHERE org_id = ANY(%s)"
    findings = check_statement(sql, tenant_tables=TABLES, column=COLUMN)
    assert [f.kind for f in findings] == ["leak", "leak"]
    assert findings[1].message == sqlcheck.ANY_REFUSED
    assert check_statement(sql, tenant_tables=TABLES, column=COLUMN, allow_any=True) == []
    # ANY on another column never binds.
    assert _kinds("SELECT body FROM t_docs WHERE id = ANY(%s)", allow_any=True) == ["leak"]
    # ANY of a constant array is no parameter: not even the reason for ANY is given.
    assert _kinds("SELECT body FROM t_docs WHERE org_id = ANY('{a,b}')") == ["leak"]


# --- kinds, positions, placeholders -------------------------------------------------------------------------


@needs_pglast
def test_a_text_the_parser_rejects_is_an_unparsed_finding_not_an_error() -> None:
    [finding] = check_statement("SELECT body FROM WHERE org_id = %s", tenant_tables=TABLES, column=COLUMN)
    assert finding.kind == "unparsed" and "syntax error" in finding.message
    assert finding.position == len("SELECT body FROM ")


@needs_pglast
@pytest.mark.parametrize(
    "sql",
    [
        "TRUNCATE t_docs",
        "COPY t_docs TO STDOUT",
        "MERGE INTO t_docs d USING src s ON d.id = s.id WHEN MATCHED THEN DELETE",
        "DO $$ BEGIN DELETE FROM t_docs; END $$",
        "SELECT query_to_xml('SELECT * FROM t_docs', true, true, '')",
    ],
)
def test_statements_the_rules_cannot_decide_are_unsupported(sql: str) -> None:
    assert "unsupported" in _kinds(sql)


@needs_pglast
def test_a_string_that_is_only_a_table_name_is_not_sql() -> None:
    assert _kinds("SELECT relname FROM pg_class WHERE relname = 't_docs'") == []


@needs_pglast
def test_every_finding_has_a_known_kind_and_a_position_in_the_text_as_given() -> None:
    sql = "SELECT 1 FROM t_notes WHERE org_id = %(org)s; SELECT body FROM t_docs WHERE id = %(id)s"
    [finding] = check_statement(sql, tenant_tables=TABLES, column=COLUMN)
    assert isinstance(finding, Finding) and finding.kind in KINDS
    # Named placeholders are longer than $n: the position is still one of the text as given.
    assert sql[finding.position :].startswith("t_docs")
    assert str(finding).startswith(f"[leak] at {finding.position}: t_docs is not bound")


@needs_pglast
def test_postgres_placeholders() -> None:
    assert _kinds("SELECT body FROM t_docs WHERE org_id = $1", placeholder="postgres") == []
    assert _kinds("SELECT body FROM t_docs WHERE org_id = $1 OR true", placeholder="postgres") == ["leak"]
    # psycopg's %s is no PostgreSQL syntax.
    assert _kinds("SELECT body FROM t_docs WHERE org_id = %s", placeholder="postgres") == ["unparsed"]
    # With psycopg placeholders, %% is a literal percent sign.
    assert _kinds("SELECT body FROM t_docs WHERE org_id = %s AND body LIKE 'a%%'") == []


def test_to_postgres_params() -> None:
    assert sqlcheck.to_postgres_params("a = %s AND b = %(x)s AND c = %(x)s AND d LIKE '5%%' %b %t") == (
        "a = $1 AND b = $2 AND c = $2 AND d LIKE '5%' $3 $4"
    )


@needs_pglast
@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("placeholder", "qmark"),
        ("column", ""),
        ("column", "Org_Id"),
        ("column", "org id"),
        ("column", '"org"'),
    ],
)
def test_invalid_options_are_refused(option: str, value: str) -> None:
    with pytest.raises(ValueError, match=option):
        check_statement("SELECT 1", tenant_tables=TABLES, **{option: value})  # type: ignore[arg-type]


def test_the_public_signature() -> None:
    """MUP-ONA B5 6(f): ``check_statement(sql, *, tenant_tables, column, placeholder, allow_any)``."""
    parameters = inspect.signature(check_statement).parameters
    assert [(p.name, p.kind, p.default) for p in parameters.values()] == [
        ("sql", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("tenant_tables", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("column", inspect.Parameter.KEYWORD_ONLY, "tenant_id"),
        ("placeholder", inspect.Parameter.KEYWORD_ONLY, "psycopg"),
        ("allow_any", inspect.Parameter.KEYWORD_ONLY, False),
    ]
    assert [field for field in Finding.__dataclass_fields__] == ["kind", "position", "message"]
    assert {"leak", "unparsed", "unsupported"} == KINDS


# --- T-SQLCHECK: Nodestro's checker cases as data -----------------------------------------------------------


def _cases() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(CASES.read_text(encoding="utf-8"))
    return data


def _statements() -> list[tuple[str, dict[str, Any]]]:
    return [
        (f"{case['id']}#{number}", statement)
        for case in _cases()["cases"]
        for number, statement in enumerate(case["statements"])
    ]


@pytest.mark.parametrize("name", ["sql_parity_before.json", "tenant_filter_cases.json"])
def test_the_fixtures_are_unchanged(name: str) -> None:
    recorded = (FIXTURES / f"{name}.sha256").read_text(encoding="utf-8").split()
    assert recorded[1] == name
    assert recorded[0] == hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest()


# MUP-ONA T-SQLCHECK
def test_the_cases_are_nodestros_112_mandatory_cases_and_37_controls() -> None:
    data = _cases()
    assert data["origin"] == {
        "repository": "nodestro-runtime",
        "commit": "2ef5b3b85181d9b31f681e6ae79e24f1f88c6416",
        "file": "tests/test_tenant_filters.py",
    }
    groups = [case["group"] for case in data["cases"]]
    assert (groups.count("mandatory"), groups.count("control"), groups.count("any")) == (112, 37, 4)
    assert len({case["id"] for case in data["cases"]}) == len(groups)
    for case in data["cases"]:
        if case["group"] == "mandatory":
            # Nodestro's check catches every one: through the engine, or (an unknown part) as dynamic.
            assert case["checker"], case["id"]
            if "expected" in case:  # a source probe names the kind Nodestro's test expects
                assert case["expected"] in case["checker"], case["id"]
        else:
            assert case["group"] == "any" or case["checker"] == [] or case["id"] == "BOUND[3]", case["id"]


# MUP-ONA T-SQLCHECK
@needs_pglast
@pytest.mark.parametrize(("name", "statement"), _statements(), ids=[name for name, _ in _statements()])
def test_the_engine_gives_nodestros_findings(name: str, statement: dict[str, Any]) -> None:
    """Every statement Nodestro's check gave its engine yields the very same findings here (kind, position,
    message), with and without ``allow_any``: no change of behavior."""
    tables = _cases()["tenant_tables"]
    for key, allow_any in (("findings", False), ("findings_allow_any", True)):
        findings = check_statement(statement["text"], tenant_tables=tables, allow_any=allow_any)
        assert [[f.kind, f.position, f.message] for f in findings] == statement[key], (name, allow_any)


# MUP-ONA T-SQLCHECK
@needs_pglast
def test_the_engine_catches_every_mandatory_case_it_is_given() -> None:
    """A mandatory case is caught by the engine unless the check could not rebuild its statement (an unknown
    part: Nodestro reports it as ``dynamic`` before the engine runs); a control passes."""
    data = _cases()
    tables = data["tenant_tables"]
    by_engine = by_reconstruction = 0
    for case in data["cases"]:
        results = [check_statement(s["text"], tenant_tables=tables) for s in case["statements"]]
        if case["group"] == "mandatory":
            if any(results):
                by_engine += 1
            else:
                assert case["dynamic"] and "dynamic" in case["checker"], case["id"]
                by_reconstruction += 1
        elif case["group"] == "control":
            for statement, findings in zip(case["statements"], results, strict=True):
                allowed = check_statement(statement["text"], tenant_tables=tables, allow_any=True)
                assert allowed == [], case["id"]
                if "ANY(" not in statement["text"]:
                    assert findings == [], case["id"]
        else:
            assert [[f.kind for f in r] for r in results] == [["leak", "leak"]], case["id"]
    assert (by_engine, by_reconstruction) == (76, 36)


# --- what the engine may depend on --------------------------------------------------------------------------


def _code(tree: ast.Module) -> list[ast.AST]:
    """Every node of ``tree`` except docstrings."""
    docstrings = {
        id(node.body[0])
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    }
    skipped: set[int] = set()
    for node in ast.walk(tree):
        if id(node) in docstrings:
            skipped.update(id(n) for n in ast.walk(node))
    return [n for n in ast.walk(tree) if id(n) not in skipped]


def test_the_engine_imports_only_pglast_and_the_standard_library() -> None:
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative imports"
            imported.add(node.module or "")
    tops = {name.split(".")[0] for name in imported}
    assert tops - {"__future__"} <= set(sys.stdlib_module_names), tops
    # pglast is imported late and by name only (importlib).
    names = {n.value for n in _code(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"pglast", "parser", "ast", "enums"} <= names


def test_the_engine_holds_no_nodestro_constant() -> None:
    for node in _code(ast.parse(ENGINE.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert "nodestro" not in node.value.lower(), node.value
        if isinstance(node, ast.Name):
            assert "nodestro" not in node.id.lower(), node.id


def test_importing_the_engine_does_not_import_the_parser() -> None:
    code = "import sys, orkexa.testing.sqlcheck; print('pglast' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "False"


# MUP-ONA T-PKG-1 (sqlcheck part)
def test_without_pglast_orkexa_imports_and_the_engine_names_the_extra_on_first_use() -> None:
    """With pglast blocked, ``import orkexa`` and the engine module work; the first check raises ImportError
    naming ``orkexa[sqlcheck]``."""
    code = (
        "import sys\n"
        "sys.modules['pglast'] = None\n"
        "import orkexa, orkexa.backends, orkexa.testing.sqlcheck as s\n"
        "try:\n"
        "    s.check_statement('SELECT 1', tenant_tables=['t'])\n"
        "except ImportError as exc:\n"
        "    print(type(exc).__name__, exc.name, exc, sep='|')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == f"ImportError|pglast|{sqlcheck.MISSING_PGLAST}"
    assert "orkexa[sqlcheck]" in sqlcheck.MISSING_PGLAST
