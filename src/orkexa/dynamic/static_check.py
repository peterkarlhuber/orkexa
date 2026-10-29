"""Static (AST) checks of generated tool code before it is ever executed.

This is a first line of defense that rejects obviously dangerous code early
and with a clear message for the Coding Agent. It is a deny/allow-list filter,
*not* a security boundary: Python offers too many indirect paths to the
interpreter for an AST check to close them all. Generated code therefore always
runs in a sandbox as well, by default an isolated :class:`~orkexa.dynamic.DockerSandbox`.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
import keyword
import re

__all__ = [
    "ALLOWED_IMPORTS",
    "FORBIDDEN_ATTRIBUTES",
    "FORBIDDEN_MODULE_ATTRIBUTES",
    "FORBIDDEN_NAMES",
    "MAX_SOURCE_CHARS",
    "TOOL_MODULE",
    "StaticReport",
    "check_tool_source",
    "check_tool_tests",
]

#: Module name under which the generated code is importable from its tests.
TOOL_MODULE = "tool"

#: Top-level modules generated code may import: pure computation, no I/O.
ALLOWED_IMPORTS = frozenset(
    {
        "__future__",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "pydantic",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "typing_extensions",
    }
)

#: Builtins and names that give access to I/O, code execution or introspection.
FORBIDDEN_NAMES = frozenset(
    {
        "breakpoint",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "exit",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "open",
        "quit",
        "setattr",
        "type",
        "vars",
        # Escape hatches equivalent to getattr() or import-by-string:
        "ImportString",
        "attrgetter",
        "methodcaller",
    }
)

#: Attributes that expose frames, code objects or globals.
FORBIDDEN_ATTRIBUTES = FORBIDDEN_NAMES | {
    "ag_frame",
    "co_code",
    "cr_frame",
    "f_back",
    "f_builtins",
    "f_code",
    "f_globals",
    "f_locals",
    "gi_code",
    "gi_frame",
    "mro",
    "tb_frame",
    "tb_next",
}

#: Modules that allowed modules re-export as attributes (``calendar.sys``,
#: ``dataclasses.inspect``, ``typing.re`` ...): reaching them by attribute would bypass
#: the import allow list, so these attribute names are rejected everywhere.
FORBIDDEN_MODULE_ATTRIBUTES = frozenset(
    {
        "_codecs",
        "_io",
        "_locale",
        "_os",
        "_socket",
        "_sys",
        "_thread",
        "_types",
        "asyncio",
        "bltns",
        "builtins",
        "code",
        "codecs",
        "codeop",
        "copyreg",
        "ctypes",
        "fcntl",
        "gc",
        "genericpath",
        "glob",
        "http",
        "importlib",
        "inspect",
        "io",
        "linecache",
        "locale",
        "marshal",
        "mmap",
        "modules",
        "multiprocessing",
        "nt",
        "ntpath",
        "os",
        "pathlib",
        "pickle",
        "pkgutil",
        "platform",
        "posix",
        "posixpath",
        "pty",
        "resource",
        "runpy",
        "select",
        "selectors",
        "shelve",
        "shutil",
        "signal",
        "site",
        "socket",
        "sqlite3",
        "ssl",
        "subprocess",
        "sys",
        "sysconfig",
        "tempfile",
        "threading",
        "types",
        "urllib",
        "warnings",
        "weakref",
        "webbrowser",
        "zipimport",
    }
)

_DUNDER_IN_STRING = re.compile(r"__\w+__")

MAX_SOURCE_CHARS = 20_000


@dataclass
class StaticReport:
    """Result of a static check; ``errors`` is empty if the code passed."""

    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class _Checker(ast.NodeVisitor):
    def __init__(self, label: str, extra_imports: frozenset[str]) -> None:
        self.label = label
        self.allowed = ALLOWED_IMPORTS | extra_imports
        self.errors: list[str] = []

    def error(self, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", "?")
        self.errors.append(f"{self.label} line {line}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root not in self.allowed:
                self.error(node, f"import of {alias.name!r} is not allowed")
            self._check_alias(node, alias)
        self.generic_visit(node)

    def _check_alias(self, node: ast.AST, alias: ast.alias) -> None:
        for name in (alias.name.split(".")[-1], alias.asname):
            if name is not None and (name in FORBIDDEN_ATTRIBUTES or _is_dunder(name)):
                self.error(node, f"import of {name!r} is not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            self.error(node, "relative imports are not allowed")
        elif (node.module or "").split(".")[0] not in self.allowed:
            self.error(node, f"import from {node.module!r} is not allowed")
        for alias in node.names:
            if alias.name == "*":
                self.error(node, "star imports are not allowed")
            else:
                self._check_alias(node, alias)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES or _is_dunder(node.id):
            self.error(node, f"use of {node.id!r} is not allowed")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if _is_dunder(node.attr) or node.attr in FORBIDDEN_ATTRIBUTES:
            self.error(node, f"access to attribute {node.attr!r} is not allowed")
        elif node.attr in FORBIDDEN_MODULE_ATTRIBUTES:
            self.error(node, f"access to module {node.attr!r} through an attribute is not allowed")
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and _DUNDER_IN_STRING.search(node.value):
            self.error(node, "strings containing dunder names are not allowed")

    def _check_string_annotation(self, annotation: ast.expr | None) -> None:
        # String annotations are evaluated by typing.get_type_hints(); check them as code.
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                parsed = ast.parse(annotation.value, mode="eval")
            except SyntaxError:
                self.error(annotation, f"invalid string annotation {annotation.value!r}")
                return
            self.visit(parsed)

    def visit_arg(self, node: ast.arg) -> None:
        self._check_string_annotation(node.annotation)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_string_annotation(node.returns)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_string_annotation(node.annotation)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.error(node, "global statements are not allowed")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.error(node, "async functions are not allowed")

    def visit_Await(self, node: ast.Await) -> None:
        self.error(node, "await is not allowed")


_ALLOWED_TOP_LEVEL = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)


def _parse(source: str, label: str, report: StaticReport) -> ast.Module | None:
    if len(source) > MAX_SOURCE_CHARS:
        report.errors.append(f"{label}: longer than {MAX_SOURCE_CHARS} characters")
        return None
    if "```" in source:
        report.errors.append(f"{label}: contains Markdown code fences; return plain Python code")
        return None
    try:
        return ast.parse(source, filename=label)
    except SyntaxError as exc:
        report.errors.append(f"{label}: syntax error in line {exc.lineno}: {exc.msg}")
        return None


def _check_top_level(module: ast.Module, label: str, report: StaticReport) -> None:
    for index, statement in enumerate(module.body):
        is_docstring = (
            index == 0
            and isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
        if not is_docstring and not isinstance(statement, _ALLOWED_TOP_LEVEL):
            report.errors.append(
                f"{label} line {statement.lineno}: only imports, constants, classes and "
                f"function definitions are allowed at module level, not {type(statement).__name__}"
            )


def _extra(extra_imports: Iterable[str]) -> frozenset[str]:
    names = frozenset(extra_imports)
    invalid = sorted(name for name in names if not name.isidentifier() or keyword.iskeyword(name))
    if invalid:
        raise ValueError(f"extra_imports must be top-level module names, got {invalid}")
    return names


def check_tool_source(source: str, function_name: str, *, extra_imports: Iterable[str] = ()) -> StaticReport:
    """Check generated tool code.

    The module must define exactly one public function, named ``function_name``;
    helpers must start with an underscore. Only imports from
    :data:`ALLOWED_IMPORTS` are permitted, and no names from
    :data:`FORBIDDEN_NAMES` or dunder names/attributes may be used.

    ``extra_imports`` permits further top-level modules, e.g. third-party
    libraries the user approved and installed for the sandbox (``"numpy"``).
    The other rules still apply to code that uses them.
    """
    report = StaticReport()
    if not function_name.isidentifier() or keyword.iskeyword(function_name) or function_name.startswith("_"):
        report.errors.append(f"invalid public function name {function_name!r}")
    module = _parse(source, "code", report)
    if module is None:
        return report
    _check_top_level(module, "code", report)
    public = [
        node.name
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
    ]
    if public != [function_name]:
        report.errors.append(
            f"code must define exactly one public function {function_name!r} "
            f"(helpers start with '_'), found {public}"
        )
    checker = _Checker("code", _extra(extra_imports))
    checker.visit(module)
    report.errors.extend(checker.errors)
    return report


def check_tool_tests(source: str, *, extra_imports: Iterable[str] = ()) -> StaticReport:
    """Check generated tests: plain ``test_*`` functions without parameters.

    Tests import the tool with ``from tool import <function_name>``.
    ``extra_imports`` permits further top-level modules, as for :func:`check_tool_source`.
    """
    report = StaticReport()
    module = _parse(source, "tests", report)
    if module is None:
        return report
    _check_top_level(module, "tests", report)
    tests = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
    if not tests:
        report.errors.append("tests must define at least one test_* function")
    for test in tests:
        arguments = test.args
        if (
            arguments.args
            or arguments.posonlyargs
            or arguments.kwonlyargs
            or arguments.vararg
            or arguments.kwarg
        ):
            report.errors.append(f"tests: {test.name} must not take parameters (no pytest fixtures)")
    checker = _Checker("tests", _extra(extra_imports) | {TOOL_MODULE})
    checker.visit(module)
    report.errors.extend(checker.errors)
    return report
