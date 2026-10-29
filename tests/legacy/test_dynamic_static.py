from __future__ import annotations

import pytest

from claude_agent.dynamic import check_tool_source, check_tool_tests

from .dynamic_helpers import MARGIN_CODE, MARGIN_TESTS

_DOC = '"""Do it.\n\n    Returns:\n        Something.\n    """'


def _function(body: str, header: str = "") -> str:
    return f"{header}\ndef f() -> str:\n    {_DOC}\n    {body}\n"


def test_valid_code_and_tests_pass() -> None:
    assert check_tool_source(MARGIN_CODE, "calculate_operating_margin").ok
    assert check_tool_tests(MARGIN_TESTS).ok


@pytest.mark.parametrize(
    ("source", "fragment"),
    [
        (_function("return ''", "import os"), "import of 'os'"),
        (_function("return ''", "import subprocess as s"), "import of 'subprocess'"),
        (_function("return ''", "from socket import socket"), "import from 'socket'"),
        (_function("return ''", "from urllib.request import urlopen"), "import from 'urllib.request'"),
        (_function("return ''", "from math import *"), "star imports"),
        (_function("return ''", "from . import x"), "relative imports"),
        (_function("return open('/etc/passwd').read()"), "'open'"),
        (_function("return eval('1')"), "'eval'"),
        (_function("return str(exec('x=1'))"), "'exec'"),
        (_function("return str(__import__('os'))"), "'__import__'"),
        (_function("return str(__builtins__)"), "'__builtins__'"),
        (_function("return str(getattr(f, 'x'))"), "'getattr'"),
        (_function("return str(().__class__.__bases__[0].__subclasses__())"), "'__class__'"),
        (_function("return str(type(f))"), "'type'"),
        (_function("return '{0.__globals__}'.format(f)"), "dunder names"),
        (_function("return str((x for x in []).gi_frame.f_globals)"), "'gi_frame'"),
        (_function("return ''", "from operator import attrgetter"), "'attrgetter'"),
        (_function("return ''", "from pydantic import ImportString"), "'ImportString'"),
        (_function("return ''", "print('side effect')"), "module level"),
        (_function("return ''", "global x"), "module level"),
        (_function("return ''") + "\ndef g() -> str:\n    return ''\n", "exactly one public function"),
        ("def f(x: \"__import__('os').system('id')\") -> str:\n    return ''\n", "'__import__'"),
        ("```python\ndef f() -> str: ...\n```", "Markdown"),
        ("def f(:\n", "syntax error"),
        ("async def f() -> str:\n    return ''\n", "async"),
    ],
)
def test_dangerous_code_is_rejected(source: str, fragment: str) -> None:
    report = check_tool_source(source, "f")
    assert not report.ok
    assert any(fragment in error for error in report.errors), report.errors


def test_function_name_must_match_and_be_public() -> None:
    assert not check_tool_source(MARGIN_CODE, "other_name").ok
    assert not check_tool_source(MARGIN_CODE, "_private").ok
    assert not check_tool_source(MARGIN_CODE, "class").ok


def test_private_helpers_are_allowed() -> None:
    source = _function("return _helper()", "def _helper() -> str:\n    return 'x'\n")
    assert check_tool_source(source, "f").ok


def test_size_limit() -> None:
    assert not check_tool_source("x = 1\n" * 10_000, "f").ok


def test_tests_need_parameterless_test_functions() -> None:
    assert not check_tool_tests("from tool import f\n").ok
    report = check_tool_tests("def test_a(tmp_path):\n    assert True\n")
    assert any("must not take parameters" in e for e in report.errors)
    assert not check_tool_tests("import os\ndef test_a():\n    assert os\n").ok
    assert check_tool_tests("from tool import f\ndef test_a():\n    assert f\n").ok


def test_extra_imports_permit_approved_modules_only() -> None:
    source = _function("return str(numpy.pi)", "import numpy\nimport numpy.linalg")
    assert not check_tool_source(source, "f").ok
    assert check_tool_source(source, "f", extra_imports={"numpy"}).ok
    assert not check_tool_source(_function("return ''", "import os"), "f", extra_imports={"numpy"}).ok
    # The other rules still apply to code using an extra module.
    assert not check_tool_source(
        _function("return str(numpy.__dict__)", "import numpy"), "f", extra_imports=["numpy"]
    ).ok
    tests = "import numpy\nfrom tool import f\n\ndef test_a():\n    assert numpy and f\n"
    assert not check_tool_tests(tests).ok
    assert check_tool_tests(tests, extra_imports=("numpy",)).ok


def test_extra_imports_must_be_module_names() -> None:
    with pytest.raises(ValueError, match="top-level module names"):
        check_tool_source(_function("return ''"), "f", extra_imports={"numpy.linalg"})
