"""``claude_agent`` 0.8.0 imports, served by Orkexa (distribution ``orkexa-claude-compat``).

Every ``claude_agent.<module>`` is an alias of the matching Orkexa module, e.g.
``claude_agent.retrieval is orkexa.retrieval`` and ``claude_agent.agent`` is
:mod:`orkexa.claude`. Nothing is implemented here.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import importlib.abc
import importlib.machinery
from importlib.metadata import PackageNotFoundError, distribution
import sys
from types import ModuleType
from typing import Any


def _check_environment() -> None:
    try:
        original = distribution("claude-agent")
    except PackageNotFoundError:
        return
    raise ImportError(
        f"The original 'claude-agent' {original.version} distribution is installed next to "
        "'orkexa-claude-compat'; both provide the package 'claude_agent' and overwrite each "
        "other's files. Repair the environment with:\n"
        "    pip uninstall -y claude-agent orkexa-claude-compat\n"
        "    pip install orkexa-claude-compat\n"
        "(or keep only claude-agent). Diagnose with: python -m orkexa.compat"
    )


_check_environment()

#: ``claude_agent.<name>`` -> Orkexa module (names not listed map to ``orkexa.<name>``).
_RENAMED = {"agent": "orkexa.claude", "async_agent": "orkexa.claude", "_base": "orkexa.claude"}


def _target(fullname: str) -> str:
    rest = fullname.removeprefix(__name__ + ".")
    return _RENAMED.get(rest, f"orkexa.{rest}")


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.spec = module.__spec__
        self.loader = getattr(module, "__loader__", None)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return self.module

    def exec_module(self, module: ModuleType) -> None:
        # The import system stamped the alias spec on the shared module; restore it.
        module.__spec__ = self.spec
        module.__loader__ = self.loader


class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolves ``claude_agent.x`` to the already existing module ``orkexa.x``."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(__name__ + "."):
            return None
        try:
            module = importlib.import_module(_target(fullname))
        except ModuleNotFoundError:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(module))


if not any(isinstance(finder, _AliasFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

from orkexa import *  # noqa: E402,F403
from orkexa import __all__ as _orkexa_all  # noqa: E402
from orkexa.claude import (  # noqa: E402,F401
    AsyncClaudeAgent,
    ClaudeAgent,
    Effort,
    Fallbacks,
    RefusalMode,
    ThinkingDisplay,
    ThinkingMode,
    ToolCall,
)

try:
    __version__ = distribution("orkexa-claude-compat").version
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0"

__all__ = sorted(
    {
        *_orkexa_all,
        "AsyncClaudeAgent",
        "ClaudeAgent",
        "Effort",
        "Fallbacks",
        "RefusalMode",
        "ThinkingDisplay",
        "ThinkingMode",
        "ToolCall",
        "__version__",
    }
)


def __getattr__(name: str) -> Any:
    return getattr(importlib.import_module("orkexa"), name)
