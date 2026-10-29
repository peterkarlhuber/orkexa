"""Diagnostics for the ``claude_agent`` compatibility layer: ``python -m orkexa.compat``.

Reports which distribution provides the package ``claude_agent`` (the Orkexa shim
``orkexa-claude-compat``, the original ``claude-agent``, both, or none), which
provider SDKs are installed and which API key variables are set (names only, never
values). The exit status is 1 if the environment has a conflict, otherwise 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
import importlib.util
import os
import sys

__all__ = ["CompatReport", "diagnose", "main"]

_SDKS = {"anthropic": "anthropic", "openai": "openai", "gemini": "google-genai"}
_KEY_VARIABLES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _version(name: str) -> str | None:
    try:
        return distribution(name).version
    except PackageNotFoundError:
        return None


@dataclass(frozen=True)
class CompatReport:
    """The state of the current environment."""

    orkexa: str | None
    shim: str | None
    original: str | None
    claude_agent_path: str | None
    sdks: dict[str, str | None] = field(default_factory=dict)
    key_variables: tuple[str, ...] = ()

    @property
    def conflict(self) -> bool:
        """Both distributions ship ``claude_agent`` and overwrite each other's files."""
        return self.shim is not None and self.original is not None

    @property
    def provider(self) -> str:
        """Who serves ``import claude_agent``."""
        if self.conflict:
            return "conflict"
        if self.shim is not None:
            return "orkexa-claude-compat"
        if self.original is not None:
            return "claude-agent"
        return "none"

    def lines(self) -> list[str]:
        out = [
            f"orkexa:               {self.orkexa or 'not installed'}",
            f"orkexa-claude-compat: {self.shim or 'not installed'}",
            f"claude-agent:         {self.original or 'not installed'}",
            f"import claude_agent:  {self.provider}"
            + (f" ({self.claude_agent_path})" if self.claude_agent_path else ""),
        ]
        for extra, sdk in _SDKS.items():
            out.append(f"SDK {sdk:<17} {self.sdks.get(extra) or 'not installed'} (orkexa[{extra}])")
        out.append("API key variables set: " + (", ".join(self.key_variables) or "none"))
        if self.conflict:
            out += [
                "",
                "CONFLICT: claude-agent and orkexa-claude-compat both install the package",
                "'claude_agent'. Repair with:",
                "    pip uninstall -y claude-agent orkexa-claude-compat",
                "    pip install orkexa-claude-compat",
            ]
        elif self.original is not None:
            out += [
                "",
                "The original claude-agent serves 'import claude_agent'. To run that code on",
                "Orkexa: pip uninstall -y claude-agent && pip install orkexa-claude-compat",
            ]
        return out


def diagnose() -> CompatReport:
    """Inspect the environment without importing ``claude_agent``."""
    spec = importlib.util.find_spec("claude_agent")
    origin = spec.origin if spec is not None else None
    return CompatReport(
        orkexa=_version("orkexa"),
        shim=_version("orkexa-claude-compat"),
        original=_version("claude-agent"),
        claude_agent_path=origin,
        sdks={extra: _version(sdk) for extra, sdk in _SDKS.items()},
        key_variables=tuple(name for name in _KEY_VARIABLES if os.environ.get(name)),
    )


def main(argv: list[str] | None = None) -> int:
    report = diagnose()
    print("\n".join(report.lines()))
    return 1 if report.conflict else 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    sys.exit(main(sys.argv[1:]))
