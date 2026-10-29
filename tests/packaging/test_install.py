"""Install ``orkexa``, ``orkexa-claude-compat`` and the original ``claude-agent`` into
fresh virtual environments and check imports, the conflict guard and uninstallation.

The original distribution is built from a copy of the reference checkout given by
``CLAUDE_AGENT_SOURCE`` (default: ``../claude-agent`` next to this repository). Without
it (e.g. in CI, where the reference repository is private) a stand-in distribution
named ``claude-agent`` 0.8.0 that ships a ``claude_agent`` package is built instead:
the conflict guard and the diagnostics only look at distribution metadata and files.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.packaging

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path(os.environ.get("CLAUDE_AGENT_SOURCE", ROOT.parent / "claude-agent"))
UV = shutil.which("uv")


def _run(*args: str | Path, check: bool = True, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY") and k != "VIRTUAL_ENV"}
    result = subprocess.run(
        [str(a) for a in args], capture_output=True, text=True, env=env, cwd=cwd, check=False
    )
    if check and result.returncode != 0:
        raise AssertionError(f"{args} failed:\n{result.stdout}\n{result.stderr}")
    return result


def _build(source: Path, out: Path) -> Path:
    if UV is not None:
        _run(UV, "build", "--wheel", source, "--out-dir", out)
    else:
        _run(sys.executable, "-m", "build", "--wheel", "--outdir", out, source)
    (wheel,) = sorted(out.glob("*.whl"))
    return wheel


class Venv:
    def __init__(self, path: Path, wheels: Path) -> None:
        self.path = path
        self.wheels = wheels
        if UV is not None:
            _run(UV, "venv", "--python", sys.executable, path)
        else:
            _run(sys.executable, "-m", "venv", path)
        self.python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    def install(self, *requirements: str | Path) -> None:
        if UV is not None:
            _run(UV, "pip", "install", "--python", self.python, "--find-links", self.wheels, *requirements)
        else:
            _run(self.python, "-m", "pip", "install", "--find-links", self.wheels, *requirements)

    def uninstall(self, *names: str) -> None:
        if UV is not None:
            _run(UV, "pip", "uninstall", "--python", self.python, *names)
        else:
            _run(self.python, "-m", "pip", "uninstall", "-y", *names)

    def py(self, code: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return _run(self.python, "-I", "-c", code, check=check, cwd=self.path)


def _stand_in_original(path: Path) -> None:
    """A minimal ``claude-agent`` 0.8.0 distribution with its own ``claude_agent`` package."""
    package = path / "src" / "claude_agent"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        '"""Stand-in for the original claude-agent."""\n__version__ = "0.8.0"\n'
    )
    (path / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["hatchling>=1.27"]\nbuild-backend = "hatchling.build"\n\n'
        '[project]\nname = "claude-agent"\nversion = "0.8.0"\nrequires-python = ">=3.10"\n\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["src/claude_agent"]\n'
    )


@pytest.fixture(scope="module")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    out = tmp_path_factory.mktemp("wheels")
    built = {
        "orkexa": _build(ROOT, out / "orkexa"),
        "shim": _build(ROOT / "compat" / "orkexa-claude-compat", out / "shim"),
    }
    copy = tmp_path_factory.mktemp("reference") / "claude-agent"
    if (REFERENCE / "pyproject.toml").exists():
        shutil.copytree(
            REFERENCE, copy, ignore=shutil.ignore_patterns(".git", ".venv", "dist", "__pycache__")
        )
    else:
        _stand_in_original(copy)
    built["original"] = _build(copy, out / "original")
    for wheel in list(built.values()):
        shutil.copy(wheel, out)
    built["dir"] = out
    yield built


def test_core_install_has_no_provider_sdk(wheels: dict[str, Path], tmp_path: Path) -> None:
    venv = Venv(tmp_path / "core", wheels["dir"])
    venv.install(wheels["orkexa"])
    out = venv.py(
        "import sys, orkexa\n"
        "from orkexa import Agent, MissingDependencyError\n"
        "assert not [m for m in sys.modules if m.split('.')[0] in ('anthropic', 'openai', 'google')]\n"
        "agent = Agent(model='openai:gpt-5.5', api_key='k')\n"
        "try:\n    agent.client\nexcept MissingDependencyError as e:\n    print('missing', e)\n"
        "print(orkexa.__version__)\n"
    ).stdout
    assert "orkexa[openai]" in out and "0.1.0" in out
    assert venv.py("import claude_agent", check=False).returncode != 0


def test_extras_install_every_sdk(wheels: dict[str, Path], tmp_path: Path) -> None:
    venv = Venv(tmp_path / "all", wheels["dir"])
    venv.install(f"orkexa[all] @ {wheels['orkexa'].as_uri()}")
    venv.py(
        "from orkexa import Agent\n"
        "for model in ('anthropic:claude-opus-5-5', 'openai:gpt-5.5', 'gemini:gemini-3-pro'):\n"
        "    assert Agent(model=model, api_key='k').client is not None\n"
        "assert Agent(model='m', provider='ollama').client is not None\n"
    )


def test_shim_serves_unchanged_imports_and_uninstalls_cleanly(
    wheels: dict[str, Path], tmp_path: Path
) -> None:
    venv = Venv(tmp_path / "shim", wheels["dir"])
    venv.install(wheels["shim"])
    venv.py(
        "from claude_agent import ClaudeAgent, Tool, EphemeralMemory, SQLiteIndex\n"
        "from claude_agent.dynamic import ToolBuilder, AutoApprover\n"
        "from claude_agent.agent import ClaudeAgent as A2\n"
        "import claude_agent.retrieval, orkexa.retrieval, orkexa\n"
        "assert claude_agent.retrieval is orkexa.retrieval and A2 is ClaudeAgent is orkexa.ClaudeAgent\n"
        "agent = ClaudeAgent(api_key='k')\n"
        "assert agent.model == 'claude-haiku-5-5' and agent.messages == []\n"
    )
    report = _run(venv.python, "-m", "orkexa.compat")
    assert "import claude_agent:  orkexa-claude-compat" in report.stdout
    venv.uninstall("orkexa-claude-compat")
    assert venv.py("import claude_agent", check=False).returncode != 0
    venv.py("import orkexa; orkexa.ClaudeAgent(api_key='k')")


@pytest.mark.parametrize("order", ["original_then_shim", "shim_then_original"])
def test_conflict_with_the_original_distribution_is_detected(
    wheels: dict[str, Path], tmp_path: Path, order: str
) -> None:
    venv = Venv(tmp_path / order, wheels["dir"])
    first, second = ("original", "shim") if order == "original_then_shim" else ("shim", "original")
    venv.install(wheels[first])
    venv.install(wheels[second])
    report = _run(venv.python, "-m", "orkexa.compat", check=False)
    assert report.returncode == 1
    assert "CONFLICT" in report.stdout
    if order == "original_then_shim":
        failed = venv.py("import claude_agent", check=False)
        assert (
            failed.returncode != 0 and "pip uninstall -y claude-agent orkexa-claude-compat" in failed.stderr
        )
    # the documented repair
    venv.uninstall("claude-agent", "orkexa-claude-compat")
    venv.install(wheels["shim"])
    venv.py("from claude_agent import ClaudeAgent; ClaudeAgent(api_key='k')")
    assert _run(venv.python, "-m", "orkexa.compat").returncode == 0
