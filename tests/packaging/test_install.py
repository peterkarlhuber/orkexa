"""Install the ``orkexa`` wheel into fresh virtual environments and check its imports
and extras."""

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


@pytest.fixture(scope="module")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Path]]:
    out = tmp_path_factory.mktemp("wheels")
    built = {"orkexa": _build(ROOT, out / "orkexa")}
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


def test_extras_install_every_sdk(wheels: dict[str, Path], tmp_path: Path) -> None:
    venv = Venv(tmp_path / "all", wheels["dir"])
    venv.install(f"orkexa[all] @ {wheels['orkexa'].as_uri()}")
    venv.py(
        "from orkexa import Agent\n"
        "for model in ('anthropic:claude-opus-5-5', 'openai:gpt-5.5', 'gemini:gemini-3-pro'):\n"
        "    assert Agent(model=model, api_key='k').client is not None\n"
        "assert Agent(model='m', provider='ollama').client is not None\n"
    )
