"""Contents of the built sdists and wheels: only what belongs in a release.

No secrets, caches, virtual environments, build leftovers, brand assets or CI files;
the wheel of ``orkexa`` contains only the ``orkexa`` package and its metadata.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
import re
import tarfile
import zipfile

import pytest

from .test_install import ROOT, UV, _run

pytestmark = pytest.mark.packaging

FORBIDDEN = re.compile(
    r"(^|/)(\.env[^/]*|\.venv|venv|__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.coverage"
    r"|htmlcov|site|dist|build|\.git|\.github|\.DS_Store)(/|$)|\.py[cod]$|\.(pem|key|p12|sqlite|db)$"
)


def _build_all(source: Path, out: Path) -> tuple[Path, Path]:
    if UV is not None:
        _run(UV, "build", source, "--out-dir", out)
    else:
        import sys

        _run(sys.executable, "-m", "build", "--outdir", out, source)
    (sdist,) = out.glob("*.tar.gz")
    (wheel,) = out.glob("*.whl")
    return sdist, wheel


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, tuple[Path, Path]]]:
    out = tmp_path_factory.mktemp("artifacts")
    yield {
        "orkexa": _build_all(ROOT, out / "orkexa"),
        "compat": _build_all(ROOT / "compat" / "orkexa-claude-compat", out / "compat"),
    }


def _names(sdist: Path, wheel: Path) -> tuple[list[str], list[str]]:
    with tarfile.open(sdist) as archive:
        # Drop the top-level "<name>-<version>/" directory.
        sdist_names = [name.split("/", 1)[1] for name in archive.getnames() if "/" in name]
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = archive.namelist()
    return sdist_names, wheel_names


@pytest.mark.parametrize("distribution", ["orkexa", "compat"])
def test_artifacts_contain_no_junk_or_secrets(
    artifacts: dict[str, tuple[Path, Path]], distribution: str
) -> None:
    for names in _names(*artifacts[distribution]):
        offending = [name for name in names if FORBIDDEN.search(name)]
        assert offending == []


def test_orkexa_wheel_holds_only_the_package(artifacts: dict[str, tuple[Path, Path]]) -> None:
    sdist_names, wheel_names = _names(*artifacts["orkexa"])
    top = {name.split("/")[0] for name in wheel_names}
    assert len(top) == 2 and "orkexa" in top and any(t.endswith(".dist-info") for t in top)
    assert "orkexa/py.typed" in wheel_names
    assert not any(name.startswith(("assets/", "compat/", "docs/assets/")) for name in sdist_names)
    assert {"README.md", "LICENSE", "pyproject.toml"} <= set(sdist_names)


def test_compat_wheel_holds_only_the_shim(artifacts: dict[str, tuple[Path, Path]]) -> None:
    _, wheel_names = _names(*artifacts["compat"])
    top = {name.split("/")[0] for name in wheel_names}
    assert "claude_agent" in top and len(top) == 2
    assert not any(name.startswith("orkexa/") for name in wheel_names)


def test_published_description_uses_only_absolute_urls(artifacts: dict[str, tuple[Path, Path]]) -> None:
    """PyPI cannot resolve repository-relative paths, so the logo would not render."""
    _, wheel = artifacts["orkexa"]
    with zipfile.ZipFile(wheel) as archive:
        (metadata,) = (n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        description = archive.read(metadata).decode("utf-8").split("\n\n", 1)[1]
    logo = "https://raw.githubusercontent.com/peterkarlhuber/orkexa/master/assets/branding/orkexa-logo.png"
    assert f'<img src="{logo}"' in description
    assert (ROOT / logo.split("/master/", 1)[1]).is_file()  # the URL names a committed file
    targets = re.findall(r'(?:src|href)="([^"]+)"|\]\(([^)\s]+)\)', description)
    relative = [t for pair in targets for t in pair if t and not re.match(r"(https?:|mailto:|#)", t)]
    assert relative == []
    for target in re.findall(r"github\.com/peterkarlhuber/orkexa/blob/master/([^)\"#\s]+)", description):
        assert (ROOT / target).exists(), target
