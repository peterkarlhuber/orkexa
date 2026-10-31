"""Repository consistency: brand assets, documentation links and community files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRANDING = ROOT / "assets" / "branding"

# The sdist leaves out assets and CI files: these checks are
# about the repository checkout only.
pytestmark = pytest.mark.skipif(not (ROOT / ".github").is_dir(), reason="needs a repository checkout")

# sha256 of the official logo as supplied; the master copy must never change.
ORIGINAL_LOGO_SHA256 = "2f0ab826a322613a297d5aafa7120356726eccd80dd5ffa6096446b514e69289"


def test_the_original_logo_is_unchanged() -> None:
    digest = hashlib.sha256((BRANDING / "orkexa-logo-original.png").read_bytes()).hexdigest()
    assert digest == ORIGINAL_LOGO_SHA256


@pytest.mark.parametrize(
    "name", ["orkexa-logo.png", "orkexa-mark.png", "favicon.ico", "favicon-32.png", "apple-touch-icon.png"]
)
def test_docs_serve_the_same_brand_assets(name: str) -> None:
    assert (ROOT / "docs" / "assets" / "branding" / name).read_bytes() == (BRANDING / name).read_bytes()


# ---- links and community files ---------------------------------------------------------------

MARKDOWN = [
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "SUPPORT.md",
    "CODE_OF_CONDUCT.md",
    "CHANGELOG.md",
    "examples/README.md",
    "assets/branding/README.md",
]
LINK = re.compile(r"\]\(([^)\s]+)\)|<img src=\"([^\"]+)\"|href=\"([^\"]+)\"")


@pytest.mark.parametrize("name", MARKDOWN)
def test_relative_links_resolve(name: str) -> None:
    """MkDocs checks the links below docs/ (``mkdocs build --strict``); this checks the rest."""
    page = ROOT / name
    for match in LINK.finditer(page.read_text(encoding="utf-8")):
        target = next(group for group in match.groups() if group)
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        path = (page.parent / target.split("#")[0]).resolve()
        assert path.exists(), f"{name}: broken link {target}"


@pytest.mark.parametrize(
    "name",
    [
        "LICENSE",
        "NOTICE",
        "CONTRIBUTING.md",
        "CODE_OF_CONDUCT.md",
        "SECURITY.md",
        "SUPPORT.md",
        "CHANGELOG.md",
        ".editorconfig",
        ".github/pull_request_template.md",
        ".github/ISSUE_TEMPLATE/bug_report.yml",
        ".github/ISSUE_TEMPLATE/feature_request.yml",
        ".github/dependabot.yml",
    ],
)
def test_community_files_exist(name: str) -> None:
    assert (ROOT / name).read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    "workflow", sorted((ROOT / ".github" / "workflows").glob("*.yml")), ids=lambda p: p.name
)
def test_workflows_default_to_read_only_permissions(workflow: Path) -> None:
    text = workflow.read_text(encoding="utf-8")
    assert re.search(r"^permissions:\n  contents: read$", text, re.MULTILINE)
    assert "pull_request_target" not in text
    assert "secrets." not in text  # publishing uses trusted publishing, not stored tokens
