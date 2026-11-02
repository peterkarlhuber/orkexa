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

# sha256 of the official logo masters as supplied; the master copies must never change.
MASTER_SHA256 = {
    "orkexa-logo-transparent.png": "7a925b11768bf68a69fcc784370d175046706d12b3de5fa5dd0f86352bcf942a",
    "orkexa-logo-white.png": "096952e28a2dd85018f8cafba52911167c2913b66cc2f074276b399e67ff98ee",
}


@pytest.mark.parametrize("name", sorted(MASTER_SHA256))
def test_the_logo_masters_are_unchanged(name: str) -> None:
    assert hashlib.sha256((BRANDING / name).read_bytes()).hexdigest() == MASTER_SHA256[name]


@pytest.mark.parametrize(
    "name",
    [
        "orkexa-logo.png",
        "orkexa-logo-on-dark.png",
        "orkexa-mark-on-dark-128.png",
        "favicon.ico",
        "favicon-32.png",
        "apple-touch-icon.png",
    ],
)
def test_docs_serve_the_same_brand_assets(name: str) -> None:
    assert (ROOT / "docs" / "assets" / "branding" / name).read_bytes() == (BRANDING / name).read_bytes()


def test_the_social_preview_has_the_github_size() -> None:
    # PNG IHDR: width and height are the big-endian 32-bit integers at bytes 16..24.
    header = (BRANDING / "social-preview.png").read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")) == (1280, 640)


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
