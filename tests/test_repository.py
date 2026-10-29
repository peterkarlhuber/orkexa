"""Repository consistency: brand assets, documentation links and community files."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
BRANDING = ROOT / "assets" / "branding"

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
