"""Derive the Orkexa brand assets in this folder from the two official logo masters.

``derive_branding.py`` (vendored unchanged from the brand kit, shared with the other
products) does every pixel operation. This wrapper only picks the files the repository
uses, renders the 1280 x 640 social preview with the shared ``social_preview.py`` template
and copies the web assets into ``docs/assets/branding/`` because MkDocs only serves files
below ``docs/``.
Run from the repository root with Pillow and NumPy:

    uv run --with pillow --with numpy python assets/branding/build.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / "docs" / "assets" / "branding"

sys.path.insert(0, str(HERE))
import derive_branding  # noqa: E402  (vendored next to this file)
import social_preview  # noqa: E402  (vendored next to this file)

PRODUCT = "orkexa"

#: Derived files kept in this folder; the docs serve the same bytes (tests/test_repository.py).
KEPT = (
    "orkexa-logo.png",
    "orkexa-logo-on-dark.png",
    "orkexa-mark-on-dark-128.png",
    "favicon.ico",
    "favicon-32.png",
    "apple-touch-icon.png",
)

#: One-line caption on the GitHub social preview.
SOCIAL_CAPTION = "One agent runtime for every major model provider"


def main() -> int:
    if derive_branding.main(["--product", PRODUCT, "--masters", str(HERE), "--check"]):
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        args = ["--product", PRODUCT, "--masters", str(HERE), "--out", str(out), "--sets", "web,favicons"]
        if derive_branding.main(args):
            return 1
        for name in KEPT:
            shutil.copyfile(out / name, HERE / name)

    # Social preview: the shared 1280 x 640 template (social_preview.py, same in every product repo).
    social = social_preview.compose(PRODUCT, HERE, SOCIAL_CAPTION, None)
    social.save(HERE / "social-preview.png", optimize=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    for name in KEPT:
        shutil.copyfile(HERE / name, DOCS / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
