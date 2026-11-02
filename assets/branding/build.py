"""Derive the Orkexa brand assets in this folder from the two official logo masters.

``derive_branding.py`` (vendored unchanged from the brand kit, shared with the other
products) does every pixel operation. This wrapper only picks the files the repository
uses, lays the light lockup out as the 1280 x 640 social preview, and copies the web
assets into ``docs/assets/branding/`` because MkDocs only serves files below ``docs/``.
Run from the repository root with Pillow and NumPy:

    uv run --with pillow --with numpy python assets/branding/build.py
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent.parent / "docs" / "assets" / "branding"

sys.path.insert(0, str(HERE))
import derive_branding  # noqa: E402  (vendored next to this file)

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

#: GitHub's recommended social preview size, and the lockup width on it (generous margin).
SOCIAL_SIZE = (1280, 640)
SOCIAL_LOCKUP_WIDTH = 800


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

    # Social preview: the light lockup, derived by the same script, centred on white.
    variants = derive_branding.variants(PRODUCT, HERE)
    lockup = derive_branding.lockup(variants["light"], SOCIAL_LOCKUP_WIDTH)
    social = Image.new("RGBA", SOCIAL_SIZE, (255, 255, 255, 255))
    offset = ((SOCIAL_SIZE[0] - lockup.width) // 2, (SOCIAL_SIZE[1] - lockup.height) // 2)
    social.alpha_composite(lockup, offset)
    social.convert("RGB").save(HERE / "social-preview.png", optimize=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    for name in KEPT:
        shutil.copyfile(HERE / name, DOCS / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
