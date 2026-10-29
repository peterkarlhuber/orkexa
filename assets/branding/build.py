"""Derive the web assets in this folder from ``orkexa-logo-original.png``.

The artwork itself is never redrawn or recolored: every derivative is a crop of the
original, extended with the original's own background color, scaled down and, for the
square tiles, given rounded transparent corners. Run from this folder with Pillow:

    uv run --with pillow python build.py
"""

from __future__ import annotations

from pathlib import Path
import shutil

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
DOCS = HERE.parent.parent / "docs" / "assets" / "branding"  # MkDocs only serves files below docs/
ORIGINAL = Image.open(HERE / "orkexa-logo-original.png").convert("RGB")
BACKGROUND = ORIGINAL.getpixel((600, 100))  # the original's (near-white) background
FULL_LOGO = (149, 279, 1106, 1020)  # symbol and wordmark with a margin of the original
SYMBOL = (312, 339, 951, 739)  # the symbol above the wordmark, without margin


def padded(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def scaled(image: Image.Image, width: int) -> Image.Image:
    return image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)


def rounded(image: Image.Image, radius_ratio: float = 0.18) -> Image.Image:
    mask = Image.new("L", (image.width * 4, image.height * 4), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, mask.width - 1, mask.height - 1), radius=round(mask.width * radius_ratio), fill=255
    )
    tile = image.convert("RGBA")
    tile.putalpha(mask.resize(image.size, Image.Resampling.LANCZOS))
    return tile


def main() -> None:
    logo = ORIGINAL.crop(FULL_LOGO)
    scaled(logo, 640).save(HERE / "orkexa-logo.png", optimize=True)

    symbol = ORIGINAL.crop(SYMBOL)
    tile = padded(symbol, 760, 760)
    for size, name in ((256, "orkexa-mark.png"), (180, "apple-touch-icon.png"), (32, "favicon-32.png")):
        rounded(tile.resize((size, size), Image.Resampling.LANCZOS)).save(HERE / name, optimize=True)
    rounded(tile.resize((256, 256), Image.Resampling.LANCZOS)).save(
        HERE / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)]
    )

    social = padded(scaled(logo, 600), 1280, 640)  # GitHub's recommended 2:1 social preview
    social.save(HERE / "social-preview.png", optimize=True)

    DOCS.mkdir(parents=True, exist_ok=True)
    for name in (
        "orkexa-logo.png",
        "orkexa-mark.png",
        "favicon.ico",
        "favicon-32.png",
        "apple-touch-icon.png",
    ):
        shutil.copyfile(HERE / name, DOCS / name)


if __name__ == "__main__":
    main()
