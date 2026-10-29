# Orkexa brand assets

`orkexa-logo-original.png` is the official Orkexa logo as supplied (1254 × 1254 px,
RGB raster). It is kept byte for byte; every other file here is derived from it by
`build.py` and must not be edited by hand.

| File | Size | Use |
|------|------|-----|
| `orkexa-logo-original.png` | 1254 × 1254 | master copy, do not modify |
| `orkexa-logo.png` | 640 × 496 | README hero and docs landing page (symbol and wordmark) |
| `orkexa-mark.png` | 256 × 256 | docs header logo (symbol on a rounded light tile) |
| `favicon.ico` | 16, 32, 48 | browser favicon |
| `favicon-32.png` | 32 × 32 | PNG favicon (docs site) |
| `apple-touch-icon.png` | 180 × 180 | home-screen icon |
| `social-preview.png` | 1280 × 640 | GitHub social preview (uploaded manually in the repository settings) |

## Rules

* The derivatives are crops, scaled copies and padded copies of the original on its own
  light background. The artwork, its proportions and its colors are not changed;
  no part of it is redrawn.
* There is no vector (SVG) version. A traced vector would be a reinterpretation of the
  raster artwork and is not provided; an official vector master would have to come from
  the logo's designer.
* There is no separate dark variant. On dark backgrounds the logo is shown on its light
  tile or light background, which keeps the navy wordmark legible without recoloring it.
* Brand colors sampled from the logo: navy `#16243B` (wordmark, symbol), blue
  `#0079FE` (accent). The docs theme uses these two colors.

## Regenerating

```bash
cd assets/branding
uv run --with pillow python build.py   # or: pip install pillow && python build.py
```

`build.py` also copies the web assets into `docs/assets/branding/`, because MkDocs only
serves files below `docs/`. `tests/test_repository.py` checks that the copies match.
