# Orkexa brand assets

`orkexa-logo-transparent.png` and `orkexa-logo-white.png` are the official Orkexa logo
masters as supplied (2048 × 640 px each: symbol left, black wordmark right; the white
file is the transparent one flattened on white). They are kept byte for byte
(`tests/test_repository.py` checks their sha256). Every other image here is derived from
the transparent master and must not be edited by hand.

| File | Size | Use |
|------|------|-----|
| `orkexa-logo-transparent.png` | 2048 × 640 | master, transparent background, do not modify |
| `orkexa-logo-white.png` | 2048 × 640 | master, white background, do not modify |
| `orkexa-logo.png` | 960 × 166 | lockup with black wordmark for light surfaces: README hero, docs landing page (light scheme), PyPI description |
| `orkexa-logo-on-dark.png` | 960 × 166 | lockup with white wordmark for dark surfaces: README hero (dark mode), docs landing page (dark scheme) |
| `orkexa-mark-on-dark-128.png` | 128 × 128 | symbol for dark surfaces: docs header logo (the header bar is navy in both schemes) |
| `favicon.ico` | 16, 32, 48 | browser favicon (symbol on a rounded white tile) |
| `favicon-32.png` | 32 × 32 | PNG favicon (docs site) |
| `apple-touch-icon.png` | 180 × 180 | home-screen icon (opaque white tile) |
| `social-preview.png` | 1280 × 640 | GitHub social preview: the white-wordmark lockup on the shared dark brand template (uploaded manually in the repository settings) |

## Rules

* Nothing is redrawn. `derive_branding.py` (the brand kit's derivation script, vendored
  unchanged and shared with the other products) only removes the light fringe of the
  background knockout, recolors the black wordmark to white for dark surfaces, lifts
  the darkest navy tones of the symbol for dark surfaces (hue and saturation kept), trims,
  pads and scales.
* Light surfaces use `orkexa-logo.png`, dark surfaces `orkexa-logo-on-dark.png`; the
  switch follows the surface's own theme mechanism (`<picture>` with
  `prefers-color-scheme` in the README, `#only-light` / `#only-dark` in MkDocs Material).
  No CSS filters or boxes behind the logo.
* Below about 40 px height, use the symbol instead of the full lockup.
* There is no vector (SVG) version; an official vector master would have to come from
  the logo's designer.

## Regenerating

```bash
uv run --with pillow --with numpy python assets/branding/build.py
```

`build.py` checks the masters, runs `derive_branding.py`, keeps the files listed above,
renders `social-preview.png` with `social_preview.py` and copies the web assets into `docs/assets/branding/`,
because MkDocs only serves files below `docs/`. `tests/test_repository.py` checks that the
copies match.

`social_preview.py` is the social preview template shared with the Nodestro and Axentra
repositories (vendored unchanged next to `derive_branding.py`): a dark background with a soft
glow in the brand blue, the white-wordmark lockup, a short accent rule and a one-line caption.
Only the logo pixels come from the masters; the background and caption are drawn by the
script. Output bytes, like those of `derive_branding.py`, depend on the Pillow version, so
`build.py` may rewrite the favicons byte-for-byte differently on another Pillow release; commit
only the files you meant to change.
