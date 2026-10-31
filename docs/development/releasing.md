# Releasing

Orkexa has not been published yet. The release workflow
(`.github/workflows/release.yml`) is prepared but runs only when a GitHub release is
published, and it cannot upload anything until the one-time setup below is done.

## One-time setup (manual)

These steps change settings of the GitHub repository and of PyPI accounts, so they are
done by the repository owner by hand:

1. **PyPI projects and trusted publishers.** On [pypi.org](https://pypi.org), add a
   *pending trusted publisher* for the project `orkexa`, with owner `peterkarlhuber`,
   repository `orkexa`, workflow `release.yml` and environment `pypi`. No API token is stored anywhere.
2. **GitHub environment.** In the repository settings, create the environment `pypi`,
   and restrict it to tags matching `v*` with required reviewers, so that every upload
   needs an explicit approval.
3. **Repository security settings.** Enable private vulnerability reporting (referenced
   by `SECURITY.md`), Dependabot alerts and security updates, and protect the default
   branch (required CI checks, no force pushes).
4. **Optional:** upload `assets/branding/social-preview.png` as the repository's social
   preview, and pin the GitHub Actions in the workflows to commit SHAs (Dependabot keeps
   SHA pins up to date).
5. **Optional documentation site:** the docs build in CI, but are not deployed. To
   publish them, enable GitHub Pages and add a deployment job (for example
   `mkdocs gh-deploy` or the `actions/deploy-pages` flow) after reviewing what the site
   exposes.

## Each release

1. Make sure CI is green on the default branch.
2. Set the version in `pyproject.toml` (`orkexa.__version__` comes from the installed
   metadata), and move the changelog entries from "unreleased" to the version with the
   date.
3. Build and check locally: `python -m build`, `twine check --strict dist/*`,
   `pytest -m packaging tests/packaging --noconftest`.
4. The published description is `README.md` with the logo and relative links
   rewritten to absolute URLs on the default branch (`hatch-fancy-pypi-readme`, see
   `pyproject.toml`); `tests/packaging/test_artifacts.py` checks that no relative path
   is left and that the logo URL names a committed file. These URLs resolve only for a
   **public** repository: make the repository public before uploading, then open
   `https://raw.githubusercontent.com/peterkarlhuber/orkexa/master/assets/branding/orkexa-logo.png`
   and check the rendered description on TestPyPI or after `twine check --strict`.
5. Create a tag `vX.Y.Z` and a GitHub release for it. Publishing the release starts
   the workflow, which checks that the tag matches the version, builds and checks the
   distribution, runs the packaging tests, creates build provenance attestations, and,
   after the `pypi` environment is approved, uploads `orkexa` through trusted
   publishing.
6. Verify the release in a clean environment: `pip install "orkexa[all]==X.Y.Z"`.

The development status classifier (`Development Status :: 3 - Alpha`) changes only
when the project's maturity does, not for a release.
