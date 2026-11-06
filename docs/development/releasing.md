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
   preview. (The GitHub Actions in the workflows are already pinned to commit SHAs;
   Dependabot keeps the pins up to date.)
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
   `pyproject.toml`; PyPI has no dark mode, so the hero keeps only the light logo); `tests/packaging/test_artifacts.py` checks that no relative path
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

## Upkeep

Some data in Orkexa is a dated snapshot of an outside source. Review each one with the
update named below and before every release; `tests/test_upkeep.py` warns once a
snapshot is older than 90 days, and fails then with `ORKEXA_CHECK_FRESHNESS=1` (the
release workflow sets it). After a review, update the date even if nothing changed.

| Snapshot | Where | Review with | Checklist |
| --- | --- | --- | --- |
| Built-in Anthropic prices | `BUILTIN_VERSION`/`BUILTIN_CHECKED_AT` in `src/orkexa/pricing/_builtin.py` | every Anthropic price change, and before a release | compare each model row with the [pricing page](https://platform.claude.com/docs/en/about-claude/pricing); add new models; bump the version and the date |
| Bedrock capability table | `bedrock_capabilities()` and `BEDROCK_CAPABILITIES_CHECKED_AT` in `src/orkexa/providers/bedrock.py` | every `boto3` minimum bump, a new Claude or Nova model on Bedrock, and before a release | check the [Converse feature table](https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-supported-models-features.html) and each model card: tools, forced tool choice, streaming, image and document input, prompt caching, structured output, strict tool schemas, CountTokens; a model that is not in the table stays `unknown` (Nodestro then turns the feature off), never guessed; extend `tests/test_bedrock.py` for each new family |
| Sandbox base image and requirements | the `FROM` digest and the "verified ... on" date in `docker/sandbox.Dockerfile`, `docker/sandbox-requirements.txt` | every Python patch release of the image's minor version, a pydantic update, and before a release | `docker buildx imagetools inspect python:3.12-slim` for the new index digest; resolve the requirements again for x86_64 and aarch64 (see the file's header) and take the hashes from PyPI; build the image and run `python -m orkexa.dynamic.docker --self-test` |
| Documented model limits | `_LIMITS` in `providers/anthropic.py`, `_DOCUMENTED_LIMITS` in `providers/openai.py` and `providers/bedrock.py`, `_DOCUMENTED_MAX_OUTPUT` in `providers/gemini.py` (each with its "checked" date) | a new model of that provider, and before a release | take context window and maximum output only from the model's own documentation page; a model without stated numbers stays out of the table (unknown) |
| Pinned Ollama runtime | `OLLAMA_VERSION` and `RUNTIME_ARTIFACTS` in `src/orkexa/local/install.py` | each Nodestro release | take the archive names and SHA-256 from the release's `sha256sum.txt`; rerun the live tests (`ORKEXA_TEST_OLLAMA_URL`, `-m live tests/test_ollama_live.py`) and the managed-server tests on Linux, macOS and Windows |

The development status classifier (`Development Status :: 3 - Alpha`) changes only
when the project's maturity does, not for a release.
