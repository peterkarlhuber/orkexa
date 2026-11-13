# Releasing

Orkexa has not been published yet, and publishing it on PyPI is not planned: consumers
install it from the repository by commit or tag. The release workflow
(`.github/workflows/release.yml`) is prepared but runs only when a GitHub release is
published, and it cannot upload anything until the one-time setup below is done.

## Versioning and releases

* **Version.** The version in `pyproject.toml` is the single source
  (`orkexa.__version__` reads the installed metadata). Development between releases
  carries a `.devN` suffix; the current version is `0.2.0.dev0`.
* **Semantic versioning from 0.2.** From 0.2.0 on, versions follow
  [SemVer](https://semver.org/): a patch release fixes without changing the API, a
  minor release may add names, and before 1.0 a minor release may also remove or
  change names. Nothing is removed in a patch release.
* **Aliases.** A kept alias (for example `orkexa.retrieval._pdf_pages` for `pdf_pages`)
  is removed only in a minor release, with a CHANGELOG entry that names the alias and
  its replacement; where possible it first warns with a `DeprecationWarning` for at
  least one minor release.
* **Public API.** `tests/test_public_api.py` lists every public name of `orkexa`,
  `orkexa.retrieval`, `orkexa.files`, `orkexa.memory` and `orkexa.embeddings`, the
  deeper paths that Nodestro imports, and checks that every re-export is the same
  object. A new name is added to the snapshot in the same change; a removed or renamed
  one needs a minor release and a CHANGELOG entry. The import contracts
  (`[tool.importlinter]` in `pyproject.toml`, checked with `lint-imports` in CI) keep
  the core modules free of `orkexa.backends.postgres` and `orkexa.testing`.
* **Tabrion (TABRION VER-1, V-B0).** Orkexa is a library: `pyproject.toml` names
  Tabrion only by a range of final versions such as `tabrion[testing,postgres]>=0.1,<0.2`,
  never by a direct URL (RC-1) and never with a pre-release specifier (RC-8). Tabrion is
  not on a package index, so `requirements/tabrion.txt` pins the one Tabrion a standalone
  install uses: exactly one line
  `tabrion @ git+https://github.com/peterkarlhuber/tabrion.git@<40-hex SHA>` on the commit
  of a final Tabrion version (now v0.1.0, `d5821faa8742dea012abda3b663549c038d76543`).
  Install it with `-r`, never `-c` (pip 25.0 refuses a direct URL in a constraints file),
  with pip 25.0 or newer:

    ```bash
    python -m pip install -r requirements/tabrion.txt -e ".[dev]"
    python -m pip install -r requirements/tabrion.txt "orkexa @ git+https://github.com/peterkarlhuber/orkexa.git@<sha>"
    ```

    `tests/architecture/test_tabrion_requirements.py` checks the file and the manifest
    (one pin, canonical URL, full SHA of a final version listed in the test, every range (only `>=0.N[.M],<0.(N+1)`, one minor line)
    of the manifest contains it). Moving the pin to a newer final Tabrion adds its SHA
    to that list in the same change. Nodestro's own Tabrion pin decides for Nodestro
    installs; this file may stay on an older final version of the same minor line.
* **Tags.** Release tags are named `orkexa-vX.Y.Z` (annotated, on the merge commit of
  the release). Until the owner approves a tag, the merge commit's SHA stands for the
  version. A tag is prepared locally and pushed only after that approval, for example:

    ```bash
    git tag -a orkexa-v0.2.0 -m "orkexa 0.2.0" <merge-sha>
    git push origin orkexa-v0.2.0   # only after approval
    ```

    Pushing a tag publishes nothing: `release.yml` starts only when a GitHub release
    is published, its tag check expects `vX.Y.Z` and so refuses an `orkexa-v` tag, and
    its upload job needs the protected `pypi` environment.
* **No PyPI.** No release is uploaded to PyPI. The steps below stay documented for the
  case that the owner decides otherwise; the workflow's tag check then has to accept
  the `orkexa-v` scheme first.

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
5. Create a tag and a GitHub release for it (the workflow still expects `vX.Y.Z`,
   not the `orkexa-vX.Y.Z` scheme above; see "Versioning and releases"). Publishing the release starts
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
