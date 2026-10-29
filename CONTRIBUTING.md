# Contributing to Orkexa

Thank you for helping. This guide covers the development setup, the checks every
change must pass, and the conventions of the repository. By participating you agree
to follow the [code of conduct](CODE_OF_CONDUCT.md).

## Setup

Python 3.10 or newer. From a clone:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pip install --no-deps -e compat/orkexa-claude-compat
```

`.[dev]` installs all provider SDKs, the test and lint tools, `pypdf` and
`sqlite-vec`. Add `.[docs]` to build the documentation.

## Checks

CI runs these on every pull request; run them before you push:

| Purpose | Command |
| --- | --- |
| Lint | `ruff check .` |
| Format | `ruff format --check .` (`ruff format .` fixes it) |
| Type check | `mypy --strict src` and `mypy --strict examples` |
| Tests (offline) | `pytest` |
| Coverage | `pytest --cov=orkexa --cov-report=term-missing` |
| Packaging | `pytest -m packaging tests/packaging --noconftest` (builds wheels, installs them into fresh virtual environments; needs network access to a package index) |
| Build | `python -m build` and `python -m build compat/orkexa-claude-compat` |
| Docs | `mkdocs build --strict` |
| Examples | covered by `pytest` (`tests/test_examples.py`) and `mypy --strict examples` |

CI also runs the suite with the lowest supported dependency versions and audits the
dependencies with `pip-audit`.

## Tests

* The default suite is **offline and deterministic**: no network, no API keys, no paid
  calls. Use `FakeProvider` or the scripted SDK clients in `tests/contract/harness.py`.
* Tests that call a real API are marked `live` and run only with `pytest -m live`;
  they need API keys and cost money. Never make them part of the default run.
* Every bug fix comes with a test that fails without the fix.
* `tests/legacy` is the unchanged `claude-agent` 0.8.0 suite. Do not edit it; it
  proves compatibility.
* Every new example needs a test in `tests/test_examples.py`.

## Conventions

* **Language:** everything in the repository is written in English with American
  spelling: code, comments, docstrings, messages, docs, tests, examples, commit
  messages. Multilingual test data (e.g. German text for the search index) is fine.
* **Compatibility:** the public API of `claude-agent` 0.8.0 (`ClaudeAgent`,
  `AsyncClaudeAgent`, `import claude_agent` through `orkexa-claude-compat`) must keep
  working. Do not rename public symbols.
* **Errors** say what failed, why, whether retrying helps, and how to fix it. They
  never contain credentials.
* **Security defaults** are not weakened without discussion: no keys in logs,
  exceptions, histories or sandboxes, no key sent to another provider or host, SDKs
  never execute tools on their own.
* **Docstrings** use the Google style; public functions and classes have one.
* **Changelog:** add user-visible changes to `CHANGELOG.md` under the unreleased
  version.

## Pull requests

1. Open an issue first for larger changes, so the design can be discussed.
2. Keep pull requests focused; split unrelated changes.
3. Describe what changed and why, and how you tested it.
4. Make sure all checks pass.

Contributions are accepted under the project's [MIT License](LICENSE).
