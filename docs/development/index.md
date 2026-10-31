# Development

The full contributor guide is
[CONTRIBUTING.md](https://github.com/peterkarlhuber/orkexa/blob/master/CONTRIBUTING.md)
in the repository. The commands at a glance:

```bash
pip install -e ".[dev]"                               # setup (all SDKs, test and lint tools)
ruff check . && ruff format --check .                 # lint and format
mypy --strict src && mypy --strict examples           # type check
pytest                                                # offline tests, deterministic
pytest --cov=orkexa --cov-report=term-missing         # with coverage
pytest -m packaging tests/packaging --noconftest      # build and install in fresh venvs
pytest -m live                                        # real APIs: needs keys, costs money
python -m build                                       # sdist and wheel in dist/
pip install -e ".[docs]" && mkdocs serve              # docs at http://127.0.0.1:8000
```

## Repository layout

| Path | Contents |
| --- | --- |
| `src/orkexa/` | The library: `core/` (runtime, types, capabilities), `providers/` (adapters), `dynamic/` (generated tools and agents), storage, usage, evaluation |
| `tests/` | Offline tests; `contract/` checks every adapter against the real SDK signatures; `anthropic/` drives the agents through a fake Anthropic client; `packaging/` builds and installs the distribution |
| `examples/` | Runnable examples, each tested offline |
| `docs/` | This documentation (MkDocs Material) |
| `assets/branding/` | The official logo and its derivatives |
| `docker/` | The image for `DockerSandbox` |

## Continuous integration

`.github/workflows/ci.yml` runs on every push to the default branch and every pull
request, with read-only repository permissions: lint, format and type checks and the
test suite on Python 3.10 to 3.13, the suite with the lowest supported dependency
versions, building both distributions and installing them into fresh virtual
environments, a strict docs build, and `pip-audit` over all runtime dependencies.
Dependabot proposes grouped dependency and GitHub Actions updates.
