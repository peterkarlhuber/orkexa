# Examples

The scripts live in [`examples/`](https://github.com/peterkarlhuber/orkexa/tree/main/examples).
Each one runs on its own (`python examples/01_text_agents.py`); providers without an
API key are skipped. All twelve run end to end in the offline test suite
(`tests/test_examples.py`) against the real adapters with scripted SDK clients, and
CI type-checks them with `mypy --strict`.

--8<-- "examples/README.md"
