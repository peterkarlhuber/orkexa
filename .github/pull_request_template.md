## Summary

<!-- What does this change and why? Link the issue if there is one. -->

## Changes

-

## Testing

<!-- Which checks did you run? Paste the relevant output. -->

- [ ] `ruff check .` and `ruff format --check .`
- [ ] `mypy --strict src` and `mypy --strict examples`
- [ ] `pytest` (offline)
- [ ] `mkdocs build --strict` (if docs changed)
- [ ] `pytest -m packaging tests/packaging --noconftest` (if packaging changed)

## Checklist

- [ ] Tests cover the change (a bug fix has a test that fails without it)
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] Docs updated if behavior changed
- [ ] No API keys, personal data or other secrets in code, tests or output
- [ ] The `claude-agent` 0.8.0 compatibility suite (`tests/legacy`) is unchanged
