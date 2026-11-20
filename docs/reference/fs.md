# Path guard

`orkexa.fs`: read files below one granted folder and nothing else. How it works, the error codes and the
remaining risks are in the [path guard guide](../guides/path-guard.md); the attack catalog for conformance tests
is [`orkexa.testing.pathguard`](testing.md#the-path-guard-attack-catalog).

```python
from orkexa.fs import SafeRoot

root = SafeRoot("/home/ada/Documents/Offer 2026", protected=[data_dir])
text = root.extract_text("specs/offer.pdf").text
```

::: orkexa.fs.SafeRoot

::: orkexa.fs.Entry

::: orkexa.fs.TextBudget

::: orkexa.fs.ExtractedText

::: orkexa.fs.FsError

::: orkexa.fs.FS_ERROR_CODES

::: orkexa.fs.DEFAULT_BLOCKED

::: orkexa.fs.root_problem

::: orkexa.fs.system_folders

::: orkexa.fs.split_relative
