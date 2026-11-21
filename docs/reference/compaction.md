# Context compaction

`orkexa.core.compaction`: fit a long conversation into a model's context window by trimming or summarizing the
older turns. How it works and how untrusted data is kept framed: [Long conversations](../guides/compaction.md).

```python
from orkexa.core.compaction import Turn, compact, context_budget

budget = context_budget(8192, output_reserve=2048, reserved=600)
result = compact([Turn("user", "...", ref=1), Turn("assistant", "...", ref=2), ...], budget=budget)
```

::: orkexa.core.compaction.compact

::: orkexa.core.compaction.acompact

::: orkexa.core.compaction.context_budget

::: orkexa.core.compaction.estimate_tokens

::: orkexa.core.compaction.split_untrusted

::: orkexa.core.compaction.Turn

::: orkexa.core.compaction.Compacted

::: orkexa.core.compaction.SUMMARY_INSTRUCTIONS
