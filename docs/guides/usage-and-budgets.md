# Usage, pricing and budgets

Every API response that reports usage is recorded immediately, also when the turn
fails later. A `UsageRecord` carries `provider`, `model`, `agent`, `turn_id`,
`request_id`, normalized token counts and the cost.

## Normalized counts (no double counting)

| Field | Anthropic | OpenAI | Gemini | Chat Completions |
| --- | --- | --- | --- | --- |
| `input_tokens` (uncached) | `input_tokens` | `input_tokens - cached_tokens - cache_write_tokens` | `prompt_token_count - cached_content_token_count` | `prompt_tokens - cached_tokens` |
| `cache_read_input_tokens` | `cache_read_input_tokens` | `cached_tokens` | `cached_content_token_count` | `cached_tokens` |
| `output_tokens` (incl. reasoning) | `output_tokens` | `output_tokens` | `candidates + thoughts` | `completion_tokens` |
| `thinking_tokens` (part of output) | thinking tokens | `reasoning_tokens` | `thoughts_token_count` | `reasoning_tokens` |

## Prices

Orkexa ships Anthropic prices. For any other provider the cost is `None` (unknown,
not zero) until you register prices:

```python
from orkexa import ModelPricing, register_pricing

register_pricing("openai", "gpt-5", ModelPricing.of("1.25", "10", "1.25", "1.25", "0.125"))
agent = Agent(model="gemini:gemini-2.5-flash", pricing=ModelPricing.of("0.3", "2.5", "0.3", "0.3", "0.03"))
```

`UsageRecord.pricing_source` says where a price came from: `"builtin"` (Orkexa's
Anthropic table), `"registered"`, `"custom"` (`pricing=`) or `None` (unknown). Registered prices are looked up for the
configured model id first, then for the id the API reported.

## Budgets

```python
from orkexa.dynamic import BudgetTracker

budget = BudgetTracker("5.00", on_unpriced="reject")
```

* Before every request the tracker refuses to start when the budget is used up.
* After every response it records the usage and raises `BudgetExceededError` when the
  limit is exceeded. One request can overshoot slightly: its cost is known only after
  it ran. Server-tool fees are added only where the provider reports them (Anthropic
  web search/fetch).
* `on_unpriced` decides what happens to models without a price: `"ignore"` (count as
  0; the 0.8.0 behavior and the default without a limit), `"warn"` (default with a
  limit; one `UnpricedUsageWarning` per model), `"reject"` (`UnpricedModelError`
  before the request is sent).

One tracker can be shared by agents of every provider (`usage_by_provider()`,
`usage_by_agent()`, `usage_by_model()`).
