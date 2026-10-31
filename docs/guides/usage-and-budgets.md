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

Costs come from the [pricing engine](pricing.md). Orkexa ships Anthropic prices; for
any other provider the cost is `None` (unknown, not zero) until you supply prices:

```python
from orkexa import ModelPricing, register_pricing
from orkexa.pricing import PriceResolver, PriceSnapshot

register_pricing("openai", "gpt-5", ModelPricing.of("1.25", "10", "1.25", "1.25", "0.125"))
agent = Agent(model="gemini:gemini-2.5-flash", pricing=ModelPricing.of("0.3", "2.5", "0.3", "0.3", "0.03"))

# Your own versioned catalog, isolated from the process-wide registry:
resolver = PriceResolver(PriceSnapshot.from_json(catalog_json))
agent = Agent(model="openai:gpt-5", pricing=resolver)
tracker = UsageTracker(price_resolver=resolver)  # for every agent sharing it
```

`UsageRecord.pricing_source` says where a price came from: `"builtin"` (Orkexa's
Anthropic table), `"registered"`, `"custom"` (`pricing=`), your catalog's `source`, or
`None` (unknown). `pricing_version` names the snapshot, `pricing_quality` its quality,
`pricing_conditions` the tariff applied (batch, service tier, fast mode, data
residency) and `cost` the itemized breakdown. The id the API reported is resolved
first, then the configured one.

A cost that could only be partly priced (an unpriced server tool, audio tokens
without an audio rate, an unknown service tier) is `None` as well; `known_cost_usd`
and `missing_cost_components` show the known part and what is missing.
`UsageSummary.known_cost_usd` and `cost_complete` do the same for aggregates.

## Budgets

```python
from orkexa.dynamic import BudgetTracker

budget = BudgetTracker("5.00", on_unpriced="reject")
```

* Before every request the tracker refuses to start when the budget is used up
  (`spent_usd >= limit_usd`; a budget of 0 sends no request at all).
* After every response it records the usage and raises `BudgetExceededError` when the
  limit is exceeded. `spent_usd` counts complete costs and the known part of
  incomplete ones.
* `price_resolver=` gives the tracker its own catalog; `price_policy=PricePolicy(...)`
  with `on_price_issue="warn" | "reject"` rejects or reports stale, low-quality or
  incomplete price bases before and after requests (see the
  [pricing guide](pricing.md#budgets-and-price-policies)).
* The limit is a **soft limit**, not a hard billing limit: a request's cost is known
  only after it ran, so the last admitted request can overshoot it, and agents that
  share a tracker and run concurrently can each overshoot it by one request. Keep a
  margin, run budget-critical agents sequentially, and use the provider's spend
  limits for hard caps.
* A native batch (`Evaluator.evaluate_many(batch=True)`) is admitted as a whole like
  one request: it is not submitted when the budget is used up or, with
  `on_unpriced="reject"`, when the model has no price. Its cost is only known when it
  has ended, so a batch with *n* items can overshoot the limit by up to *n* requests
  (at batch prices). All of its results are recorded before `BudgetExceededError` is
  raised, so the ledger stays complete. To keep a tight limit, evaluate with
  `batch=False` (every request is admitted separately) or split the cases into smaller
  batches.
* Batch usage counts in `agent.usage` and the tracker, not in `agent.last_turn_usage`,
  which covers the most recent `ask()` only.
* `on_unpriced` decides what happens to models without a price: `"ignore"` (count as
  0; the default without a limit), `"warn"` (default with a
  limit; one `UnpricedUsageWarning` per model), `"reject"` (`UnpricedModelError`
  before the request is sent, and after a request whose cost turned out unknown or
  incomplete; the request is recorded first).

One tracker can be shared by agents of every provider (`usage_by_provider()`,
`usage_by_agent()`, `usage_by_model()`, `usage_by_turn()`, `usage_by_workflow()`).

## Workflows

`workflow(label)` labels every request sent inside the block, including agents that
run as tools or in asyncio tasks. `Orchestrator.ask()` opens a workflow per run
(`orchestrator.last_workflow`), so the cost of one multi-agent run is one entry:

```python
from orkexa.usage import workflow

with workflow("nightly-report"):
    researcher.ask("...")
    writer.ask("...")
tracker.usage_by_workflow()["nightly-report"].estimated_cost_usd
```

Requests outside any workflow are listed under `None`, like requests without a turn
(batches) in `usage_by_turn()`.
