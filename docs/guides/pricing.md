# Pricing engine

Orkexa calculates the cost of every model request from the usage the provider
reported and a price it can name: which price list, which version, which tariff.
When it cannot name one, the cost is **unknown** (`None`), never 0 USD.

The engine lives in `orkexa.pricing` and has no runtime dependency on a pricing
service, a database or a hosting environment. Applications bring their own price
catalogs as plain Python objects or JSON; Orkexa resolves, validates and calculates.

```python
from decimal import Decimal
from orkexa.pricing import MeteredUsage, ModelPrice, PriceResolver, TokenRates

resolver = PriceResolver(
    [
        ModelPrice(
            provider="openai",
            model="gpt-5",
            rates=TokenRates.of("1.25", "10", "0.125"),
            source="acme-contract",
            quality="contract",
        )
    ],
    version="acme-2026-11",
)
cost = resolver.calculate("openai", "gpt-5", MeteredUsage(input_tokens=120_000, output_tokens=8_000))
cost.total  # Decimal('0.23')
cost.components  # input 0.15, output 0.08
cost.references[0]  # snapshot acme-2026-11, source acme-contract, quality contract
```

`examples/12_pricing_snapshots.py` shows the complete flow offline.

## Principles

* **Usage is authoritative.** Costs are computed from the token counts and units the
  provider reported, never from text.
* **Unknown stays unknown.** A missing price, an unknown tariff, an unpriced fee or an
  incomplete usage report is listed in `CostBreakdown.missing`; `total` is then `None`.
  The sum of what *could* be priced is still available as `known_total`.
* **No guessing.** No prefix matches, no prices across providers, no invented prices
  for unknown models, no standard rate for an unknown tariff.
* **One price basis per request.** A request is priced with exactly one immutable
  snapshot, even while prices are updated concurrently. Costs already calculated never
  change when prices change later.
* **Exact amounts.** Every amount is a `Decimal`; Orkexa never rounds internally (use
  `round_cost()` at the boundary where you display or book an amount).

## Data model

| Type | Purpose |
| --- | --- |
| `TokenRates` | Prices per **million** tokens (`TOKENS_PER_RATE_UNIT`): `input`, `output` (includes reasoning), `cache_read`, `cache_write_5m`, `cache_write_1h`, and `special` rates for token classes such as `audio_input`. A rate of `None` means unknown. |
| `LongContextTier` | Rates for prompts longer than `above_input_tokens` (uncached input + cache reads + cache writes). The tier with the highest threshold below the prompt length prices the **whole** request. |
| `Tariff` / `TariffOption` | A pricing dimension (`service_tier`, `speed`, `inference_geo`, `region`, ...) with its allowed values. An option multiplies every token rate (`multiplier`) or replaces the rates (`rates`, with its own `long_context`). |
| `FeeRate` | A non-token charge: `price` per `per` units of a reported unit (`web_search_request`, `web_search_call`, `google_search_query`). `0` declares a unit free. |
| `ModelPrice` | The complete price of one model of one provider: rates, `currency` (ISO 4217, default `USD`), explicit `aliases`, `match_dated_versions`, tiers, tariffs, fees, `source`, `source_url`, `version`, `valid_from`/`valid_until` (timezone-aware, end exclusive), `checked_at`, `quality` and `notes`. |
| `PriceSnapshot` | An immutable, versioned set of `ModelPrice`s: one price basis. |
| `MeteredUsage` | The billable quantities of one request. |
| `CostBreakdown` | Components, missing parts, price references, `known_total` and `total`. |

`PriceQuality` is one of `"verified"` (checked against the official price list),
`"contract"` (an individual agreement), `"unverified"` (the default for prices you
register) and `"estimated"`.

Floats are rejected everywhere a price is expected: write `"0.10"`, not `0.10`.

### Supported cost components

| Component | How it is priced |
| --- | --- |
| Input tokens | `rates.input` (uncached input only) |
| Output tokens incl. reasoning/thinking | `rates.output`; `thinking_tokens` is the reasoning share and is never charged again |
| Cache reads | `rates.cache_read`, separate from input |
| Cache writes, 5 minutes / 1 hour | `rates.cache_write_5m` / `rates.cache_write_1h`; writes without a reported lifetime are priced only when both rates are equal |
| Long context | `long_context` tiers |
| Batch, service tiers, fast mode, data residency, regions, contract variants | `tariffs` (multipliers or replacement rates) |
| Server tools and other per-unit fees | `fees` |
| Audio and image tokens | `rates.special` (`audio_input`, `audio_cache_read`, `audio_output`, `image_output`, ...) |
| Several model calls of one request (Anthropic iterations, compaction, advisor) | each priced at its own model, then combined |
| Several requests of a workflow | `UsageTracker.usage_by_workflow()` / `CostBreakdown.combine()` |

Kinds of charge Orkexa does not model (see [limitations](#known-limitations)) are
never shown as calculated: if the usage reports them, the cost is incomplete.

## Resolution

`PriceSnapshot.lookup(provider, model, at=...)` and `PriceResolver.resolve(...)`
resolve deterministically, in this order:

1. the exact model id,
2. an explicit alias,
3. the undated id of a dated id (`<id>-YYYYMMDD`, `<id>-YYYY-MM-DD`, `<id>@YYYYMMDD`),
   only for prices with `match_dated_versions=True`.

Only prices of the same provider are considered, and only those valid at `at`
(default: now). A snapshot rejects two prices for the same provider and id (or alias)
with overlapping validity, so a lookup never has two answers. The result is a
`PriceLookup` with `price`, `matched_by`, `snapshot_version` and, when nothing was
found, a `reason`; `lookup.require()` raises `PriceNotFoundError` instead.

Inside an agent, the id the API reported is resolved first; if it is unknown, the id
the agent was configured with (`requested_model`, e.g. an Azure deployment name) is
used and marked `matched_by="requested"`.

## Tariffs

Usage reports conditions such as `service_tier="batch"`, `speed="fast"` or
`inference_geo="us"`. For each tariff of the price, the reported value (or the
tariff's `default` when nothing is reported) must be one of its options; the
multipliers of all applied options are multiplied. A reported value the price does
not list makes every token component unknown (`tariff:service_tier=priority`).

A dimension the price does not declare accepts only neutral values
(`NEUTRAL_CONDITIONS`: `service_tier` `standard`/`default`, `speed` `standard`,
`inference_geo` `global`). `standard` and `default` are synonyms for a declared
service-tier tariff whose default is one of them.

## Snapshots and versions

A `PriceSnapshot` never changes. Updating prices means a new snapshot with a new
version: `snapshot.with_prices([...])` (version `"<old>+<content hash>"` unless you
pass one), `resolver.register(...)` or `resolver.publish(snapshot)`.

```python
snapshot = PriceSnapshot("acme-2026-11", prices, source="acme-pricing-service")
text = snapshot.to_json()  # store it anywhere
same = PriceSnapshot.from_json(text)  # equal, same content_hash()
```

### Serialization format

```json
{
  "format": "orkexa.price-snapshot/1",
  "version": "acme-2026-11",
  "created_at": "2026-10-30T00:00:00+00:00",
  "source": "acme-pricing-service",
  "prices": [
    {
      "provider": "openai",
      "model": "gpt-5",
      "currency": "USD",
      "rates": {"input": "1.25", "output": "10", "cache_read": "0.125"},
      "aliases": ["gpt-5-prod"],
      "match_dated_versions": true,
      "long_context": [{"above_input_tokens": 272000, "rates": {"input": "2.5", "output": "15"}}],
      "tariffs": [
        {"dimension": "service_tier", "default": "default",
         "options": [{"value": "default", "multiplier": "1"}, {"value": "flex", "multiplier": "0.5"}]}
      ],
      "fees": [{"unit": "web_search_call", "price": "10", "per": 1000}],
      "source": "acme-contract",
      "version": "2026-11",
      "valid_from": "2026-11-01T00:00:00+00:00",
      "checked_at": "2026-10-30T00:00:00+00:00",
      "quality": "contract"
    }
  ]
}
```

Amounts are decimal **strings**, timestamps ISO 8601 with offset. Optional keys may be
omitted; unknown keys and other `format` values are rejected (`PricingError`), so a
newer or corrupted file is never half-read. `content_hash()` (SHA-256 of the canonical
prices) detects modified copies. `CostBreakdown`, `MeteredUsage` and `UsageRecord`
have `to_dict()`/`from_dict()` with the same conventions.

## Resolvers

A `PriceResolver` holds one active snapshot and keeps previous ones by version
(`max_history`, default 32). Resolvers are isolated: two instances never share prices,
and none of them touches global state.

| Method | Purpose |
| --- | --- |
| `snapshot` | The active snapshot. |
| `publish(snapshot)` | Atomically make a snapshot active (versions must be unique per content). |
| `register(*prices, version=None)` | Add or replace prices; publishes the derived snapshot. |
| `get_snapshot(version)`, `versions` | Earlier snapshots, for audits and recalculation. |
| `resolve(provider, model, at=None, snapshot=None)` | A `PriceLookup`. |
| `basis(snapshot=None)` | A pinned `PricingBasis`. |
| `calculate(provider, model, usage, requested_model=None, at=None, snapshot=None)` | Resolve and price against one snapshot. |

`PricingBasis(snapshot, override=None)` is what an agent pins per request. Its
`override` is an agent's `pricing=` price.

### Integrating external price data

1. Produce a snapshot in your system (database, pricing service, contract file) and
   serialize it with the format above, or build `ModelPrice` objects directly.
2. Create a resolver: `resolver = PriceResolver(PriceSnapshot.from_json(text))`.
3. Hand it to Orkexa:
   * per agent: `Agent(model=..., pricing=resolver)` (or a `PriceSnapshot`),
   * for everything sharing a tracker: `UsageTracker(price_resolver=resolver)` or
     `BudgetTracker(limit, price_resolver=resolver)`,
   * without agents: `resolver.calculate(provider, model, MeteredUsage(...))`.
4. To update prices, `resolver.publish(new_snapshot)`. Running requests keep their
   pinned snapshot; new requests use the new one.
5. Persist `UsageRecord.to_dict()` (with `pricing_version` and the itemized `cost`)
   next to the snapshot versions you keep, to reproduce every amount later.

Which price wins: an agent's `pricing=` (a `ModelPrice` override that matches the
provider and model, a legacy `ModelPricing`, or a whole resolver/snapshot), then the
tracker's `price_resolver`, then the process-wide `default_resolver()`.

## Usage records and summaries

Every `UsageRecord` carries the price basis: `provider`, `model` (reported) and
`requested_model`, `pricing_source`, `pricing_version` (snapshot), `pricing_quality`
(the lowest involved), `pricing_conditions` (tariff options applied),
`estimated_cost_usd` (complete cost or `None`), `known_cost_usd`,
`missing_cost_components`, `pricing_note` and the itemized `cost`.

`UsageSummary.estimated_cost_usd` is `None` as soon as one request's cost is unknown
or incomplete; `known_cost_usd` is the sum of what is known (a lower bound),
`unpriced_requests` counts the affected requests, `missing_cost_components` lists
what is missing and `cost_complete` says whether the total is complete. Records
written before these fields existed aggregate unchanged.

## Budgets and price policies

`BudgetTracker` prices through the same engine and adds:

* `price_resolver=`: the catalog for agents without their own `pricing=`;
* `price_policy=PricePolicy(max_age=..., accepted_quality=..., allow_incomplete=...)`:
  which prices are acceptable; checked before each request (the resolved price) and
  after it (every price used for the bill);
* `on_price_issue="ignore" | "warn" | "reject"` (default `"warn"`): a violation warns
  once per provider, model and issue (`UnreliablePriceWarning`) or raises
  `UnreliablePriceError`: before the request (nothing is sent) or after it (the
  request is recorded first);
* `spent_usd` counts complete costs and the known part of incomplete ones;
* `on_unpriced="reject"` also raises after a request whose cost turned out unknown or
  incomplete (it is recorded first).

Defaults are unchanged (`on_unpriced` `"ignore"` without a limit, `"warn"` with one;
the default policy accepts every price).

!!! warning "Budgets are not hard billing limits"
    A request's cost is known only after it ran. The last admitted request can
    overshoot the limit, concurrent agents can each overshoot it by one request, and a
    batch by all of its items. Use the provider's spend limits for hard caps.

## Provider data

| Provider | Reported for pricing |
| --- | --- |
| Anthropic | tokens per iteration and model, cache write lifetimes, `service_tier` (`batch` for batches), `speed`, `inference_geo`, web search/fetch, code execution (only without web tools) |
| OpenAI (Responses) | tokens, cache reads/writes, reasoning, the `service_tier` OpenAI reports (else the requested one), audio tokens, hosted-tool calls (`web_search_call`, `file_search_call`, `code_interpreter_call`, `image_generation_call`) |
| Chat Completions / compatible | tokens, cache reads, reasoning, reported `service_tier`, audio tokens |
| Gemini | tokens incl. thoughts, cached tokens, audio input/cache/output and image output tokens, Google Search grounding (`google_search_query`, `google_search_grounded_prompt`) |
| Bedrock | uncached input, output, cache reads, cache writes split by lifetime (5 minutes / 1 hour), the `service_tier` Bedrock reports, and always `aws_region` (the source region) |

## Amazon Bedrock

Bedrock is its own price source. Orkexa ships **no** Bedrock prices, and Anthropic's
first-party prices in `BUILTIN_SNAPSHOT` are never used for Claude on Bedrock:
prices are looked up for provider `"bedrock"` only. Until you register a price, the
cost of every Bedrock request is unknown (`None`), never 0.

AWS prices depend on the region and on the kind of model id
([Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/)), so a Bedrock
price has to say what it applies to:

* **Per model id or profile id.** The usage record carries the id the agent used
  (`eu.anthropic.claude-sonnet-5-5`, `anthropic.claude-sonnet-4-6`, an ARN), and the
  lookup is exact. Global cross-Region profiles are priced differently from
  geographic and in-region use (AWS documents roughly 10 % savings for global
  profiles, see
  [global cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/global-cross-region-inference.html)),
  so register `global.` and `eu.` ids as separate prices. Use `aliases=` for ARNs
  of the same model and price.
* **Per region.** Every Bedrock usage record carries the tariff condition
  `aws_region`: the source region of the client, which is the region AWS prices a
  cross-Region request by
  ([cross-Region inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)).
  A price must declare the regions it covers with a tariff; a region it does not
  list, and a price without an `aws_region` tariff, make the cost unknown
  (`tariff:aws_region=eu-west-3`).
* **Per service tier.** `BedrockOptions(service_tier="priority")` or `"flex"` is
  reported back as `service_tier` and needs a tariff option; `default` (or no
  report) is the standard price. Any other reported tier, such as `reserved`, is
  unknown unless the price lists it.
* **Cache tokens** are separate from input: `input_tokens` holds only uncached input,
  cache reads use `cache_read`, and cache writes use `cache_write_5m` /
  `cache_write_1h` (Bedrock reports the lifetime of each write).

```python
from orkexa import Agent, BedrockOptions
from orkexa.pricing import ModelPrice, Tariff, TokenRates

price = ModelPrice(
    provider="bedrock",
    model="eu.anthropic.claude-sonnet-5-5",
    # Illustrative numbers in USD per million tokens; copy your own from the AWS price list.
    rates=TokenRates.of("3.30", "16.50", "0.33", cache_write_5m="4.125", cache_write_1h="6.60"),
    tariffs=(
        Tariff.of("aws_region", "eu-central-1", **{"eu-west-1": "1", "eu-west-3": "1"}),
        Tariff.of("service_tier", "default", priority="1.75", flex="0.5"),
    ),
    source="my-aws-price-list",
    quality="unverified",
)
agent = Agent(
    provider="bedrock",
    model="eu.anthropic.claude-sonnet-5-5",
    options=BedrockOptions(region_name="eu-central-1"),
    pricing=price,
)
```

A region with different rates gets a `TariffOption` with its own `rates` instead of a
multiplier. For catalogs shared by many agents, put the prices into a
`PriceSnapshot` and pass a `PriceResolver` (see [Resolvers](#resolvers)).

Anthropic models on Bedrock are billed through AWS Marketplace and appear on the AWS
bill under the model provider. Orkexa's figures are estimates from the prices you
register; the AWS bill is authoritative.

## Built-in prices

`BUILTIN_SNAPSHOT` (version `orkexa-builtin-2026-10-31`) holds Anthropic's first-party
Claude API list prices, checked against the
[official price list](https://platform.claude.com/docs/en/about-claude/pricing) on
2026-10-31 (`quality="verified"`): standard rates, cache writes and reads, the Claude
Haiku 5.5 long-context tier, batch (50 %), fast mode (2x, Opus 5.5 / Opus 5 /
Opus 4.8), US-only data residency (1.1x, Claude 4.6 and later), web search ($10 per
1,000) and web fetch (free). Priority Tier is contractual and therefore unknown.
Orkexa ships no prices for other providers or for Bedrock/Vertex; these prices are
never applied to the `bedrock` provider (see [Amazon Bedrock](#amazon-bedrock)).

## Concurrency

Snapshots and prices are immutable and can be shared between threads and asyncio
tasks. `PriceResolver.publish()`/`register()` replace the active snapshot atomically;
readers never see a partially updated catalog. Agents pin one `PricingBasis` per
request, so the pre-request budget check and the bill use the same snapshot even if
it is replaced while the request runs. Calculations use a local decimal context and
do not depend on the global one.

## Errors

| Error | When |
| --- | --- |
| `PricingError` (`ValueError`) | invalid price data, overlapping prices in a snapshot, malformed or unsupported serialized data |
| `PriceNotFoundError` (`PricingError`, `LookupError`) | `PriceLookup.require()` without a price |
| `UnpricedModelError` | budget with `on_unpriced="reject"`: no price before a request, or an unknown/incomplete cost after it |
| `UnreliablePriceError` (`UnpricedModelError`) | budget with `on_price_issue="reject"`: a price fails the policy |
| `UnpricedUsageWarning`, `UnreliablePriceWarning` | the `"warn"` variants |

`calculate_cost()` itself never raises for missing prices; it reports them.

## Backward compatibility and migration

* `ModelPricing`, `MODEL_PRICING`, `PROVIDER_PRICING`, `register_pricing()` and
  `resolve_pricing()` keep their behavior. The legacy dictionaries feed the
  `default_resolver()`; changes to them are picked up by the next request.
  `register_pricing()` also accepts a full `ModelPrice`.
* `pricing=ModelPricing(...)` still applies to every request of an agent;
  `ModelPricing.to_model_price(provider, model)` converts it explicitly.
* `UsageRecord` and `UsageSummary` gained fields with defaults; existing code that
  constructs or reads them keeps working.
* Behavior changes (corrections):
  * Claude Sonnet 5.5 cache reads cost $0.10/MTok (the official 0.05x rate), not
    $0.20.
  * Anthropic fast mode and US data residency are now priced (they were unknown).
  * OpenAI and compatible responses that report a non-standard `service_tier`
    (`flex`, `priority`, ...) are unknown unless the price has that tariff; they were
    priced at standard rates.
  * Fee-bearing hosted tools (OpenAI web search etc., Gemini grounding), audio/image
    tokens and Anthropic code execution without web tools make a cost incomplete
    unless the price declares them; before, they were silently left out.
  * The reported model id is resolved before the configured one.
  * `pricing_source` is set whenever a price was used, also when the cost is
    incomplete.
  * `BudgetTracker.spent_usd` includes the known part of incomplete costs, and
    `on_unpriced="reject"` also raises after a request whose cost turned out unknown.
* Custom providers: `Provider.usage_record(pricing=...)` now receives a pinned
  `PricingBasis`; pass it on to `record_token_usage(custom_pricing=...)` as before.
  A custom `has_pricing()` override is still honored by the budget pre-check.
* Custom trackers that override only `before_request()` keep working; override
  `check_request()` to see the full `PriceLookup`.

## Known limitations

* Orkexa ships Anthropic prices only; every other price must be supplied, including
  Claude on Bedrock, per model or profile id and `aws_region`.
* Charges that usage does not report are not modeled: Anthropic code-execution
  container hours, Gemini context-cache storage per hour, OpenAI container and vector
  store storage, and per-call fees of search-enabled Chat Completions models. Where
  usage hints at them, the cost is incomplete.
* Long-context tiers use the total prompt length (input + cache reads + cache writes);
  a provider that counts differently needs adjusted thresholds.
* Gemini's per-modality details are counted from `prompt_tokens_details`,
  `cache_tokens_details` and `candidates_tokens_details`; reasoning and tool-use
  prompt tokens are billed at the text rates.
* Currencies are never converted; a USD budget ignores amounts in other currencies
  (they count as incomplete).
* Budgets are soft limits (see above).
