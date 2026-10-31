# ADR 0008 – Pricing engine: one immutable snapshot per request, unknown is not zero

Status: accepted

## Context

Orkexa estimated costs from a flat `ModelPricing` table with Anthropic prices and a
global registry for other providers. It could not express aliases, validity periods,
tariffs or provenance, priced OpenAI `flex`/`priority` usage at standard rates, left
server-tool fees and audio tokens out silently and could not tell which price version
produced an amount. Work package PRICING-1 asked for a production-ready, versionable
pricing engine without a runtime dependency on external pricing services.

## Decision

* **Data model.** `ModelPrice` describes one model of one provider completely:
  Decimal token rates per million tokens, explicit currency, explicit aliases, opt-in
  dated ids, long-context tiers, tariffs, fees, special token rates, provenance,
  version, validity, check date and quality. Everything is immutable.
* **Snapshots are the unit of versioning.** A `PriceSnapshot` is an immutable,
  versioned, JSON-serializable set of prices; updates create new snapshots. This
  replaces the "price version" field per request suggested in the work package: the
  snapshot version names the complete basis, the entry's `version` its origin.
* **Resolvers are instances, not globals.** `PriceResolver` holds the active snapshot
  and swaps it atomically. The process-wide `default_resolver()` exists only to keep
  the legacy API (`MODEL_PRICING`, `register_pricing()`) working; it mirrors the legacy
  dictionaries instead of replacing them.
* **One pinned basis per request.** The runtime pins a `PricingBasis` before each
  request and uses it for the budget pre-check and for the bill, so concurrent catalog
  updates cannot mix versions.
* **Tariffs instead of special cases.** Batch, service tiers, fast mode, data
  residency, regions and contract variants are tariff dimensions with options
  (multiplier or replacement rates). A reported value the price does not list makes
  the cost unknown; only documented neutral values (`default`, `standard`, `global`)
  need no declaration.
* **Completeness is explicit.** `CostBreakdown.missing` lists every component that could
  not be priced; `total` is `None` unless the list is empty, `known_total` keeps the
  known part. Usage records and summaries expose both.
* **Resolution order** exact id, explicit alias, opt-in dated id; never a prefix,
  never another provider. Inside agents the reported id is tried before the
  configured one (deployment names), and the match kind is recorded.

## Deviations from the work package

* Prices are not fetched from anywhere. The built-in catalog contains Anthropic list
  prices only (checked against the official page); other providers' prices are data the
  application supplies. This keeps Orkexa free of network and hosting dependencies.
* Charges the providers' usage data does not report (code-execution container hours,
  context-cache storage) are not estimated; they mark a cost as incomplete when usage
  hints at them. Estimating them would produce seemingly precise but wrong amounts.
* Budgets stay soft limits; the engine cannot make them hard billing limits because
  costs are known only after a request.
* Snapshots are not persisted by Orkexa; they are serializable for external storage.

## Consequences

* Costs are reproducible: a usage record names the snapshot version, the price entry,
  the tariff and every component.
* Some costs that used to be shown are now `None` (non-standard OpenAI tiers, unpriced
  tool fees), because they were wrong or incomplete; applications fix this by adding the
  missing tariff or fee to their prices.
* The legacy API keeps working and is covered by its original tests.
