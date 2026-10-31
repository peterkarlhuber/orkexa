# Pricing

The pricing engine (`orkexa.pricing`). See the [pricing guide](../guides/pricing.md)
for concepts, the serialization format and examples.

::: orkexa.pricing
    options:
      members: false

## Price data

::: orkexa.pricing.ModelPrice

::: orkexa.pricing.TokenRates

::: orkexa.pricing.LongContextTier

::: orkexa.pricing.Tariff

::: orkexa.pricing.TariffOption

::: orkexa.pricing.FeeRate

::: orkexa.pricing.PriceQuality

::: orkexa.pricing.NEUTRAL_CONDITIONS

::: orkexa.pricing.TOKENS_PER_RATE_UNIT

## Snapshots and resolution

::: orkexa.pricing.PriceSnapshot

::: orkexa.pricing.SNAPSHOT_FORMAT

::: orkexa.pricing.PriceLookup

::: orkexa.pricing.PriceResolver

::: orkexa.pricing.PricingBasis

::: orkexa.pricing.pricing_basis

::: orkexa.pricing.default_resolver

::: orkexa.pricing.BUILTIN_SNAPSHOT

## Cost calculation

::: orkexa.pricing.MeteredUsage

::: orkexa.pricing.calculate_cost

::: orkexa.pricing.CostBreakdown

::: orkexa.pricing.CostComponent

::: orkexa.pricing.PriceReference

::: orkexa.pricing.round_cost

## Policies

::: orkexa.pricing.PricePolicy

## Legacy API

::: orkexa.pricing.ModelPricing

::: orkexa.pricing.MODEL_PRICING

::: orkexa.pricing.PROVIDER_PRICING

::: orkexa.pricing.register_pricing

::: orkexa.pricing.resolve_pricing

## Errors

`PricingError` and `PriceNotFoundError` live in `orkexa.errors` (and are re-exported
from `orkexa.pricing`); they are documented in the [errors reference](errors.md).
