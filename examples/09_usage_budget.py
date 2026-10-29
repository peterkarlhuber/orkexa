"""9. Usage, cost and budget across providers.

Orkexa ships prices for Anthropic models only. For other providers the cost is
*unknown* (``estimated_cost_usd is None``), never silently zero; register your own
prices with :func:`orkexa.register_pricing` or ``pricing=``. A
:class:`~orkexa.dynamic.BudgetTracker` decides deterministically what to do with
unpriced requests: ``on_unpriced="ignore" | "warn" | "reject"``.
"""

from __future__ import annotations

from _setup import available_models

from orkexa import Agent, ModelPricing, register_pricing
from orkexa.dynamic import BudgetExceededError, BudgetTracker, UnpricedModelError


def main() -> None:
    models = available_models()
    # Example prices (USD per million tokens): input, output, cache write 5m, cache write 1h, cache read.
    for model in models:
        provider, _, model_id = model.partition(":")
        if provider != "anthropic":
            register_pricing(provider, model_id, ModelPricing.of("1.25", "10", "1.25", "1.25", "0.125"))

    budget = BudgetTracker("0.50", on_unpriced="reject")
    for model in models:
        agent = Agent(model=model, usage_tracker=budget)
        try:
            agent.ask("Name three rivers in Austria.")
        except UnpricedModelError as exc:
            print(f"refused before sending: {exc}")
        except BudgetExceededError as exc:
            print(f"budget: {exc}")

    for provider, usage in budget.usage_by_provider().items():
        print(
            f"{provider}: in={usage.input_tokens} out={usage.output_tokens} "
            f"cached={usage.cache_read_input_tokens} reasoning={usage.thinking_tokens} "
            f"cost=${usage.estimated_cost_usd}"
        )
    for record in budget.records:
        print(f"  {record.provider}:{record.model} turn={record.turn_id} pricing={record.pricing_source}")
    print(f"spent ${budget.spent_usd:.6f}, remaining ${budget.remaining_usd:.6f}")


if __name__ == "__main__":
    main()
