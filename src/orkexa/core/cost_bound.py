"""Upper bounds for the cost of one model request, before it is sent.

A budget tracker reserves this bound before a request and settles it with the real
cost afterwards (see :meth:`orkexa.BudgetTracker.try_reserve`), so requests that run at
the same time cannot together spend more than the budget.

The bound is ``input_tokens * highest input rate + max_tokens * highest output rate``:

* **Input tokens** are bounded by the UTF-8 size of everything the request sends
  (messages, system prompt, tool definitions, output schema) plus a fixed allowance for
  the provider's framing. Byte-level tokenizers never produce more tokens than bytes.
* **Output tokens** are bounded by ``max_tokens``, which covers reasoning as well.
* **Rates** are the highest of the price: every long-context tier, every tariff option
  and every input class (uncached, cache read, cache writes, special classes), times
  the highest tariff multiplier.

There is **no** bound (``None``) when the cost cannot be capped in advance: no price,
a price in another currency than USD, provider-native tools (server tools with fees,
MCP toolsets), images, binary documents or files referenced by id or URL. An unknown
bound is never treated as zero; the tracker counts such requests separately.
"""

from __future__ import annotations

from decimal import Decimal
import json
from typing import Any

from ..pricing import ModelPrice, TokenRates
from .types import (
    DocumentPart,
    ImagePart,
    Message,
    ModelRequest,
    SearchResultPart,
    TextPart,
    ToolResultPart,
)

__all__ = ["FRAMING_TOKENS", "MESSAGE_TOKENS", "input_token_bound", "request_cost_bound"]

FRAMING_TOKENS = 4096
"""Allowance for what the provider adds around a request (tool-use system prompt, roles)."""

MESSAGE_TOKENS = 64
"""Allowance per message and per tool for the provider's separators."""

_MILLION = Decimal(1_000_000)


def _bounded_part(part: Any) -> bool:
    if isinstance(part, ImagePart):
        return False
    if isinstance(part, DocumentPart):
        return part.source.type == "text"
    if isinstance(part, ToolResultPart) and isinstance(part.content, list):
        return all(_bounded_part(child) for child in part.content)
    if isinstance(part, SearchResultPart):
        return all(isinstance(child, TextPart) for child in part.content)
    return True


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _message_size(message: Message) -> int:
    return len(message.model_dump_json().encode("utf-8"))


def input_token_bound(request: ModelRequest) -> int | None:
    """Most input tokens ``request`` can be billed for, ``None`` if it cannot be bounded."""
    if request.native_tools:
        return None
    for message in request.messages:
        if not all(_bounded_part(part) for part in message.parts()):
            return None
    size = sum(_message_size(message) for message in request.messages)
    size += len((request.system or "").encode("utf-8"))
    size += sum(_size([tool.name, tool.description, tool.input_schema]) for tool in request.tools)
    if request.output_format is not None:
        size += _size(request.output_format.model_json_schema())
    return size + FRAMING_TOKENS + MESSAGE_TOKENS * (len(request.messages) + len(request.tools) + 1)


def _rate_sets(price: ModelPrice) -> list[TokenRates]:
    sets = [price.rates, *(tier.rates for tier in price.long_context)]
    for tariff in price.tariffs:
        for option in tariff.options:
            if option.rates is not None:
                sets.append(option.rates)
                sets.extend(tier.rates for tier in option.long_context or ())
    return sets


def _highest(rates: list[TokenRates], kind: str) -> Decimal:
    values: list[Decimal] = []
    for item in rates:
        if kind == "input":
            classes = [item.input, item.cache_read, item.cache_write_5m, item.cache_write_1h]
            classes += [rate for name, rate in item.special.items() if not name.endswith("_output")]
        else:
            classes = [
                item.output,
                *(rate for name, rate in item.special.items() if name.endswith("_output")),
            ]
        values.extend(value for value in classes if value is not None)
    return max(values, default=Decimal(0))


def request_cost_bound(request: ModelRequest, price: ModelPrice | None) -> Decimal | None:
    """Most USD ``request`` can cost with ``price``, ``None`` when no such bound exists."""
    if price is None or price.currency != "USD":
        return None
    input_tokens = input_token_bound(request)
    if input_tokens is None:
        return None
    rates = _rate_sets(price)
    multiplier = max(
        [Decimal(1), *(option.multiplier for tariff in price.tariffs for option in tariff.options)]
    )
    total = Decimal(input_tokens) * _highest(rates, "input") + Decimal(request.max_tokens) * _highest(
        rates, "output"
    )
    return total * multiplier / _MILLION
