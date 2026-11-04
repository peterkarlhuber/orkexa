# Tools and structured output

## Tools

`@Tool` derives a strict JSON schema from type hints and a Google-style docstring
(the `Args:` and `Returns:` sections are required). Pydantic models, enums,
`Literal`, lists and optional parameters work. The same `Tool` object is rendered
per provider:

| Provider | Rendering | Enforcement |
| --- | --- | --- |
| Anthropic | tool with `input_schema` (`strict: true`) | API + local validation |
| OpenAI | strict `function` (`additionalProperties: false`, all properties required, optionals nullable) | API + local validation |
| Gemini | `FunctionDeclaration` with inlined schema | local validation |
| Compatible | `function` (strict only if `strict_tool_schemas` is declared) | local validation |

Tool arguments from the model are untrusted input: they are validated with Pydantic
before the function runs. Invalid input raises `ToolInputError` (or, with
`raise_tool_errors=False`, goes back to the model as an error result so it can retry).
A schema a provider cannot express (e.g. a nullable optional parameter with a non-`None`
default under OpenAI strict mode) raises `ToolSchemaError` before any request.

Orkexa always executes tools itself, after `before_tool`. Automatic function calling of
SDKs (Gemini) is disabled; tool calls are never executed by an SDK.

```python
agent = Agent(
    model="gemini:gemini-3.8-flash",
    tools=[price, multiply],
    max_tool_rounds=8,  # MaxToolRoundsError after that
    max_parallel_tools=4,  # tool calls of one response run concurrently
    max_tool_calls_per_round=64,  # default; MaxToolCallsError above, before any call runs
    tool_timeout=10,  # seconds per call
    raise_tool_errors=False,  # failures go back to the model
    tool_choice="required",  # "auto", "none", "required" or {"name": "price"}
)
```

## Structured output

```python
from pydantic import BaseModel
from orkexa import Agent


class Invoice(BaseModel):
    number: str
    total: float
    currency: str = "EUR"


invoice = Agent(model="openai:gpt-6.1-sol", output_format=Invoice).ask("Extract: Invoice 17, 120.50 EUR")
```

`ask()` returns an `Invoice` instance; the agent is typed `Agent[Invoice]`. The JSON is
always validated locally. If it does not validate, `StructuredOutputError` carries the
validation error. With OpenAI strict mode, `null` for an optional field means "use the
default".

Tools and structured output together: most providers send both in one request. Ollama turns the
JSON schema into a grammar, so a model that may only write JSON can never call a tool. For
`provider="ollama"` (`Provider.structured_output_with_tools = False`) the agent therefore offers the
tools without the schema; once the model answers without a tool call, it asks once more, without
tools and with the schema, for the structured answer (`orkexa.core.runtime.FORMAT_PROMPT`). That
extra request is not counted as a tool round and stays in the history.
