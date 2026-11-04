# Quickstart

## One agent, any provider

```python
from orkexa import Agent

agent = Agent(model="openai:gpt-6.1-sol", system_prompt="Answer briefly.")
print(agent.ask("What is the capital of Austria?"))
print(agent.ask("And its population?"))  # the agent keeps the conversation
print(agent.usage.total_tokens)
```

The model string is `"provider:model"`. Without a registered prefix the model belongs
to the default provider `anthropic` (`Agent(model="claude-sonnet-5-5")`). You can also
pass `provider=` separately; it must agree with a prefix.

## Switching provider

Only the model string changes; tools, structured output, hooks, memory and usage
tracking stay the same:

```python
from pydantic import BaseModel
from orkexa import Agent


class City(BaseModel):
    name: str
    population: int


for model in ("anthropic:claude-sonnet-5-5", "openai:gpt-6.1-sol", "gemini:gemini-3.8-flash"):
    city = Agent(model=model, output_format=City).ask("Largest city in Austria?")
    print(model, city.population)
```

## Local models

```python
from orkexa import Agent

agent = Agent(provider="ollama", model="llama3.1:8b")  # http://localhost:11434/v1
agent = Agent(provider="vllm", model="Qwen/Qwen3-8B")  # http://localhost:8000/v1
agent = Agent(
    provider="compatible",
    model="my-model",
    base_url="https://gateway.example/v1",
    api_key="gateway-key",
    capabilities={"tools": "supported"},
)
```

These agents need a server that already runs at that address; without one, the first
`ask()` raises `ProviderConnectionError`. For Ollama, `orkexa-local status` shows whether a
server answers, and [Local models, embeddings and retrieval modes](../guides/local-models.md)
shows how to install and run a pinned Ollama and read a model's capabilities from it.

Capabilities of compatible endpoints are *unknown* until you declare them; a request
that needs an unknown capability fails with `UnknownCapabilityError` naming the
`capabilities=` override. See [Providers and capabilities](providers.md).

## Async

```python
import asyncio
from orkexa import AsyncAgent


async def main() -> None:
    agent = AsyncAgent(model="gemini:gemini-3.8-flash", max_parallel_tools=4)
    print(await agent.ask("Hello!"))


asyncio.run(main())
```

## Anthropic-native agents

`ClaudeAgent` and `AsyncClaudeAgent` take Anthropic-specific constructor arguments and
keep `messages` as Messages API dicts, native SDK objects in hooks and SDK
exceptions. They run on the same runtime as `Agent`.

```python
from orkexa import ClaudeAgent

agent = ClaudeAgent(model="claude-opus-5-5", thinking="adaptive")
```

## Without an API key

The built-in `fake` provider answers from a script, with no SDK and no network. Use it
to try the API, and in your own unit tests:

```python
from orkexa import Agent, Tool
from orkexa.providers.fake import FakeModel, text, tool_calls


@Tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First summand.
        b: Second summand.

    Returns:
        The sum.
    """
    return a + b


model = FakeModel([tool_calls(("add", {"a": 17, "b": 25})), text("42")])
agent = Agent(model="fake:scripted", client=model, tools=[add])
print(agent.ask("What is 17 + 25?"))  # 42, after the agent ran add(17, 25)
print(len(model.requests))  # 2: every request the runtime sent
```

A string in the script is a text answer, an exception is raised as if the API had
raised it, and a callable receives the request and returns the answer. Tool docstrings
need an `Args:` and a `Returns:` section; see
[Tools and structured output](../guides/tools-and-structured-output.md).
