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

## The claude-agent API

`ClaudeAgent` and `AsyncClaudeAgent` keep the exact `claude-agent` 0.8.0 constructor,
defaults, `messages` format (Messages API dicts), native SDK objects in hooks and SDK
exceptions. They run on the same runtime as `Agent`.

```python
from orkexa import ClaudeAgent

agent = ClaudeAgent(model="claude-opus-5-5", thinking="adaptive")
```
