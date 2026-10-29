# Multi-agent and dynamic agents

## Mixed providers

Agents of different providers share tools, trackers, judges and orchestration:

```python
from orkexa import Agent, ClaudeAgent, Criterion, Evaluator, Tool, UsageTracker, refine

usage = UsageTracker()
researcher = Agent(model="openai:gpt-5", usage_tracker=usage)


@Tool
def research(question: str) -> str:
    """Ask the research agent.

    Args:
        question: A self-contained question.

    Returns:
        The answer.
    """
    return researcher.ask(question)


lead = ClaudeAgent(model="claude-sonnet-5-5", tools=[research], usage_tracker=usage)
judge = Evaluator(
    criteria=[Criterion(name="sourced", description="Every claim names its source.")],
    model="gemini:gemini-2.5-flash",
    usage_tracker=usage,
)
result = refine(lead, "Write a sourced brief on the Danube.", judge)
```

## Dynamic agents

`orkexa.dynamic` (Coding Agent, ToolBuilder, registries, Agent Architect, factory,
Orchestrator, approvers, sandboxes) accepts provider-qualified models everywhere:

```python
from orkexa.dynamic import (
    AgentArchitect,
    CallbackApprover,
    DynamicPolicy,
    Orchestrator,
    ProviderSettings,
    ToolRegistry,
)

credentials = {
    "anthropic": ProviderSettings(api_key="sk-ant-..."),
    "openai": ProviderSettings(api_key="sk-..."),
    "gemini": ProviderSettings(api_key="..."),
}
orchestrator = Orchestrator(
    architect=AgentArchitect(model="anthropic:claude-sonnet-5-5", credentials=credentials),
    approver=CallbackApprover(my_review),
    tools=ToolRegistry(),
    policy=DynamicPolicy(allowed_models=["openai:gpt-5", "gemini:gemini-2.5-flash"], budget_usd=5),
    model="openai:gpt-5",
    credentials=credentials,
)
```

* Every created agent receives only the `ProviderSettings` of its own model's
  provider; a key is never handed to another provider. The legacy `client=`/`api_key=`
  arguments are used for Anthropic models only.
* The policy has the last word: models (bare or qualified names), effort, token and
  round limits, allowed tools, delegation and depth limits (`RecursionLimitError`),
  agent and tool counts, budget.
* An `effort` chosen by the architect is dropped for models whose provider reports
  reasoning as `unsupported`.
* Generated code is checked statically, tested in a sandbox and approved before it is
  registered, and every call runs in the sandbox, never in the host process.
  A tool built by one provider's Coding Agent can be used by an agent of any provider.
