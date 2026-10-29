# Migrating from claude-agent

There are three ways, from "change nothing" to "provider-neutral".

## 1. Unchanged `import claude_agent` code

```bash
pip uninstall claude-agent
pip install orkexa-claude-compat
python -m orkexa.compat     # "import claude_agent:  orkexa-claude-compat"
```

`orkexa-claude-compat` contains only the package `claude_agent`: every module is an
alias of the Orkexa module (`claude_agent.retrieval is orkexa.retrieval`,
`claude_agent.agent` is `orkexa.claude`), so classes, exceptions and `isinstance`
checks are identical. Nothing is reimplemented.

Tested: the complete `claude-agent` 0.8.0 test suite runs unchanged through the shim
(`tests/legacy`, 326 tests), and `examples/12_unchanged_claude_agent_script.py` is the
0.8.0 example byte for byte.

**Never install both distributions.** Both contain `claude_agent/`, and pip would
overwrite one with the other. The shim raises an `ImportError` with the repair
commands if the original is installed, and `python -m orkexa.compat` exits with 1.
Repair: `pip uninstall -y claude-agent orkexa-claude-compat && pip install orkexa-claude-compat`.

### Documented differences of the shim

| Area | 0.8.0 | Orkexa |
| --- | --- | --- |
| sandbox temp directory prefix | `claude-agent-sandbox-` | `orkexa-sandbox-` |
| Docker image / mount | `claude-agent-sandbox`, `/opt/claude-agent/claude_agent` | `orkexa-sandbox`, `/opt/orkexa/orkexa` |
| install hints in errors | `claude-agent[vector]` | `orkexa[vector] (formerly claude-agent[vector])` |
| `AutoApprover()` | silent | emits `UnsafeOptInWarning` (pass `acknowledge_unsafe=True`) |
| `BudgetTracker(limit)` with unpriced models | counts them as 0 | counts them as 0 **and warns**; `on_unpriced="ignore"` restores silence |
| `UsageRecord` | 14 fields | 4 more trailing fields: `provider`, `turn_id`, `workflow`, `pricing_source` |

The legacy test for the temp directory prefix is the only changed line in the copied
0.8.0 suite (marked with a comment in `tests/legacy/test_dynamic_sandbox.py`).

## 2. `from orkexa import ClaudeAgent`

Replace the import; everything else stays. `ClaudeAgent` is a facade over the shared
runtime with the 0.8.0 constructor, defaults (`model="claude-haiku-5-5"`,
`effort="medium"`), Messages API dict history, native SDK objects in
`on_request`/`on_response`/`last_response` and unchanged SDK exceptions.

## 3. Provider-neutral `Agent`

| claude-agent | Orkexa `Agent` |
| --- | --- |
| `ClaudeAgent(model="claude-opus-5-5")` | `Agent(model="anthropic:claude-opus-5-5")` (or any provider) |
| `thinking=`, `cache_control=`, `fallbacks=`, `request_options=` | `options=AnthropicOptions(...)` |
| `tool_choice={"type": "any"}` | `tool_choice="required"` (`"auto"`, `"none"`, `{"name": ...}`) |
| `messages` = Messages API dicts | `messages` = neutral `Message` objects (`agent.conversation()`) |
| `on_request(kwargs)`, `on_response(sdk_obj)` | same hooks, receive native objects; plus `on_event` with neutral events |
| `anthropic.RateLimitError` etc. | `orkexa.RateLimitError` etc. (redacted, `retryable`, `provider`, cause = SDK error) |
| `effort="medium"` default | `effort=None` default (not every model has reasoning) |

### Histories

`Agent.load_history()` reads 0.8.0 history files (plain Messages API lists) and the
versioned Orkexa format (`{"format": "orkexa.history", "version": 1, ...}`).
`ClaudeAgent.load_history()` reads both as well. The assistant turns of a 0.8.0 file
are marked as Anthropic output, so continuing them with another provider follows
`history_policy` (see [History](history.md)).

### Dynamic agents

`orkexa.dynamic` keeps every 0.8.0 class and signature and adds `credentials=` (a
mapping provider → `ProviderSettings`) to `CodingAgent`, `AgentArchitect`,
`Orchestrator` and `AgentFactory`. The legacy `client=`/`api_key=` arguments are the
Anthropic credentials and are only used for Anthropic models. `DynamicPolicy.
allowed_models` accepts provider-qualified names; a bare name means the provider it
resolves to (`"claude-opus-5-5"` = `"anthropic:claude-opus-5-5"`).
