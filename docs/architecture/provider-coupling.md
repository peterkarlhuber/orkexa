# Provider coupling in `claude-agent` 0.8.0

Which generic parts of the reference depend on Anthropic types, schemas, clients or
APIs, and how Orkexa removes or confines each dependency.

## Hard dependencies

| Location | Coupling | Kind | Orkexa resolution |
| --- | --- | --- | --- |
| `pyproject.toml` | `anthropic>=1.0,<3` is a required dependency | packaging | moved to the `anthropic` extra; the core imports no provider SDK |
| `agent.py`, `async_agent.py` | `from anthropic import Anthropic / AsyncAnthropic` default clients | client | Anthropic adapter creates the client lazily (`MissingDependencyError` without the extra) |
| `evaluation.py` | `from anthropic import Anthropic, transform_schema` | client + schema | `transform_schema` used only inside the Anthropic batch adapter |
| `dynamic/coding_agent.py`, `architect.py`, `factory.py`, `orchestrator.py` | `from anthropic import Anthropic` type hints, `ClaudeAgent(...)` construction | client + class | build generic `orkexa.Agent`; client typed `Any` |
| `dynamic/_sandbox_runner.py` | `claude_agent` namespace import | path | `orkexa` namespace import |
| `dynamic/sandbox.py` | `claude-agent-sandbox-` prefix, `/opt/claude-agent/claude_agent` mount | path | `orkexa-sandbox-`, `/opt/orkexa/orkexa` |

## Data formats bound to the Messages API

| Location | Coupling | Orkexa resolution |
| --- | --- | --- |
| `_base.py` history (`self.messages`) | list of Messages-API dicts (`tool_use`, `tool_result`, `thinking`, server-tool blocks) | neutral `Message`/`Part` model (`orkexa.core.types`); lossless Anthropic codec keeps unknown blocks and fields; `ClaudeAgent` stores the legacy dict list |
| `_base.py` `_process_response` | reads `response.stop_reason`, `response.content[*].type`, `parsed_output`, `stop_details`; `block.model_dump()` | adapters decode into `ModelResponse` (normalized `StopReason`, parts, `parsed`, `stop_details`) |
| `_base.py` `_tool_result` | `{"type": "tool_result", "tool_use_id", "content", "is_error"}` | neutral `ToolResultPart(call_id, name, content, is_error)` rendered per provider (function_call_output, functionResponse, role `tool`) |
| `tools.py` `Tool.schema()` | Anthropic tool definition format (`input_schema`, `strict`) | neutral `ToolDefinition`; provider-specific schema conversion and validation in `orkexa.core.schema` |
| `content.py` | Anthropic content block dicts (`source.type` base64/url/file/text) | kept as Orkexa's input block format; adapters translate (OpenAI `input_image`/`input_file`, Gemini `inline_data`/`file_data`, Chat Completions `image_url`/`file`) |
| `content.ToolContent` / `retrieval.Hit.block` | Anthropic `search_result` blocks | neutral `SearchResultPart`; native on Anthropic, rendered as attributed text for other providers |
| `retrieval.citations()` | Anthropic `search_result_location` citations | neutral `Citation(kind="api_native" \| "retrieval")`, `retrieval_sources()` for Orkexa-verified hits |
| `memory.MemoryBackend.definition` | Anthropic client-side tool type `memory_20250818` | native on Anthropic; `MemoryBackend.tool()` standard function tool elsewhere |
| `files.FileStore(files_api=...)` | Anthropic Files API (`client.files.upload`) and `file` source blocks | stays an Anthropic-only extension; file references are tagged with their provider and refused elsewhere |
| `usage.record_api_usage` | Anthropic `usage` fields (`cache_creation`, `iterations`, `server_tool_use`, `service_tier`, `speed`, `inference_geo`) | used by the Anthropic adapter only; other adapters normalize into the same `UsageRecord` fields |
| `pricing.MODEL_PRICING` | Claude prices | Anthropic table of a per-provider catalog; other providers unpriced unless configured |
| `evaluation.evaluate_many` | `client.messages.batches`, `output_config.format` | provider batch hook; Anthropic implementation unchanged |

## Anthropic-only request features (kept as typed provider options)

`output_config.effort`, `thinking {type, display}`, `cache_control`/`cache_ttl`,
`fallbacks` + beta headers, `betas` routing to `client.beta.messages`,
`eager_input_streaming`, `request_options` merge, server tools/MCP toolsets,
`pause_turn`, compaction blocks, `stop_details`, `count_tokens` field subset,
`_request_id` from stream headers. All of them live in
`orkexa.providers.anthropic` (`AnthropicOptions`); none is part of the generic runtime.
The generic `effort` option is mapped only where the provider has a matching
concept (Anthropic effort, OpenAI reasoning effort, Gemini thinking level).

## Model-specific validation

`_NO_FORCED_TOOL_CHOICE`, `_NO_DISABLED_THINKING`, `_DISABLED_THINKING_UP_TO_HIGH`,
`_BETWEEN_TOOLS_MODEL`, `_NO_SYSTEM_MESSAGES` and `model_family()` move to the
Anthropic adapter and its capability table. The generic runtime only asks the
capability system (`forced_tool_choice`, `system_messages`, …).
