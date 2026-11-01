# Feature matrix

## Per provider and API mode

✓ = implemented and covered by offline contract/integration tests with real SDK
response objects; ✗ = not supported (documented, raises a clear error);
"declare" = works when declared with `capabilities=`. No live API calls were run in
the automated tests.

| Feature | Anthropic | OpenAI Responses | Gemini | Compatible Chat Completions | Compatible Responses | Bedrock Converse |
| --- | --- | --- | --- | --- | --- | --- |
| text answer | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| tools, several calls per response | ✓ | ✓ | ✓ | ✓ (declare) | ✓ (declare) | ✓ (Claude; Nova: declare) |
| strict tool schemas | ✓ | ✓ | local validation | declare | declare | ✓ (Claude 4.5+) |
| streaming (sync + async), history parity | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| async agent + async tools | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (worker threads) |
| structured output | ✓ | ✓ (strict schema) | ✓ | ✓ (json_schema / json_object) | ✓ (declare) | ✓ (Claude 4.5+), ✗ Nova |
| refusal → `RefusalError` | ✓ | ✓ | ✓ (safety) | ✓ | ✓ | ✓ (guardrail, content filter) |
| truncation → `IncompleteResponseError` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| usage normalization | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| error mapping + redaction | SDK errors (`ClaudeAgent`) / `ProviderError` (`Agent`) | ✓ | ✓ | ✓ | ✓ | ✓ |
| native reasoning replay | thinking signatures | encrypted reasoning items | thought signatures | never replayed | reasoning items | Claude reasoning blocks |
| save/load + continue | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cross-provider continuation | `history_policy="convert"` | ✓ | ✓ | ✓ | ✓ | ✓ |
| token counting | ✓ | ✓ | ✓ | ✗ | ✗ | Claude 4.5/4.6 |
| batch | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| native memory tool | ✓ | function tool | function tool | function tool | function tool | function tool |
| files / images / documents | ✓ (+ Files API) | ✓ | ✓ | declare | declare | ✓ (bytes; S3 on Nova; no Files API) |
| RAG search results | native blocks + citations | attributed text | attributed text | attributed text | attributed text | attributed text |
| server tools | ✓ | ✓ (native dicts) | ✓ (native dicts) | ✗ | declare | ✗ |
| cost estimate | built-in prices | registered/custom | registered/custom | registered/custom | registered/custom | registered/custom (per `aws_region`) |

For Bedrock, the column shows Claude and Amazon Nova; whether Nova returns several
tool calls in one response is not documented, so `parallel_tool_calls` is unknown
there until declared.

## Amazon Bedrock per model family

On Bedrock the capabilities depend on the foundation model behind the model id or
inference profile (`bedrock_capabilities()` in `orkexa.providers.bedrock`). ✓ =
supported, ✗ = unsupported (raises before a request), unknown = declare it with
`capabilities=` if your model supports it.

| Capability | Claude 5.x | Claude 4.5–4.8 | Older Claude (3.x, 4.0, 4.1) | Amazon Nova | Other models, opaque ARNs |
| --- | --- | --- | --- | --- | --- |
| tools, streaming | ✓ | ✓ | ✓ | ✓ | unknown |
| parallel tool calls | ✓ | ✓ | ✓ | unknown | unknown |
| forced tool choice | ✗ Sonnet 5.5, Opus 5.5, Fable 5.1, Mythos 5.1; else unknown | ✓ 4.5/4.6, unknown 4.7/4.8 | ✓ | ✓ | unknown |
| strict tool schemas, structured output | ✓ | ✓ | unknown | ✗ structured output, strict schemas unknown | unknown |
| reasoning (`effort`) | ✓ (`outputConfig.effort`) | unknown | unknown | unknown | unknown |
| image input | ✓ | ✓ | ✓ (✗ Claude 3.5 Haiku) | ✓ (✗ Nova Micro) | unknown |
| document input | ✓ | ✓ | ✓ | ✓ (✗ Nova Micro) | unknown |
| prompt caching (`prompt_caching=True`) | ✓ (5 min, 1 h) | ✓ (5 min, 1 h) | ✓ Claude 3.7 Sonnet and 4.x, else unknown | ✓ (5 min only) | unknown |
| token counting | ✗ | ✓ 4.5/4.6, ✗ 4.7/4.8 | unknown | unknown | unknown |
| server tools, native memory tool, batch, system messages | ✗ | ✗ | ✗ | ✗ | ✗ |

"Opaque ARNs" are application inference profiles, provisioned throughput, custom and
imported models: their ids do not name the foundation model. Foundation model ARNs and
inference profile ids or ARNs are classified like the model they name. The values
follow the AWS model cards checked on 2026-11-01; a model AWS documents differently
later is corrected by `capabilities=` until Orkexa is updated.

The test matrix lives in `tests/contract/test_contract.py` (11 scenarios × 6 adapters
plus cross-provider continuation into every other adapter).
