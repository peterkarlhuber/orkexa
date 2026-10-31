# Feature matrix

## Per provider and API mode

✓ = implemented and covered by offline contract/integration tests with real SDK
response objects; ✗ = not supported (documented, raises a clear error);
"declare" = works when declared with `capabilities=`. No live API calls were run in
the automated tests.

| Feature | Anthropic | OpenAI Responses | Gemini | Compatible Chat Completions | Compatible Responses |
| --- | --- | --- | --- | --- | --- |
| text answer | ✓ | ✓ | ✓ | ✓ | ✓ |
| tools, several calls per response | ✓ | ✓ | ✓ | ✓ (declare) | ✓ (declare) |
| strict tool schemas | ✓ | ✓ | local validation | declare | declare |
| streaming (sync + async), history parity | ✓ | ✓ | ✓ | ✓ | ✓ |
| async agent + async tools | ✓ | ✓ | ✓ | ✓ | ✓ |
| structured output | ✓ | ✓ (strict schema) | ✓ | ✓ (json_schema / json_object) | ✓ (declare) |
| refusal → `RefusalError` | ✓ | ✓ | ✓ (safety) | ✓ | ✓ |
| truncation → `IncompleteResponseError` | ✓ | ✓ | ✓ | ✓ | ✓ |
| usage normalization | ✓ | ✓ | ✓ | ✓ | ✓ |
| error mapping + redaction | SDK errors (`ClaudeAgent`) / `ProviderError` (`Agent`) | ✓ | ✓ | ✓ | ✓ |
| native reasoning replay | thinking signatures | encrypted reasoning items | thought signatures | never replayed | reasoning items |
| save/load + continue | ✓ | ✓ | ✓ | ✓ | ✓ |
| cross-provider continuation | `history_policy="convert"` | ✓ | ✓ | ✓ | ✓ |
| token counting | ✓ | ✓ | ✓ | ✗ | ✗ |
| batch | ✓ | ✗ | ✗ | ✗ | ✗ |
| native memory tool | ✓ | function tool | function tool | function tool | function tool |
| files / images / documents | ✓ (+ Files API) | ✓ | ✓ | declare | declare |
| RAG search results | native blocks + citations | attributed text | attributed text | attributed text | attributed text |
| server tools | ✓ | ✓ (native dicts) | ✓ (native dicts) | ✗ | declare |
| cost estimate | built-in prices | registered/custom | registered/custom | registered/custom | registered/custom |

The test matrix lives in `tests/contract/test_contract.py` (11 scenarios × 5 adapters
plus cross-provider continuation into every other adapter).
