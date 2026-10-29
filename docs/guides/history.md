# History and provider switching

The conversation is a list of neutral `Message` objects (`agent.conversation()`).
Each part keeps lossless provider metadata (Anthropic thinking signatures, OpenAI
reasoning items with encrypted content, Gemini thought signatures), so a conversation
continues with the same provider exactly as the API produced it.

```python
agent.save_history("chat.json")  # {"format": "orkexa.history", "version": 1, ...}
other = Agent(model="openai:gpt-5")
other.load_history("chat.json")  # also reads claude-agent 0.8.0 files
```

## Switching provider mid-conversation

Reasoning parts and native-only blocks cannot be replayed to another provider:

* `history_policy="strict"` (default): `HistoryConversionError` before the request.
* `history_policy="convert"`: they are dropped with a `HistoryConversionWarning`; text,
  tool calls, tool results, images and documents are converted.

Provider file references (`uploaded_file()`) never move between providers.

## Switching model within a provider

Reasoning is bound to the model that produced it (thinking signatures, encrypted
reasoning items and Gemini thought signatures are only valid for that model). When
`agent.model` changes to another model of the same provider, the reasoning parts and
Gemini thought signatures of the old model's messages are not sent, under both
policies, with a `HistoryConversionWarning`. Nothing visible is lost: text, tool calls
and tool results stay, and the stored history is unchanged (switching back replays
them). Server-tool blocks of the provider stay as well.

`Message.model` is the model id the agent requested (`"gpt-5.5"`); the version the API
reported is `ModelResponse.model` and `UsageRecord.model`. Messages without a model
(`claude-agent` 0.8.0 histories, which do not record one) count as coming from the
current model, as in 0.8.0.
See `examples/07_history_migration.py`.
