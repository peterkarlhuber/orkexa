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
See `examples/07_history_migration.py`.
