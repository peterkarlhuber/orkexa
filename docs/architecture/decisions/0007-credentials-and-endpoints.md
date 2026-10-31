# ADR 0007 – Credentials and endpoints

Status: accepted

* Keys are passed per provider (`api_key=`, `ProviderConfig`, or the SDK's own
  environment variable for first-party endpoints). `ProviderConfig.api_key` is a
  `SecretStr`; `repr()` and serialized configs never contain it.
* `CompatibleProvider` uses only an explicit `api_key` (or none: a placeholder is sent
  to local servers that ignore it). It never reads `OPENAI_API_KEY`.
* `OpenAIProvider`/`AnthropicProvider` with a `base_url` outside their first-party
  hosts require an explicit `api_key`; otherwise the SDK would send the environment
  key to a third-party host.
* `AgentFactory` and `Orchestrator` pass credentials only to agents of the matching
  provider (`credentials={"openai": ProviderConfig(...)}`); the single
  `client`/`api_key` applies to the default provider only.
* Provider errors are re-raised as `ProviderError` with secrets redacted from the
  message; sandboxes start with an empty environment.
