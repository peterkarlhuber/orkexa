# GDPR

The General Data Protection Regulation (Regulation (EU) 2016/679) applies to the
controllers and processors of personal data, which are the organizations that build
and run a system, not the library they use. Orkexa supports their obligations with
technical defaults and mechanisms; it does not determine purposes, legal bases or
retention periods, and it does not know whether the data it handles is personal data.

| GDPR topic | How Orkexa supports it | What you still need to do |
| --- | --- | --- |
| Data protection by design and by default (Art. 25) | Nothing persisted by default; usage records without content; OpenAI `store=False`; credentials isolated per provider; no network on import | Minimize what goes into prompts, tool results, memory and indexes; choose providers and settings deliberately |
| Storage limitation (Art. 5(1)(e)) | Stores are explicit objects with `delete()`; histories are saved only on request | Define and enforce retention periods for every store, saved history and log |
| Right to erasure (Art. 17) | `delete()` in every store, overwriting in SQLite stores, `SQLiteIndex.optimize()`, `agent.reset()` ([details](../guides/memory-files-rag.md#deleting-data)) | Find a person's data across your stores (e.g. a namespace or source per person), delete it at providers, in backups and logs |
| Rights of access and portability (Art. 15, 20) | Stores can list and export their contents (`view`, `list`, `get_bytes`, `sources`); histories are JSON | Identify the data subject's records and answer the request |
| Security of processing (Art. 32) | Secret redaction in errors, tracebacks and `repr`, `SecretStr` keys, owner-only permissions for stored data, an isolated sandbox by default, path validation in stores and registries, see [Security](../guides/security.md) | Encryption at rest, access control, secrets management, patching, monitoring |
| Processors and transfers (Art. 28, Chapter V) | Data goes only to the endpoints you configure ([data flows](data-flows.md)); on Amazon Bedrock, `BedrockOptions(data_residency="eu_only")` refuses models and inference profiles that can route outside EU regions ([details](../guides/security.md#amazon-bedrock-data-residency)) | Data processing agreements with providers, transfer mechanisms, records of processing (Art. 30); enforce regions with AWS IAM and service control policies |
| Automated decisions (Art. 22) | `before_tool` hooks and approvers can put a person in front of consequential actions | Decide whether your system makes decisions with legal or similarly significant effects, and provide the required safeguards |
| Data protection impact assessment (Art. 35) | This documentation describes the library's data handling as input | Carry out the assessment for your system |

## Practical defaults for personal data

* Keep personal data out of system prompts and tool descriptions; pass it per request.
* Use one store `namespace` (SQLite) or directory per data subject or tenant, so that
  access and erasure requests map to one place.
* Do not log prompts or tool results by default; Orkexa's own log messages do not
  contain them, but your `on_event`, `on_text` and tool hooks may.
* Prefer `SQLiteMemory`/`SQLiteFileStore` over file-based stores when you need
  deletion that overwrites data, and call `SQLiteIndex.optimize()` after deleting
  indexed personal data.
* Use a local endpoint (`ollama`, `vllm`) when data must not leave your
  infrastructure; Orkexa reads no keys from the environment for these.
* On Amazon Bedrock, combine `data_residency="eu_only"` with service control policies
  that deny other regions and global inference profiles, and check whether model
  invocation logging is enabled in your account: it stores full prompts and
  responses. `eu_only` verifies Orkexa's configuration; it is not a certification of
  residency or of GDPR compliance.
