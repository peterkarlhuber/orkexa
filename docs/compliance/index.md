# Data protection and AI regulation

Orkexa is a software library. It is not an AI system, not a general-purpose AI model
and not a certified product, and it cannot make a system compliant on its own. This
section describes what Orkexa does with data, which technical controls it provides,
and which obligations stay with the people who build and run systems with it, so that
those people can carry out their own assessment.

!!! warning "Not legal advice"
    These pages describe technical behavior, last reviewed on 2026-10-29. They are not
    legal advice and do not state that any system is compliant with the GDPR, the EU AI
    Act or ISO/IEC 42001. Have your assessment reviewed by qualified counsel and, where
    applicable, your data protection officer.

* [Data flows and retention](data-flows.md): what leaves the process, what is stored
  where, and how to delete it.
* [GDPR](gdpr.md): controls that support data protection by design.
* [EU AI Act](eu-ai-act.md): roles, risk classification, transparency, record-keeping,
  human oversight and the general-purpose AI boundary.
* [ISO/IEC 42001](iso-iec-42001.md): how Orkexa fits into an AI management system.

## Responsibility boundaries

| Party | Responsible for |
| --- | --- |
| **Orkexa project** | The library's code and documentation: correct, documented behavior of the controls below, security fixes ([SECURITY.md](https://github.com/peterkarlhuber/orkexa/blob/master/SECURITY.md)), and an accurate description of limits. The project processes no personal data of your users and operates no service. |
| **Library safeguards** | Defaults and mechanisms in the code: no persistence unless you configure a store, deletion in every store, credential isolation, secret redaction, local validation, approval gates and limits for generated code. They work only as configured by the integrator. |
| **Model providers** (Anthropic, OpenAI, Google, operators of compatible endpoints) | Their models and APIs: processing, retention and location of the data you send them, their own obligations as providers of general-purpose AI models, their contractual terms and data processing agreements. |
| **Integrators** (developers who build a system with Orkexa) | The system's purpose and design: legal basis, data minimization in prompts and tools, choice of provider, region and retention settings, configuration of stores, approvals and logging, transparency notices, risk classification under the AI Act, impact assessments, documentation. |
| **Operators / deployers** (organizations that run the system) | Operation: access control, secrets management, monitoring, incident response, retention schedules, answering data subject requests, human oversight in practice, staff AI literacy. |

## Mapping of technical controls

The tests named here run offline in CI on every change.

| Topic | Orkexa control | Verified by | Remains with the integrator / operator |
| --- | --- | --- | --- |
| Storage limitation | Agents keep history in memory only; nothing is written to disk unless you configure a file or SQLite store or call `save_history()` | `tests/test_data_controls.py::test_an_agent_writes_nothing_to_disk_by_default` | Retention periods for stores and saved histories |
| Erasure | `delete()` in every store; SQLite stores overwrite deleted rows; `SQLiteIndex.optimize()`; `agent.reset()` | `tests/test_data_controls.py` (erasure tests), `test_reset_forgets_the_conversation` | Erasure at providers, in backups and in your own logs |
| Data minimization in metadata | Usage records hold counts, model, provider, ids and cost, never prompt or output text | `test_usage_records_hold_metadata_but_no_content` | What you put into prompts, tools and logs |
| Provider-side storage | OpenAI requests use `store=False` by default | `tests/test_providers.py::test_openai_request_is_stateless_with_encrypted_reasoning` | Provider retention settings and agreements |
| Credential protection | Keys and `base_url` credentials hidden in `repr`, redacted from errors, never in history files, never sent to another host or provider; provider errors carry no link to the SDK exception, so tracebacks and logs cannot print request headers; `compatible` never reads `OPENAI_API_KEY` | `tests/test_providers.py` (`test_api_key_is_hidden_in_repr_and_redacted`, `test_compatible_never_reads_openai_api_key`, `test_openai_custom_base_url_needs_api_key`), `tests/test_security.py` (redaction), `tests/test_secret_exposure.py`, `test_saved_histories_contain_no_credentials` | Secrets management and key rotation; tracebacks of `ClaudeAgent` errors (raw SDK exceptions) |
| Access to stored data | Histories, registry records and new SQLite databases (with their journal and WAL files) are created `0600`, directories `0700`, independent of the umask; atomic replacement | `tests/test_private_storage.py` | File system and ACL configuration (especially on Windows), encryption at rest, existing files created by older versions |
| Security of generated code | Static check, `DockerSandbox` by default (fails closed without Docker), `SubprocessSandbox` only as an acknowledged opt-in, sandbox with an empty environment, approval before registration, policy limits | `tests/test_sandbox_policy.py`, `tests/legacy/test_dynamic_sandbox.py::test_environment_is_empty`, `tests/legacy/test_dynamic_tools.py::test_build_approval_denied_marks_rejected` | Docker or stronger isolation in operation, review of generated code |
| Human oversight | `before_tool` can refuse any tool call; approvers gate generated tools and agents; an LLM review cannot approve on its own; `AutoApprover` warns | `tests/test_runtime.py::test_before_and_after_tool_hooks`, `tests/legacy/test_dynamic_evaluation.py::test_llm_approver_pass_still_needs_then`, `tests/test_mixed_providers.py::test_auto_approver_is_an_explicit_unsafe_opt_in` | Who reviews, with what authority and competence |
| Traceability | Per-request usage records with provider, model, request id, turn id and workflow label; neutral events; versioned history files | `tests/test_usage_budget.py`, `tests/test_history_models.py` | Log storage, integrity and retention |
| Transparency | Explicit errors instead of silent fallbacks (capabilities, refusals, truncation); `retrieval_sources()` separates retrieved passages from model claims | `tests/test_runtime.py`, `tests/contract/test_contract.py` | Informing users that they interact with AI; labeling generated content |
