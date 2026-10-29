# EU AI Act

The Artificial Intelligence Act (Regulation (EU) 2024/1689) regulates AI systems and
general-purpose AI models placed on the market or used in the EU. Its obligations fall
on roles: providers and deployers of AI systems, providers of general-purpose AI
models, importers and distributors. A software library is not itself an AI system in
the sense of the Act; an AI system comes into being when someone combines a model,
code such as Orkexa, data and a purpose.

## Timeline

Dates as published by the European Commission
([regulatory framework for AI](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai),
checked on 2026-10-29), including the changes of the Digital Omnibus on AI, which
entered into force on 27 July 2026:

| Date | What applies |
| --- | --- |
| 1 August 2024 | Entry into force |
| 2 February 2025 | Prohibited practices; AI literacy |
| 2 August 2025 | Governance; obligations for providers of general-purpose AI models |
| 2 August 2026 | General application, including the transparency obligations and enforcement |
| December 2026 | Additional prohibited practice added by the Digital Omnibus |
| 2 December 2027 | High-risk AI systems in the use cases of Annex III |
| 2 August 2028 | High-risk AI systems that are products, or safety components of products, under Annex I |

Check the current text on EUR-Lex before relying on these dates.

## Roles

| Role | Typical party when Orkexa is used |
| --- | --- |
| Provider of a general-purpose AI model | Anthropic, OpenAI, Google, or whoever trains or substantially modifies the model behind a compatible endpoint |
| Provider of an AI system | The organization that builds a system with Orkexa and places it on the market or puts it into service under its name |
| Deployer | The organization that uses the system under its authority |
| Orkexa project | None of these roles through the library alone; it supplies a software component |

Orkexa does not train, fine-tune or distribute models. Using Orkexa to call a model
does not make you a provider of a general-purpose AI model; modifying a model can, and
needs its own assessment.

## Risk classification

The risk class depends on the system's intended purpose, not on the library. Assess,
for the system you build:

1. Whether any use falls under the prohibited practices (Art. 5).
2. Whether the system is a product or safety component under Annex I, or is used in an
   Annex III area (for example employment, education, access to essential services,
   law enforcement, migration, administration of justice), and whether the exception
   of Art. 6(3) applies.
3. Whether the transparency obligations of Art. 50 apply (systems that interact with
   people, generate synthetic content, recognize emotions, or produce deep fakes).
4. Otherwise, the system has minimal risk, and AI literacy (Art. 4) and voluntary
   codes remain.

Agents that take actions through tools (sending messages, changing records, running
generated code) deserve particular attention in step 2, because the purpose of the
tools, not the model, often decides the classification.

## How Orkexa supports common obligations

| Obligation area | Orkexa mechanism | Limits |
| --- | --- | --- |
| Risk management and testing (Art. 9, 15) | Deterministic `FakeProvider` and scripted SDK clients for offline tests; `Evaluator` for output checks, including judges from another provider | Evaluations measure what you define; they do not establish accuracy or robustness for your use case |
| Record-keeping and logs (Art. 12, 19, 26(6)) | Usage records per request (provider, model, request id, turn id, workflow), neutral events via `on_event`, versioned history files, registry records with code hashes and approval status | Orkexa keeps nothing beyond the process unless you store it; log storage, integrity and retention are yours |
| Transparency to deployers (Art. 13) | Explicit capability errors, typed provider errors, `retrieval_sources()` separating retrieved passages from model text, documented limits | Instructions for use of your system |
| Human oversight (Art. 14) | `before_tool` can stop any tool call; approvers gate generated tools and agents; `LLMApprover` can only make decisions stricter and always needs a final approver; `AutoApprover` warns | See below |
| Transparency to people (Art. 50) | None built in | Tell people they interact with an AI system; mark synthetic content where required |
| Accuracy, robustness, cybersecurity (Art. 15) | Local validation of tool input and structured output, limits on tool rounds and parallelism, timeouts, sandboxes, credential isolation | `SubprocessSandbox` (an explicit opt-in) is not a security boundary; `DockerSandbox` isolation is not audited; prompt injection is not prevented |

### Human oversight is a human

An `LLMApprover`, a judge model or any automated review is not human oversight. It can
reject requests, and it adds its verdict to the approval request so that a person sees
it, but the final approval of a high-impact action needs a natural person with the
competence, authority and time to understand the action and to refuse it, for example
through a `ConsoleApprover`, a `CallbackApprover` connected to a review queue, or a
`before_tool` hook that waits for a decision. `AutoApprover` removes oversight
entirely and is meant for trusted, low-risk development setups.

## General-purpose AI boundary

The obligations of providers of general-purpose AI models (Art. 53 to 55: technical
documentation, information for downstream providers, copyright policy, training data
summary, and for models with systemic risk, evaluations and incident reporting) lie
with the model providers. As a downstream provider you can rely on the information
they supply, and you remain responsible for your system's own obligations.
