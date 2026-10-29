# ISO/IEC 42001

ISO/IEC 42001:2023 specifies requirements for an AI management system: the policies,
roles, processes and controls with which an organization governs the AI systems it
develops, provides or uses. Organizations are certified against it; software libraries
are not. Orkexa is not certified and makes no claim of conformity. The standard's text
is published by ISO and IEC and is not reproduced here.

For an organization that operates such a management system, Orkexa is a third-party
software component of its AI systems. The table lists the management-system topics
where Orkexa provides technical evidence or mechanisms, described in general terms
rather than in the standard's wording.

| Management-system topic | What Orkexa provides |
| --- | --- |
| Inventory of AI systems and components | A declared dependency with a changelog, versioned history files, explicit provider and model ids per request |
| Supplier and third-party relationships | A clear separation between the library and the model providers ([responsibilities](index.md#responsibility-boundaries)), and [data flows](data-flows.md) that show which provider receives what |
| Risk and impact assessment | Documented capabilities and limits per provider, [security defaults and remaining risks](../guides/security.md), inputs for the AI Act classification ([EU AI Act](eu-ai-act.md)) |
| Data management | Explicit stores with deletion, no persistence by default, usage metadata without content ([GDPR](gdpr.md)) |
| Verification and validation | Offline test doubles, contract tests against the real SDK signatures, `Evaluator` for output quality checks |
| Operation and monitoring | Usage records, neutral events, typed errors with a `retryable` flag, budgets |
| Human oversight and approval | `before_tool` hooks, approvers for generated tools and agents |
| Change management | Versioned releases, a changelog, CI on every change, and a release process with build provenance ([Releasing](../development/releasing.md)) |

What the organization must provide itself: the AI policy, roles and responsibilities,
competence and awareness, risk criteria, impact assessments, objectives, internal
audits, management reviews and continual improvement.
