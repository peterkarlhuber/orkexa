# Security policy

## Supported versions

Orkexa is in alpha and has not been released yet. Security fixes are made on the
default branch (`master`) and will be part of the next release. Once releases exist,
the latest minor release line receives fixes.

| Version | Supported |
| --- | --- |
| `master` (unreleased 0.1.0) | yes |

## Reporting a vulnerability

Please do not report vulnerabilities in public issues, pull requests or discussions.

1. If the repository's **Security** tab offers **Report a vulnerability** (GitHub
   private vulnerability reporting), use it. Only the maintainers can see the report.
2. If that option is not available, open an issue titled "Security contact request"
   **without any details** of the vulnerability, so that a private channel can be
   arranged.

Include what you can: the affected component and version or commit, a description of
the impact, steps or a minimal script to reproduce, and whether the issue is already
public. Never include real API keys or personal data; use placeholders.

This is a volunteer-maintained project without a guaranteed response time. The aim
is to acknowledge reports promptly and to coordinate fixes with the reporter before
details are published.

## Scope

In scope are vulnerabilities in the code of this repository, for example credential
leaks (keys in logs, exceptions, histories, sandboxes or requests to the wrong host),
path traversal in stores or registries, bypasses of the static check, the sandboxes'
documented restrictions or the approval gate, and unsafe defaults.

Out of scope, because they are documented limits rather than vulnerabilities (see
[Security](docs/guides/security.md)):

* `SubprocessSandbox` is not a security boundary; code it runs can use the network and
  read the user's files. Use `DockerSandbox` or stronger isolation for untrusted code.
* `AutoApprover` approves everything; it is an explicit unsafe opt-in.
* Prompt injection through tool results, search hits or files is not prevented by
  Orkexa itself.
* Vulnerabilities in model providers' APIs or SDKs, Docker or the Python runtime;
  please report those to their maintainers.
