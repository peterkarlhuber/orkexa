# ADR 0012 – MacOSSandbox removed (Docker-first)

Status: accepted (2026-11-02)

## Context

Orkexa #28 added an experimental `MacOSSandbox` (ORK-014): each call started a Python process that confined
itself with Apple's Seatbelt (`sandbox_init`, a deny-default profile), plus resource limits, a memory watchdog
and an empty environment. It was reported as `maturity="experimental"` and never used by Nodestro.

An independent security review (2026-11-02) found it **not releasable**:

* **MS-1:** sandboxed code could read the arguments and environment of other processes of the same user
  (including the host process and its secrets) through `sysctl(KERN_PROCARGS2)`.
* **MS-2 to MS-10:** further findings; details are in the review report (see Evidence).

The review branch prototyped fixes for MS-1, MS-2, MS-4, MS-5 and MS-7 and a cleanup for MS-3 (leftover System V
shared memory segments, which Seatbelt cannot deny). Instead of hardening and re-reviewing a same-user Seatbelt
sandbox further, the owner decided on 2026-11-02 to rely on containers.

## Decision

**Docker-first.** `DockerSandbox` is the default sandbox for untrusted (model-written) code on all platforms.
The native macOS Seatbelt sandbox is never activated and is removed from Orkexa:

* `orkexa/dynamic/_macos_confine.py`, the Seatbelt profile, the memory watchdog and their tests are deleted, and
  the `Platforms` workflow no longer requires a native sandbox on macOS (it still runs the offline suite there).
* The name `orkexa.dynamic.MacOSSandbox` stays importable for one deprecation cycle so that status reports
  (nodestro-runtime's sandbox selection) keep importing. It never runs code: `probe()` always returns
  `available=False`, `reason="removed"`, `detail="removed; use DockerSandbox (Docker-first)"`; its capabilities
  claim no isolation; creating one emits a `DeprecationWarning`; `validate()` and `call()` raise `SandboxError`
  (`call()` as `ToolExecutionError`) before any file is written. It will be deleted in a later release.
* `SubprocessSandbox` stays trusted-only (no security boundary). Nodestro's own `LinuxSandbox` is unaffected.
  Docker is also the Windows path (ORK-015).

## Evidence

* Review report: <https://claude.ai/artifact/9jAFwehAYsrkQyxSWFzsv1>
* Decision document: <https://claude.ai/artifact/Uy4q3zfte6PDhAC9zBwaDY>
* Review branch commits `c084cae` (MS-1, MS-2, MS-4, MS-5, MS-7 fixes, MS-3 cleanup) and `4282be9` (kill test),
  reachable through the merge of the removal branch `claude/project-thread-qz41p9`; that branch first reverts
  them, then removes the backend.
* GitHub Actions runs 41427847803 and 41430332558 (macOS review runs of the branch).

## Consequences

* On macOS and Windows, untrusted tools need Docker; without it they stay blocked (fail closed).
* Code that constructs `MacOSSandbox` keeps working for reporting but gets a `DeprecationWarning`; code that
  tries to run tools with it fails loudly instead of silently running unconfined.
* Generic attack probes from the removed macOS attack suite may be ported to a Docker attack suite later.
