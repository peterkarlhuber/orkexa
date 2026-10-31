# ADR 0005 – Flat subsystem modules, new `core/` and `providers/` packages

Status: accepted

The subsystems (`tools`, `content`, `memory`, `files`, `retrieval`, `evaluation`,
`usage`, `pricing`, `errors`, `dynamic/`) are flat modules. The runtime lives in
`core/` (data model, runtime, capabilities, conversion, schemas) and the adapters in
`providers/`. Reasons: each subsystem is one importable module with a stable path, and
splitting the subsystems into packages would add indirection without benefit.
