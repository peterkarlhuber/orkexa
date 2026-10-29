# ADR 0005 – Flat subsystem modules, new `core/` and `providers/` packages

Status: accepted

The subsystems (`tools`, `content`, `memory`, `files`, `retrieval`, `evaluation`,
`usage`, `pricing`, `errors`, `dynamic/`) keep the 0.8.0 module names. New code lives
in `core/` (data model, runtime, capabilities, conversion, schemas) and `providers/`.
Reasons: each 0.8.0 module maps 1:1 to an Orkexa module, so the legacy shim can alias
modules instead of copying code, and documented submodule imports keep working.
Splitting the subsystems into packages would add indirection without benefit.
