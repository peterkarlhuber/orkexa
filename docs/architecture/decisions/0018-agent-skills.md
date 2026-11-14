# ADR 0018 – Agent Skills packages

Status: accepted (2026-11-14, NAL-B3; development approval E-1 of the owner)

## Context

Products built on Orkexa (first Nodestro) want reusable instructions for agents in the open Agent Skills format
([specification](https://agentskills.io/specification)): a folder with a `SKILL.md` (YAML front matter with
`name` and `description`, then Markdown instructions) and optional `scripts/`, `references/` and `assets/`. A
skill is imported from outside, so it is untrusted input: a ZIP file can carry path traversal, symbolic links or
a ZIP bomb, and its front matter is YAML, whose full feature set (anchors, tags, implicit types) is a needless
risk. The format has no versions, signatures or permissions of its own.

## Decision

1. **`orkexa.skills` is pure.** Input is ZIP bytes (`read_zip`) or `{relative path: bytes}` (`parse_skill`);
   output is pydantic models and bytes. No file system, network or model access, no new dependency.
2. **Front matter without a YAML library.** A strict subset parser reads what skills use: plain, quoted and
   block scalars, flow and block sequences, and one level of mapping (`metadata`). Anchors, aliases, tags, flow
   mappings and deeper nesting are refused. Every scalar stays a string.
3. **Unsafe packages are refused before parsing.** Defaults (`SkillLimits`): ZIP at most 10 MB, unpacked at most
   30 MB, at most 500 files, depth at most 6, `SKILL.md` at most 100 KB; no absolute paths, drive letters,
   backslashes, `.`/`..` segments, control characters, non-NFC names, names that differ only in case, symbolic
   links or other special files, encrypted entries, duplicates, or entries that expand more than 100 times. The
   declared size of an entry is checked against what is read (one byte more is read to catch a lying header).
   `__MACOSX/`, `.DS_Store` and `Thumbs.db` are dropped.
4. **Checks follow the specification strictly**, with one difference from `skills-ref`: an unknown front matter
   key is a warning, not an error, so skills from other clients still import. `allowed-tools` is accepted (a list
   is joined with spaces) and reported as a warning: it never grants tools. Long instructions (over 500 lines or
   about 5,000 tokens) and scripts are warnings.
5. **Loading stages.** `catalog_entry` gives name and description (stage 1), `render_activation` the instructions
   in a `<skill_content name="...">` wrapper with a list of resource paths but not their content (stage 2). Reading
   one resource (stage 3) is left to the product, which decides what a model may read (`is_text_path`).
6. **Byte-exact packages.** Files keep their bytes; `package_digest` (`sha256:` over sorted paths and bytes)
   identifies a version, and `write_zip` writes a reproducible ZIP in a folder named after the skill, every file
   with mode `0644`, so a round trip gives the same files and the same digest.

## Consequences

* Storage, versions, assignment to agents and the tools that load a skill at run time belong to the product
  (Nodestro runtime and API); this module stays a library both can test against.
* A skill can only add text to a prompt. Permissions keep coming from the agent's tools and policies.
* Scripts are stored and shown but never executed here. Running them is a later, separate decision (NAL-B4).
