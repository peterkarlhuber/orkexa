# Skills

Agent Skills packages ([agentskills.io](https://agentskills.io/specification)): read, check, describe and write
them. The module is pure: it needs only pydantic, never touches the disk or the network and never runs a script.
The design is in [ADR 0018](../architecture/decisions/0018-agent-skills.md).

```python
from orkexa.skills import read_zip, validate, render_activation

package = read_zip(data)  # raises SkillPackageError for unsafe or oversized packages
errors = [issue for issue in validate(package) if issue.severity == "error"]
prompt_part = render_activation(package)
```

::: orkexa.skills.read_zip

::: orkexa.skills.parse_skill

::: orkexa.skills.validate

::: orkexa.skills.catalog_entry

::: orkexa.skills.render_activation

::: orkexa.skills.package_digest

::: orkexa.skills.write_zip

::: orkexa.skills.is_text_path

::: orkexa.skills.SkillPackage

::: orkexa.skills.SkillFile

::: orkexa.skills.SkillIssue

::: orkexa.skills.SkillLimits

::: orkexa.skills.SkillPackageError

::: orkexa.skills.parse_frontmatter

::: orkexa.skills.split_frontmatter

::: orkexa.skills.FrontmatterError
