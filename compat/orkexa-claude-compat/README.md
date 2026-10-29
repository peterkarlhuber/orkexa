# orkexa-claude-compat

Lets code written for `claude-agent` 0.8.0 run **unchanged** on
[Orkexa](https://github.com/peterkarlhuber/orkexa):

```python
from claude_agent import ClaudeAgent, Tool
from claude_agent.dynamic import CodingAgent, ToolBuilder, ToolRegistry
```

The package `claude_agent` installed by this distribution contains no business
logic: every `claude_agent.<module>` is the corresponding Orkexa module
(`claude_agent.retrieval is orkexa.retrieval`), so patches and `isinstance` checks
work across both names.

## Installation

The original `claude-agent` distribution ships the same top-level package
`claude_agent`; the two must not be installed together (pip cannot detect such file
conflicts). Replace it:

```bash
pip uninstall -y claude-agent
pip install orkexa-claude-compat
python -m orkexa.compat          # verifies which distribution provides claude_agent
```

If both are installed, `import claude_agent` raises an `ImportError` that explains
how to repair the environment. To go back: `pip uninstall -y orkexa-claude-compat`
and `pip install claude-agent`.
