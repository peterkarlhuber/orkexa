# MCP client

`orkexa.mcp` connects an agent to a [Model Context Protocol](https://modelcontextprotocol.io) server and offers the
server's tools as ordinary Orkexa tools. It needs only the standard library.

```python
from orkexa import Agent
from orkexa.mcp import HttpTransport, MCPClient, StdioTransport, mcp_tools

# A server started as a child process (the MCP stdio transport). The child gets a minimal
# environment, never the parent's API keys, unless env= passes one.
with MCPClient(StdioTransport(["my-mcp-server", "--read-only"])) as client:
    agent = Agent(model="ollama:qwen3.5:4b", tools=mcp_tools(client, allow=lambda tool: tool.read_only))
    print(agent.ask("Which invoices arrived this week?"))

# A Streamable HTTP endpoint. Loopback only unless allow_remote=True (then https:// is required);
# redirects are never followed.
client = MCPClient(HttpTransport("http://127.0.0.1:7341/mcp", headers={"Authorization": "Bearer ..."}))
```

## Untrusted results

A tool result is text written by a third party (an email, a web page, a database row). `mcp_tools` wraps every
result in an `<untrusted_data source="mcp:server/tool">` frame that the text cannot close and tells the model to
treat it as data. An error message from the server is framed the same way. The frame lowers the risk of prompt injection; it does not remove it. Keep a `before_tool` policy
or an approver for every tool with side effects, and prefer servers that enforce permissions themselves.

## Limits

* Protocol revisions 2025-06-18, 2025-03-26 and 2024-11-05.
* Tools only: resources, prompts, sampling and elicitation are not supported. Requests the server sends to the
  client are answered with "method not found" (except `ping`).
* The input is checked against the top level of the tool's schema; the server validates the rest.
* A message is at most 16 MiB.

::: orkexa.mcp.MCPClient

::: orkexa.mcp.StdioTransport

::: orkexa.mcp.HttpTransport

::: orkexa.mcp.Transport

::: orkexa.mcp.mcp_tools

::: orkexa.mcp.MCPTool

::: orkexa.mcp.MCPToolInfo

::: orkexa.mcp.MCPToolResult

::: orkexa.mcp.MCPServerInfo

::: orkexa.mcp.MCPError

::: orkexa.mcp.frame_untrusted
