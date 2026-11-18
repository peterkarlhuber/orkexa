"""A Model Context Protocol (MCP) client: tools of an MCP server as Orkexa tools.

The client speaks JSON-RPC 2.0 over one of two transports:

* :class:`StdioTransport` starts the server as a child process and talks over its standard
  input and output (newline-delimited JSON, the MCP ``stdio`` transport).
* :class:`HttpTransport` posts to a Streamable HTTP endpoint and reads a JSON or
  ``text/event-stream`` answer. Only loopback hosts are allowed unless ``allow_remote=True``.

:func:`mcp_tools` turns the server's tools into :class:`MCPTool` objects that any
:class:`~orkexa.Agent` accepts. Tool results are text written by a third party: by default
they are wrapped in an ``<untrusted_data>`` frame that tells the model to treat them as data,
never as instructions. Side effects still need a ``before_tool`` policy; the frame does not
prevent prompt injection on its own.

Example::

    from orkexa import Agent
    from orkexa.mcp import MCPClient, StdioTransport, mcp_tools

    with MCPClient(StdioTransport(["my-mcp-server"])) as client:
        agent = Agent(model="ollama:qwen3.5:4b", tools=mcp_tools(client, include={"search"}))
        print(agent.ask("Find the newest invoice."))
"""

from __future__ import annotations

from ._client import (
    PROTOCOL_VERSION,
    HttpTransport,
    MCPClient,
    MCPError,
    MCPServerInfo,
    MCPTool,
    MCPToolInfo,
    MCPToolResult,
    StdioTransport,
    Transport,
    frame_untrusted,
    mcp_tools,
)

__all__ = [
    "PROTOCOL_VERSION",
    "HttpTransport",
    "MCPClient",
    "MCPError",
    "MCPServerInfo",
    "MCPTool",
    "MCPToolInfo",
    "MCPToolResult",
    "StdioTransport",
    "Transport",
    "frame_untrusted",
    "mcp_tools",
]
