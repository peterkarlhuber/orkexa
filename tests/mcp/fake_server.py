"""A tiny MCP server over stdio for the client tests (standard library only)."""

from __future__ import annotations

import json
import sys

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the text.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
            "additionalProperties": False,
        },
        "annotations": {"readOnlyHint": True},
    },
    {
        "name": "mail.send",
        "description": "Send a mail.",
        "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
    },
]


def send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message:
            continue
        if mode == "silent" and method == "tools/call":
            continue
        if method == "initialize":
            version = "1999-01-01" if mode == "old" else "2025-06-18"
            sys.stdout.write("not json\n")
            send({"jsonrpc": "2.0", "method": "ping", "id": "srv-1"})
            send({"jsonrpc": "2.0", "method": "sampling/createMessage", "id": "srv-2"})
            result = {
                "protocolVersion": version,
                "serverInfo": {"name": "fake", "version": "1"},
                "capabilities": {},
            }
        elif method == "tools/list":
            cursor = (message.get("params") or {}).get("cursor")
            result = {"tools": TOOLS[:1], "nextCursor": "p2"} if not cursor else {"tools": TOOLS[1:]}
        elif method == "tools/call":
            params = message["params"]
            if params["name"] == "echo":
                text = params["arguments"]["text"]
                if text == "crash":
                    sys.exit(3)
                result = {"content": [{"type": "text", "text": text}], "isError": text == "fail"}
            else:
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": message["id"],
                        "error": {"code": -32602, "message": "Unknown tool"},
                    }
                )
                continue
        else:
            send({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "nope"}})
            continue
        send({"jsonrpc": "2.0", "id": message["id"], "result": result})


if __name__ == "__main__":
    main()
