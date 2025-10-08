"""
Minimal MCP server for integration with LangChain MCPAdapter.
This server exposes a few example tools via stdio transport.
"""

import json
import sys
from typing import Any, Dict

# ------------------------
# Example tool implementations
# ------------------------

def hello_tool(name: str) -> str:
    """Simple example tool."""
    return f"Hello, {name}! This is MCP tool response."

def sum_tool(a: float, b: float) -> float:
    """Example numeric tool."""
    return a + b

# Map tool names to functions
TOOLS = {
    "hello_tool": hello_tool,
    "sum_tool": sum_tool,
}

# ------------------------
# MCP server loop (stdio transport)
# ------------------------
def read_message() -> Dict[str, Any]:
    """Read a JSON message from stdin."""
    line = sys.stdin.readline()
    if not line:
        return {}
    return json.loads(line)

def send_message(message: Dict[str, Any]) -> None:
    """Write a JSON message to stdout."""
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def main():
    send_message({"type": "ready", "tools": list(TOOLS.keys())})
    while True:
        try:
            msg = read_message()
            if not msg:
                continue

            tool_name = msg.get("tool")
            args = msg.get("args", {})

            if tool_name in TOOLS:
                result = TOOLS[tool_name](**args)
                send_message({"type": "result", "tool": tool_name, "result": result})
            else:
                send_message({"type": "error", "message": f"Unknown tool: {tool_name}"})
        except Exception as e:
            send_message({"type": "error", "message": str(e)})

if __name__ == "__main__":
    main()
