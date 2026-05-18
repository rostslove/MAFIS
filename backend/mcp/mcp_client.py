"""In-process client for the local tool registry."""

import importlib.util
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MCP-Client")


class MCPToolClient:
    """Loads tool metadata and dispatches local calls."""

    def __init__(self):
        self._server_module: Optional[Any] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._connected = False

    async def connect(self, server_script: Optional[str] = None):
        """Load the local MCP tool registry."""
        try:
            if server_script is None:
                server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
            server_path = os.path.abspath(server_script)
            server_dir = os.path.dirname(server_path)
            if server_dir not in sys.path:
                sys.path.insert(0, server_dir)

            spec = importlib.util.spec_from_file_location("graph_automl_mcp_server", server_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Cannot load MCP server module from {server_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            self._server_module = module
            self._tools_cache = module.mcp.list_tools()
            self._connected = True
            logger.info("MCP tools connected locally: %s", [t["name"] for t in self._tools_cache])

        except Exception as exc:
            logger.error("MCP connection failed: %s", exc)
            self._connected = False
            raise

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a local MCP tool. Returns parsed JSON result."""
        if not self._connected or self._server_module is None:
            return {"error": "MCP client not connected"}

        try:
            logger.info("MCP tool call: %s(%s)", name, arguments)
            result = self._server_module.mcp.call_tool(name, arguments)

            if isinstance(result, str):
                try:
                    return json.loads(result)
                except json.JSONDecodeError:
                    return {"result": result}
            if isinstance(result, dict):
                return result
            return {"result": result}

        except Exception as exc:
            logger.error("MCP tool call '%s' failed: %s", name, exc)
            return {"error": str(exc)}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return cached list of available tools with descriptions."""
        return self._tools_cache or []

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to OpenAI-compatible tool calling format."""
        if not self._tools_cache:
            return []

        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for tool in self._tools_cache
        ]

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def cleanup(self):
        """Mark the local tool registry as disconnected."""
        self._connected = False
        self._server_module = None
        logger.info("MCP client disconnected")
