"""
MCP Client wrapper for the multi-agent system.

Launches the MCP server as a subprocess and provides a clean interface
for agents to call tools via MCP protocol.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("MCP-Client")


class MCPToolClient:
    """Client that connects to the MCP tool server via stdio."""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._connected = False

    async def connect(self, server_script: str = "mcp_server.py"):
        """Launch MCP server subprocess and establish connection."""
        try:
            # Ensure MCP server subprocess can find backend modules
            server_dir = os.path.dirname(os.path.abspath(server_script))
            python_path = os.environ.get("PYTHONPATH", "")
            if server_dir not in python_path:
                python_path = f"{server_dir}{os.pathsep}{python_path}" if python_path else server_dir

            env = {
                **os.environ,
                "PYTHONPATH": python_path,
            }

            server_params = StdioServerParameters(
                command="python",
                args=[server_script],
                env=env,
            )

            logger.info(f"Launching MCP server: python {server_script}")

            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            self.session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self.session.initialize()

            # Cache tools
            tools_response = await self.session.list_tools()
            self._tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }
                for tool in tools_response.tools
            ]

            self._connected = True
            logger.info(f"MCP connected. Available tools: {[t['name'] for t in self._tools_cache]}")

        except Exception as e:
            logger.error(f"MCP connection failed: {e}")
            self._connected = False
            raise

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server. Returns parsed JSON result."""
        if not self._connected or not self.session:
            return {"error": "MCP client not connected"}

        try:
            logger.info(f"MCP tool call: {name}({arguments})")

            result = await self.session.call_tool(name, arguments)

            # Parse the result content
            if result.content:
                text_parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        text_parts.append(block.text)

                combined = "\n".join(text_parts)

                # Try to parse as JSON
                try:
                    parsed = json.loads(combined)
                    logger.info(f"MCP tool '{name}' returned successfully")
                    return parsed
                except json.JSONDecodeError:
                    return {"result": combined}
            else:
                return {"result": None}

        except Exception as e:
            logger.error(f"MCP tool call '{name}' failed: {e}")
            return {"error": str(e)}

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return cached list of available tools with descriptions."""
        if self._tools_cache is not None:
            return self._tools_cache

        if not self._connected or not self.session:
            return []

        try:
            tools_response = await self.session.list_tools()
            self._tools_cache = [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                }
                for tool in tools_response.tools
            ]
            return self._tools_cache
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []

    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Convert MCP tool schemas to OpenAI-compatible tool calling format."""
        if not self._tools_cache:
            return []

        tools = []
        for tool in self._tools_cache:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            })
        return tools

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def cleanup(self):
        """Close the MCP connection."""
        try:
            await self._exit_stack.aclose()
            self._connected = False
            logger.info("MCP client disconnected")
        except Exception as e:
            logger.error(f"MCP cleanup error: {e}")
