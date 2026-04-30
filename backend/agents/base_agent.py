import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from openai import OpenAI

from .schemas import ToolCall

logger = logging.getLogger("BaseAgent")

# OpenRouter configuration
# Default to a free model with good tool calling support
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Optional: identify app to OpenRouter for analytics
OPENROUTER_HEADERS = {
    "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "http://localhost"),
    "X-Title": os.environ.get("OPENROUTER_TITLE", "Industrial Learning Agent"),
}

# Shared OpenAI client (reused across agents)
_llm_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    """Get or create the OpenAI client configured for OpenRouter."""
    global _llm_client
    if _llm_client is None:
        if not OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set — LLM calls will fail")
        _llm_client = OpenAI(
            api_key=OPENROUTER_API_KEY or "missing-key",
            base_url=OPENROUTER_BASE_URL,
            default_headers=OPENROUTER_HEADERS,
        )
    return _llm_client


class BaseAgent(ABC):
    SYSTEM_PROMPT: str = ""

    # Override in subclass to restrict which MCP tools this agent can use.
    # If empty, ALL tools are available (not recommended).
    ALLOWED_TOOLS: List[str] = []

    def __init__(self, name: str, mcp_client=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.mcp_client = mcp_client  # MCPToolClient instance
        self._tool_call_log: List[ToolCall] = []

    # ============== MCP Tool Calling ==============

    async def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool via MCP client and log the call."""
        if not self.mcp_client or not self.mcp_client.is_connected:
            error_msg = "MCP client not connected"
            self.logger.error(f"[{self.name}] {error_msg}")
            tc = ToolCall(tool_name=name, arguments=arguments, result=None, success=False, error=error_msg)
            self._tool_call_log.append(tc)
            return {"error": error_msg}

        try:
            result = await self.mcp_client.call_tool(name, arguments)
            success = "error" not in result
            tc = ToolCall(tool_name=name, arguments=arguments, result=result, success=success,
                          error=result.get("error") if not success else None)
            self._tool_call_log.append(tc)
            self.logger.info(f"[{self.name}] MCP tool '{name}' -> {'OK' if success else 'ERROR'}")
            return result
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"[{self.name}] MCP tool '{name}' failed: {error_msg}")
            tc = ToolCall(tool_name=name, arguments=arguments, result=None, success=False, error=error_msg)
            self._tool_call_log.append(tc)
            return {"error": error_msg}

    def get_tool_calls(self) -> List[ToolCall]:
        """Return the tool call log and reset."""
        calls = list(self._tool_call_log)
        self._tool_call_log = []
        return calls

    # ============== LLM Call with MCP Tool Calling (via OpenRouter) ==============

    async def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_rounds: int = 10,
    ) -> Dict[str, Any]:
        """
        Call the LLM with tool calling via MCP.

        Uses OpenRouter (OpenAI-compatible) API.
        Tools are discovered from MCP server and passed in OpenAI format.
        """
        try:
            system = system_prompt or self.SYSTEM_PROMPT
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ]

            # Get tool schemas from MCP client, filtered to this agent's allowed tools
            tools = None
            if self.mcp_client and self.mcp_client.is_connected:
                all_tools = self.mcp_client.get_tools_for_ollama()
                if self.ALLOWED_TOOLS:
                    tools = [t for t in all_tools if t["function"]["name"] in self.ALLOWED_TOOLS]
                else:
                    tools = all_tools

            client = get_llm_client()

            for round_num in range(max_rounds):
                self.logger.info(f"[{self.name}] LLM round {round_num + 1}/{max_rounds} (model={LLM_MODEL})")

                kwargs: Dict[str, Any] = {"model": LLM_MODEL, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                tool_calls = getattr(msg, "tool_calls", None) or []

                # No tool calls → final response
                if not tool_calls:
                    text = msg.content or ""
                    self.logger.info(f"[{self.name}] Final response ({len(text)} chars)")
                    return {"full_response": text, "success": True, "rounds": round_num + 1}

                # Append assistant message (with tool calls) to history
                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                })

                # Process tool calls via MCP
                for tool_call in tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments
                    if isinstance(fn_args, str):
                        try:
                            fn_args = json.loads(fn_args)
                        except json.JSONDecodeError:
                            fn_args = {}

                    self.logger.info(f"[{self.name}] MCP tool call: {fn_name}({fn_args})")

                    result = await self.call_mcp_tool(fn_name, fn_args)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str, ensure_ascii=False),
                    })

            self.logger.warning(f"[{self.name}] Max tool rounds reached ({max_rounds})")
            return {"full_response": "", "success": False, "error": "Max tool rounds reached"}

        except Exception as e:
            error = str(e)
            self.logger.error(f"[{self.name}] LLM error: {error}")
            return {"full_response": "", "success": False, "error": error}

    # ============== Legacy LLM Call (no tools) ==============

    async def call_ollama(self, user_message: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Legacy call without tool support (name kept for backward compat, now uses OpenRouter)."""
        try:
            system = system_prompt or self.SYSTEM_PROMPT
            client = get_llm_client()
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
            )
            text = response.choices[0].message.content or ""
            return {"full_response": text, "success": True}
        except Exception as e:
            return {"full_response": "", "success": False, "error": str(e)}

    # ============== Text Parsing Utilities ==============

    @staticmethod
    def extract_field(text: str, field_name: str) -> str:
        try:
            start = text.find(field_name + ":")
            if start == -1:
                return ""
            start += len(field_name) + 1
            while start < len(text) and text[start] in " \t":
                start += 1
            end = text.find("\n", start)
            if end == -1:
                end = len(text)
            return text[start:end].strip()
        except Exception:
            return ""

    @staticmethod
    def extract_list(text: str, section_name: str) -> List[str]:
        try:
            start = text.find(section_name + ":")
            if start == -1:
                return []
            start += len(section_name) + 1
            end = text.find("\n\n", start)
            if end == -1:
                end = len(text)
            section = text[start:end]
            items = []
            for line in section.split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    items.append(line[2:].strip())
            return items
        except Exception:
            return []

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass
