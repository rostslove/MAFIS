"""Base agent: OpenAI-compatible LLM tool calling + MCP."""

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from .schemas import ToolCall

logger = logging.getLogger("BaseAgent")


def _load_project_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_project_env()

DEFAULT_LLM_MODEL = "qwen2.5-coder:7b"
DEFAULT_LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_LLM_MODEL)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL", DEFAULT_LLM_BASE_URL)
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "ollama"

_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
            default_headers={
                "HTTP-Referer": "http://localhost",
                "X-Title": "GraphAutoML Agent",
            },
        )
    return _client


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """Extract first JSON object from text (handles ```json fences and bare braces)."""
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # Greedy brace-matching scan for first balanced JSON object
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


class BaseAgent(ABC):
    SYSTEM_PROMPT: str = ""
    ALLOWED_TOOLS: List[str] = []

    def __init__(self, name: str, mcp_client=None):
        self.name = name
        self.logger = logging.getLogger(name)
        self.mcp_client = mcp_client
        self._tool_call_log: List[ToolCall] = []

    async def call_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Call an MCP tool and log the call."""
        if not self.mcp_client or not self.mcp_client.is_connected:
            tc = ToolCall(name, arguments, None, False, "MCP not connected")
            self._tool_call_log.append(tc)
            return {"error": "MCP not connected"}

        try:
            result = await self.mcp_client.call_tool(name, arguments)
            success = isinstance(result, dict) and "error" not in result
            tc = ToolCall(name, arguments, result, success, result.get("error") if isinstance(result, dict) else None)
            self._tool_call_log.append(tc)
            self.logger.info(f"[{self.name}] {name} -> {'OK' if success else 'ERR'}")
            return result
        except Exception as e:
            tc = ToolCall(name, arguments, None, False, str(e))
            self._tool_call_log.append(tc)
            self.logger.error(f"[{self.name}] {name} failed: {e}")
            return {"error": str(e)}

    def get_tool_calls(self) -> List[ToolCall]:
        calls = list(self._tool_call_log)
        self._tool_call_log = []
        return calls

    async def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        max_rounds: int = 8,
        use_tools: bool = True,
    ) -> Dict[str, Any]:
        """LLM tool-calling loop. Returns {full_response, success, rounds}."""
        try:
            messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt or self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            tools = None
            if use_tools and self.mcp_client and self.mcp_client.is_connected:
                all_tools = self.mcp_client.get_tools_for_openai()
                if self.ALLOWED_TOOLS:
                    tools = [t for t in all_tools if t["function"]["name"] in self.ALLOWED_TOOLS]
                else:
                    tools = all_tools

            client = get_llm_client()

            for round_num in range(max_rounds):
                kwargs: Dict[str, Any] = {"model": LLM_MODEL, "messages": messages}
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None) or []

                if not tool_calls:
                    return {"full_response": msg.content or "", "success": True, "rounds": round_num + 1}

                messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in tool_calls
                    ],
                })

                for tc in tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    result = await self.call_mcp_tool(tc.function.name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, default=str, ensure_ascii=False),
                    })

            return {"full_response": "", "success": False, "error": "Max tool rounds reached"}

        except Exception as e:
            self.logger.error(f"[{self.name}] LLM error: {e}")
            return {"full_response": "", "success": False, "error": str(e)}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        ...
