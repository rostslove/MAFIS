"""Scribe: asks the LLM to turn evaluation facts into a report."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent, extract_json_block
from .schemas import DataContext, ScribeReport
from .structured import parse_scribe_report_object

logger = logging.getLogger("Scribe")


class Scribe(BaseAgent):
    """Reporting agent called once after graph evaluation."""

    ALLOWED_TOOLS = ["generate_report"]

    SYSTEM_PROMPT = """You are the Scribe Agent for MultiAgentFedot.IndustrialSystem (MAFIS).
Write a concise technical report from the provided factual evaluation payload.

Return one strict JSON object:
{
  "title": "...",
  "summary": "...",
  "methodology": "...",
  "results": "...",
  "recommendations": ["..."]
}

Use only facts present in the payload. Do not invent metrics, causes, or
recommendations. Return JSON only."""

    def __init__(self, name: str = "Scribe", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, all_iterations: List[Dict[str, Any]], data_context: DataContext) -> ScribeReport:
        self._tool_call_log = []
        report = ScribeReport()

        try:
            tool_report = await self.call_mcp_tool(
                "generate_report",
                {"evaluations_json": json.dumps(all_iterations, ensure_ascii=False)},
            )
            if not isinstance(tool_report, dict):
                tool_report = {}

            payload = {
                "task_type": data_context.task_type,
                "primary_metric": data_context.primary_metric,
                "data_profile": data_context.profile,
                "tool_report": tool_report,
                "iterations": all_iterations,
            }
            response = await self.call_llm(
                "Write the report JSON for this payload:\n"
                f"{json.dumps(payload, ensure_ascii=False)}",
                max_rounds=1,
                use_tools=False,
            )
            raw_text = response.get("full_response", "") or response.get("error", "")
            parsed = extract_json_block(raw_text)
            llm_report = parse_scribe_report_object(parsed) if parsed else None
            if llm_report:
                report.title = llm_report.title
                report.summary = llm_report.summary
                report.methodology = llm_report.methodology
                report.results = llm_report.results
                report.recommendations = llm_report.recommendations
            report.best_graph_mermaid = tool_report.get("best_graph_mermaid", "")
            report.full_response = raw_text
            report.tool_calls = self.get_tool_calls()
            logger.info("[Scribe] report ready")
            return report

        except Exception as exc:
            logger.exception("[Scribe] failed")
            report.full_response = str(exc)
            report.tool_calls = self.get_tool_calls()
            return report
