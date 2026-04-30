import json
import logging
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from .schemas import DataContext, ScribeReport

logger = logging.getLogger("Scribe")


class Scribe(BaseAgent):
    """Scribe: generates final report via MCP tools. Called ONCE at end."""

    ALLOWED_TOOLS = ["generate_report"]

    SYSTEM_PROMPT = """You are a Solution Architect writing a comprehensive ML report.

You have access to MCP tool:
1. generate_report - Compile iteration history into a structured report.

Call generate_report with iteration data, then provide a polished summary.

Your report should include:
TITLE: Solution name
SUMMARY: Executive summary
METHODOLOGY: What was tried
RESULTS: Final metrics
RECOMMENDATIONS: Next steps
"""

    def __init__(self, name: str = "Scribe", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, all_iterations: List[Dict[str, Any]], data_context: DataContext) -> ScribeReport:
        """Generate final report."""
        self._tool_call_log = []
        result = ScribeReport()

        try:
            logger.info("[Scribe] Generating final report")

            # Build iterations summary for context
            iterations_text = ""
            for it in all_iterations:
                it_num = it.get("iteration", "?")
                eng = it.get("engineer", {})
                crit = it.get("critic", {})
                iterations_text += f"""
--- Iteration {it_num} ---
Best model: {eng.get('best_model', 'N/A')} (score: {eng.get('best_score', 0):.4f})
Fedot score: {eng.get('fedot_result', {}).get('score', 0):.4f}
Winner: {crit.get('winner', 'N/A')}
Suggested changes: {crit.get('suggested_fedot_changes', {})}
"""

            context = f"""Generate a comprehensive report for this ML experiment.

Data: {data_context.csv_path}
Task: {data_context.task_type}
Samples: {data_context.profile.get('n_samples', 'N/A')}
Features: {data_context.profile.get('n_features', 'N/A')}

ITERATION RESULTS:
{iterations_text}

Please call generate_report with the iteration data, then provide a polished summary.
"""

            response = await self.call_llm(context)
            text = response.get("full_response", "")

            result.title = self.extract_field(text, "TITLE") or "ML Experiment Report"
            result.summary = self.extract_field(text, "SUMMARY") or text[:500]
            result.methodology = self.extract_field(text, "METHODOLOGY") or ""
            result.results = self.extract_field(text, "RESULTS") or ""
            result.recommendations = self.extract_list(text, "RECOMMENDATIONS") or ["Monitor model performance"]
            result.full_response = text
            result.tool_calls = self.get_tool_calls()

            logger.info(f"[Scribe] Report: {result.title}")
            return result

        except Exception as e:
            logger.error(f"[Scribe] Error: {e}")
            return ScribeReport(
                title="ML Experiment Report",
                summary="Report generation failed",
                recommendations=["Review logs"],
                full_response=str(e),
                tool_calls=self.get_tool_calls(),
            )
