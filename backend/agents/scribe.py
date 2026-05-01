"""Scribe: turns a GraphAutoML evaluation into a compact final report."""

import logging
from typing import Any, Dict, List

import json

from .base_agent import BaseAgent
from .schemas import DataContext, ScribeReport

logger = logging.getLogger("Scribe")


class Scribe(BaseAgent):
    """Reporting agent called once after graph evaluation."""

    ALLOWED_TOOLS = ["generate_report"]

    SYSTEM_PROMPT = """You are the Scribe Agent for GraphAutoML.
Write a concise technical report from the approved graph evaluation summary.
Return JSON only:
{
  "title": "...",
  "summary": "...",
  "methodology": "...",
  "results": "...",
  "recommendations": ["..."]
}"""

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

            llm_report: Dict[str, Any] = {}
            best_score = tool_report.get("best_score", 0)
            n_evaluations = tool_report.get("n_evaluations", tool_report.get("n_iterations", len(all_iterations)))

            report.title = llm_report.get("title") or f"GraphAutoML report: {data_context.task_type}"
            report.summary = llm_report.get("summary") or (
                f"Completed {n_evaluations} approved graph evaluation(s). Best test score: {best_score:.4f}."
            )
            report.methodology = llm_report.get("methodology") or (
                "Architect prepared a graph, Engineer trained the approved graph, "
                "and Critic evaluated graph quality using metrics, validation stability, diagnostics and data profile."
            )
            report.results = llm_report.get("results") or self._default_results(tool_report)
            report.recommendations = llm_report.get("recommendations") or self._default_recommendations(all_iterations)
            report.best_graph_mermaid = tool_report.get("best_graph_mermaid", "")
            report.full_response = json.dumps(llm_report, ensure_ascii=False) if llm_report else json.dumps(tool_report, ensure_ascii=False)
            report.tool_calls = self.get_tool_calls()
            logger.info("[Scribe] report ready")
            return report

        except Exception as exc:
            logger.exception("[Scribe] failed")
            report.title = "GraphAutoML report"
            report.summary = f"Report generation failed: {exc}"
            report.recommendations = ["Review backend logs and rerun the orchestration"]
            report.tool_calls = self.get_tool_calls()
            return report

    @staticmethod
    def _default_results(tool_report: Dict[str, Any]) -> str:
        summaries = tool_report.get("evaluation_summaries", tool_report.get("iteration_summaries", []))
        if not summaries:
            return "No successful evaluation summaries were produced."
        lines = [
            f"Evaluation {item.get('iteration')}: "
            f"{item.get('primary_metric') or 'score'}={Scribe._fmt(item.get('primary_metric_value', item.get('graph_score', 0)))}, "
            f"ranking_score={Scribe._fmt(item.get('graph_score', 0))}, "
            f"decision={item.get('winner', 'n/a')}"
            for item in summaries
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _default_recommendations(iterations: List[Dict[str, Any]]) -> List[str]:
        if not iterations:
            return ["Run at least one approved graph evaluation"]
        last = iterations[-1].get("critic", {})
        diagnostics = []
        diagnostics.extend(iterations[-1].get("engineer", {}).get("diagnostics", []) or [])
        diagnostics.extend(last.get("diagnostics", []) or [])
        recommendations = []
        for diagnostic in diagnostics:
            for rec in diagnostic.get("recommendations", []) or []:
                if rec not in recommendations:
                    recommendations.append(rec)
        if recommendations:
            return recommendations[:4]
        suggestions = last.get("suggested_mutations", [])
        if suggestions:
            return ["Review the final Critic mutations before approving a new graph", "Use explicit node parameters for manual refinement"]
        return ["Export and validate the best graph on a held-out dataset"]
