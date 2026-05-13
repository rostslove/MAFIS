"""Scribe: asks the LLM to turn evaluation facts into a report."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent, extract_json_block, extract_json_value
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
recommendations. When previous_evaluations and current_vs_best are present,
explicitly compare the current run against the saved best evaluation. If the
current run is below the saved best, say so clearly, include the metric delta,
and recommend keeping/restoring the best graph rather than presenting the
latest accepted run as the best result. Return JSON only."""

    def __init__(self, name: str = "Scribe", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(
        self,
        all_iterations: List[Dict[str, Any]],
        data_context: DataContext,
        previous_evaluations: List[Dict[str, Any]] | None = None,
        best_evaluation: Dict[str, Any] | None = None,
        current_vs_best: Dict[str, Any] | None = None,
    ) -> ScribeReport:
        self._tool_call_log = []
        report = ScribeReport()

        try:
            previous_evaluations = previous_evaluations or []
            best_evaluation = best_evaluation or {}
            current_vs_best = current_vs_best or {}
            report.best_evaluation = best_evaluation
            report.current_vs_best = current_vs_best
            report.best_graph = best_evaluation.get("graph", {}) if isinstance(best_evaluation, dict) else {}

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
                "previous_evaluations": previous_evaluations,
                "best_evaluation": best_evaluation,
                "current_vs_best": current_vs_best,
            }
            response = await self.call_llm(
                "Write the report JSON for this payload:\n"
                f"{json.dumps(payload, ensure_ascii=False)}",
                max_rounds=1,
                use_tools=False,
            )
            raw_text = response.get("full_response", "") or response.get("error", "")
            parsed = self._report_payload_from_model_text(raw_text)
            llm_report = parse_scribe_report_object(parsed) if parsed else None
            if llm_report and self._has_report_content(llm_report.dict()):
                report.title = llm_report.title
                report.summary = llm_report.summary
                report.methodology = llm_report.methodology
                report.results = llm_report.results
                report.recommendations = llm_report.recommendations
            else:
                self._fill_from_tool_report(report, tool_report)
            self._attach_best_comparison(report, current_vs_best)
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

    @staticmethod
    def _report_payload_from_model_text(raw_text: str) -> Dict[str, Any] | None:
        parsed = extract_json_value(raw_text) or extract_json_block(raw_text)
        if isinstance(parsed, list):
            parsed = next((item for item in parsed if isinstance(item, dict)), None)
        if not isinstance(parsed, dict):
            return None
        fields = {"title", "summary", "methodology", "results", "recommendations"}
        if fields.intersection(parsed):
            return parsed
        for key in ("report", "scribe_report", "final_report", "result"):
            nested = parsed.get(key)
            if isinstance(nested, dict) and fields.intersection(nested):
                return nested
        return None

    @staticmethod
    def _has_report_content(payload: Dict[str, Any]) -> bool:
        return bool(
            str(payload.get("title") or payload.get("summary") or payload.get("methodology") or payload.get("results") or "").strip()
            or payload.get("recommendations")
        )

    @staticmethod
    def _fill_from_tool_report(report: ScribeReport, tool_report: Dict[str, Any]) -> None:
        if not isinstance(tool_report, dict):
            return
        summaries = tool_report.get("evaluation_summaries") or tool_report.get("iteration_summaries") or []
        best_score = tool_report.get("best_score")
        report.title = report.title or "MAFIS report"
        report.summary = report.summary or f"Evaluated {tool_report.get('n_evaluations', len(summaries) or 0)} graph run(s)."
        if best_score is not None:
            report.results = report.results or f"Best observed score: {best_score}."
        if not report.recommendations:
            report.recommendations = ["Review train/test metrics and keep the graph with the best hold-out score."]

    @staticmethod
    def _attach_best_comparison(report: ScribeReport, comparison: Dict[str, Any]) -> None:
        if not isinstance(comparison, dict) or comparison.get("status") != "current_below_best":
            return
        metric = comparison.get("primary_metric") or "score"
        try:
            current = float(comparison.get("current_score"))
            best = float(comparison.get("best_score"))
            delta = current - best
            current_text = f"{current:.4f}"
            best_text = f"{best:.4f}"
            delta_text = f"{abs(delta):.4f}"
        except (TypeError, ValueError):
            current_text = str(comparison.get("current_score", ""))
            best_text = str(comparison.get("best_score", ""))
            delta_text = str(comparison.get("delta", ""))
        sentence = (
            f"Cross-run comparison: the current run reached {metric}={current_text}, "
            f"which is below the saved best {metric}={best_text} by {delta_text}."
        )
        if sentence not in report.summary:
            report.summary = f"{report.summary} {sentence}".strip()
        if sentence not in report.results:
            report.results = f"{report.results} {sentence}".strip()
        recommendation = "Keep or restore the saved best graph before continuing optimization."
        if recommendation not in report.recommendations:
            report.recommendations.insert(0, recommendation)
