import json
import logging
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from .schemas import (
    DataContext, EngineerResult, CriticFeedback, is_ts_task, EXPLAIN_METHODS,
)

logger = logging.getLogger("Critic")


class Critic(BaseAgent):
    """Critic: evaluates models via MCP tools, provides structured feedback."""

    ALLOWED_TOOLS = [
        "analyze_errors", "get_feature_importance",
        "explain_model", "get_additional_metrics",
    ]

    SYSTEM_PROMPT = """You are an ML Critic. Your job is to evaluate model results and provide actionable feedback.

You have access to MCP tools:
1. analyze_errors - Analyze prediction errors and compare baselines vs Fedot.
2. get_feature_importance - Feature importance from tree-based models (classification/regression only).
3. explain_model - Explain Fedot.Industrial model (methods: point, recurrence, shap, lime).
4. get_additional_metrics - Calculate extra metrics from Fedot (e.g. rmse, mae, mape).
5. compare_results - Compare all model results.

WORKFLOW:
1. Call analyze_errors with baseline results and Fedot score.
2. For time series: call explain_model. For tabular: call get_feature_importance.
3. Optionally call get_additional_metrics for deeper analysis.
4. Provide your assessment as text with:

WINNER: architect or fedot
ASSESSMENT: Your overall analysis
STRENGTHS: list
WEAKNESSES: list
FEDOT_CHANGES: JSON of suggested config changes (e.g. {"timeout": 10, "preset": "best_quality"})
SHOULD_STOP: true/false (true if scores are good enough)
"""

    def __init__(self, name: str = "Critic", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, engineer_result: EngineerResult, data_context: DataContext, iteration: int) -> CriticFeedback:
        """Run the Critic agent."""
        self._tool_call_log = []
        result = CriticFeedback()

        try:
            logger.info(f"[Critic] Iteration {iteration}, task: {data_context.task_type}")
            is_ts = is_ts_task(data_context.task_type)

            # Build baseline results JSON for tools
            baseline_json = json.dumps([
                {"name": r.name, "score": r.score, "metrics": r.metrics, "error": r.error}
                for r in engineer_result.baseline_results
            ])
            fedot_score = engineer_result.fedot_result.get("score", 0)
            fedot_metrics = engineer_result.fedot_result.get("metrics", {})
            fedot_config = engineer_result.fedot_config_used.to_dict() if engineer_result.fedot_config_used else {}

            context = f"""Iteration: {iteration}
Task type: {data_context.task_type}
Is time series: {is_ts}
CSV path: {data_context.csv_path}
Target column: {data_context.target_column}

BASELINE RESULTS:
{baseline_json}

FEDOT RESULT:
  Score: {fedot_score:.4f}
  Metrics: {json.dumps(fedot_metrics)}
  Config: {json.dumps(fedot_config)}

COMPARISON:
  {engineer_result.comparison.get('comparison_text', '')}
  {engineer_result.comparison.get('cross_iteration_text', 'No previous iterations to compare')}

Please:
1. Call analyze_errors with the baseline results and fedot score.
2. {'Call explain_model for model interpretability.' if is_ts else 'Call get_feature_importance for feature analysis.'}
3. Provide your text assessment with WINNER, ASSESSMENT, STRENGTHS, WEAKNESSES, FEDOT_CHANGES, SHOULD_STOP.
"""

            response = await self.call_llm(context)
            text = response.get("full_response", "")

            # Parse structured feedback from text
            result = self._parse_feedback(text, engineer_result)
            result.tool_calls = self.get_tool_calls()

            # Extract explanation from tool calls
            for tc in self._tool_call_log:
                if tc.tool_name == "explain_model" and tc.success:
                    result.explanation = tc.result if isinstance(tc.result, dict) else {}

            if not result.score_assessment:
                logger.warning("[Critic] No assessment, using fallback")
                self._fallback_feedback(result, engineer_result, data_context)

            logger.info(f"[Critic] Done. Winner: {result.winner}, Stop: {result.should_stop}")
            return result

        except Exception as e:
            logger.error(f"[Critic] Error: {e}")
            self._fallback_feedback(result, engineer_result, data_context)
            result.tool_calls = self.get_tool_calls()
            return result

    def _parse_feedback(self, text: str, engineer_result: EngineerResult) -> CriticFeedback:
        """Parse structured feedback from LLM text response."""
        fb = CriticFeedback()
        fb.full_response = text

        fb.winner = self.extract_field(text, "WINNER").lower().strip()
        if fb.winner not in ("architect", "fedot"):
            # Determine from scores
            best_bl = max((r.score for r in engineer_result.baseline_results), default=0)
            fedot_score = engineer_result.fedot_result.get("score", 0)
            fb.winner = "architect" if best_bl > fedot_score else "fedot"

        fb.score_assessment = self.extract_field(text, "ASSESSMENT")
        fb.strengths = self.extract_list(text, "STRENGTHS")
        fb.weaknesses = self.extract_list(text, "WEAKNESSES")

        # Parse FEDOT_CHANGES as JSON
        changes_str = self.extract_field(text, "FEDOT_CHANGES")
        if changes_str:
            try:
                fb.suggested_fedot_changes = json.loads(changes_str)
            except json.JSONDecodeError:
                fb.suggested_fedot_changes = {}

        stop_str = self.extract_field(text, "SHOULD_STOP").lower().strip()
        fb.should_stop = stop_str in ("true", "yes", "1")

        return fb

    def _fallback_feedback(self, fb: CriticFeedback, engineer_result: EngineerResult, dc: DataContext):
        """Deterministic fallback."""
        best_bl = max((r.score for r in engineer_result.baseline_results), default=0)
        fedot_score = engineer_result.fedot_result.get("score", 0)
        best_score = max(best_bl, fedot_score)

        fb.winner = "architect" if best_bl > fedot_score else "fedot"
        fb.score_assessment = f"Baseline best: {best_bl:.4f}, Fedot: {fedot_score:.4f}"
        fb.strengths = fb.strengths or ["Models trained successfully"]
        fb.weaknesses = fb.weaknesses or ["LLM analysis unavailable"]

        if fedot_score < best_bl:
            fb.suggested_fedot_changes = {"timeout": 10, "preset": "best_quality"}
        elif fedot_score < 0.7:
            changes = {"timeout": 15}
            if is_ts_task(dc.task_type):
                changes["strategy"] = "kernel_automl"
            fb.suggested_fedot_changes = changes

        fb.should_stop = best_score > 0.95
        fb.full_response = fb.full_response or "Fallback analysis"
