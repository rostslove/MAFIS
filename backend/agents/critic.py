"""Critic: validates graph quality and suggests structural mutations."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext, EngineerResult

logger = logging.getLogger("Critic")


class Critic(BaseAgent):
    """Evaluation agent for graph-vs-baseline analysis."""

    ALLOWED_TOOLS = [
        "validate_graph",
        "analyze_errors",
        "get_node_importance",
        "explain_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Critic for GraphAutoML.
You inspect a trained pipeline graph, baseline scores, cross-validation, and node importance.
Suggest graph mutations only in this JSON shape:
{
  "assessment": "...",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "suggested_mutations": [
    {"type": "replace", "node_id": "model", "new_operation": "rf"},
    {"type": "add", "node": {"id": "scale2", "operation": "scaling", "params": {}, "inputs": []}, "rewire_input_of": "model"},
    {"type": "set_params", "node_id": "model", "params": {"n_estimators": 200}}
  ],
  "should_stop": false
}
Return JSON only."""

    def __init__(self, name: str = "Critic", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(
        self,
        architect_result: ArchitectResult,
        engineer_result: EngineerResult,
        data_context: DataContext,
        iteration: int,
    ) -> CriticFeedback:
        self._tool_call_log = []
        feedback = CriticFeedback()
        graph = architect_result.graph
        graph_json = json.dumps(graph, ensure_ascii=False)

        try:
            baseline_json = json.dumps(
                [
                    {"name": r.name, "score": r.score, "metrics": r.metrics, "error": r.error}
                    for r in engineer_result.baseline_results
                ],
                ensure_ascii=False,
            )

            validation = await self._validate_graph(graph_json, data_context)
            error_analysis = await self.call_mcp_tool(
                "analyze_errors",
                {
                    "baseline_results_json": baseline_json,
                    "graph_score": engineer_result.graph_score,
                    "task_type": data_context.task_type,
                },
            )
            explanation = await self.call_mcp_tool("explain_graph", {"top_k": 10})
            node_importance = await self._maybe_get_node_importance(graph, graph_json, data_context)

            feedback.winner = "graph" if engineer_result.graph_score >= engineer_result.best_baseline_score else "baseline"
            feedback.assessment = self._build_assessment(engineer_result, validation, error_analysis)
            feedback.strengths = self._strengths(engineer_result, validation, error_analysis)
            feedback.weaknesses = self._weaknesses(engineer_result, validation, error_analysis)
            feedback.explanation = explanation if isinstance(explanation, dict) else {}
            feedback.node_importance = (
                node_importance.get("node_importance", {})
                if isinstance(node_importance, dict)
                else {}
            )

            llm_feedback = await self._ask_llm_for_mutations(
                graph,
                data_context,
                engineer_result,
                validation if isinstance(validation, dict) else {},
                error_analysis if isinstance(error_analysis, dict) else {},
                feedback,
            )
            feedback.suggested_mutations = (
                llm_feedback.get("suggested_mutations")
                or self._fallback_mutations(graph, data_context, engineer_result)
            )
            feedback.should_stop = bool(
                llm_feedback.get("should_stop")
                if "should_stop" in llm_feedback
                else self._should_stop(engineer_result, feedback)
            )
            feedback.full_response = json.dumps(llm_feedback, ensure_ascii=False) if llm_feedback else "Deterministic critic fallback"
            feedback.tool_calls = self.get_tool_calls()
            logger.info("[Critic] winner=%s, stop=%s", feedback.winner, feedback.should_stop)
            return feedback

        except Exception as exc:
            logger.exception("[Critic] failed")
            feedback.winner = "graph" if engineer_result.graph_score >= engineer_result.best_baseline_score else "baseline"
            feedback.assessment = f"Critic fallback after error: {exc}"
            feedback.weaknesses = ["Critic LLM/tool analysis failed"]
            feedback.suggested_mutations = self._fallback_mutations(graph, data_context, engineer_result)
            feedback.should_stop = False
            feedback.tool_calls = self.get_tool_calls()
            return feedback

    async def _validate_graph(self, graph_json: str, dc: DataContext) -> Dict[str, Any]:
        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "cv_folds": 3,
        }
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length
        validation = await self.call_mcp_tool("validate_graph", args)
        return validation if isinstance(validation, dict) else {}

    async def _maybe_get_node_importance(self, graph: Dict[str, Any], graph_json: str, dc: DataContext) -> Dict[str, Any]:
        n_samples = int(dc.profile.get("n_samples") or 0)
        if n_samples > 3000 or len(graph.get("nodes", [])) > 5:
            return {"skipped": "dataset or graph too large for ablation"}

        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
        }
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length
        importance = await self.call_mcp_tool("get_node_importance", args)
        return importance if isinstance(importance, dict) else {}

    async def _ask_llm_for_mutations(
        self,
        graph: Dict[str, Any],
        dc: DataContext,
        engineer: EngineerResult,
        validation: Dict[str, Any],
        error_analysis: Dict[str, Any],
        initial: CriticFeedback,
    ) -> Dict[str, Any]:
        context = f"""Task: {dc.task_type}
Profile: {json.dumps(dc.profile, ensure_ascii=False)}
Graph: {json.dumps(graph, ensure_ascii=False)}
Graph score: {engineer.graph_score}
Graph metrics: {json.dumps(engineer.graph_metrics, ensure_ascii=False)}
Best baseline: {engineer.best_baseline_name} = {engineer.best_baseline_score}
Validation: {json.dumps(validation, ensure_ascii=False)}
Error analysis: {json.dumps(error_analysis, ensure_ascii=False)}
Initial assessment: {initial.assessment}
Allowed operations are constrained by the graph task type. Suggest at most 2 concrete mutations.
Return JSON only."""
        response = await self.call_llm(context, max_rounds=1, use_tools=False)
        parsed = extract_json_block(response.get("full_response", ""))
        return parsed or {}

    @staticmethod
    def _build_assessment(engineer: EngineerResult, validation: Dict[str, Any], error_analysis: Dict[str, Any]) -> str:
        cv = ""
        if isinstance(validation, dict) and "score_mean" in validation:
            cv = f", CV mean {validation.get('score_mean', 0):.4f} +/- {validation.get('score_std', 0):.4f}"
        delta = error_analysis.get("delta") if isinstance(error_analysis, dict) else None
        delta_text = f", delta vs baseline {delta}" if delta is not None else ""
        return (
            f"Graph score {engineer.graph_score:.4f}, best baseline "
            f"{engineer.best_baseline_name or 'none'} {engineer.best_baseline_score:.4f}"
            f"{cv}{delta_text}."
        )

    @staticmethod
    def _strengths(engineer: EngineerResult, validation: Dict[str, Any], error_analysis: Dict[str, Any]) -> List[str]:
        strengths = []
        if engineer.graph_score > 0:
            strengths.append("The proposed graph trained successfully")
        if engineer.graph_score >= engineer.best_baseline_score:
            strengths.append("The graph matches or beats the strongest baseline")
        if validation.get("score_std", 1) < 0.05:
            strengths.append("Cross-validation variance is low")
        if error_analysis.get("graph_beats_baselines"):
            strengths.append("Baseline comparison favors the graph")
        return strengths or ["The pipeline produced evaluable metrics"]

    @staticmethod
    def _weaknesses(engineer: EngineerResult, validation: Dict[str, Any], error_analysis: Dict[str, Any]) -> List[str]:
        weaknesses = []
        if engineer.graph_score <= 0:
            weaknesses.append("The graph score is zero or unavailable")
        if engineer.best_baseline_score > engineer.graph_score:
            weaknesses.append("A simple baseline currently outperforms the graph")
        if validation.get("score_std", 0) > 0.1:
            weaknesses.append("Cross-validation variance is high")
        if error_analysis.get("failed_baselines"):
            weaknesses.append(f"Some baselines failed: {error_analysis['failed_baselines']}")
        return weaknesses or ["No major weakness detected from current metrics"]

    def _fallback_mutations(self, graph: Dict[str, Any], dc: DataContext, engineer: EngineerResult) -> List[Dict[str, Any]]:
        nodes = graph.get("nodes", [])
        root_id = self._root_id(nodes)
        if not root_id:
            return []

        mutations: List[Dict[str, Any]] = []
        baseline_op = self._baseline_to_operation(dc.task_type, engineer.best_baseline_name)
        current_root = next((n for n in nodes if n.get("id") == root_id), {})

        if engineer.best_baseline_score > engineer.graph_score and baseline_op and current_root.get("operation") != baseline_op:
            mutations.append({"type": "replace", "node_id": root_id, "new_operation": baseline_op})

        if dc.task_type in ("classification", "regression"):
            has_scaling = any(n.get("operation") in ("scaling", "normalization") for n in nodes)
            if not has_scaling:
                mutations.append(
                    {
                        "type": "add",
                        "node": {"id": "scale_auto", "operation": "scaling", "params": {}, "inputs": []},
                        "rewire_input_of": root_id,
                    }
                )
        elif dc.task_type.startswith("ts_") and not any("fourier" in n.get("operation", "") for n in nodes):
            mutations.append(
                {
                    "type": "add",
                    "node": {"id": "freq_auto", "operation": "fourier_basis", "params": {}, "inputs": []},
                    "rewire_input_of": root_id,
                }
            )

        return mutations[:2]

    @staticmethod
    def _root_id(nodes: List[Dict[str, Any]]) -> str:
        ids = {n.get("id") for n in nodes}
        children = {inp for n in nodes for inp in n.get("inputs", [])}
        roots = [node_id for node_id in ids - children if node_id]
        return roots[0] if roots else ""

    @staticmethod
    def _baseline_to_operation(task_type: str, baseline_name: str) -> str:
        mapping = {
            "classification": {"logreg": "logit", "rf": "rf", "xgb": "xgboost"},
            "regression": {"ridge": "ridge", "rf": "treg", "xgb": "xgbreg"},
            "ts_classification": {
                "logreg": "industrial_stat_clf",
                "rf": "industrial_stat_clf",
                "xgb": "industrial_freq_clf",
            },
            "ts_regression": {
                "ridge": "industrial_stat_reg",
                "rf": "industrial_stat_reg",
                "xgb": "industrial_freq_reg",
            },
        }
        return mapping.get(task_type, {}).get(baseline_name, "")

    @staticmethod
    def _should_stop(engineer: EngineerResult, feedback: CriticFeedback) -> bool:
        if engineer.graph_score <= 0:
            return False
        if feedback.winner != "graph":
            return False
        return engineer.graph_score >= 0.95 or (
            engineer.best_baseline_score > 0 and engineer.graph_score - engineer.best_baseline_score > 0.02
        )
