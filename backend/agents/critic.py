"""Critic: validates graph quality and suggests structural mutations."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext, EngineerResult

logger = logging.getLogger("Critic")


class Critic(BaseAgent):
    """Evaluation agent for standalone graph quality analysis."""

    ALLOWED_TOOLS = [
        "validate_graph",
        "get_node_importance",
        "explain_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Critic for GraphAutoML.
You inspect a trained pipeline graph, data profile, cross-validation, and node importance.
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
            if engineer_result.graph_error:
                validation = {
                    "skipped": "Graph training failed; cross-validation would repeat the same failure.",
                    "diagnostics": engineer_result.diagnostics,
                }
            else:
                validation = await self._validate_graph(graph_json, data_context)
            if engineer_result.graph_error or engineer_result.graph_score <= 0:
                explanation = {
                    "skipped": "Graph was not trained successfully; explanation is unavailable.",
                    "error": engineer_result.graph_error,
                }
                node_importance = {"skipped": "Graph was not trained successfully; node ablation is unavailable."}
            else:
                explanation = await self.call_mcp_tool("explain_graph", {"top_k": 10})
                node_importance = await self._maybe_get_node_importance(graph, graph_json, data_context)

            feedback.assessment = self._build_assessment(engineer_result, validation, data_context)
            feedback.strengths = self._strengths(engineer_result, validation, data_context)
            feedback.weaknesses = self._weaknesses(engineer_result, validation, data_context)
            feedback.diagnostics = self._collect_diagnostics(engineer_result, validation, node_importance)
            feedback.explanation = explanation if isinstance(explanation, dict) else {}
            feedback.node_importance = (
                node_importance.get("node_importance", {})
                if isinstance(node_importance, dict)
                else {}
            )

            # Keep Critic deterministic during evaluation: LLM limits should
            # not hide actionable feedback from the user.
            llm_feedback = {}
            feedback.suggested_mutations = (
                llm_feedback.get("suggested_mutations")
                or self._fallback_mutations(graph, data_context, engineer_result, validation if isinstance(validation, dict) else {})
            )
            feedback.improvement_plan = self._build_improvement_plan(
                graph,
                data_context,
                engineer_result,
                validation if isinstance(validation, dict) else {},
                feedback.suggested_mutations,
            )
            feedback.should_stop = bool(
                llm_feedback.get("should_stop")
                if "should_stop" in llm_feedback
                else self._should_stop(engineer_result, feedback, validation if isinstance(validation, dict) else {}, data_context.task_type)
            )
            feedback.winner = "accepted" if feedback.should_stop else "needs_revision"
            feedback.full_response = json.dumps(llm_feedback, ensure_ascii=False) if llm_feedback else "Deterministic critic fallback"
            feedback.tool_calls = self.get_tool_calls()
            logger.info("[Critic] decision=%s, stop=%s", feedback.winner, feedback.should_stop)
            return feedback

        except Exception as exc:
            logger.exception("[Critic] failed")
            feedback.winner = "needs_revision"
            feedback.assessment = f"Critic fallback after error: {exc}"
            feedback.weaknesses = ["Critic LLM/tool analysis failed"]
            feedback.diagnostics = list(engineer_result.diagnostics)
            feedback.suggested_mutations = self._fallback_mutations(graph, data_context, engineer_result, {})
            feedback.improvement_plan = self._build_improvement_plan(graph, data_context, engineer_result, {}, feedback.suggested_mutations)
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
        initial: CriticFeedback,
    ) -> Dict[str, Any]:
        context = f"""Task: {dc.task_type}
Profile: {json.dumps(dc.profile, ensure_ascii=False)}
Graph: {json.dumps(graph, ensure_ascii=False)}
Graph score: {engineer.graph_score}
Graph metrics: {json.dumps(engineer.graph_metrics, ensure_ascii=False)}
Validation: {json.dumps(validation, ensure_ascii=False)}
Initial assessment: {initial.assessment}
Allowed operations are constrained by the graph task type. Suggest at most 2 concrete mutations.
Return JSON only."""
        response = await self.call_llm(context, max_rounds=1, use_tools=False)
        parsed = extract_json_block(response.get("full_response", ""))
        return parsed or {}

    @staticmethod
    def _build_assessment(engineer: EngineerResult, validation: Dict[str, Any], dc: DataContext) -> str:
        if engineer.graph_error:
            return f"Graph failed during training: {engineer.graph_error[:220]}."
        cv = ""
        if isinstance(validation, dict) and "score_mean" in validation:
            cv = f", CV mean {validation.get('score_mean', 0):.4f} +/- {validation.get('score_std', 0):.4f}"
        issues = dc.profile.get("issues", [])
        issue_text = f" Data issues: {issues}." if issues and issues != ["none"] else ""
        return (
            f"Graph score {engineer.graph_score:.4f}{cv}. "
            f"Assessment is based on graph metrics, validation stability, node behavior, and data profile."
            f"{issue_text}"
        )

    @staticmethod
    def _strengths(engineer: EngineerResult, validation: Dict[str, Any], dc: DataContext) -> List[str]:
        strengths = []
        if engineer.graph_error:
            return ["The failure was captured and converted into diagnostics"]
        if engineer.graph_score > 0:
            strengths.append("The proposed graph trained successfully")
        if validation.get("score_std", 1) < 0.05:
            strengths.append("Cross-validation variance is low")
        if dc.task_type in ("classification", "regression") and len(engineer.graph_metrics) > 1:
            strengths.append("The graph produced multiple interpretable quality metrics")
        if not dc.profile.get("issues") or dc.profile.get("issues") == ["none"]:
            strengths.append("The data profile does not show obvious structural issues")
        return strengths or ["The pipeline produced evaluable metrics"]

    @staticmethod
    def _weaknesses(engineer: EngineerResult, validation: Dict[str, Any], dc: DataContext) -> List[str]:
        weaknesses = []
        if engineer.graph_error:
            weaknesses.append("The graph could not be trained")
            weaknesses.extend(Critic._diagnostic_recommendations(engineer.diagnostics))
            return weaknesses
        if engineer.graph_score <= 0:
            weaknesses.append("The graph score is zero or unavailable")
        if engineer.graph_score < Critic._target_quality_floor(dc.task_type):
            weaknesses.append("The graph metric is below the target quality floor for this task")
        if validation.get("score_std", 0) > 0.1:
            weaknesses.append("Cross-validation variance is high")
        issues = dc.profile.get("issues", []) or []
        for issue in issues:
            if issue != "none":
                weaknesses.append(f"Data profile issue: {issue}")
        return weaknesses

    def _fallback_mutations(self, graph: Dict[str, Any], dc: DataContext, engineer: EngineerResult, validation: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = graph.get("nodes", [])
        root_id = self._root_id(nodes)
        if not root_id:
            return []

        mutations: List[Dict[str, Any]] = []
        for diagnostic in engineer.diagnostics:
            for mutation in diagnostic.get("suggested_mutations", []) or []:
                if isinstance(mutation, dict):
                    mutations.append(mutation)
        if mutations:
            return mutations[:2]

        current_root = next((n for n in nodes if n.get("id") == root_id), {})

        score_floor = self._target_quality_floor(dc.task_type)
        unstable = validation.get("score_std", 0) > 0.1
        if engineer.graph_score < score_floor or unstable:
            candidate = self._next_model_candidate(dc.task_type, current_root.get("operation", ""))
            if candidate:
                mutations.append({"type": "replace", "node_id": root_id, "new_operation": candidate})

        if dc.task_type.startswith("ts_") and not any("fourier" in n.get("operation", "") for n in nodes):
            mutations.append(
                {
                    "type": "add",
                    "node": {"id": "freq_auto", "operation": "fourier_basis", "params": {}, "inputs": []},
                    "rewire_input_of": root_id,
                }
            )

        return mutations[:2]

    def _build_improvement_plan(
        self,
        graph: Dict[str, Any],
        dc: DataContext,
        engineer: EngineerResult,
        validation: Dict[str, Any],
        mutations: List[Dict[str, Any]],
    ) -> List[str]:
        plan: List[str] = []
        if engineer.graph_error:
            plan.append("First fix the graph/runtime error before judging model quality.")
            plan.extend(self._diagnostic_recommendations(engineer.diagnostics))
            return self._unique_strings(plan)

        quality_floor = self._target_quality_floor(dc.task_type)
        if engineer.graph_score < quality_floor:
            plan.append(
                f"The graph score is below the standalone target floor for {dc.task_type} "
                f"({engineer.graph_score:.4f} < {quality_floor:.2f})."
            )
        else:
            plan.append("The graph passes the current standalone quality check; keep it unless domain constraints require changes.")

        if validation.get("score_std", 0) > 0.05:
            plan.append("Cross-validation variance is noticeable; prefer simpler models or stronger regularization.")

        if dc.profile.get("is_imbalanced"):
            plan.append("Target classes look imbalanced; prefer metrics such as F1/precision and consider class weighting.")

        for mutation in mutations:
            if mutation.get("type") == "replace":
                plan.append(
                    f"Architect can draft a new graph by replacing node '{mutation.get('node_id')}' "
                    f"with '{mutation.get('new_operation')}'."
                )
            elif mutation.get("type") == "add":
                node = mutation.get("node", {})
                plan.append(f"Architect can draft a new graph by adding '{node.get('operation')}' before the model.")
            elif mutation.get("type") == "set_params":
                plan.append(f"Architect can draft a new graph by updating parameters of node '{mutation.get('node_id')}'.")

        return self._unique_strings(plan)[:6]

    @staticmethod
    def _collect_diagnostics(
        engineer: EngineerResult,
        validation: Dict[str, Any],
        node_importance: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        diagnostics = list(engineer.diagnostics)
        for payload in (validation, node_importance):
            if isinstance(payload, dict):
                for item in payload.get("diagnostics", []) or []:
                    if isinstance(item, dict):
                        diagnostics.append({**item, "agent": item.get("agent") or "Critic"})
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in diagnostics:
            key = (item.get("kind"), item.get("summary"), item.get("technical_message"))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @staticmethod
    def _diagnostic_recommendations(diagnostics: List[Dict[str, Any]]) -> List[str]:
        recs: List[str] = []
        for diagnostic in diagnostics:
            for rec in diagnostic.get("recommendations", []) or []:
                if rec not in recs:
                    recs.append(rec)
        return recs[:4]

    @staticmethod
    def _unique_strings(items: List[str]) -> List[str]:
        out: List[str] = []
        for item in items:
            if item and item not in out:
                out.append(item)
        return out

    @staticmethod
    def _root_id(nodes: List[Dict[str, Any]]) -> str:
        ids = {n.get("id") for n in nodes}
        children = {inp for n in nodes for inp in n.get("inputs", [])}
        roots = [node_id for node_id in ids - children if node_id]
        return roots[0] if roots else ""

    @staticmethod
    def _target_quality_floor(task_type: str) -> float:
        return {
            "classification": 0.75,
            "regression": 0.60,
            "ts_classification": 0.70,
            "ts_regression": 0.50,
            "ts_forecasting": 0.0,
        }.get(task_type, 0.0)

    @staticmethod
    def _next_model_candidate(task_type: str, current: str) -> str:
        candidates = {
            "classification": ["xgboost", "rf", "logit"],
            "regression": ["xgbreg", "treg", "ridge"],
            "ts_classification": ["industrial_freq_clf", "industrial_stat_clf"],
            "ts_regression": ["industrial_freq_reg", "industrial_stat_reg"],
        }.get(task_type, [])
        return next((candidate for candidate in candidates if candidate != current), "")

    @staticmethod
    def _should_stop(engineer: EngineerResult, feedback: CriticFeedback, validation: Dict[str, Any], task_type: str) -> bool:
        if engineer.graph_score <= 0:
            return False
        if feedback.weaknesses:
            return False
        if validation.get("score_std", 0) > 0.1:
            return False
        return engineer.graph_score >= Critic._target_quality_floor(task_type)
