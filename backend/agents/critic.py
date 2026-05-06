"""Critic: asks the LLM to evaluate graph quality and propose safe mutations."""

import json
import logging
from typing import Any, Dict, List, Tuple

from .base_agent import BaseAgent, extract_json_block
from .schemas import ArchitectResult, CriticFeedback, DataContext, EngineerResult
from .structured import CriticFeedbackObject, parse_critic_feedback_object

logger = logging.getLogger("Critic")


class Critic(BaseAgent):
    """Evaluation agent for model-driven graph critique."""

    ALLOWED_TOOLS = [
        "validate_graph",
        "get_node_importance",
        "explain_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Critic for MultiAgentFedot.IndustrialSystem (MAFIS).
You inspect the current graph, data profile, training result, validation result,
runtime diagnostics, node importance, and the allowed operation catalog.

Return one strict JSON object with exactly this shape:
{
  "assessment": "short overall judgement",
  "strengths": ["short evidence-backed strength"],
  "weaknesses": ["short evidence-backed weakness"],
  "suggested_mutations": [
    {"type": "remove", "node_id": "existing_node_id"},
    {"type": "replace", "node_id": "existing_node_id", "new_operation": "allowed_operation"},
    {"type": "add", "node": {"id": "new_node_id", "operation": "allowed_operation", "inputs": []}, "rewire_input_of": "existing_node_id"},
    {"type": "connect", "node_id": "existing_node_id", "input_id": "existing_node_id"}
  ],
  "improvement_plan": ["short next step"],
  "questions_for_user": ["short question when user input would materially improve the next graph"],
  "should_stop": false
}

Mutations of type remove/replace/add/connect modify the pipeline graph itself.
Critic may only edit graph nodes and connections. Do not suggest hyperparameters,
node params, model params, strategy changes, or set_strategy mutations. If a
model-parameter error appears in diagnostics, describe it as a graph execution
issue and propose only a structural recovery when appropriate. Use only
operations listed in allowed_catalog. If the graph failed, focus on recovery
mutations. If the graph is good enough and no revision is needed, return
suggested_mutations=[] and should_stop=true.
If engineer_result.finetune_error or fallback_used is present, treat metrics as
fallback baseline metrics, not as a clean Fedot.Industrial finetune result.
Preserve and reason from diagnostics. Do not recommend re-adding an operation
that diagnostics identify as failed/skipped/bypassed unless the payload shows a
different configuration that directly addresses that failure.
When diagnostics include kind="failure_localization", treat primary_suspect,
confidence, evidence, and ruled_out_nodes as factual runtime evidence. Prefer
mutations that address a high-confidence primary_suspect over nodes that merely
produced fallback metrics. If runtime_issue is
fedot_industrial_output_mode_compatibility, explain it as an adapter/runtime
compatibility issue, not as a model hyperparameter.
When diagnostics include kind="runtime_attempts_summary", reason over every
attempt in recovery_attempts. Do not claim that removing one node cleanly solves
the graph if another attempt exposes a different structural/runtime error. If
multiple original nodes produce distinct failures, include the relevant node
mutations or mention the remaining blocker in improvement_plan.
Analyze traceback text when present. If traceback points to categorical encoder,
one_hot_encoding, label_encoding, categorical_ids, or a similar categorical
metadata failure, explicitly consider removing the explicit encoder from the
graph. Data profile includes automatically detected categorical_feature_names;
when those names are present, treat categorical columns as known and do not ask
the user to confirm them. Ask about categorical columns only if the data profile
does not list them or the traceback shows ambiguous metadata.
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
            recovered_from_node_skip = self._has_node_skip_recovery(engineer_result)
            if engineer_result.graph_error:
                validation = {
                    "skipped": "Graph training failed; cross-validation would repeat the same failure.",
                    "diagnostics": engineer_result.diagnostics,
                }
            elif recovered_from_node_skip:
                validation = {
                    "skipped": (
                        "Engineer trained a recovered graph after skipping failed node(s); "
                        "cross-validation should run after Architect applies the recovery mutation."
                    ),
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
            elif recovered_from_node_skip:
                explanation = await self.call_mcp_tool("explain_graph", {"top_k": 10})
                node_importance = {
                    "skipped": (
                        "Node ablation is skipped because the approved graph differs from "
                        "the recovered graph Engineer actually trained."
                    )
                }
            else:
                explanation = await self.call_mcp_tool("explain_graph", {"top_k": 10})
                node_importance = await self._maybe_get_node_importance(graph, graph_json, data_context)

            diagnostics = self._collect_diagnostics(engineer_result, validation, node_importance)
            llm_feedback, raw_response, parse_error = await self._ask_llm_for_feedback(
                graph=graph,
                dc=data_context,
                engineer=engineer_result,
                validation=validation if isinstance(validation, dict) else {},
                explanation=explanation if isinstance(explanation, dict) else {},
                node_importance=node_importance if isinstance(node_importance, dict) else {},
                diagnostics=diagnostics,
            )
            if parse_error:
                diagnostics.append(
                    {
                        "agent": "Critic",
                        "kind": "critic_llm_parse_error",
                        "summary": "Critic model response was not valid feedback JSON.",
                        "technical_message": raw_response[:2000],
                        "recommendations": ["Run Critic again or simplify the graph feedback context."],
                        "recoverable": True,
                    }
                )

            feedback.assessment = llm_feedback.assessment
            feedback.strengths = llm_feedback.strengths
            feedback.weaknesses = llm_feedback.weaknesses
            feedback.questions_for_user = llm_feedback.questions_for_user
            feedback.diagnostics = diagnostics
            feedback.explanation = explanation if isinstance(explanation, dict) else {}
            feedback.node_importance = (
                node_importance.get("node_importance", {})
                if isinstance(node_importance, dict)
                else {}
            )
            feedback.suggested_mutations = self._validated_llm_mutations(
                llm_feedback.mutations_as_dicts(),
                graph,
                data_context,
                feedback.diagnostics,
            )
            feedback.improvement_plan = llm_feedback.improvement_plan
            feedback.should_stop = self._safe_should_stop(
                requested=bool(llm_feedback.should_stop),
                engineer=engineer_result,
                has_recovery=recovered_from_node_skip,
                suggested_mutations=feedback.suggested_mutations,
            )
            feedback.winner = "accepted" if feedback.should_stop else "needs_revision"
            feedback.full_response = raw_response
            feedback.tool_calls = self.get_tool_calls()
            logger.info("[Critic] decision=%s, stop=%s", feedback.winner, feedback.should_stop)
            return feedback

        except Exception as exc:
            logger.exception("[Critic] failed")
            feedback.winner = "needs_revision"
            feedback.assessment = "Critic could not complete model-driven review."
            feedback.weaknesses = ["Critic model/tool analysis failed"]
            feedback.diagnostics = list(engineer_result.diagnostics) + [
                {
                    "agent": "Critic",
                    "kind": "critic_error",
                    "summary": "Critic failed before returning feedback.",
                    "technical_message": str(exc),
                    "recommendations": ["Run the evaluation again after checking backend logs."],
                    "recoverable": True,
                }
            ]
            feedback.suggested_mutations = []
            feedback.improvement_plan = []
            feedback.should_stop = False
            feedback.full_response = str(exc)
            feedback.tool_calls = self.get_tool_calls()
            return feedback

    async def _validate_graph(self, graph_json: str, dc: DataContext) -> Dict[str, Any]:
        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "cv_folds": 3,
        }
        if dc.primary_metric:
            args["primary_metric"] = dc.primary_metric
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
        if dc.primary_metric:
            args["primary_metric"] = dc.primary_metric
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length
        importance = await self.call_mcp_tool("get_node_importance", args)
        return importance if isinstance(importance, dict) else {}

    async def _ask_llm_for_feedback(
        self,
        graph: Dict[str, Any],
        dc: DataContext,
        engineer: EngineerResult,
        validation: Dict[str, Any],
        explanation: Dict[str, Any],
        node_importance: Dict[str, Any],
        diagnostics: List[Dict[str, Any]],
    ) -> Tuple[CriticFeedbackObject, str, bool]:
        payload = {
            "task_type": dc.task_type,
            "primary_metric": dc.primary_metric,
            "data_profile": dc.profile,
            "allowed_catalog": self._allowed_catalog(dc.task_type),
            "current_graph": graph,
            "engineer_result": {
                "graph_score": engineer.graph_score,
                "graph_metrics": engineer.graph_metrics,
                "train_metrics": engineer.train_metrics,
                "test_metrics": engineer.test_metrics,
                "split_info": engineer.split_info,
                "graph_error": engineer.graph_error,
                "finetune_error": engineer.finetune_error,
                "finetune_traceback": engineer.finetune_traceback[-6000:],
                "fallback_used": engineer.fallback_used,
                "training_notes": engineer.training_notes,
                "diagnostics": engineer.diagnostics,
                "runtime_attempts_summary": self._runtime_attempts_summary(engineer.diagnostics),
            },
            "validation": validation,
            "explanation": explanation,
            "node_importance": node_importance,
            "collected_diagnostics": diagnostics,
            "iteration_history": [
                {
                    "iteration": item.iteration,
                    "graph": item.graph,
                    "graph_score": item.graph_score,
                    "winner": item.winner,
                    "suggested_mutations": item.suggested_mutations,
                }
                for item in (dc.iteration_history or [])
            ],
        }
        message = (
            "Review this payload and return only the Critic feedback JSON object. "
            "Do not call tools in this step; the validation and explanation payloads are already provided.\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
        response = await self.call_llm(message, max_rounds=1, use_tools=False)
        raw_response = str(response.get("full_response") or response.get("error") or "")
        parsed = extract_json_block(raw_response)
        feedback = parse_critic_feedback_object(parsed or {})
        if feedback is None:
            return CriticFeedbackObject(), raw_response, True
        return feedback, raw_response, False

    @staticmethod
    def _allowed_catalog(task_type: str) -> Dict[str, Any]:
        from graph_engine import OPERATIONS

        ops = OPERATIONS.get(task_type, {})
        return {
            "preprocessing": list(ops.get("preprocessing", []) or []),
            "models": list(ops.get("models", []) or []),
        }

    def _validated_llm_mutations(
        self,
        mutations: List[Dict[str, Any]],
        graph: Dict[str, Any],
        dc: DataContext,
        diagnostics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        from graph_engine import PipelineGraph

        validated: List[Dict[str, Any]] = []
        current_graph = graph
        for mutation in mutations:
            error = self._mutation_validation_error(mutation, current_graph, dc)
            if error:
                diagnostics.append(
                    {
                        "agent": "Critic",
                        "kind": "critic_mutation_rejected",
                        "summary": "Critic model proposed a mutation that was rejected by graph validation.",
                        "technical_message": f"{error}: {json.dumps(mutation, ensure_ascii=False)}",
                        "recommendations": ["Ask Critic to revise feedback or let Architect propose a graph without selected mutations."],
                        "recoverable": True,
                    }
                )
                continue
            if mutation not in validated:
                validated.append(mutation)
            try:
                current_graph = PipelineGraph.from_dict(current_graph).apply_mutation(mutation).to_dict()
            except Exception:
                pass
        return validated[:3]

    @staticmethod
    def _mutation_validation_error(mutation: Dict[str, Any], graph: Dict[str, Any], dc: DataContext) -> str:
        from graph_engine import OPERATIONS, PipelineGraph

        if not isinstance(mutation, dict):
            return "mutation is not an object"
        kind = mutation.get("type")
        allowed_kinds = {"remove", "replace", "add", "connect"}
        if kind not in allowed_kinds:
            return f"unsupported mutation type '{kind}'"

        nodes = graph.get("nodes", []) or []
        node_ids = {node.get("id") for node in nodes if isinstance(node, dict)}
        allowed_operations = set(
            (OPERATIONS.get(dc.task_type, {}) or {}).get("preprocessing", [])
            + (OPERATIONS.get(dc.task_type, {}) or {}).get("models", [])
        )

        if kind in {"remove", "replace", "connect"} and mutation.get("node_id") not in node_ids:
            return f"node_id '{mutation.get('node_id')}' does not exist"
        if kind == "replace" and mutation.get("new_operation") not in allowed_operations:
            return f"operation '{mutation.get('new_operation')}' is not allowed for task '{dc.task_type}'"
        if kind == "add":
            node = mutation.get("node")
            if not isinstance(node, dict):
                return "add mutation requires node object"
            if node.get("id") in node_ids:
                return f"node id '{node.get('id')}' already exists"
            if node.get("operation") not in allowed_operations:
                return f"operation '{node.get('operation')}' is not allowed for task '{dc.task_type}'"
        if kind == "connect" and mutation.get("input_id") not in node_ids:
            return f"input_id '{mutation.get('input_id')}' does not exist"

        try:
            candidate = PipelineGraph.from_dict(graph).apply_mutation(mutation)
            ok, message = candidate.validate()
            if not ok:
                return message
        except Exception as exc:
            return str(exc)
        return ""

    @staticmethod
    def _safe_should_stop(
        requested: bool,
        engineer: EngineerResult,
        has_recovery: bool,
        suggested_mutations: List[Dict[str, Any]],
    ) -> bool:
        if not requested:
            return False
        if (
            engineer.graph_error
            or engineer.finetune_error
            or engineer.fallback_used
            or engineer.graph_score <= 0
            or has_recovery
        ):
            return False
        if suggested_mutations:
            return False
        return True

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
    def _runtime_attempts_summary(diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
        for item in diagnostics or []:
            if isinstance(item, dict) and item.get("kind") == "runtime_attempts_summary":
                return item
        return {}

    @staticmethod
    def _has_node_skip_recovery(engineer: EngineerResult) -> bool:
        return any(
            isinstance(item, dict) and item.get("kind") == "node_skipped_after_runtime_error"
            for item in (engineer.diagnostics or [])
        )
