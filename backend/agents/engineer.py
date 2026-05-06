"""Engineer: trains the approved Architect graph through MCP tools."""

import json
import logging
from typing import Any, Dict, List, Optional

from .base_agent import BaseAgent
from .schemas import ArchitectResult, DataContext, EngineerResult

logger = logging.getLogger("Engineer")


class Engineer(BaseAgent):
    """Technical agent responsible for fitting a proposed graph."""

    ALLOWED_TOOLS = [
        "train_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Engineer. You receive a validated pipeline graph.
Your role is to train the graph as-is and report metrics.
If non-root operations crash during training, skip those operations, retry the
remaining graph, and report skipped nodes as recovery feedback for Critic."""

    def __init__(self, name: str = "Engineer", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, architect_result: ArchitectResult, data_context: DataContext) -> EngineerResult:
        self._tool_call_log = []
        result = EngineerResult()
        graph_json = json.dumps(architect_result.graph, ensure_ascii=False)

        try:
            graph_run = await self._train_graph(graph_json, data_context)

            if isinstance(graph_run, dict):
                result.graph_score = float(graph_run.get("score") or 0)
                result.graph_metrics = graph_run.get("metrics", {}) or {}
                result.train_metrics = graph_run.get("train_metrics", {}) or {}
                result.test_metrics = graph_run.get("test_metrics", {}) or {}
                result.split_info = graph_run.get("split_info", {}) or {}
                result.assumption_graph = graph_run.get("assumption_graph", {}) or {}
                result.assumption_mermaid = graph_run.get("assumption_mermaid", "") or ""
                result.industrial_strategy = (
                    graph_run.get("industrial_strategy")
                    or data_context.industrial_strategy
                    or "tabular"
                )
                result.industrial_strategy_params = (
                    graph_run.get("industrial_strategy_params", {})
                    or dict(data_context.industrial_strategy_params or {})
                )
                result.graph_error = graph_run.get("error", "") or ""
                result.finetune_error = graph_run.get("finetune_error", "") or ""
                result.finetune_traceback = graph_run.get("finetune_traceback", "") or ""
                result.fallback_used = graph_run.get("fallback_used", "") or ""
                result.target_info = graph_run.get("target_info", {}) or result.target_info
                result.training_notes.extend(graph_run.get("training_notes", []) or [])
                result.diagnostics.extend(self._extract_diagnostics(graph_run))
                self._append_training_diagnostic(result)
                self._append_failure_localization(result, architect_result.graph)
                result.diagnostics = self._unique_diagnostics(result.diagnostics)
                if result.graph_error:
                    result.graph_metrics["error"] = result.graph_error

            if result.target_info.get("fedot_receives_raw_target") and result.target_info.get("reference_encoded"):
                result.training_notes.append(
                    "Fedot graph used the raw target values. Reference label mapping for readable diagnostics: "
                    f"{result.target_info.get('reference_encoding', {})}"
                )

            result.tool_calls = self.get_tool_calls()
            logger.info(
                "[Engineer] test_primary=%.4f",
                result.graph_score,
            )
            return result

        except Exception as exc:
            logger.exception("[Engineer] failed")
            result.graph_metrics = {"error": str(exc)}
            result.graph_error = str(exc)
            result.diagnostics.append(
                {
                    "agent": "Engineer",
                    "kind": "engineer_error",
                    "summary": "Engineer could not complete training.",
                    "technical_message": str(exc),
                    "recoverable": True,
                }
            )
            result.tool_calls = self.get_tool_calls()
            return result

    async def _train_graph(self, graph_json: str, dc: DataContext) -> dict:
        trained = await self._train_direct(graph_json, dc)
        if self._should_attempt_node_recovery(trained):
            trained = await self._recover_by_skipping_node(graph_json, dc, trained)
        strategy_name = trained.get("industrial_strategy") or dc.industrial_strategy
        if strategy_name and strategy_name != "tabular":
            trained.setdefault("training_notes", []).append(
                f"Fedot.Industrial executed the '{strategy_name}' strategy with its own internal search. "
                "MAFIS did not run separate node-level hyperparameter tuning."
            )
        return trained

    async def _train_direct(self, graph_json: str, dc: DataContext) -> dict:
        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "test_size": dc.test_size,
            "industrial_strategy": dc.industrial_strategy or "tabular",
            "industrial_strategy_params": dict(dc.industrial_strategy_params or {}),
        }
        if dc.primary_metric:
            args["primary_metric"] = dc.primary_metric
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length
        trained = await self.call_mcp_tool("train_graph", args)
        if not isinstance(trained, dict):
            return {"score": 0, "error": "train_graph returned no data"}
        return trained

    async def _recover_by_skipping_node(
        self,
        graph_json: str,
        dc: DataContext,
        failed_run: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Try bypass recovery for crashes in optional graph nodes."""
        graph = self._parse_graph_json(graph_json)
        if not graph:
            return failed_run

        root_id = self._root_id(graph.get("nodes", []))
        candidates = [
            node for node in self._candidate_skip_nodes(graph)
            if node.get("id") and node.get("id") != root_id
        ]
        if not candidates:
            return failed_run

        original_error = self._failure_message(failed_run)
        attempts: List[Dict[str, Any]] = []
        best_fallback: Dict[str, Any] | None = None
        for node in candidates:
            recovered_graph = self._graph_with_node_skipped(graph, str(node["id"]))
            recovered_json = json.dumps(recovered_graph, ensure_ascii=False)
            recovered = await self._train_direct(recovered_json, dc)
            if not isinstance(recovered, dict):
                recovered = {"score": 0, "error": "train_graph returned no data"}

            if self._recovery_resolved(failed_run, recovered):
                diagnostic = self._nodes_skipped_diagnostic([node], original_error, recovered_graph)
                attempts_with_current = attempts + [self._attempt_record(node, recovered, recovered_graph)]
                attempts_summary = self._runtime_attempts_summary_diagnostic(
                    graph=graph,
                    original_run=failed_run,
                    attempts=attempts_with_current,
                )
                localization = self._failure_localization_diagnostic(
                    graph=graph,
                    run=failed_run,
                    attempts=attempts_with_current,
                )
                recovered.setdefault("diagnostics", [])
                extra_diagnostics = [diagnostic]
                if attempts_summary:
                    extra_diagnostics.append(attempts_summary)
                if localization:
                    extra_diagnostics.append(localization)
                recovered["diagnostics"] = (
                    list(failed_run.get("diagnostics", []) or [])
                    + extra_diagnostics
                    + list(recovered.get("diagnostics", []) or [])
                )
                recovered.setdefault("training_notes", [])
                recovered["training_notes"].append(
                    f"Engineer skipped failed node '{node.get('id')}' ({node.get('operation')}) "
                    "and retrained the remaining graph. Critic should decide whether to remove or replace it."
                )
                recovered["recovered_from_error"] = True
                recovered["original_error"] = original_error
                recovered["effective_graph"] = recovered_graph
                recovered["skipped_nodes"] = diagnostic["failed_nodes"]
                logger.info(
                    "[Engineer] recovered by skipping node %s (%s)",
                    node.get("id"),
                    node.get("operation"),
                )
                return recovered

            best_fallback = self._best_fallback_candidate(
                current=best_fallback,
                recovered=recovered,
                nodes=[node],
                recovered_graph=recovered_graph,
            )
            attempts.append(self._attempt_record(node, recovered, recovered_graph))

        if len(candidates) > 1:
            recovered_graph = self._graph_with_nodes_skipped(
                graph,
                [str(node["id"]) for node in candidates if node.get("id")],
            )
            recovered_json = json.dumps(recovered_graph, ensure_ascii=False)
            recovered = await self._train_direct(recovered_json, dc)
            if not isinstance(recovered, dict):
                recovered = {"score": 0, "error": "train_graph returned no data"}

            all_optional_attempt = {
                "id": "__all_optional__",
                "operation": "skip_all_optional_nodes",
                "variant": "skip_all_optional_nodes",
                "skipped_nodes": [
                    {"id": node.get("id"), "operation": node.get("operation")}
                    for node in candidates
                ],
                "remaining_graph": self._graph_signature(recovered_graph),
                "error": self._failure_message(recovered),
                "error_signature": self._error_signature(self._failure_message(recovered)),
                "phase": self._error_phase(recovered),
                "score": self._primary_score(recovered),
                "fallback_used": recovered.get("fallback_used", ""),
                "finetune_error": recovered.get("finetune_error", ""),
                "fallback_error": recovered.get("fallback_error", ""),
            }
            attempts.append(all_optional_attempt)

            if self._recovery_resolved(failed_run, recovered):
                diagnostic = self._nodes_skipped_diagnostic(candidates, original_error, recovered_graph)
                attempts_summary = self._runtime_attempts_summary_diagnostic(
                    graph=graph,
                    original_run=failed_run,
                    attempts=attempts,
                )
                localization = self._failure_localization_diagnostic(
                    graph=graph,
                    run=failed_run,
                    attempts=attempts,
                )
                recovered.setdefault("diagnostics", [])
                extra_diagnostics = [diagnostic]
                if attempts_summary:
                    extra_diagnostics.append(attempts_summary)
                if localization:
                    extra_diagnostics.append(localization)
                recovered["diagnostics"] = (
                    list(failed_run.get("diagnostics", []) or [])
                    + extra_diagnostics
                    + list(recovered.get("diagnostics", []) or [])
                )
                skipped = ", ".join(
                    f"'{node.get('id')}' ({node.get('operation')})" for node in candidates
                )
                recovered.setdefault("training_notes", [])
                recovered["training_notes"].append(
                    f"Engineer skipped failed optional nodes {skipped} and retrained the remaining graph. "
                    "Critic should decide whether to remove or replace them."
                )
                recovered["recovered_from_error"] = True
                recovered["original_error"] = original_error
                recovered["effective_graph"] = recovered_graph
                recovered["skipped_nodes"] = diagnostic["failed_nodes"]
                logger.info(
                    "[Engineer] recovered by skipping optional nodes: %s",
                    ", ".join(str(node.get("id")) for node in candidates),
                )
                return recovered
            best_fallback = self._best_fallback_candidate(
                current=best_fallback,
                recovered=recovered,
                nodes=candidates,
                recovered_graph=recovered_graph,
            )

        if best_fallback:
            recovered = best_fallback["run"]
            diagnostic = self._fallback_only_diagnostic(
                best_fallback["nodes"],
                original_error,
                best_fallback["graph"],
            )
            attempts_summary = self._runtime_attempts_summary_diagnostic(
                graph=graph,
                original_run=failed_run,
                attempts=attempts,
            )
            localization = self._failure_localization_diagnostic(
                graph=graph,
                run=failed_run,
                attempts=attempts,
            )
            recovered.setdefault("diagnostics", [])
            extra_diagnostics = [diagnostic]
            if attempts_summary:
                extra_diagnostics.append(attempts_summary)
            if localization:
                extra_diagnostics.append(localization)
            recovered["diagnostics"] = (
                list(failed_run.get("diagnostics", []) or [])
                + extra_diagnostics
                + list(recovered.get("diagnostics", []) or [])
            )
            recovered.setdefault("training_notes", [])
            skipped = ", ".join(
                f"'{node.get('id')}' ({node.get('operation')})"
                for node in best_fallback["nodes"]
            )
            recovered["training_notes"].append(
                f"Engineer found direct-fit fallback metrics after bypassing {skipped}, "
                "but Fedot.Industrial finetune still failed. Treat this as a baseline, not a clean recovery."
            )
            recovered["fallback_from_error"] = True
            recovered["original_error"] = original_error
            recovered["effective_graph"] = best_fallback["graph"]
            return recovered

        problem_nodes = [
            {"id": node.get("id"), "operation": node.get("operation")}
            for node in candidates
        ]
        failed_run.setdefault("diagnostics", [])
        attempts_summary = self._runtime_attempts_summary_diagnostic(
            graph=graph,
            original_run=failed_run,
            attempts=attempts,
        )
        localization = self._failure_localization_diagnostic(
            graph=graph,
            run=failed_run,
            attempts=attempts,
        )
        failed_run["diagnostics"].append(
            {
                "agent": "Engineer",
                "kind": "node_skip_recovery_failed",
                "summary": "Engineer tried to skip optional nodes, but the graph still failed during training.",
                "technical_message": original_error,
                "problem_nodes": problem_nodes,
                "recovery_attempts": attempts,
                "recoverable": True,
            }
        )
        if attempts_summary:
            failed_run["diagnostics"].append(attempts_summary)
        if localization:
            failed_run["diagnostics"].append(localization)
        failed_run["error"] = original_error
        return failed_run

    @staticmethod
    def _parse_graph_json(graph_json: str) -> Dict[str, Any]:
        try:
            graph = json.loads(graph_json)
            return graph if isinstance(graph, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _root_id(nodes: List[Dict[str, Any]]) -> str:
        ids = {node.get("id") for node in nodes if node.get("id")}
        children = {
            inp
            for node in nodes
            for inp in (node.get("inputs", []) or [])
            if inp
        }
        roots = [node_id for node_id in ids - children if node_id]
        return roots[0] if roots else ""

    @classmethod
    def _candidate_skip_nodes(cls, graph: Dict[str, Any]) -> List[Dict[str, Any]]:
        nodes = graph.get("nodes", []) or []
        depths = cls._node_depths(nodes)
        return sorted(
            [node for node in nodes if isinstance(node, dict)],
            key=lambda node: depths.get(node.get("id"), 0),
            reverse=True,
        )

    @staticmethod
    def _node_depths(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
        by_id = {node.get("id"): node for node in nodes if node.get("id")}
        cache: Dict[str, int] = {}

        def depth(node_id: str) -> int:
            if node_id in cache:
                return cache[node_id]
            node = by_id.get(node_id, {})
            parent_ids = [inp for inp in (node.get("inputs", []) or []) if inp in by_id]
            cache[node_id] = 1 + max((depth(parent_id) for parent_id in parent_ids), default=-1)
            return cache[node_id]

        for node_id in by_id:
            depth(str(node_id))
        return cache

    @staticmethod
    def _graph_with_node_skipped(graph: Dict[str, Any], node_id: str) -> Dict[str, Any]:
        nodes = graph.get("nodes", []) or []
        failed = next((node for node in nodes if node.get("id") == node_id), {})
        bypass_inputs = list(failed.get("inputs", []) or [])
        new_nodes = []
        for node in nodes:
            if node.get("id") == node_id:
                continue
            updated = dict(node)
            inputs: List[str] = []
            for current_input in node.get("inputs", []) or []:
                if current_input == node_id:
                    inputs.extend(bypass_inputs)
                else:
                    inputs.append(current_input)
            updated["inputs"] = Engineer._unique_inputs(inputs, str(updated.get("id") or ""))
            new_nodes.append(updated)

        recovered = dict(graph)
        recovered["nodes"] = new_nodes
        return recovered

    @classmethod
    def _graph_with_nodes_skipped(cls, graph: Dict[str, Any], node_ids: List[str]) -> Dict[str, Any]:
        recovered = dict(graph)
        for node_id in node_ids:
            recovered = cls._graph_with_node_skipped(recovered, node_id)
        return recovered

    @staticmethod
    def _unique_inputs(inputs: List[str], node_id: str) -> List[str]:
        unique: List[str] = []
        for item in inputs:
            if not item or item == node_id or item in unique:
                continue
            unique.append(item)
        return unique

    @staticmethod
    def _nodes_skipped_diagnostic(
        nodes: List[Dict[str, Any]],
        original_error: str,
        recovered_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem_nodes = [
            {"id": node.get("id"), "operation": node.get("operation")}
            for node in nodes
        ]
        if len(problem_nodes) == 1:
            summary = (
                f"Node '{problem_nodes[0].get('id')}' ({problem_nodes[0].get('operation')}) "
                "failed during training, so Engineer skipped it and retrained the remaining graph."
            )
        else:
            skipped = ", ".join(
                f"'{node.get('id')}' ({node.get('operation')})" for node in problem_nodes
            )
            summary = (
                f"Optional nodes {skipped} failed during training, so Engineer skipped them "
                "and retrained the remaining graph."
            )
        return {
            "agent": "Engineer",
            "kind": "node_skipped_after_runtime_error",
            "summary": summary,
            "technical_message": original_error[:2000],
            "failed_nodes": problem_nodes,
            "problem_nodes": problem_nodes,
            "effective_graph": recovered_graph,
            "recoverable": True,
            "recovered": True,
        }

    @staticmethod
    def _fallback_only_diagnostic(
        nodes: List[Dict[str, Any]],
        original_error: str,
        recovered_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        bypassed_nodes = [
            {"id": node.get("id"), "operation": node.get("operation")}
            for node in nodes
        ]
        if len(bypassed_nodes) == 1:
            summary = (
                f"Engineer got direct-fit fallback metrics after bypassing "
                f"'{bypassed_nodes[0].get('id')}' ({bypassed_nodes[0].get('operation')}), "
                "but Fedot.Industrial finetune still failed."
            )
        else:
            skipped = ", ".join(
                f"'{node.get('id')}' ({node.get('operation')})" for node in bypassed_nodes
            )
            summary = (
                f"Engineer got direct-fit fallback metrics after bypassing {skipped}, "
                "but Fedot.Industrial finetune still failed."
            )
        return {
            "agent": "Engineer",
            "kind": "finetune_fallback_after_node_skip",
            "summary": summary,
            "technical_message": original_error[:2000],
            "bypassed_nodes": bypassed_nodes,
            "effective_graph": recovered_graph,
            "recoverable": True,
            "recovered": False,
        }

    @classmethod
    def _attempt_record(
        cls,
        node: Dict[str, Any],
        recovered: Dict[str, Any],
        recovered_graph: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        error = cls._failure_message(recovered)
        return {
            "id": node.get("id"),
            "operation": node.get("operation"),
            "variant": f"skip_{node.get('id')}",
            "skipped_nodes": [{"id": node.get("id"), "operation": node.get("operation")}],
            "remaining_graph": cls._graph_signature(recovered_graph),
            "phase": cls._error_phase(recovered),
            "error_signature": cls._error_signature(error),
            "error": error,
            "score": cls._primary_score(recovered),
            "fallback_used": recovered.get("fallback_used", ""),
            "finetune_error": recovered.get("finetune_error", ""),
            "fallback_error": recovered.get("fallback_error", ""),
        }

    @classmethod
    def _runtime_attempts_summary_diagnostic(
        cls,
        graph: Dict[str, Any],
        original_run: Dict[str, Any],
        attempts: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not attempts:
            return None
        original_error = cls._failure_message(original_run)
        rows = [{
            "variant": "approved_graph",
            "skipped_nodes": [],
            "remaining_graph": cls._graph_signature(graph),
            "phase": cls._error_phase(original_run),
            "error_signature": cls._error_signature(original_error),
            "error": original_error,
            "score": cls._primary_score(original_run),
            "fallback_used": original_run.get("fallback_used", ""),
            "finetune_error": original_run.get("finetune_error", ""),
            "fallback_error": original_run.get("fallback_error", ""),
        }]
        rows.extend(attempts)
        return {
            "agent": "Engineer",
            "kind": "runtime_attempts_summary",
            "summary": "Fedot.Industrial failed across multiple graph variants; review the attempt sequence before applying graph mutations.",
            "technical_message": original_error[:2000],
            "recovery_attempts": rows,
            "recommendations": cls._runtime_recommendations(rows),
            "recoverable": True,
        }

    @staticmethod
    def _graph_signature(graph: Optional[Dict[str, Any]]) -> str:
        if not isinstance(graph, dict):
            return ""
        nodes = graph.get("nodes", []) or []
        by_id = {node.get("id"): node for node in nodes if isinstance(node, dict)}
        roots = []
        children = {
            inp
            for node in nodes
            for inp in (node.get("inputs", []) or [])
            if inp
        }
        for node in nodes:
            if isinstance(node, dict) and node.get("id") not in children:
                roots.append(node.get("id"))

        def render(node_id: str) -> str:
            node = by_id.get(node_id, {})
            op = str(node.get("operation") or node_id)
            inputs = [render(inp) for inp in (node.get("inputs", []) or []) if inp in by_id]
            return f"{' + '.join(inputs)} -> {op}" if inputs else op

        return " | ".join(render(str(root)) for root in roots if root) or "empty_graph"

    @classmethod
    def _runtime_recommendations(cls, attempts: List[Dict[str, Any]]) -> List[str]:
        signatures = {str(item.get("error_signature") or "") for item in attempts if isinstance(item, dict)}
        recommendations: List[str] = []
        if "categorical_encoder_metadata" in signatures:
            recommendations.append(
                "Retry with automatically detected categorical metadata; if the encoder still fails, remove or replace the explicit categorical encoder node."
            )
        if "sklearn_dimensionality" in signatures:
            recommendations.append(
                "Do not keep the shape-changing preprocessing path with a sklearn tabular model until a shape adapter is available."
            )
        if "fedot_predict_preprocessor_state" in signatures:
            recommendations.append(
                "Treat the finetune result as failed at prediction time; direct-fit metrics are only fallback baseline metrics."
            )
        return recommendations

    @classmethod
    def _append_failure_localization(
        cls,
        result: EngineerResult,
        graph: Dict[str, Any],
    ) -> None:
        if not (result.graph_error or result.finetune_error):
            return
        diagnostic = cls._failure_localization_diagnostic(
            graph=graph,
            run={
                "error": result.graph_error,
                "finetune_error": result.finetune_error,
                "finetune_traceback": result.finetune_traceback,
                "fallback_used": result.fallback_used,
            },
            attempts=cls._recovery_attempts_from_diagnostics(result.diagnostics),
        )
        if diagnostic:
            result.diagnostics.append(diagnostic)

    @staticmethod
    def _recovery_attempts_from_diagnostics(diagnostics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        attempts: List[Dict[str, Any]] = []
        for item in diagnostics or []:
            if not isinstance(item, dict):
                continue
            for attempt in item.get("recovery_attempts", []) or []:
                if isinstance(attempt, dict):
                    attempts.append(attempt)
        return attempts

    @classmethod
    def _failure_localization_diagnostic(
        cls,
        graph: Dict[str, Any],
        run: Dict[str, Any],
        attempts: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Infer the most likely graph node from traceback and recovery evidence."""
        nodes = [node for node in (graph.get("nodes", []) or []) if isinstance(node, dict)]
        if not nodes:
            return None

        attempts = attempts or []
        error_text = cls._failure_message(run)
        traceback_text = str(run.get("finetune_traceback") or run.get("fallback_traceback") or "")
        combined = f"{error_text}\n{traceback_text}".lower()
        if not combined.strip():
            return None

        scores: Dict[str, int] = {}
        evidence_by_node: Dict[str, List[str]] = {}
        global_evidence: List[str] = []
        runtime_issue = ""

        def add(node: Dict[str, Any], points: int, evidence: str) -> None:
            node_id = str(node.get("id") or "")
            if not node_id:
                return
            scores[node_id] = scores.get(node_id, 0) + points
            evidence_by_node.setdefault(node_id, [])
            if evidence not in evidence_by_node[node_id]:
                evidence_by_node[node_id].append(evidence)

        def operation_matches(names: set[str]) -> List[Dict[str, Any]]:
            return [node for node in nodes if str(node.get("operation") or "") in names]

        if (
            "categorical_encoders.py" in combined
            or "categorical_ids" in combined
            or "one_hot_encoding" in combined
            or "label_encoding" in combined
        ):
            global_evidence.append(
                "Traceback points to FEDOT categorical encoder metadata, not to a sampler/model."
            )
            for node in operation_matches({"one_hot_encoding", "label_encoding"}):
                add(
                    node,
                    7,
                    f"Graph contains explicit categorical encoder node '{node.get('id')}' ({node.get('operation')}).",
                )

        if (
            "expected 2d array, got 1d array" in combined
            or "found array with dim" in combined
            or "invalid shape" in combined
            or "must be 2 or 3 dimensional" in combined
        ):
            global_evidence.append(
                "A sklearn/FEDOT operation received data with a dimensionality it does not accept."
            )
            for node in operation_matches({"scaling", "normalization", "resample"}):
                add(
                    node,
                    5,
                    f"Node '{node.get('id')}' ({node.get('operation')}) can change feature shape before a sklearn tabular model.",
                )

        if "unexpected keyword argument 'output_mode'" in combined or 'unexpected keyword argument "output_mode"' in combined:
            runtime_issue = "fedot_industrial_output_mode_compatibility"
            global_evidence.append(
                "Traceback shows a Fedot.Industrial/FEDOT predict API compatibility issue around output_mode."
            )
            if "fedotpreprocessingstrategy" in combined:
                for node in operation_matches({"scaling", "normalization", "resample", "one_hot_encoding", "label_encoding"}):
                    add(
                        node,
                        1,
                        f"output_mode error is raised inside a preprocessing strategy while node '{node.get('id')}' exists in that path.",
                    )

        if "keyerror: 'default'" in combined or "ids_relevant_features" in combined:
            runtime_issue = "fedot_predict_preprocessor_state"
            global_evidence.append(
                "Fedot predict failed because pipeline preprocessing state for source 'default' was missing after finetune."
            )

        original_signature = cls._error_signature(error_text)
        ruled_out_nodes: List[Dict[str, Any]] = []
        for attempt in attempts:
            if not isinstance(attempt, dict) or not attempt.get("id"):
                continue
            node = next((candidate for candidate in nodes if candidate.get("id") == attempt.get("id")), None)
            if node is None:
                continue
            attempt_signature = cls._error_signature(str(attempt.get("error") or attempt.get("finetune_error") or ""))
            if attempt_signature and original_signature and attempt_signature != original_signature:
                add(
                    node,
                    4,
                    f"Bypassing '{node.get('id')}' changed the error signature from {original_signature} to {attempt_signature}.",
                )
            elif attempt_signature and original_signature and attempt_signature == original_signature:
                ruled_out_nodes.append({"id": node.get("id"), "operation": node.get("operation")})
                global_evidence.append(
                    f"Bypassing '{node.get('id')}' kept the same error signature ({original_signature})."
                )

        if not scores and not runtime_issue:
            return None

        suspect: Dict[str, Any] | None = None
        confidence = "low"
        if scores:
            suspect_id, score = max(scores.items(), key=lambda item: item[1])
            if not (runtime_issue and score <= 1):
                suspect_node = next((node for node in nodes if node.get("id") == suspect_id), {})
                suspect = {"id": suspect_node.get("id"), "operation": suspect_node.get("operation")}
                confidence = "high" if score >= 7 else "medium" if score >= 4 else "low"

        if suspect:
            summary = (
                f"Engineer localized the likely failure source to '{suspect.get('id')}' "
                f"({suspect.get('operation')}) with {confidence} confidence."
            )
            problem_nodes = [suspect]
            evidence = evidence_by_node.get(str(suspect.get("id")), []) + global_evidence
        else:
            summary = "Engineer localized this as a runtime compatibility issue rather than a specific graph node."
            problem_nodes = []
            evidence = global_evidence

        return {
            "agent": "Engineer",
            "kind": "failure_localization",
            "summary": summary,
            "technical_message": error_text[:2000],
            "primary_suspect": suspect or {},
            "problem_nodes": problem_nodes,
            "confidence": confidence,
            "evidence": cls._unique_text(evidence),
            "ruled_out_nodes": cls._unique_node_records(ruled_out_nodes),
            "runtime_issue": runtime_issue,
            "recommendations": cls._runtime_recommendations([
                {"error_signature": cls._error_signature(error_text)}
            ]),
            "recoverable": True,
        }

    @staticmethod
    def _error_signature(message: str) -> str:
        text = str(message or "").lower()
        if "categorical_ids" in text or "categorical_encoders.py" in text or "no attribute 'size'" in text:
            return "categorical_encoder_metadata"
        if (
            "expected 2d array, got 1d array" in text
            or "found array with dim" in text
            or "invalid shape" in text
            or "must be 2 or 3 dimensional" in text
        ):
            return "sklearn_dimensionality"
        if "unexpected keyword argument 'output_mode'" in text or 'unexpected keyword argument "output_mode"' in text:
            return "fedot_output_mode_api"
        if "keyerror: 'default'" in text or "ids_relevant_features" in text:
            return "fedot_predict_preprocessor_state"
        if "mix of binary and continuous targets" in text:
            return "classification_prediction_shape"
        if "early stopping" in text and "eval metric" in text:
            return "model_eval_set_required"
        return text[:120]

    @classmethod
    def _error_phase(cls, payload: Dict[str, Any]) -> str:
        text = cls._failure_message(payload).lower()
        traceback_text = str(
            payload.get("finetune_traceback")
            or payload.get("fallback_traceback")
            or ""
        ).lower()
        combined = f"{text}\n{traceback_text}"
        if "keyerror: 'default'" in combined or "ids_relevant_features" in combined:
            return "finetune_predict"
        if payload.get("fallback_error") and not payload.get("fallback_used"):
            return "direct_fit"
        if payload.get("finetune_error"):
            return "finetune_fit"
        if payload.get("error"):
            return "train_graph"
        return "completed"

    @staticmethod
    def _unique_text(items: List[str]) -> List[str]:
        unique: List[str] = []
        for item in items:
            text = str(item or "").strip()
            if text and text not in unique:
                unique.append(text)
        return unique

    @staticmethod
    def _unique_node_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in items:
            key = (item.get("id"), item.get("operation"))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    @classmethod
    def _should_attempt_node_recovery(cls, payload: Dict[str, Any]) -> bool:
        if not isinstance(payload, dict):
            return True
        if payload.get("error"):
            return True
        if not payload.get("finetune_error"):
            return False

        primary_value = cls._primary_score(payload)
        metrics_missing = not any(
            isinstance(payload.get(key), dict) and payload.get(key)
            for key in ("test_metrics", "metrics", "graph_metrics")
        )
        if metrics_missing:
            return True
        primary_metric = cls._primary_metric(payload)
        zero_is_failure_metrics = {"accuracy", "f1", "precision", "recall", "roc_auc"}
        return primary_metric in zero_is_failure_metrics and primary_value <= 0

    @staticmethod
    def _primary_score(payload: Dict[str, Any]) -> float:
        for key in ("score",):
            if key not in payload or payload.get(key) is None:
                continue
            try:
                return float(payload.get(key))
            except (TypeError, ValueError):
                pass
        for section in ("test_metrics", "metrics", "graph_metrics"):
            metrics = payload.get(section)
            if not isinstance(metrics, dict):
                continue
            for key in ("primary_metric_value", "primary_score"):
                try:
                    return float(metrics.get(key) or 0)
                except (TypeError, ValueError):
                    continue
        return 0.0

    @staticmethod
    def _primary_metric(payload: Dict[str, Any]) -> str:
        for section in ("test_metrics", "metrics", "graph_metrics"):
            metrics = payload.get(section)
            if isinstance(metrics, dict) and metrics.get("primary_metric"):
                return str(metrics.get("primary_metric"))
        return ""

    @classmethod
    def _recovery_resolved(cls, original: Dict[str, Any], recovered: Dict[str, Any]) -> bool:
        if not isinstance(recovered, dict) or recovered.get("error"):
            return False
        if original.get("finetune_error"):
            return not recovered.get("finetune_error") and not recovered.get("fallback_used")
        return not cls._should_attempt_node_recovery(recovered)

    @classmethod
    def _best_fallback_candidate(
        cls,
        current: Dict[str, Any] | None,
        recovered: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        recovered_graph: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        if not isinstance(recovered, dict) or recovered.get("error"):
            return current
        if not (recovered.get("finetune_error") or recovered.get("fallback_used")):
            return current
        score = cls._primary_score(recovered)
        if current and score <= current.get("score", 0):
            return current
        return {
            "score": score,
            "run": recovered,
            "nodes": nodes,
            "graph": recovered_graph,
        }

    @staticmethod
    def _failure_message(payload: Dict[str, Any]) -> str:
        if not isinstance(payload, dict):
            return "train_graph returned no data"
        if payload.get("error"):
            return str(payload.get("error"))
        if payload.get("finetune_error"):
            return f"Fedot.Industrial finetune failed: {payload.get('finetune_error')}"
        if payload.get("fallback_error"):
            return f"Direct-fit fallback failed: {payload.get('fallback_error')}"
        return "Training returned unusable metrics"

    @staticmethod
    def _append_training_diagnostic(result: EngineerResult) -> None:
        """Surface factual training path issues even when fallback metrics exist."""
        if result.graph_error:
            technical_message = result.graph_error
            kind = "runtime_error"
            summary = (
                "Fedot.Industrial finetune raised an exception during training."
                if result.finetune_error
                else "Engineer could not finish training the proposed graph."
            )
        elif result.finetune_error:
            technical_message = f"Fedot.Industrial finetune failed: {result.finetune_error}"
            kind = "finetune_fallback"
            summary = (
                "Fedot.Industrial finetune failed, so Engineer reported direct-fit fallback metrics."
            )
        else:
            return

        if any(
            isinstance(item, dict) and item.get("technical_message") == technical_message
            for item in result.diagnostics
        ):
            return

        result.diagnostics.append({
            "agent": "Engineer",
            "kind": kind,
            "summary": summary,
            "technical_message": technical_message,
            "finetune_error": result.finetune_error,
            "finetune_traceback": result.finetune_traceback[-6000:],
            "fallback_used": result.fallback_used,
            "recoverable": True,
        })

    @staticmethod
    def _extract_diagnostics(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        diagnostics = payload.get("diagnostics", []) if isinstance(payload, dict) else []
        clean = []
        for item in diagnostics:
            if isinstance(item, dict):
                item = {**item, "agent": item.get("agent") or "Engineer"}
                clean.append(item)
        return clean

    @staticmethod
    def _unique_diagnostics(diagnostics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in diagnostics:
            key = (item.get("kind"), item.get("summary"), item.get("technical_message"))
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique
