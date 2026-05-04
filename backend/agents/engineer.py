"""Engineer: trains the approved Architect graph through MCP tools."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from .schemas import ArchitectResult, DataContext, EngineerResult

logger = logging.getLogger("Engineer")


class Engineer(BaseAgent):
    """Technical agent responsible for fitting and tuning a proposed graph."""

    ALLOWED_TOOLS = [
        "tune_graph_hyperparameters",
        "train_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Engineer. You receive a validated pipeline graph.
Your role is to tune graph hyperparameters when safe and train the graph as-is.
If a non-root operation crashes during training, skip that operation, retry the
remaining graph, and report the skipped node as recovery feedback for Critic."""

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
                result.training_strategy = graph_run.get("training_strategy", {}) or {}
                result.tuned_nodes = graph_run.get("tuned_nodes", []) or []
                result.graph_error = graph_run.get("error", "") or ""
                result.target_info = graph_run.get("target_info", {}) or result.target_info
                result.training_notes.extend(graph_run.get("training_notes", []) or [])
                result.diagnostics.extend(self._extract_diagnostics(graph_run))
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
                    "recommendations": ["Check the graph and dataset format, then run again."],
                    "recoverable": True,
                }
            )
            result.tool_calls = self.get_tool_calls()
            return result

    async def _train_graph(self, graph_json: str, dc: DataContext) -> dict:
        if dc.task_type in ("classification", "ts_classification"):
            trained = await self._train_without_tuning(graph_json, dc)
            if trained.get("error"):
                trained = await self._recover_by_skipping_node(graph_json, dc, trained)
            if trained.get("training_strategy"):
                trained.setdefault("training_notes", []).append(
                    "Node-level hyperparameter tuning was skipped because the selected Fedot.Industrial "
                    "training strategy performs its own branch AutoML search."
                )
            else:
                if trained.get("recovered_from_error"):
                    trained.setdefault("training_notes", []).append(
                        "Fedot hyperparameter tuning was skipped for classification. After node-skip recovery, "
                        "the remaining graph was trained and scored with external sklearn metrics."
                    )
                else:
                    trained.setdefault("training_notes", []).append(
                        "Fedot hyperparameter tuning was skipped for classification because Fedot.Industrial 0.5 "
                        "prints internal ROC-AUC shape tracebacks for binary probability matrices during tuning. "
                        "The graph was trained as-is and scored with external sklearn metrics. To improve it, use "
                        "manual graph mutations or explicit node parameters instead of Fedot's classification tuner."
                    )
            return trained

        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "iterations": 20,
            "test_size": dc.test_size,
        }
        if dc.primary_metric:
            args["primary_metric"] = dc.primary_metric
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length

        tuned = await self.call_mcp_tool("tune_graph_hyperparameters", args)
        if isinstance(tuned, dict) and not tuned.get("error") and float(tuned.get("score") or 0) != 0:
            return tuned

        logger.warning("[Engineer] tuning failed or returned zero score, training graph without tuning")
        trained = await self._train_without_tuning(graph_json, dc)
        if isinstance(tuned, dict) and tuned.get("error") and trained.get("error"):
            trained.setdefault("diagnostics", [])
            trained["diagnostics"].extend(self._extract_diagnostics(tuned))
            trained["tuning_error"] = tuned.get("error")
        if trained.get("error"):
            trained = await self._recover_by_skipping_node(graph_json, dc, trained)
        return trained

    async def _train_without_tuning(self, graph_json: str, dc: DataContext) -> dict:
        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "test_size": dc.test_size,
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
        """Try one-node bypass recovery for crashes in optional graph nodes."""
        graph = self._parse_graph_json(graph_json)
        if not graph or graph.get("training_strategy"):
            return failed_run

        root_id = self._root_id(graph.get("nodes", []))
        candidates = [
            node for node in self._candidate_skip_nodes(graph)
            if node.get("id") and node.get("id") != root_id
        ]
        if not candidates:
            return failed_run

        original_error = str(failed_run.get("error") or "")
        attempts: List[Dict[str, Any]] = []
        for node in candidates:
            recovered_graph = self._graph_with_node_skipped(graph, str(node["id"]))
            recovered_json = json.dumps(recovered_graph, ensure_ascii=False)
            recovered = await self._train_without_tuning(recovered_json, dc)
            if not isinstance(recovered, dict):
                recovered = {"score": 0, "error": "train_graph returned no data"}

            if not recovered.get("error"):
                diagnostic = self._node_skipped_diagnostic(node, original_error, recovered_graph)
                recovered.setdefault("diagnostics", [])
                recovered["diagnostics"] = (
                    list(failed_run.get("diagnostics", []) or [])
                    + [diagnostic]
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

            attempts.append({
                "id": node.get("id"),
                "operation": node.get("operation"),
                "error": recovered.get("error"),
            })

        failed_run.setdefault("diagnostics", [])
        failed_run["diagnostics"].append(
            {
                "agent": "Engineer",
                "kind": "node_skip_recovery_failed",
                "summary": "Engineer tried to skip optional nodes, but the graph still failed during training.",
                "technical_message": original_error,
                "problem_nodes": [
                    {"id": item.get("id"), "operation": item.get("operation")}
                    for item in attempts
                ],
                "recovery_attempts": attempts,
                "recommendations": [
                    "Ask Architect to simplify the graph or replace preprocessing nodes before evaluating quality."
                ],
                "recoverable": True,
            }
        )
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

    @staticmethod
    def _unique_inputs(inputs: List[str], node_id: str) -> List[str]:
        unique: List[str] = []
        for item in inputs:
            if not item or item == node_id or item in unique:
                continue
            unique.append(item)
        return unique

    @staticmethod
    def _node_skipped_diagnostic(
        node: Dict[str, Any],
        original_error: str,
        recovered_graph: Dict[str, Any],
    ) -> Dict[str, Any]:
        problem_node = {"id": node.get("id"), "operation": node.get("operation")}
        return {
            "agent": "Engineer",
            "kind": "node_skipped_after_runtime_error",
            "summary": (
                f"Node '{node.get('id')}' ({node.get('operation')}) failed during training, "
                "so Engineer skipped it and retrained the remaining graph."
            ),
            "technical_message": original_error[:2000],
            "failed_nodes": [problem_node],
            "problem_nodes": [problem_node],
            "effective_graph": recovered_graph,
            "suggested_mutations": [{"type": "remove", "node_id": node.get("id")}],
            "recommendations": [
                "Critic should decide whether Architect removes this node permanently or replaces it with a safer alternative.",
                "If the dataset contains missing values, prefer explicit imputation before adding preprocessing transforms.",
            ],
            "recoverable": True,
            "recovered": True,
        }

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
