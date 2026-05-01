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
Do not change graph structure; only node parameters may be tuned."""

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
                result.tuned_nodes = graph_run.get("tuned_nodes", []) or []
                result.graph_error = graph_run.get("error", "") or ""
                result.target_info = graph_run.get("target_info", {}) or result.target_info
                result.training_notes.extend(graph_run.get("training_notes", []) or [])
                result.diagnostics.extend(self._extract_diagnostics(graph_run))
                result.diagnostics = self._unique_diagnostics(result.diagnostics)
                if result.graph_error:
                    result.graph_metrics["error"] = result.graph_error

            if result.target_info.get("reference_encoded"):
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
