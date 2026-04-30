"""Engineer: trains the Architect graph and baseline models through MCP tools."""

import json
import logging
from typing import Any, Dict, List

from .base_agent import BaseAgent
from .schemas import ArchitectResult, BaselineResult, DataContext, EngineerResult

logger = logging.getLogger("Engineer")


class Engineer(BaseAgent):
    """Technical agent responsible for fitting and tuning a proposed graph."""

    ALLOWED_TOOLS = [
        "get_baselines",
        "train_baseline",
        "tune_graph_hyperparameters",
        "train_graph",
    ]

    SYSTEM_PROMPT = """You are an ML Engineer. You receive a validated pipeline graph.
Your role is to train simple baselines, tune graph hyperparameters, and train the graph as-is.
Do not change graph structure; only node parameters may be tuned."""

    def __init__(self, name: str = "Engineer", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, architect_result: ArchitectResult, data_context: DataContext) -> EngineerResult:
        self._tool_call_log = []
        result = EngineerResult()
        graph_json = json.dumps(architect_result.graph, ensure_ascii=False)

        try:
            await self._train_baselines(result, data_context)
            graph_run = await self._train_graph(graph_json, data_context)

            if isinstance(graph_run, dict):
                result.graph_score = float(graph_run.get("score") or 0)
                result.graph_metrics = graph_run.get("metrics", {}) or {}
                result.tuned_nodes = graph_run.get("tuned_nodes", []) or []
                result.graph_error = graph_run.get("error", "") or ""
                result.diagnostics.extend(self._extract_diagnostics(graph_run))
                result.diagnostics = self._unique_diagnostics(result.diagnostics)
                if result.graph_error:
                    result.graph_metrics["error"] = result.graph_error

            if result.baseline_results:
                best = max(result.baseline_results, key=lambda r: r.score)
                result.best_baseline_name = best.name
                result.best_baseline_score = best.score

            result.tool_calls = self.get_tool_calls()
            logger.info(
                "[Engineer] graph=%.4f, best baseline=%s %.4f",
                result.graph_score,
                result.best_baseline_name or "none",
                result.best_baseline_score,
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

    async def _train_baselines(self, result: EngineerResult, dc: DataContext) -> None:
        response = await self.call_mcp_tool("get_baselines", {"task_type": dc.task_type})
        names = response.get("baselines", []) if isinstance(response, dict) else []

        for name in names:
            baseline = await self.call_mcp_tool(
                "train_baseline",
                {
                    "csv_path": dc.csv_path,
                    "target_column": dc.target_column,
                    "baseline_name": name,
                    "task_type": dc.task_type,
                },
            )
            if isinstance(baseline, dict):
                result.baseline_results.append(
                    BaselineResult(
                        name=baseline.get("name", name),
                        score=float(baseline.get("score") or 0),
                        metrics=baseline.get("metrics", {}) or {},
                        error=baseline.get("error"),
                    )
                )

    async def _train_graph(self, graph_json: str, dc: DataContext) -> dict:
        args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "iterations": 20,
        }
        if dc.forecast_length:
            args["forecast_length"] = dc.forecast_length

        tuned = await self.call_mcp_tool("tune_graph_hyperparameters", args)
        if isinstance(tuned, dict) and not tuned.get("error") and float(tuned.get("score") or 0) != 0:
            return tuned

        logger.warning("[Engineer] tuning failed or returned zero score, training graph without tuning")
        fallback_args = {
            "graph_json": graph_json,
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
        }
        if dc.forecast_length:
            fallback_args["forecast_length"] = dc.forecast_length
        trained = await self.call_mcp_tool("train_graph", fallback_args)
        if not isinstance(trained, dict):
            return {"score": 0, "error": "train_graph returned no data"}

        if isinstance(tuned, dict) and tuned.get("error") and trained.get("error"):
            trained.setdefault("diagnostics", [])
            trained["diagnostics"].extend(self._extract_diagnostics(tuned))
            trained["tuning_error"] = tuned.get("error")
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
