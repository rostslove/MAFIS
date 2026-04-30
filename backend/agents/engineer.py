import json
import logging
from typing import Dict, Any, Optional, List

from .base_agent import BaseAgent
from .schemas import (
    DataContext, FedotConfig, ArchitectResult, EngineerResult,
    PipelineResult, is_ts_task,
)

logger = logging.getLogger("Engineer")


class Engineer(BaseAgent):
    """Engineer: trains baselines via MCP, calls Fedot.Industrial via MCP, compares results."""

    ALLOWED_TOOLS = ["train_baseline", "call_fedot", "compare_results"]

    SYSTEM_PROMPT = """You are an ML Engineer. Your job is to train models and evaluate them.

You have access to MCP tools:
1. train_baseline - Train a sklearn baseline pipeline (only classification/regression).
2. call_fedot - Call Fedot.Industrial with FULL parameter control (all task types including time series).
3. compare_results - Compare all models and determine the best.

WORKFLOW:
1. For classification/regression: train each baseline using train_baseline.
2. For time series tasks: skip baselines, go directly to call_fedot.
3. Call call_fedot with the Architect's proposed configuration.
4. Call compare_results with all results as a JSON array.

After using tools, provide a training summary.
"""

    def __init__(self, name: str = "Engineer", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(self, architect_result: ArchitectResult, data_context: DataContext) -> EngineerResult:
        """Run the Engineer agent."""
        self._tool_call_log = []
        result = EngineerResult()

        try:
            task_type = data_context.task_type
            is_ts = is_ts_task(task_type)
            baselines = architect_result.selected_baselines
            fedot_config = architect_result.fedot_config

            self.logger.info(f"[Engineer] Task: {task_type}, TS: {is_ts}, Baselines: {baselines}")

            # Build context
            config_str = json.dumps(fedot_config.to_dict()) if fedot_config else "{}"
            baselines_str = ", ".join(baselines) if baselines and not is_ts else "N/A (time series)"

            context = f"""Task type: {task_type}
Is time series: {is_ts}
CSV path: {data_context.csv_path}
Target column: {data_context.target_column}

Baselines to train: {baselines_str}

Fedot.Industrial configuration from Architect:
{config_str}

Please:
"""
            if not is_ts and baselines:
                context += f"1. Train each baseline: {baselines_str} using train_baseline tool.\n"
            context += f"{'2' if not is_ts and baselines else '1'}. Call call_fedot with the configuration above.\n"
            context += f"{'3' if not is_ts and baselines else '2'}. Call compare_results with all results as JSON array.\n"

            response = await self.call_llm(context)

            # Extract results from tool call log
            result = self._build_result_from_log(data_context)

            # Guarantee: if LLM didn't call required tools, run them manually
            has_fedot = bool(result.fedot_result and result.fedot_result.get("score", 0) > 0)
            has_baselines = len(result.baseline_results) > 0 or is_ts
            if not has_fedot or not has_baselines:
                self.logger.warning(f"[Engineer] Missing tool calls (fedot={has_fedot}, baselines={has_baselines}), running fallback")
                result = await self._manual_execution(architect_result, data_context)

            result.tool_calls = self.get_tool_calls()
            self.logger.info(f"[Engineer] Done. Best: {result.best_model} ({result.best_score:.4f})")
            return result

        except Exception as e:
            self.logger.error(f"[Engineer] Error: {e}")
            try:
                result = await self._manual_execution(architect_result, data_context)
                result.tool_calls = self.get_tool_calls()
                return result
            except Exception as e2:
                self.logger.error(f"[Engineer] Manual fallback failed: {e2}")
                return EngineerResult(tool_calls=self.get_tool_calls())

    def _build_result_from_log(self, dc: DataContext) -> EngineerResult:
        """Build EngineerResult from MCP tool call log."""
        result = EngineerResult()

        for tc in self._tool_call_log:
            if not tc.success or not isinstance(tc.result, dict):
                continue

            if tc.tool_name == "train_baseline":
                pr = PipelineResult(
                    name=tc.result.get("name", "unknown"),
                    score=tc.result.get("score", 0),
                    metrics=tc.result.get("metrics", {}),
                    error=tc.result.get("error"),
                )
                result.baseline_results.append(pr)

            elif tc.tool_name == "call_fedot":
                result.fedot_result = tc.result
                config_used = tc.result.get("config_used", {})
                if config_used:
                    result.fedot_config_used = FedotConfig.from_dict(config_used)

            elif tc.tool_name == "compare_results":
                result.comparison = tc.result
                result.best_model = tc.result.get("best_overall", "")
                result.best_score = tc.result.get("best_overall_score", 0)

        # If no comparison was called, compute best manually
        if not result.comparison:
            all_scores = [(r.name, r.score) for r in result.baseline_results if r.score > 0]
            fedot_score = result.fedot_result.get("score", 0) if result.fedot_result else 0
            if fedot_score > 0:
                all_scores.append(("fedot_industrial", fedot_score))

            if all_scores:
                best = max(all_scores, key=lambda x: x[1])
                result.best_model = best[0]
                result.best_score = best[1]
            else:
                result.best_model = "none"
                result.best_score = 0.0

            # Build comparison dict
            best_bl = max((r.score for r in result.baseline_results), default=0)
            result.comparison = {
                "best_overall": result.best_model,
                "best_overall_score": result.best_score,
                "fedot_score": fedot_score,
                "best_baseline_score": best_bl,
                "architect_wins": best_bl > fedot_score,
                "comparison_text": f"Baseline best: {best_bl:.4f} vs Fedot: {fedot_score:.4f}",
            }

        # Cross-iteration comparison: Fedot current vs best score from previous iterations
        fedot_score = result.fedot_result.get("score", 0) if result.fedot_result else 0
        if dc.iteration_history:
            prev_best = max((r.best_score for r in dc.iteration_history), default=0)
            prev_fedot_best = max((r.fedot_score for r in dc.iteration_history), default=0)
            delta = fedot_score - prev_fedot_best
            result.comparison["previous_best_score"] = prev_best
            result.comparison["previous_fedot_best"] = prev_fedot_best
            result.comparison["fedot_improvement"] = delta
            result.comparison["improved_over_previous"] = delta > 0
            result.comparison["cross_iteration_text"] = (
                f"Fedot now: {fedot_score:.4f} vs previous best Fedot: {prev_fedot_best:.4f} "
                f"({'+' if delta >= 0 else ''}{delta:.4f})"
            )

        return result

    async def _manual_execution(self, architect_result, data_context):
        """Fallback: manually call tools via MCP."""
        dc = data_context
        result = EngineerResult()

        # Default baselines per task type (used if Architect didn't populate)
        _DEFAULT_BASELINES = {
            "classification": ["baseline_lr", "svc", "xgb_clf", "rf"],
            "regression": ["baseline_lr", "ridge", "svr", "xgb_reg"],
        }

        # Train baselines — ALWAYS for classification/regression
        if not is_ts_task(dc.task_type):
            baselines_to_train = architect_result.selected_baselines or _DEFAULT_BASELINES.get(dc.task_type, [])
            if baselines_to_train:
                self.logger.info(f"[Engineer] Manual: training baselines {baselines_to_train}")
            for name in baselines_to_train:
                r = await self.call_mcp_tool("train_baseline", {
                    "csv_path": dc.csv_path,
                    "target_column": dc.target_column,
                    "pipeline_name": name,
                    "task_type": dc.task_type,
                })
                if isinstance(r, dict):
                    result.baseline_results.append(PipelineResult(
                        name=r.get("name", name), score=r.get("score", 0),
                        metrics=r.get("metrics", {}), error=r.get("error"),
                    ))

        # Call Fedot
        fedot_args = {
            "csv_path": dc.csv_path,
            "target_column": dc.target_column,
            "problem": dc.task_type,
        }
        if architect_result.fedot_config:
            cfg = architect_result.fedot_config.to_dict()
            for k, v in cfg.items():
                if k not in fedot_args and v is not None:
                    # MCP tools expect strings for complex types
                    if isinstance(v, (dict, list)):
                        fedot_args[k] = json.dumps(v)
                    else:
                        fedot_args[k] = v

        fedot_r = await self.call_mcp_tool("call_fedot", fedot_args)
        if isinstance(fedot_r, dict):
            result.fedot_result = fedot_r
            if fedot_r.get("config_used"):
                result.fedot_config_used = FedotConfig.from_dict(fedot_r["config_used"])

        # Compare
        all_results = [
            {"name": r.name, "score": r.score, "type": "baseline", "metrics": r.metrics}
            for r in result.baseline_results
        ]
        all_results.append({"name": "fedot_industrial", "score": fedot_r.get("score", 0) if fedot_r else 0, "type": "automl"})

        comp_r = await self.call_mcp_tool("compare_results", {"results_json": json.dumps(all_results)})
        if isinstance(comp_r, dict):
            result.comparison = comp_r
            result.best_model = comp_r.get("best_overall", "")
            result.best_score = comp_r.get("best_overall_score", 0)

        return result
