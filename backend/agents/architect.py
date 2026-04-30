import json
import logging
from typing import Dict, Any, Optional, Tuple

from .base_agent import BaseAgent
from .schemas import (
    DataContext, FedotConfig, ArchitectResult, CriticFeedback,
    ALL_PROBLEM_TYPES, is_ts_task,
)

logger = logging.getLogger(__name__)


class Architect(BaseAgent):
    """Architect: analyzes data, selects baselines, proposes Fedot.Industrial config."""

    ALLOWED_TOOLS = [
        "get_data_profile", "get_available_baselines",
        "get_available_operations", "propose_fedot_config", "mutate_fedot_config",
    ]

    SYSTEM_PROMPT = """You are an ML Architect. Your job is to analyze a dataset and propose the best ML strategy.

You have access to MCP tools:
1. get_data_profile - Analyze the dataset (statistics, issues, recommendations).
2. get_available_baselines - Get sklearn baselines (only for classification/regression).
3. get_available_operations - Get Fedot.Industrial models/preprocessing/strategies for ANY task type.
4. propose_fedot_config - Propose a FedotIndustrial configuration with all parameters.
5. mutate_fedot_config - Modify existing config based on Critic feedback.

SUPPORTED TASK TYPES: classification, regression, ts_forecasting, ts_classification, ts_regression, anomaly_detection.

WORKFLOW:
1. Call get_data_profile to understand the data.
2. Call get_available_operations to see available models/strategies for this task.
3. For classification/regression: call get_available_baselines.
4. Call propose_fedot_config with your configuration.
   - For ts_forecasting: MUST set task_params='{"forecast_length": N}'.
   - Choose appropriate strategy (e.g. "forecasting_assumptions" for ts_forecasting).
5. If previous feedback exists, call mutate_fedot_config to apply suggested changes.

After using tools, provide:
ANALYSIS: Your data analysis.
REASONING: Why you chose this configuration.
"""

    def __init__(self, name: str = "Architect", mcp_client=None):
        super().__init__(name=name, mcp_client=mcp_client)

    async def execute(
        self,
        data_context: DataContext,
        iteration: int,
        prev_feedback: Optional[CriticFeedback] = None,
        task_type: str = "classification",
    ) -> Tuple[Dict[str, Any], ArchitectResult]:
        """Run the Architect agent. Returns (baselines_dict, ArchitectResult)."""
        self._tool_call_log = []
        result = ArchitectResult()

        try:
            logger.info(f"[Architect] Iteration {iteration}, task: {task_type}")
            context = self._build_user_message(data_context, iteration, prev_feedback, task_type)
            response = await self.call_llm(context)

            if not response.get("success"):
                logger.warning("[Architect] LLM failed, using fallback")
                fb = self._fallback(data_context, task_type)
                return fb.baseline_pipelines, fb

            text = response.get("full_response", "")

            # Extract Fedot config from tool call results
            fedot_config = self._extract_fedot_config_from_log(task_type, data_context)
            baselines = self._extract_baselines_from_log()

            # If LLM didn't call get_available_baselines, use defaults for non-TS tasks
            if not baselines and task_type in _DEFAULT_BASELINES:
                default = _DEFAULT_BASELINES[task_type]
                baselines = {name: {"model": info, "steps": []} for name, info in default.items()}
                logger.info(f"[Architect] LLM didn't select baselines, using defaults: {list(baselines.keys())}")

            result.fedot_config = fedot_config
            result.baseline_pipelines = baselines
            result.selected_baselines = list(baselines.keys()) if baselines else []
            result.analysis = self.extract_field(text, "ANALYSIS") or text[:500]
            result.reasoning = self.extract_field(text, "REASONING") or ""
            result.tool_calls = self.get_tool_calls()

            logger.info(f"[Architect] Done. Baselines: {result.selected_baselines}, Config: {fedot_config.to_dict() if fedot_config else 'default'}")
            return baselines, result

        except Exception as e:
            logger.error(f"[Architect] Error: {e}")
            fb = self._fallback(data_context, task_type)
            return fb.baseline_pipelines, fb

    def _build_user_message(self, dc: DataContext, iteration, prev_feedback, task_type):
        profile = dc.profile
        msg = f"""Iteration: {iteration}
Task Type: {task_type}
Is Time Series: {is_ts_task(task_type)}
CSV Path: {dc.csv_path}
Target Column: {dc.target_column}

Data Profile:
- Samples: {profile.get('n_samples', 'Unknown')}
- Features: {profile.get('n_features', 'Unknown')}
- Issues: {', '.join(profile.get('issues', ['Unknown']))}
"""
        if dc.forecast_length:
            msg += f"- Forecast Length: {dc.forecast_length}\n"

        # Cross-iteration memory
        if dc.iteration_history:
            msg += "\nPREVIOUS ITERATIONS:\n"
            for rec in dc.iteration_history:
                msg += (
                    f"  Iter {rec.iteration}: best={rec.best_model} ({rec.best_score:.4f}), "
                    f"fedot={rec.fedot_score:.4f}, winner={rec.winner}"
                )
                if rec.failed_baselines:
                    msg += f", FAILED: {rec.failed_baselines}"
                msg += f", config={rec.fedot_config_used}\n"
            msg += "\nAvoid repeating failed approaches. Build on what worked.\n"

        msg += "\nPlease use tools to analyze data and propose configuration.\n"

        if prev_feedback:
            msg += f"""
PREVIOUS ITERATION FEEDBACK:
- Winner: {prev_feedback.winner}
- Assessment: {prev_feedback.score_assessment}
- Suggested Fedot changes: {json.dumps(prev_feedback.suggested_fedot_changes)}
- Weaknesses: {prev_feedback.weaknesses}

Use mutate_fedot_config to apply the suggested changes.
"""
        return msg

    def _extract_fedot_config_from_log(self, task_type, dc):
        """Extract FedotConfig from tool call results in the log."""
        for tc in self._tool_call_log:
            if tc.tool_name in ("propose_fedot_config", "mutate_fedot_config") and tc.success:
                result = tc.result
                if isinstance(result, dict):
                    cfg = result.get("fedot_config") or result.get("new_config")
                    if cfg:
                        return FedotConfig.from_dict(cfg)

        # Default config
        config = FedotConfig(problem=task_type)
        if task_type == "ts_forecasting" and dc.forecast_length:
            config.task_params = {"forecast_length": dc.forecast_length}
        return config

    def _extract_baselines_from_log(self):
        """Extract baseline pipeline names from tool call results."""
        for tc in self._tool_call_log:
            if tc.tool_name == "get_available_baselines" and tc.success:
                result = tc.result
                if isinstance(result, dict):
                    pipelines = result.get("pipelines", {})
                    # Return pipeline names as dict (actual Pipeline objects are in MCP server)
                    return {name: info for name, info in pipelines.items()}
        return {}

    def _fallback(self, dc, task_type):
        config = FedotConfig(problem=task_type)
        if task_type == "ts_forecasting" and dc.forecast_length:
            config.task_params = {"forecast_length": dc.forecast_length}

        # Include default sklearn baselines for classification/regression
        baselines = _DEFAULT_BASELINES.get(task_type, {})

        result = ArchitectResult(
            baseline_pipelines={name: {"model": info, "steps": []} for name, info in baselines.items()},
            selected_baselines=list(baselines.keys()),
            fedot_config=config,
            analysis="Fallback: LLM unavailable, using all default baselines",
            reasoning="Default configuration",
            tool_calls=self.get_tool_calls(),
        )
        return result


# Default baselines used by fallback path. Matches mcp_server.py _get_pipelines().
_DEFAULT_BASELINES = {
    "classification": {
        "baseline_lr": "LogisticRegression",
        "svc": "SVC",
        "xgb_clf": "XGBClassifier",
        "rf": "RandomForest",
    },
    "regression": {
        "baseline_lr": "LinearRegression",
        "ridge": "Ridge",
        "svr": "SVR",
        "xgb_reg": "XGBRegressor",
    },
}
