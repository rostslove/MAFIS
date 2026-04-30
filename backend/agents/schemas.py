"""Shared data types for inter-agent communication."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


# ============== Available Operations Registry ==============

AVAILABLE_OPERATIONS = {
    "ts_forecasting": {
        "models": [
            "ar", "stl_arima", "ets", "eigen_forecaster", "topo_forecaster",
            "lagged_forecaster", "deepar_model", "tcn_model", "glm",
            "nbeats_model",
        ],
        "preprocessing": [
            "smoothing", "gaussian_filter", "exog_ts",
            "wavelet_basis", "fourier_basis", "eigen_basis",
        ],
    },
    "ts_classification": {
        "models": [
            "industrial_stat_clf", "industrial_freq_clf", "industrial_manifold_clf",
            "inception_model", "resnet_model", "pdl_clf",
        ],
        "preprocessing": [
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "riemann_extractor", "minirocket_extractor",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "channel_filtration", "bagging", "scaling", "normalization",
        ],
    },
    "ts_regression": {
        "models": [
            "industrial_stat_reg", "industrial_freq_reg", "industrial_manifold_reg",
            "inception_model", "resnet_model", "nbeats_model", "tcn_model",
            "tst_model", "deepar_model",
        ],
        "preprocessing": [
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "riemann_extractor", "minirocket_extractor",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "channel_filtration", "bagging", "scaling", "normalization",
        ],
    },
    "anomaly_detection": {
        "models": [
            "one_class_svm", "sst", "unscented_kalman_filter", "stat_detector",
            "arima_detector", "iforest_detector", "conv_ae_detector", "lstm_ae_detector",
        ],
        "preprocessing": [
            "channel_filtration", "gaussian_filter", "smoothing",
        ],
    },
    "classification": {
        "models": [
            "industrial_stat_clf", "industrial_freq_clf", "industrial_manifold_clf",
            "inception_model", "resnet_model",
        ],
        "preprocessing": [
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "riemann_extractor", "minirocket_extractor",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "channel_filtration", "bagging", "scaling", "normalization",
        ],
    },
    "regression": {
        "models": [
            "industrial_stat_reg", "industrial_freq_reg", "industrial_manifold_reg",
            "inception_model", "resnet_model", "nbeats_model", "tcn_model",
        ],
        "preprocessing": [
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "riemann_extractor", "minirocket_extractor",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "scaling", "normalization",
        ],
    },
}

INDUSTRIAL_STRATEGIES = {
    "federated_automl": {
        "description": "Distributed AutoML with RAF ensemble",
        "params": {"timeout": "int - time per worker"},
        "applicable": ["classification", "regression", "ts_classification", "ts_regression"],
    },
    "kernel_automl": {
        "description": "Kernel-based ensemble AutoML",
        "params": {},
        "applicable": ["classification", "regression", "ts_classification", "ts_regression"],
    },
    "forecasting_assumptions": {
        "description": "Forecasting with preset pipeline assumptions",
        "params": {},
        "applicable": ["ts_forecasting"],
    },
    "forecasting_exogenous": {
        "description": "Forecasting with exogenous variables",
        "params": {"exog_variable": "np.array", "data_type": "str"},
        "applicable": ["ts_forecasting"],
    },
    "lora_strategy": {
        "description": "Low-rank adaptation strategy for fine-tuning",
        "params": {},
        "applicable": ["classification", "regression", "ts_classification", "ts_regression"],
    },
    "sampling_strategy": {
        "description": "Data sampling strategy (CUR or Random)",
        "params": {"sampling_algorithm": "CUR|Random", "sampling_range": "list[float]"},
        "applicable": ["classification", "regression"],
    },
}

METRICS_BY_TASK = {
    "classification": ["f1", "accuracy", "precision", "roc_auc", "logloss"],
    "ts_classification": ["f1", "accuracy", "precision", "roc_auc", "logloss"],
    "regression": ["r2", "rmse", "mse", "mae"],
    "ts_regression": ["r2", "rmse", "mse", "mae"],
    "ts_forecasting": ["rmse", "mse", "mae", "mape", "smape", "r2"],
    "anomaly_detection": ["f1", "accuracy", "precision", "roc_auc"],
}

EXPLAIN_METHODS = ["point", "recurrence", "shap", "lime"]

ALL_PROBLEM_TYPES = [
    "classification", "regression",
    "ts_forecasting", "ts_classification", "ts_regression",
    "anomaly_detection",
]

TS_PROBLEM_TYPES = [
    "ts_forecasting", "ts_classification", "ts_regression", "anomaly_detection",
]


def is_ts_task(task_type: str) -> bool:
    return task_type in TS_PROBLEM_TYPES


# ============== Data Types ==============

@dataclass
class FedotConfig:
    """Full FedotIndustrial configuration with TS support."""
    problem: str = "classification"
    timeout: float = 5.0
    n_jobs: int = -1
    preset: str = "auto"
    metric: Optional[str] = None
    cv_folds: Optional[int] = None
    seed: Optional[int] = None
    available_operations: Optional[List[str]] = None
    backend_method: str = "cpu"
    use_meta_rules: bool = True
    strategy: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None
    logging_level: int = 20
    # Time series specific
    task_params: Optional[Dict[str, Any]] = None  # e.g. {"forecast_length": 14}
    pop_size: Optional[int] = None
    max_depth: Optional[int] = None
    with_tuning: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, filtering None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FedotConfig":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class IterationRecord:
    """Record of one iteration's key results for cross-iteration memory."""
    iteration: int = 0
    best_model: str = ""
    best_score: float = 0.0
    fedot_score: float = 0.0
    fedot_config_used: Dict[str, Any] = field(default_factory=dict)
    winner: str = ""
    suggested_changes: Dict[str, Any] = field(default_factory=dict)
    failed_baselines: List[str] = field(default_factory=list)


@dataclass
class DataContext:
    """Shared data context passed between agents.

    NOTE: Does NOT store raw data arrays. All tools work with csv_path directly.
    This keeps DataContext lightweight and serializable.
    """
    csv_path: str
    target_column: str
    task_type: str
    profile: Dict[str, Any] = field(default_factory=dict)
    n_train_samples: int = 0
    n_val_samples: int = 0
    # TS-specific context
    forecast_length: Optional[int] = None
    is_time_series: bool = False
    # Cross-iteration memory
    iteration_history: List[IterationRecord] = field(default_factory=list)


@dataclass
class ToolCall:
    """Record of a single tool call for logging."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool = True
    error: Optional[str] = None


@dataclass
class ArchitectResult:
    """Output from Architect agent."""
    baseline_pipelines: Dict[str, Any] = field(default_factory=dict)
    fedot_config: Optional[FedotConfig] = None
    analysis: str = ""
    reasoning: str = ""
    selected_baselines: List[str] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_pipelines": list(self.baseline_pipelines.keys()),
            "fedot_config": self.fedot_config.to_dict() if self.fedot_config else {},
            "analysis": self.analysis,
            "reasoning": self.reasoning,
            "selected_baselines": self.selected_baselines,
            "tool_calls": [
                {"tool": tc.tool_name, "args": tc.arguments, "result": str(tc.result)[:500]}
                for tc in self.tool_calls
            ],
        }


@dataclass
class PipelineResult:
    """Result from training a single pipeline."""
    name: str
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class EngineerResult:
    """Output from Engineer agent."""
    baseline_results: List[PipelineResult] = field(default_factory=list)
    fedot_result: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    best_model: str = ""
    comparison: Dict[str, Any] = field(default_factory=dict)
    fedot_config_used: Optional[FedotConfig] = None
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_results": [
                {"name": r.name, "score": r.score, "metrics": r.metrics, "error": r.error}
                for r in self.baseline_results
            ],
            "fedot_result": self.fedot_result,
            "best_score": self.best_score,
            "best_model": self.best_model,
            "comparison": self.comparison,
            "fedot_config_used": self.fedot_config_used.to_dict() if self.fedot_config_used else {},
            "tool_calls": [
                {"tool": tc.tool_name, "args": tc.arguments, "result": str(tc.result)[:500]}
                for tc in self.tool_calls
            ],
        }


@dataclass
class CriticFeedback:
    """Structured feedback from Critic agent."""
    score_assessment: str = ""
    winner: str = ""
    suggested_fedot_changes: Dict[str, Any] = field(default_factory=dict)
    suggested_pipeline_changes: List[str] = field(default_factory=list)
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)  # from explain()
    should_stop: bool = False
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    full_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_assessment": self.score_assessment,
            "winner": self.winner,
            "suggested_fedot_changes": self.suggested_fedot_changes,
            "suggested_pipeline_changes": self.suggested_pipeline_changes,
            "error_analysis": self.error_analysis,
            "feature_importance": self.feature_importance,
            "explanation": self.explanation,
            "should_stop": self.should_stop,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "full_response": self.full_response,
            "tool_calls": [
                {"tool": tc.tool_name, "args": tc.arguments, "result": str(tc.result)[:500]}
                for tc in self.tool_calls
            ],
        }


@dataclass
class ScribeReport:
    """Output from Scribe agent."""
    title: str = ""
    summary: str = ""
    methodology: str = ""
    results: str = ""
    recommendations: List[str] = field(default_factory=list)
    full_response: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "methodology": self.methodology,
            "results": self.results,
            "recommendations": self.recommendations,
            "full_response": self.full_response,
            "tool_calls": [
                {"tool": tc.tool_name, "args": tc.arguments, "result": str(tc.result)[:500]}
                for tc in self.tool_calls
            ],
        }
