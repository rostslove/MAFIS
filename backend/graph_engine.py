"""Pipeline graph: build, validate, mutate, convert to Fedot, visualize."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from path_utils import normalize_csv_path

logger = logging.getLogger("GraphEngine")


def _add_local_fedot_industrial_to_path() -> str:
    """Prefer a sibling Fedot.Industrial source checkout when it exists."""
    candidates = []
    env_path = os.environ.get("FEDOT_INDUSTRIAL_PATH")
    if env_path:
        candidates.append(Path(env_path))

    project_root = Path(__file__).resolve().parents[1]
    candidates.extend([
        project_root.parent / "Fedot.Industrial",
        project_root.parent / "FEDOT.Industrial",
        Path("/opt/Fedot.Industrial"),
    ])

    for candidate in candidates:
        if (candidate / "fedot_ind").is_dir():
            source_path = str(candidate.resolve())
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
            return source_path
    return ""


FEDOT_INDUSTRIAL_SOURCE = _add_local_fedot_industrial_to_path()

try:
    import fedot_ind  # noqa: F401
    from fedot_ind.core.repository.initializer_industrial_models import IndustrialModels

    FEDOT_IND_VERSION = getattr(fedot_ind, "__version__", "unknown")
    IndustrialModels().setup_repository()
except ImportError as exc:
    raise ImportError(
        "Fedot.Industrial 0.5.0 is required. Install it from a local clone with "
        "`git clone https://github.com/aimclub/Fedot.Industrial.git`, "
        "`cd Fedot.Industrial`, `poetry install`, then run this backend inside "
        "that Poetry environment. You can also set FEDOT_INDUSTRIAL_PATH to the clone path."
    ) from exc
except Exception as exc:
    logger.warning("Fedot.Industrial repository setup failed: %s", exc)

try:
    from fedot_ind.core.repository import model_repository as fedot_ind_model_repository
except Exception as exc:
    fedot_ind_model_repository = None
    logger.warning("Fedot.Industrial model repository import failed: %s", exc)

try:
    from fedot_ind.api.utils.industrial_strategy import IndustrialStrategy
except Exception as exc:
    IndustrialStrategy = None
    logger.warning("Fedot.Industrial strategy import failed: %s", exc)

from fedot.core.data.data import InputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PipelineNode
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams


SUPPORTED_TASKS = ["classification", "regression", "ts_classification", "ts_regression", "ts_forecasting"]
TABULAR_TASKS = ["classification", "regression"]
TS_TASKS = ["ts_classification", "ts_regression", "ts_forecasting"]


def is_ts_task(task_type: str) -> bool:
    return task_type in TS_TASKS


FEDOT_MODEL_REPOSITORY_REFERENCE = "fedot_ind.core.repository.model_repository"
FEDOT_MODEL_JSON_REFERENCE = "fedot_ind.core.repository.data.industrial_model_repository.json"
FEDOT_DATA_OPERATION_JSON_REFERENCE = "fedot_ind.core.repository.data.industrial_data_operation_repository.json"
FEDOT_STRATEGY_REFERENCE = "fedot_ind.api.utils.industrial_strategy.IndustrialStrategy"


def _unique(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _repo_dict(name: str) -> Dict[str, Any]:
    if fedot_ind_model_repository is None:
        return {}
    value = getattr(fedot_ind_model_repository, name, {}) or {}
    return dict(value) if isinstance(value, dict) else {}


def _repo_keys(name: str) -> List[str]:
    return list(_repo_dict(name).keys())


def _load_fedot_industrial_json(filename: str) -> Dict[str, Any]:
    if not FEDOT_INDUSTRIAL_SOURCE:
        return {}
    path = Path(FEDOT_INDUSTRIAL_SOURCE) / "fedot_ind" / "core" / "repository" / "data" / filename
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to read Fedot.Industrial repository JSON %s: %s", path, exc)
    return {}


INDUSTRIAL_MODEL_REPOSITORY_JSON = _load_fedot_industrial_json("industrial_model_repository.json")
INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON = _load_fedot_industrial_json(
    "industrial_data_operation_repository.json"
)


def _json_operations(repository: Dict[str, Any]) -> Dict[str, Any]:
    operations = repository.get("operations", {})
    return operations if isinstance(operations, dict) else {}


def _json_operation_names_by_meta(repository: Dict[str, Any], metas: set[str]) -> List[str]:
    names = []
    for name, spec in _json_operations(repository).items():
        if isinstance(spec, dict) and spec.get("meta") in metas:
            names.append(name)
    return names


def _framework_operation_names() -> set[str]:
    names = set()
    for repo_name in (
        "SKLEARN_REG_MODELS",
        "SKLEARN_CLF_MODELS",
        "FEDOT_PREPROC_MODEL",
        "INDUSTRIAL_PREPROC_MODEL",
        "INDUSTRIAL_CLF_PREPROC_MODEL",
        "INDUSTRIAL_REG_AUTOML_MODEL",
        "INDUSTRIAL_CLF_AUTOML_MODEL",
        "ANOMALY_DETECTION_MODELS",
        "NEURAL_MODEL",
        "NEURAL_MODEL_FOR_CLF",
        "FORECASTING_MODELS",
        "FORECASTING_PREPROC",
        "PRIMARY_FORECASTING_MODELS",
    ):
        names.update(_repo_keys(repo_name))
    names.update(_json_operations(INDUSTRIAL_MODEL_REPOSITORY_JSON).keys())
    names.update(_json_operations(INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON).keys())
    return names


def _only_framework_ops(names: List[str]) -> List[str]:
    framework_names = _framework_operation_names()
    if not framework_names:
        return _unique(names)
    return _unique([name for name in names if name in framework_names])


def _operation_repository_info(operation: str) -> Dict[str, Any]:
    for repository, reference in (
        (INDUSTRIAL_MODEL_REPOSITORY_JSON, FEDOT_MODEL_JSON_REFERENCE),
        (INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON, FEDOT_DATA_OPERATION_JSON_REFERENCE),
    ):
        operation_spec = _json_operations(repository).get(operation)
        if not isinstance(operation_spec, dict):
            continue
        meta_name = operation_spec.get("meta", "")
        metadata = repository.get("metadata", {})
        meta_spec = metadata.get(meta_name, {}) if isinstance(metadata, dict) else {}
        if not isinstance(meta_spec, dict):
            meta_spec = {}
        return {
            "source": reference,
            "meta": meta_name,
            "description": meta_spec.get("description", ""),
            "tags": operation_spec.get("tags", []),
            "presets": operation_spec.get("presets", []),
            "input_type": operation_spec.get("input_type") or meta_spec.get("input_type"),
            "output_type": operation_spec.get("output_type") or meta_spec.get("output_type"),
        }
    if operation in _framework_operation_names():
        return {"source": FEDOT_MODEL_REPOSITORY_REFERENCE, "meta": "", "description": ""}
    return {"source": "MAFIS fallback", "meta": "", "description": ""}


# Local fallback is used only when Fedot.Industrial repository introspection is unavailable.
FALLBACK_OPERATIONS: Dict[str, Dict[str, List[str]]] = {
    "classification": {
        "preprocessing": [],
        "models": ["rf", "xgboost", "logit", "knn", "lgbm", "mlp", "dt"],
    },
    "regression": {
        "preprocessing": [],
        "models": ["xgbreg", "treg", "ridge", "lasso", "lgbmreg", "knnreg", "dtreg", "sgdr"],
    },
    "ts_classification": {
        "preprocessing": [
            "scaling", "normalization", "channel_filtration",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "minirocket_extractor", "riemann_extractor",
        ],
        "models": [
            "industrial_stat_clf", "industrial_freq_clf", "industrial_manifold_clf",
            "inception_model", "resnet_model",
        ],
    },
    "ts_regression": {
        "preprocessing": [
            "scaling", "normalization", "channel_filtration",
            "wavelet_basis", "fourier_basis", "eigen_basis",
            "quantile_extractor", "topological_extractor", "recurrence_extractor",
            "minirocket_extractor",
        ],
        "models": [
            "industrial_stat_reg", "industrial_freq_reg", "industrial_manifold_reg",
            "inception_model", "resnet_model", "nbeats_model", "tcn_model",
        ],
    },
    "ts_forecasting": {
        "preprocessing": ["smoothing", "gaussian_filter", "exog_ts", "eigen_basis"],
        "models": [
            "ar", "stl_arima", "ets", "lagged_forecaster", "eigen_forecaster",
            "topo_forecaster", "glm", "tcn_model", "nbeats_model",
        ],
    },
}


def _build_framework_operations() -> Dict[str, Dict[str, List[str]]]:
    if not (_repo_keys("SKLEARN_CLF_MODELS") or _repo_keys("SKLEARN_REG_MODELS")):
        return FALLBACK_OPERATIONS

    sklearn_class = set(_json_operation_names_by_meta(
        INDUSTRIAL_MODEL_REPOSITORY_JSON,
        {"sklearn_class", "custom_class"},
    ))
    sklearn_class.update(_repo_keys("SKLEARN_CLF_MODELS"))
    sklearn_reg = set(_json_operation_names_by_meta(
        INDUSTRIAL_MODEL_REPOSITORY_JSON,
        {"sklearn_regr", "custom_regr"},
    ))
    sklearn_reg.update(_repo_keys("SKLEARN_REG_MODELS"))

    tabular_class_excluded = {
        "one_class_svm",
        "pdl_clf",
        "industrial_stat_clf",
        "industrial_freq_clf",
        "industrial_manifold_clf",
    }
    tabular_reg_excluded = {
        "pdl_reg",
        "industrial_stat_reg",
        "industrial_freq_reg",
        "industrial_manifold_reg",
    }

    tabular_preproc_common = _json_operation_names_by_meta(
        INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON,
        {"custom_preprocessing", "sklearn_categorical", "dimension_transformation"},
    )
    class_preproc = tabular_preproc_common + _json_operation_names_by_meta(
        INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON,
        {"classification_preprocessing"},
    ) + _repo_keys("INDUSTRIAL_CLF_PREPROC_MODEL")
    reg_preproc = tabular_preproc_common + _json_operation_names_by_meta(
        INDUSTRIAL_DATA_OPERATION_REPOSITORY_JSON,
        {"regression_preprocessing"},
    )

    industrial_preproc = [
        name
        for name in _repo_keys("INDUSTRIAL_PREPROC_MODEL")
        if name not in {"bagging", "isolation_forest_class", "isolation_forest_reg"}
    ]
    ts_scalers = [name for name in _repo_keys("FEDOT_PREPROC_MODEL") if name in {"scaling", "normalization"}]

    ts_class_models = _repo_keys("INDUSTRIAL_CLF_AUTOML_MODEL") + _repo_keys("NEURAL_MODEL_FOR_CLF")
    ts_class_models = [
        name
        for name in ts_class_models
        if name not in {"ensembled_featuresbagging", "pdl_clf", "one_class_svm"}
    ]

    ts_reg_neural_allow = {
        "inception_model", "resnet_model", "nbeats_model", "tcn_model", "tst_model", "xcm_model"
    }
    ts_reg_models = _repo_keys("INDUSTRIAL_REG_AUTOML_MODEL") + [
        name for name in _repo_keys("NEURAL_MODEL") if name in ts_reg_neural_allow
    ]
    ts_reg_models = [
        name
        for name in ts_reg_models
        if name not in {"ensembled_featuresbagging", "pdl_reg"}
    ]

    forecasting_models = _repo_keys("FORECASTING_MODELS") + _repo_keys("PRIMARY_FORECASTING_MODELS")
    forecasting_preproc = _repo_keys("FORECASTING_PREPROC")

    return {
        "classification": {
            "preprocessing": _only_framework_ops(class_preproc),
            "models": _only_framework_ops([
                name for name in sorted(sklearn_class)
                if name not in tabular_class_excluded and not name.startswith("industrial_")
            ]),
        },
        "regression": {
            "preprocessing": _only_framework_ops(reg_preproc),
            "models": _only_framework_ops([
                name for name in sorted(sklearn_reg)
                if name not in tabular_reg_excluded and not name.startswith("industrial_")
            ]),
        },
        "ts_classification": {
            "preprocessing": _only_framework_ops(ts_scalers + industrial_preproc),
            "models": _only_framework_ops(ts_class_models),
        },
        "ts_regression": {
            "preprocessing": _only_framework_ops(ts_scalers + industrial_preproc),
            "models": _only_framework_ops(ts_reg_models),
        },
        "ts_forecasting": {
            "preprocessing": _only_framework_ops(forecasting_preproc + ["eigen_basis"]),
            "models": _only_framework_ops(forecasting_models),
        },
    }


# Atomic operations available per task type. Used for validation and proposal.
OPERATIONS: Dict[str, Dict[str, List[str]]] = _build_framework_operations()

METRICS_BY_TASK: Dict[str, List[str]] = {
    "classification": ["roc_auc", "f1", "accuracy", "precision"],
    "regression": ["r2", "rmse", "mse", "mae"],
    "ts_classification": ["f1", "accuracy", "roc_auc"],
    "ts_regression": ["r2", "rmse", "mae"],
    "ts_forecasting": ["rmse", "mae", "mape", "smape"],
}


# Sensible default graph per task - used as starter / fallback.
DEFAULT_GRAPHS: Dict[str, List[Dict[str, Any]]] = {
    "classification": [
        {"id": "model", "operation": "rf", "params": {}, "inputs": []},
    ],
    "regression": [
        {"id": "model", "operation": "ridge", "params": {}, "inputs": []},
    ],
    "ts_classification": [
        {"id": "stat_feat", "operation": "quantile_extractor", "params": {}, "inputs": []},
        {"id": "freq_feat", "operation": "fourier_basis", "params": {}, "inputs": []},
        {"id": "model", "operation": "industrial_stat_clf", "params": {}, "inputs": ["stat_feat", "freq_feat"]},
    ],
    "ts_regression": [
        {"id": "stat_feat", "operation": "quantile_extractor", "params": {}, "inputs": []},
        {"id": "freq_feat", "operation": "fourier_basis", "params": {}, "inputs": []},
        {"id": "model", "operation": "industrial_stat_reg", "params": {}, "inputs": ["stat_feat", "freq_feat"]},
    ],
    "ts_forecasting": [
        {"id": "smooth", "operation": "smoothing", "params": {}, "inputs": []},
        {"id": "model", "operation": "ar", "params": {}, "inputs": ["smooth"]},
    ],
}


def _graph_template(task_type: str, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "nodes": [
            {
                "id": node.get("id"),
                "operation": node.get("operation"),
                "params": dict(node.get("params", {}) or {}),
                "inputs": list(node.get("inputs", []) or []),
            }
            for node in nodes
        ],
    }


INDUSTRIAL_GRAPH_TEMPLATES: Dict[str, List[Dict[str, Any]]] = {
    "classification": [
        {
            "name": "tabular_direct_model",
            "description": "Direct FEDOT/Fedot.Industrial table model over numeric CSV features.",
            "graph": _graph_template("classification", DEFAULT_GRAPHS["classification"]),
        },
    ],
    "regression": [
        {
            "name": "tabular_direct_model",
            "description": "Direct FEDOT/Fedot.Industrial table model over numeric CSV features.",
            "graph": _graph_template("regression", DEFAULT_GRAPHS["regression"]),
        },
    ],
    "ts_classification": [
        {
            "name": "industrial_feature_fusion",
            "description": "Parallel statistical and frequency feature branches merged by Fedot.Industrial.",
            "graph": _graph_template("ts_classification", DEFAULT_GRAPHS["ts_classification"]),
        },
    ],
    "ts_regression": [
        {
            "name": "industrial_feature_fusion",
            "description": "Parallel statistical and frequency feature branches merged by Fedot.Industrial.",
            "graph": _graph_template("ts_regression", DEFAULT_GRAPHS["ts_regression"]),
        },
    ],
    "ts_forecasting": [
        {
            "name": "forecasting_smoothing_ar",
            "description": "Fedot.Industrial forecasting preprocessing followed by an autoregressive model.",
            "graph": _graph_template("ts_forecasting", DEFAULT_GRAPHS["ts_forecasting"]),
        },
    ],
}


OPERATION_DESCRIPTIONS: Dict[str, str] = {
    "rf": "Random forest classifier; robust starter model for tabular classification.",
    "xgboost": "Gradient boosting classifier; useful for nonlinear tabular patterns.",
    "logit": "Linear logistic model; good for small or mostly linear classification tasks.",
    "knn": "Nearest-neighbor classifier; sensitive to feature scale.",
    "lgbm": "LightGBM classifier; fast gradient boosting when available.",
    "mlp": "Neural network classifier; needs enough samples.",
    "dt": "Decision tree classifier; interpretable but can overfit.",
    "ridge": "Regularized linear regressor; stable default for tabular regression.",
    "lasso": "Sparse linear regressor; can suppress weak features.",
    "xgbreg": "Gradient boosting regressor; strong nonlinear tabular model.",
    "treg": "Tree ensemble regressor; robust nonlinear regression model.",
    "lgbmreg": "LightGBM regressor; fast gradient boosting when available.",
    "knnreg": "Nearest-neighbor regressor; sensitive to feature scale.",
    "dtreg": "Decision tree regressor; interpretable but can overfit.",
    "sgdr": "Linear stochastic-gradient regressor; useful on larger tables.",
    "scaling": "Scale time-series channels before feature extraction.",
    "normalization": "Normalize time-series channels before feature extraction.",
    "channel_filtration": "Filter noisy or weak time-series channels.",
    "wavelet_basis": "Wavelet decomposition for localized time-frequency patterns.",
    "fourier_basis": "Frequency-domain representation for periodic signals.",
    "eigen_basis": "Basis decomposition for compact signal representation.",
    "quantile_extractor": "Statistical time-series features based on quantiles.",
    "topological_extractor": "Topological descriptors of signal shape.",
    "recurrence_extractor": "Recurrence-plot based time-series features.",
    "minirocket_extractor": "Fast convolutional time-series feature extractor.",
    "riemann_extractor": "Riemannian features for multichannel signals.",
    "industrial_stat_clf": "Classifier over statistical industrial time-series features.",
    "industrial_freq_clf": "Classifier over frequency-domain industrial features.",
    "industrial_manifold_clf": "Classifier over manifold/topological signal features.",
    "industrial_stat_reg": "Regressor over statistical industrial time-series features.",
    "industrial_freq_reg": "Regressor over frequency-domain industrial features.",
    "industrial_manifold_reg": "Regressor over manifold/topological signal features.",
    "inception_model": "Deep time-series model with inception-style blocks.",
    "resnet_model": "Deep residual model for time-series data.",
    "smoothing": "Smooth a forecasting series before modeling.",
    "gaussian_filter": "Gaussian filtering for noisy forecasting series.",
    "exog_ts": "Use exogenous time-series features.",
    "ar": "Autoregressive forecasting model.",
    "stl_arima": "Seasonal-trend decomposition plus ARIMA.",
    "ets": "Exponential smoothing forecasting model.",
    "lagged_forecaster": "Forecasting model over lagged features.",
    "eigen_forecaster": "Forecasting with eigen-basis representation.",
    "topo_forecaster": "Forecasting using topological signal descriptors.",
    "glm": "Generalized linear forecasting model.",
    "tcn_model": "Temporal convolutional neural forecasting/regression model.",
    "nbeats_model": "Deep neural forecasting/regression model.",
}


def _build_operation_description(operation: str) -> str:
    repo_info = _operation_repository_info(operation)
    description = str(repo_info.get("description") or "").strip()
    tags = repo_info.get("tags") or []
    presets = repo_info.get("presets") or []
    details = []
    if tags:
        details.append(f"tags: {', '.join(str(tag) for tag in tags[:6])}")
    if presets:
        details.append(f"presets: {', '.join(str(preset) for preset in presets[:4])}")
    if description and details:
        return f"{description} ({'; '.join(details)})."
    if description:
        return description
    return OPERATION_DESCRIPTIONS.get(operation, "")


for _task_ops in OPERATIONS.values():
    for _operation in _task_ops.get("preprocessing", []) + _task_ops.get("models", []):
        repo_description = _build_operation_description(_operation)
        if repo_description:
            OPERATION_DESCRIPTIONS[_operation] = repo_description


OPERATION_PARAM_HINTS: Dict[str, List[Dict[str, Any]]] = {
    "xgboost": [
        {"name": "scale_pos_weight", "description": "Multiplier for the positive class contribution in binary boosting."},
        {"name": "use_eval_set", "description": "Controls whether FEDOT creates an internal validation split for boosting fit calls."},
        {"name": "early_stopping_rounds", "description": "Number of validation rounds without improvement before stopping; null disables eval-set early stopping."},
        {"name": "max_depth", "description": "Maximum depth of each boosting tree."},
        {"name": "learning_rate", "description": "Shrinkage applied to each boosting step."},
        {"name": "n_estimators", "description": "Number of boosting trees."},
    ],
    "rf": [
        {"name": "class_weight", "description": "Class weighting policy used by the classifier."},
        {"name": "max_depth", "description": "Maximum depth of each tree."},
        {"name": "n_estimators", "description": "Number of trees in the forest."},
    ],
    "logit": [
        {"name": "class_weight", "description": "Class weighting policy used by the linear classifier."},
        {"name": "C", "description": "Inverse regularization strength for logistic regression."},
        {"name": "max_iter", "description": "Maximum number of solver iterations."},
    ],
    "lgbm": [
        {"name": "class_weight", "description": "Class weighting policy used by LightGBM."},
        {"name": "use_eval_set", "description": "Controls whether FEDOT creates an internal validation split for LightGBM fit calls."},
        {"name": "early_stopping_rounds", "description": "Number of validation rounds without improvement before stopping; null disables eval-set early stopping."},
        {"name": "num_leaves", "description": "Maximum number of leaves in each LightGBM tree."},
        {"name": "learning_rate", "description": "Shrinkage applied to each boosting step."},
        {"name": "n_estimators", "description": "Number of boosting trees."},
    ],
    "dt": [
        {"name": "class_weight", "description": "Class weighting policy used by the tree classifier."},
        {"name": "max_depth", "description": "Maximum depth of the decision tree."},
    ],
    "ridge": [
        {"name": "alpha", "description": "L2 regularization strength."},
    ],
    "xgbreg": [
        {"name": "use_eval_set", "description": "Controls whether FEDOT creates an internal validation split for boosting fit calls."},
        {"name": "early_stopping_rounds", "description": "Number of validation rounds without improvement before stopping; null disables eval-set early stopping."},
        {"name": "max_depth", "description": "Maximum depth of each boosting tree."},
        {"name": "learning_rate", "description": "Shrinkage applied to each boosting step."},
        {"name": "n_estimators", "description": "Number of boosting trees."},
    ],
    "lgbmreg": [
        {"name": "use_eval_set", "description": "Controls whether FEDOT creates an internal validation split for LightGBM fit calls."},
        {"name": "early_stopping_rounds", "description": "Number of validation rounds without improvement before stopping; null disables eval-set early stopping."},
        {"name": "num_leaves", "description": "Maximum number of leaves in each LightGBM tree."},
        {"name": "learning_rate", "description": "Shrinkage applied to each boosting step."},
        {"name": "n_estimators", "description": "Number of boosting trees."},
    ],
    "treg": [
        {"name": "max_depth", "description": "Maximum depth of each tree."},
        {"name": "n_estimators", "description": "Number of trees in the ensemble."},
    ],
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if k != "hyperopt-dist"}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        return getattr(value, "__name__", str(value))
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _build_industrial_search_space() -> Dict[str, Dict[str, Any]]:
    """Return Fedot.Industrial parameter ranges in a JSON-safe form."""
    try:
        from fedot_ind.core.tuning.search_space import get_industrial_search_space

        class _SearchSpaceContext:
            custom_search_space = None
            replace_default_search_space = False

        raw_space = get_industrial_search_space(_SearchSpaceContext())
        if not isinstance(raw_space, dict):
            return {}
        return {
            str(operation): _json_safe(params)
            for operation, params in raw_space.items()
            if isinstance(params, dict)
        }
    except Exception as exc:
        logger.warning("Fedot.Industrial search space import failed: %s", exc)
        return {}


INDUSTRIAL_SEARCH_SPACE: Dict[str, Dict[str, Any]] = _build_industrial_search_space()


def _operation_search_space(operation: str) -> Dict[str, Any]:
    return INDUSTRIAL_SEARCH_SPACE.get(operation, {})


FEDERATED_AUTOML_REFERENCE = "fedot_ind.core.ensemble.random_automl_forest.RAFEnsembler"


STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "federated_automl": (
        "RAFEnsembler-based strategy: train several internal AutoML branches on sample folds "
        "and stack their predictions through a final head model."
    ),
    "kernel_automl": "KernelEnsembler-based strategy over transformed feature representations.",
    "forecasting_assumptions": "Forecasting strategy that evaluates built-in forecasting assumptions.",
    "forecasting_exogenous": "Forecasting strategy that augments the model with exogenous time-series variables.",
    "lora_strategy": "LoRA model strategy exposed by Fedot.Industrial for neural adaptation flows.",
    "sampling_strategy": "Big-dataset strategy that samples the training tensor before fitting task models.",
}


STRATEGY_REQUIREMENTS: Dict[str, str] = {
    "federated_automl": "supervised target, train/test split, branch AutoML params",
    "kernel_automl": "kernel ensemble configuration and dedicated prediction path",
    "forecasting_assumptions": "forecasting task contract and forecast horizon",
    "forecasting_exogenous": "forecasting task plus exogenous time-series columns",
    "lora_strategy": "LoRA/model-specific data and training configuration",
    "sampling_strategy": "sampling_algorithm, sampling_range, and task-specific runner",
}


def _framework_strategy_names() -> List[str]:
    if IndustrialStrategy is not None:
        try:
            strategy = IndustrialStrategy({}, "federated_automl", {})
            return list(strategy.industrial_strategy_fit.keys())
        except Exception as exc:
            logger.warning("Fedot.Industrial strategy catalog introspection failed: %s", exc)
    return [
        "federated_automl",
        "kernel_automl",
        "forecasting_assumptions",
        "forecasting_exogenous",
        "lora_strategy",
        "sampling_strategy",
    ]


FRAMEWORK_TRAINING_STRATEGIES = _framework_strategy_names()
WIRED_STRATEGY_TASKS: Dict[str, set[str]] = {
    "federated_automl": {"classification", "regression", "ts_classification", "ts_regression"},
}


TRAINING_STRATEGY_HINTS: Dict[str, List[Dict[str, Any]]] = {
    task_type: [
        {
            "name": strategy_name,
            "description": STRATEGY_DESCRIPTIONS.get(strategy_name, ""),
            "fedot_industrial_reference": (
                FEDERATED_AUTOML_REFERENCE
                if strategy_name == "federated_automl"
                else FEDOT_STRATEGY_REFERENCE
            ),
            "source": FEDOT_STRATEGY_REFERENCE,
            "status": (
                "wired"
                if task_type in WIRED_STRATEGY_TASKS.get(strategy_name, set())
                else "upstream only"
            ),
        }
        for strategy_name in FRAMEWORK_TRAINING_STRATEGIES
    ]
    for task_type in SUPPORTED_TASKS
}


def get_training_strategy_hints(task_type: str) -> List[Dict[str, Any]]:
    return TRAINING_STRATEGY_HINTS.get(task_type, [])


FEDERATED_AUTOML_PIPELINE_EFFECTS = [
    "The approved graph stays as the user-facing contract.",
    "Fedot.Industrial splits the train part into branch folds.",
    "Each branch runs AutoML over available_operations.",
    "Branch predictions are stacked into a final head model.",
]


FEDERATED_AUTOML_EDITABLE_PARAMS = [
    {
        "name": "data_type",
        "description": "Controls whether branch AutoML searches tabular operations or time-series feature extraction operations.",
    },
    {
        "name": "available_operations",
        "description": "Candidate operations that branch AutoML may place inside generated branch pipelines.",
    },
    {
        "name": "timeout",
        "description": "AutoML budget used inside strategy training.",
    },
    {
        "name": "n_splits",
        "description": "Number of branch folds/models trained before stacking.",
    },
    {
        "name": "n_jobs",
        "description": "Parallelism passed to branch AutoML/models where supported.",
    },
]


FEDERATED_AUTOML_RUNTIME_NOTICE = (
    "Fedot.Industrial training strategies can take noticeably longer than direct graph execution, "
    "because they may train several internal AutoML branch pipelines before scoring the approved graph."
)


def _strategy_default_operations(task_type: str, preferred: List[str]) -> List[str]:
    available = OPERATIONS.get(task_type, {}).get("models", [])
    return [operation for operation in preferred if operation in available]


TRAINING_STRATEGIES: Dict[str, List[Dict[str, Any]]] = {
    "classification": [
        {
            "name": "federated_automl",
            "label": "Federated AutoML ensemble",
            "description": "Split the training set into worker folds, train one AutoML branch per fold, and join branches with an xgboost head.",
            "fedot_industrial_reference": FEDERATED_AUTOML_REFERENCE,
            "pipeline_effects": FEDERATED_AUTOML_PIPELINE_EFFECTS,
            "editable_params": FEDERATED_AUTOML_EDITABLE_PARAMS,
            "runtime_notice": FEDERATED_AUTOML_RUNTIME_NOTICE,
            "default_params": {
                "timeout": 10,
                "data_type": "table",
                "problem": "classification",
                "n_jobs": 1,
                "available_operations": _strategy_default_operations(
                    "classification",
                    ["rf", "xgboost", "logit", "dt", "lgbm", "catboost"],
                ),
            },
        },
    ],
    "regression": [
        {
            "name": "federated_automl",
            "label": "Federated AutoML ensemble",
            "description": "Split the training set into worker folds, train one AutoML branch per fold, and join branches with a tree-regression head.",
            "fedot_industrial_reference": FEDERATED_AUTOML_REFERENCE,
            "pipeline_effects": FEDERATED_AUTOML_PIPELINE_EFFECTS,
            "editable_params": FEDERATED_AUTOML_EDITABLE_PARAMS,
            "runtime_notice": FEDERATED_AUTOML_RUNTIME_NOTICE,
            "default_params": {
                "timeout": 10,
                "data_type": "table",
                "problem": "regression",
                "n_jobs": 1,
                "available_operations": _strategy_default_operations(
                    "regression",
                    ["treg", "xgbreg", "ridge", "lasso", "dtreg", "lgbmreg", "sgdr", "catboostreg"],
                ),
            },
        },
    ],
    "ts_classification": [
        {
            "name": "federated_automl",
            "label": "Federated AutoML TS ensemble",
            "description": "Train multiple AutoML branches on fixed-window time-series folds and join branch predictions with a classification head.",
            "fedot_industrial_reference": FEDERATED_AUTOML_REFERENCE,
            "pipeline_effects": FEDERATED_AUTOML_PIPELINE_EFFECTS,
            "editable_params": FEDERATED_AUTOML_EDITABLE_PARAMS,
            "runtime_notice": FEDERATED_AUTOML_RUNTIME_NOTICE,
            "default_params": {"timeout": 10, "data_type": "time_series", "problem": "classification", "n_jobs": 1},
        },
    ],
    "ts_regression": [
        {
            "name": "federated_automl",
            "label": "Federated AutoML TS ensemble",
            "description": "Train multiple AutoML branches on fixed-window time-series folds and join branch predictions with a regression head.",
            "fedot_industrial_reference": FEDERATED_AUTOML_REFERENCE,
            "pipeline_effects": FEDERATED_AUTOML_PIPELINE_EFFECTS,
            "editable_params": FEDERATED_AUTOML_EDITABLE_PARAMS,
            "runtime_notice": FEDERATED_AUTOML_RUNTIME_NOTICE,
            "default_params": {"timeout": 10, "data_type": "time_series", "problem": "regression", "n_jobs": 1},
        },
    ],
}


def _build_fedot_industrial_strategy_catalog() -> List[Dict[str, Any]]:
    catalog = []
    for strategy_name in FRAMEWORK_TRAINING_STRATEGIES:
        wired = strategy_name == "federated_automl"
        catalog.append({
            "strategy": strategy_name,
            "status": "wired" if wired else "upstream only",
            "current_scope": (
                "classification, regression, ts_classification, ts_regression"
                if wired
                else "not exposed in MAFIS execution yet"
            ),
            "requires": STRATEGY_REQUIREMENTS.get(strategy_name, ""),
            "description": STRATEGY_DESCRIPTIONS.get(strategy_name, ""),
            "fedot_industrial_reference": (
                FEDERATED_AUTOML_REFERENCE if wired else FEDOT_STRATEGY_REFERENCE
            ),
            "source": FEDOT_STRATEGY_REFERENCE,
        })
    return catalog


FEDOT_INDUSTRIAL_STRATEGY_CATALOG = _build_fedot_industrial_strategy_catalog()


def get_training_strategies(task_type: str) -> List[Dict[str, Any]]:
    return TRAINING_STRATEGIES.get(task_type, [])


def get_fedot_industrial_strategy_catalog() -> List[Dict[str, Any]]:
    return FEDOT_INDUSTRIAL_STRATEGY_CATALOG


def get_training_strategy_spec(task_type: str, name: str) -> Optional[Dict[str, Any]]:
    return next((item for item in get_training_strategies(task_type) if item.get("name") == name), None)


def normalize_training_strategy(task_type: str, strategy: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not strategy:
        return None
    if isinstance(strategy, str):
        strategy = {"name": strategy}
    if not isinstance(strategy, dict):
        return None

    name = str(strategy.get("name") or strategy.get("strategy") or "").strip()
    if not name or name in ("default", "graph", "none"):
        return None
    spec = get_training_strategy_spec(task_type, name)
    params = dict((spec or {}).get("default_params", {}))
    incoming_params = strategy.get("params") or strategy.get("strategy_params") or {}
    if isinstance(incoming_params, dict):
        params.update(incoming_params)
    return {"name": name, "params": params}


def get_operation_catalog(task_type: str) -> Dict[str, List[Dict[str, Any]]]:
    ops = OPERATIONS.get(task_type, {})
    return {
        group: [
            {
                "operation": name,
                "description": OPERATION_DESCRIPTIONS.get(name, ""),
                "param_hints": OPERATION_PARAM_HINTS.get(name, []),
                "industrial_search_space": _operation_search_space(name),
                "source": _operation_repository_info(name).get("source", ""),
                "fedot_industrial_meta": _operation_repository_info(name).get("meta", ""),
                "tags": _operation_repository_info(name).get("tags", []),
                "presets": _operation_repository_info(name).get("presets", []),
            }
            for name in names
        ]
        for group, names in ops.items()
    }


@dataclass
class GraphNode:
    id: str
    operation: str
    params: Dict[str, Any] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)


@dataclass
class PipelineGraph:
    """A directed acyclic graph of atomic operations forming an ML pipeline."""

    task_type: str
    nodes: List[GraphNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"task_type": self.task_type, "nodes": [asdict(n) for n in self.nodes]}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineGraph":
        nodes = [GraphNode(**n) for n in d.get("nodes", [])]
        task_type = d.get("task_type", "classification")
        return cls(task_type=task_type, nodes=nodes)

    @classmethod
    def default(cls, task_type: str) -> "PipelineGraph":
        spec = DEFAULT_GRAPHS.get(task_type, DEFAULT_GRAPHS["classification"])
        return cls.from_dict({"task_type": task_type, "nodes": spec})

    def valid_operations(self) -> set[str]:
        ops = OPERATIONS.get(self.task_type, {})
        return set(ops.get("preprocessing", []) + ops.get("models", []))

    def validate(self) -> Tuple[bool, str]:
        if not self.nodes:
            return False, "Graph has no nodes"
        if self.task_type not in OPERATIONS:
            return False, f"Unknown task_type '{self.task_type}'"

        ids = [n.id for n in self.nodes]
        if len(set(ids)) != len(ids):
            return False, "Duplicate node IDs"

        valid_ops = self.valid_operations()
        id_set = set(ids)
        for n in self.nodes:
            if n.operation not in valid_ops:
                return False, f"Operation '{n.operation}' not allowed for task '{self.task_type}'"
            for inp in n.inputs:
                if inp not in id_set:
                    return False, (
                        f"Node '{n.id}' references missing input '{inp}'. "
                        "Each input must be the id of an existing node in graph.nodes; "
                        "use inputs=[] when a node consumes raw data."
                    )

        # Single root (node with no children)
        children: set[str] = set()
        for n in self.nodes:
            children.update(n.inputs)
        roots = id_set - children
        if len(roots) != 1:
            return False, f"Pipeline must have exactly one root, got {len(roots)}"

        # Acyclicity check via topological sort
        try:
            self._topo_sorted()
        except ValueError as e:
            return False, str(e)

        root_node = next(n for n in self.nodes if n.id == self.root_id())
        if root_node.operation not in OPERATIONS[self.task_type].get("models", []):
            return False, f"Root node '{root_node.id}' must be a model operation"

        return True, "OK"

    def _topo_sorted(self) -> List[GraphNode]:
        nodes_by_id = {n.id: n for n in self.nodes}
        order: List[GraphNode] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def dfs(nid: str) -> None:
            if nid in visited:
                return
            if nid in visiting:
                raise ValueError(f"Cycle detected at node '{nid}'")
            visiting.add(nid)
            for inp in nodes_by_id[nid].inputs:
                dfs(inp)
            visiting.remove(nid)
            visited.add(nid)
            order.append(nodes_by_id[nid])

        for nid in nodes_by_id:
            dfs(nid)
        return order

    def root_id(self) -> str:
        children: set[str] = set()
        for n in self.nodes:
            children.update(n.inputs)
        for n in self.nodes:
            if n.id not in children:
                return n.id
        raise ValueError("No root node")

    def to_fedot_pipeline(self) -> Pipeline:
        nodes_by_id: Dict[str, PipelineNode] = {}
        for spec in self._topo_sorted():
            parents = [nodes_by_id[i] for i in spec.inputs] or None
            fedot_node = PipelineNode(operation_type=spec.operation, nodes_from=parents)
            if spec.params:
                try:
                    fedot_node.parameters = spec.params
                except Exception:
                    pass
            nodes_by_id[spec.id] = fedot_node
        return Pipeline(nodes_by_id[self.root_id()])

    def to_mermaid(self) -> str:
        lines = ["graph LR"]
        for n in self.nodes:
            label = n.operation
            if n.params:
                p = json.dumps(n.params)
                if len(p) > 30:
                    p = p[:27] + "..."
                label = f"{n.operation}<br/>{p}"
            lines.append(f'    {n.id}["{label}"]')
        for n in self.nodes:
            for inp in n.inputs:
                lines.append(f"    {inp} --> {n.id}")
        return "\n".join(lines)

    def apply_mutation(self, mutation: Dict[str, Any]) -> "PipelineGraph":
        """Return a new graph with the mutation applied. Strategy mutations are
        handled at the DataContext level by the orchestrator, not here."""
        nodes = [GraphNode(**asdict(n)) for n in self.nodes]
        op = mutation.get("type")

        if op == "add":
            spec = mutation.get("node", {})
            nodes.append(GraphNode(
                id=spec["id"],
                operation=spec["operation"],
                params=spec.get("params", {}),
                inputs=spec.get("inputs", []),
            ))
            # Optionally re-route an existing node to consume the new node
            after = mutation.get("rewire_input_of")
            if after:
                for n in nodes:
                    if n.id == after:
                        n.inputs = [spec["id"]]

        elif op == "remove":
            nid = mutation["node_id"]
            nodes = [n for n in nodes if n.id != nid]
            for n in nodes:
                n.inputs = [i for i in n.inputs if i != nid]

        elif op == "replace":
            nid = mutation["node_id"]
            new_op = mutation["new_operation"]
            for n in nodes:
                if n.id == nid:
                    old_op = n.operation
                    n.operation = new_op
                    if "params" in mutation:
                        n.params = mutation["params"]
                    elif old_op != new_op:
                        n.params = {}

        elif op == "connect":
            nid = mutation["node_id"]
            new_input = mutation["input_id"]
            for n in nodes:
                if n.id == nid and new_input not in n.inputs:
                    n.inputs.append(new_input)

        elif op in ("set_strategy", "clear_strategy"):
            # Strategy mutations are no longer part of the graph; the orchestrator
            # applies them to DataContext.industrial_strategy. Return graph unchanged.
            return PipelineGraph(task_type=self.task_type, nodes=nodes)

        else:
            raise ValueError(f"Unknown mutation type: {op}")

        return PipelineGraph(task_type=self.task_type, nodes=nodes)


# ============== Data loading ==============

def load_input_data(
    csv_path: str,
    target: str,
    task_type: str,
    forecast_length: Optional[int] = None,
) -> InputData:
    """Load CSV into a Fedot InputData object."""
    csv_path = normalize_csv_path(csv_path)
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV")

    if task_type == "ts_forecasting":
        series = df[target].astype(float).values
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=forecast_length or 14))
        return InputData(
            idx=np.arange(len(series)),
            features=series,
            target=series,
            task=task,
            data_type=DataTypesEnum.ts,
        )

    y = df[target].values
    numeric_df = df.drop(columns=[target]).select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError(
            "No numeric feature columns found after removing the target column. "
            "Encode categorical features or choose another target."
        )
    X = numeric_df.values.astype(float)

    if task_type in ("classification", "ts_classification"):
        task = Task(TaskTypesEnum.classification)
    else:
        if y.dtype == object:
            try:
                y = y.astype(float)
            except ValueError as exc:
                raise ValueError(
                    "Regression target contains non-numeric values. "
                    "Choose classification or convert the target to numbers."
                ) from exc
        task = Task(TaskTypesEnum.regression)

    if task_type in ("ts_classification", "ts_regression"):
        return InputData(
            idx=np.arange(len(X)),
            # Fedot.Industrial channel-independent TS operations expect
            # channel-first tensors; one CSV row is one fixed-window series.
            features=X[np.newaxis, :, :],
            target=y,
            task=task,
            data_type=DataTypesEnum.ts,
        )

    return InputData(
        idx=np.arange(len(X)),
        features=X,
        target=y,
        task=task,
        data_type=DataTypesEnum.table,
    )


def input_sample_count(input_data: InputData) -> int:
    """Return the supervised sample count for table and channel-first TS data."""
    features = np.asarray(input_data.features)
    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        return len(features)
    if input_data.data_type == DataTypesEnum.ts and features.ndim == 3:
        return features.shape[1]
    return len(features)


def slice_input_data(input_data: InputData, sample_indices) -> InputData:
    """Slice InputData by supervised sample indices, preserving TS channel axis."""
    indices = np.asarray(sample_indices)
    features = np.asarray(input_data.features)
    if input_data.data_type == DataTypesEnum.ts and features.ndim == 3 and input_data.task.task_type != TaskTypesEnum.ts_forecasting:
        sliced_features = features[:, indices, :]
    else:
        sliced_features = features[indices]
    return InputData(
        idx=np.arange(len(indices)),
        features=sliced_features,
        target=np.asarray(input_data.target)[indices],
        task=input_data.task,
        data_type=input_data.data_type,
        supplementary_data=input_data.supplementary_data,
    )


def _basic_split_input_data(input_data: InputData, test_size: float = 0.2) -> Tuple[InputData, InputData]:
    """Local fallback train/val split for shapes Fedot.Industrial splitters do not cover."""
    n = input_sample_count(input_data)
    if n < 2:
        raise ValueError("At least two samples are required for a train/test split.")
    cut = int(n * (1 - test_size))
    cut = min(max(cut, 1), n - 1)

    if input_data.task.task_type == TaskTypesEnum.ts_forecasting:
        train = InputData(
            idx=input_data.idx[:cut], features=input_data.features[:cut],
            target=input_data.target[:cut], task=input_data.task, data_type=input_data.data_type,
        )
        val = InputData(
            idx=input_data.idx[cut:], features=input_data.features[cut:],
            target=input_data.target[cut:], task=input_data.task, data_type=input_data.data_type,
        )
        return train, val

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    train_idx, val_idx = perm[:cut], perm[cut:]
    return slice_input_data(input_data, train_idx), slice_input_data(input_data, val_idx)


def split_input_data(input_data: InputData, test_size: float = 0.2) -> Tuple[InputData, InputData]:
    """Train/val split for InputData, using Fedot.Industrial split hooks where safe."""
    test_size = min(max(float(test_size) if test_size is not None else 0.2, 0.05), 0.5)
    if input_data.data_type == DataTypesEnum.table:
        try:
            from fedot_ind.core.repository.industrial_implementations.abstract import split_any_industrial

            return split_any_industrial(
                input_data,
                split_ratio=1 - test_size,
                shuffle=True,
                stratify=input_data.task.task_type == TaskTypesEnum.classification,
                random_seed=42,
            )
        except Exception as exc:
            logger.warning("Fedot.Industrial split failed; using local split: %s", exc)
    return _basic_split_input_data(input_data, test_size=test_size)


# ============== Metrics ==============

def compute_metrics(task_type: str, y_true, y_pred, primary_metric: Optional[str] = None) -> Dict[str, Any]:
    """Compute metrics for the task.

    ``primary_score`` is always higher-is-better. For regression losses such as
    RMSE/MAE/MSE, the raw metric is reported unchanged and primary_score is the
    negative loss so ranking still works consistently.
    """
    from sklearn import metrics as skm
    from sklearn.preprocessing import LabelEncoder

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    out: Dict[str, Any] = {}
    try:
        if task_type in ("classification", "ts_classification"):
            out["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
            out["f1"] = float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
            out["precision"] = float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0))
            try:
                encoder = LabelEncoder()
                combined = np.concatenate([y_true.astype(str), y_pred.astype(str)])
                encoder.fit(combined)
                encoded_true = encoder.transform(y_true.astype(str))
                encoded_pred = encoder.transform(y_pred.astype(str))
                out["roc_auc"] = float(skm.roc_auc_score(encoded_true, encoded_pred))
            except Exception:
                pass
            selected = primary_metric if primary_metric in out else ("roc_auc" if "roc_auc" in out else "accuracy")
            out["primary_score"] = out[selected]
        else:
            out["r2"] = float(skm.r2_score(y_true, y_pred))
            out["mse"] = float(skm.mean_squared_error(y_true, y_pred))
            out["rmse"] = float(np.sqrt(out["mse"]))
            out["mae"] = float(skm.mean_absolute_error(y_true, y_pred))
            denom = np.where(np.abs(y_true) < 1e-12, np.nan, np.abs(y_true))
            mape = np.nanmean(np.abs((y_true - y_pred) / denom)) * 100
            if np.isfinite(mape):
                out["mape"] = float(mape)
            smape_denom = np.abs(y_true) + np.abs(y_pred)
            smape_values = np.where(smape_denom < 1e-12, np.nan, 2 * np.abs(y_pred - y_true) / smape_denom)
            smape = np.nanmean(smape_values) * 100
            if np.isfinite(smape):
                out["smape"] = float(smape)
            selected = primary_metric if primary_metric in out else "r2"
            out["primary_score"] = -out[selected] if selected in {"rmse", "mse", "mae", "mape", "smape"} else out[selected]
        out["primary_metric"] = selected
        out["primary_metric_value"] = out[selected]
        out["primary_score_direction"] = "higher_is_better"
    except Exception as e:
        out["primary_score"] = 0.0
        out["primary_metric"] = primary_metric or ""
        out["primary_metric_value"] = 0.0
        out["primary_score_direction"] = "higher_is_better"
        out["error"] = str(e)
    return out
