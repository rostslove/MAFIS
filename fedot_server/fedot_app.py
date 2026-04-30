from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional, Any, Dict, List
import pandas as pd
import numpy as np
import logging
from fedot_ind.api.main import FedotIndustrial
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import threading
import uuid

app = FastAPI(title="Fedot.Industrial API", version="3.0.0")

# Thread-safe model storage: {session_id: {"model": FedotIndustrial, "data": {...}}}
_model_lock = threading.Lock()
_models: Dict[str, Dict[str, Any]] = {}
_latest_session: Optional[str] = None


def _store_model(model, data: dict) -> str:
    """Store a trained model and return session ID."""
    global _latest_session
    session_id = str(uuid.uuid4())[:8]
    with _model_lock:
        # Keep max 5 models to avoid memory leak
        if len(_models) >= 5:
            oldest = next(iter(_models))
            del _models[oldest]
        _models[session_id] = {"model": model, "data": data}
        _latest_session = session_id
    return session_id


def _get_model(session_id: Optional[str] = None):
    """Get a stored model by session ID (or latest)."""
    with _model_lock:
        sid = session_id or _latest_session
        if sid and sid in _models:
            return _models[sid]["model"], _models[sid]["data"]
    return None, {}


# ============== Request/Response Models ==============

class TrainPredictParams(BaseModel):
    datapath: str
    target: str
    problem: str = "classification"
    # Core params
    timeout: Optional[float] = 5.0
    n_jobs: Optional[int] = 2
    logging_level: Optional[int] = 20
    preset: Optional[str] = "auto"
    metric: Optional[str] = None
    cv_folds: Optional[int] = None
    seed: Optional[int] = None
    # Operations
    available_operations: Optional[List[str]] = None
    use_meta_rules: Optional[bool] = None
    backend_method: Optional[str] = None
    # Strategy
    strategy: Optional[str] = None
    strategy_params: Optional[Dict[str, Any]] = None
    # Time series specific
    task_params: Optional[Dict[str, Any]] = None  # {"forecast_length": 14}
    pop_size: Optional[int] = None
    max_depth: Optional[int] = None
    with_tuning: Optional[bool] = None


class TrainPredictResponse(BaseModel):
    score: float
    predictions: list
    shape: tuple
    problem: str
    config_used: Dict[str, Any] = {}
    metrics: Dict[str, float] = {}
    status: str = "success"
    error: Optional[str] = None


class ExplainParams(BaseModel):
    method: str = "point"  # point, recurrence, shap, lime
    metric: Optional[str] = None
    window: int = 5
    samples: int = 1
    threshold: int = 90


class ExplainResponse(BaseModel):
    explanation: Dict[str, Any] = {}
    method: str = ""
    status: str = "success"
    error: Optional[str] = None


class MetricsParams(BaseModel):
    metric_names: List[str] = ["f1", "accuracy"]


class MetricsResponse(BaseModel):
    metrics: Dict[str, float] = {}
    status: str = "success"
    error: Optional[str] = None


class AvailableOpsResponse(BaseModel):
    problem: str
    models: List[str]
    preprocessing: List[str]
    strategies: List[str]
    metrics: List[str]


class ParamInfo(BaseModel):
    name: str
    type: str
    default: Any
    description: str
    options: Optional[List[str]] = None


# ============== Operations/Strategy Registry ==============

OPERATIONS_BY_TASK = {
    "ts_forecasting": {
        "models": ["ar", "stl_arima", "ets", "eigen_forecaster", "topo_forecaster",
                    "lagged_forecaster", "deepar_model", "tcn_model", "glm", "nbeats_model"],
        "preprocessing": ["smoothing", "gaussian_filter", "exog_ts",
                          "wavelet_basis", "fourier_basis", "eigen_basis"],
    },
    "ts_classification": {
        "models": ["industrial_stat_clf", "industrial_freq_clf", "industrial_manifold_clf",
                    "inception_model", "resnet_model", "pdl_clf"],
        "preprocessing": ["quantile_extractor", "topological_extractor", "recurrence_extractor",
                          "riemann_extractor", "minirocket_extractor",
                          "wavelet_basis", "fourier_basis", "eigen_basis",
                          "channel_filtration", "bagging", "scaling", "normalization"],
    },
    "ts_regression": {
        "models": ["industrial_stat_reg", "industrial_freq_reg", "industrial_manifold_reg",
                    "inception_model", "resnet_model", "nbeats_model", "tcn_model",
                    "tst_model", "deepar_model"],
        "preprocessing": ["quantile_extractor", "topological_extractor", "recurrence_extractor",
                          "riemann_extractor", "minirocket_extractor",
                          "wavelet_basis", "fourier_basis", "eigen_basis",
                          "channel_filtration", "bagging", "scaling", "normalization"],
    },
    "anomaly_detection": {
        "models": ["one_class_svm", "sst", "unscented_kalman_filter", "stat_detector",
                    "arima_detector", "iforest_detector", "conv_ae_detector", "lstm_ae_detector"],
        "preprocessing": ["channel_filtration", "gaussian_filter", "smoothing"],
    },
    "classification": {
        "models": ["industrial_stat_clf", "industrial_freq_clf", "industrial_manifold_clf",
                    "inception_model", "resnet_model"],
        "preprocessing": ["quantile_extractor", "topological_extractor", "recurrence_extractor",
                          "riemann_extractor", "minirocket_extractor",
                          "wavelet_basis", "fourier_basis", "eigen_basis",
                          "channel_filtration", "bagging", "scaling", "normalization"],
    },
    "regression": {
        "models": ["industrial_stat_reg", "industrial_freq_reg", "industrial_manifold_reg",
                    "inception_model", "resnet_model", "nbeats_model", "tcn_model"],
        "preprocessing": ["quantile_extractor", "topological_extractor", "recurrence_extractor",
                          "riemann_extractor", "minirocket_extractor",
                          "wavelet_basis", "fourier_basis", "eigen_basis",
                          "scaling", "normalization"],
    },
}

STRATEGIES_BY_TASK = {
    "ts_forecasting": ["forecasting_assumptions", "forecasting_exogenous",
                       "federated_automl", "kernel_automl"],
    "ts_classification": ["federated_automl", "kernel_automl", "lora_strategy", "sampling_strategy"],
    "ts_regression": ["federated_automl", "kernel_automl", "lora_strategy", "sampling_strategy"],
    "anomaly_detection": [],
    "classification": ["federated_automl", "kernel_automl", "lora_strategy", "sampling_strategy"],
    "regression": ["federated_automl", "kernel_automl", "lora_strategy", "sampling_strategy"],
}

METRICS_BY_TASK = {
    "classification": ["f1", "accuracy", "precision", "roc_auc", "logloss"],
    "ts_classification": ["f1", "accuracy", "precision", "roc_auc", "logloss"],
    "regression": ["r2", "rmse", "mse", "mae"],
    "ts_regression": ["r2", "rmse", "mse", "mae"],
    "ts_forecasting": ["rmse", "mse", "mae", "mape", "smape", "r2"],
    "anomaly_detection": ["f1", "accuracy", "precision", "roc_auc"],
}


# ============== Endpoints ==============

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "fedot-industrial", "version": "3.0.0"}


@app.get("/available_operations/{problem}", response_model=AvailableOpsResponse)
async def get_available_operations(problem: str):
    """Return available operations, strategies, and metrics for a given problem type."""
    if problem not in OPERATIONS_BY_TASK:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown problem type: {problem}. Available: {list(OPERATIONS_BY_TASK.keys())}"
        )
    ops = OPERATIONS_BY_TASK[problem]
    return AvailableOpsResponse(
        problem=problem,
        models=ops["models"],
        preprocessing=ops["preprocessing"],
        strategies=STRATEGIES_BY_TASK.get(problem, []),
        metrics=METRICS_BY_TASK.get(problem, []),
    )


@app.get("/params", response_model=List[ParamInfo])
async def get_params():
    """Return all accepted FedotIndustrial parameters."""
    return [
        ParamInfo(name="problem", type="str", default="classification",
                  description="Task type",
                  options=list(OPERATIONS_BY_TASK.keys())),
        ParamInfo(name="timeout", type="float", default=5.0,
                  description="Training timeout in minutes"),
        ParamInfo(name="n_jobs", type="int", default=2,
                  description="Parallel jobs (-1 = all CPUs)"),
        ParamInfo(name="metric", type="str", default=None,
                  description="Evaluation metric"),
        ParamInfo(name="preset", type="str", default="auto",
                  description="Model preset",
                  options=["auto", "best_quality", "fast_train", "stable", "light", "ultra-light"]),
        ParamInfo(name="strategy", type="str", default=None,
                  description="Industrial strategy",
                  options=["federated_automl", "kernel_automl", "forecasting_assumptions",
                           "forecasting_exogenous", "lora_strategy", "sampling_strategy"]),
        ParamInfo(name="task_params", type="dict", default=None,
                  description="Task-specific params, e.g. {'forecast_length': 14} for ts_forecasting"),
        ParamInfo(name="available_operations", type="list", default=None,
                  description="Specific ML operations to use"),
        ParamInfo(name="pop_size", type="int", default=20,
                  description="Population size for evolutionary optimization"),
        ParamInfo(name="max_depth", type="int", default=6,
                  description="Maximum pipeline depth"),
        ParamInfo(name="with_tuning", type="bool", default=True,
                  description="Enable hyperparameter tuning"),
    ]


@app.post("/mcp", response_model=TrainPredictResponse)
async def train_and_predict(params: TrainPredictParams) -> Dict[str, Any]:
    try:
        logger.info(f"Training Fedot.Industrial: problem={params.problem}")
        logger.info(f"Data: {params.datapath}, target: {params.target}")

        if not Path(params.datapath).exists():
            raise FileNotFoundError(f"File not found: {params.datapath}")

        data = pd.read_csv(params.datapath)
        logger.info(f"Data shape: {data.shape}")

        if params.target not in data.columns:
            raise ValueError(f"Target column '{params.target}' not found")

        y = data[params.target]
        X = data.drop(columns=[params.target])

        # Build FedotIndustrial config. Newer API expects nested config dict
        # with industrial_config / automl_config / learning_config / compute_config sections.
        # We build both flat and nested forms and try nested first, falling back to flat.

        industrial_section = {"problem": params.problem}
        if params.task_params:
            industrial_section["task_params"] = params.task_params
        if params.strategy:
            industrial_section["strategy"] = params.strategy
        if params.strategy_params:
            industrial_section["strategy_params"] = params.strategy_params
        if params.use_meta_rules is not None:
            industrial_section["use_meta_rules"] = params.use_meta_rules

        automl_section = {"task": params.problem}
        if params.task_params:
            automl_section["task_params"] = params.task_params
        if params.available_operations:
            automl_section["available_operations"] = params.available_operations

        learning_params = {}
        if params.timeout is not None:
            learning_params["timeout"] = params.timeout
        if params.n_jobs is not None:
            learning_params["n_jobs"] = params.n_jobs
        if params.with_tuning is not None:
            learning_params["with_tuning"] = params.with_tuning
        if params.pop_size:
            learning_params["pop_size"] = params.pop_size
        if params.max_depth:
            learning_params["max_depth"] = params.max_depth
        if params.cv_folds:
            learning_params["cv_folds"] = params.cv_folds
        if params.seed is not None:
            learning_params["seed"] = params.seed

        learning_section = {"learning_strategy": "from_scratch"}
        if learning_params:
            learning_section["learning_strategy_params"] = learning_params
        if params.metric:
            learning_section["optimisation_loss"] = {"quality_loss": params.metric}

        compute_section = {"backend": params.backend_method or "cpu"}

        nested_config = {
            "industrial_config": industrial_section,
            "automl_config": automl_section,
            "learning_config": learning_section,
            "compute_config": compute_section,
        }

        # Also build flat kwargs for fallback
        flat_kwargs = {"problem": params.problem}
        for k, v in {
            "timeout": params.timeout,
            "n_jobs": params.n_jobs,
            "logging_level": params.logging_level,
            "preset": params.preset,
            "metric": params.metric,
            "cv_folds": params.cv_folds,
            "seed": params.seed,
            "use_meta_rules": params.use_meta_rules,
            "available_operations": params.available_operations,
            "backend_method": params.backend_method,
            "strategy": params.strategy,
            "strategy_params": params.strategy_params,
            "task_params": params.task_params,
            "pop_size": params.pop_size,
            "max_depth": params.max_depth,
            "with_tuning": params.with_tuning,
        }.items():
            if v is not None:
                flat_kwargs[k] = v

        logger.info(f"Trying nested config: {nested_config}")
        try:
            industrial = FedotIndustrial(**nested_config)
            fedot_kwargs = nested_config
        except TypeError as e:
            logger.warning(f"Nested config failed ({e}), falling back to flat kwargs")
            logger.info(f"Flat kwargs: {flat_kwargs}")
            industrial = FedotIndustrial(**flat_kwargs)
            fedot_kwargs = flat_kwargs

        logger.info("Fitting model...")
        industrial.fit(input_data=(X, y))

        logger.info("Predicting...")
        preds = industrial.predict(X)

        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = _calculate_metrics(params.problem, y, preds, industrial, X)

        score = metrics.get("primary_score", 0)
        logger.info(f"Score: {score:.4f}, all metrics: {metrics}")

        # Store for explain/get_metrics (thread-safe)
        session_id = _store_model(industrial, {"X": X, "y": y, "preds": preds, "problem": params.problem})

        return {
            "score": score,
            "predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
            "shape": data.shape,
            "problem": params.problem,
            "config_used": fedot_kwargs,
            "metrics": metrics,
            "status": "success",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/explain", response_model=ExplainResponse)
async def explain_model(params: ExplainParams):
    """Explain the last trained model using point/recurrence/shap/lime methods."""
    model, data = _get_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model trained yet. Call /mcp first.")

    try:
        explaining_config = {
            "method": params.method,
            "metric": params.metric or "rmse",
            "window": params.window,
            "samples": params.samples,
            "threshold": params.threshold,
            "name": "explain_result",
        }

        logger.info(f"Explaining model with config: {explaining_config}")
        explanation_result = model.explain(explaining_config=explaining_config)

        # Convert to serializable format
        explanation = {}
        if explanation_result is not None:
            if isinstance(explanation_result, dict):
                explanation = {k: str(v)[:1000] for k, v in explanation_result.items()}
            elif hasattr(explanation_result, "to_dict"):
                explanation = explanation_result.to_dict()
            else:
                explanation = {"raw": str(explanation_result)[:2000]}

        return ExplainResponse(
            explanation=explanation,
            method=params.method,
            status="success",
        )

    except Exception as e:
        logger.error(f"Explain error: {e}", exc_info=True)
        return ExplainResponse(
            explanation={},
            method=params.method,
            status="error",
            error=str(e),
        )


@app.post("/get_metrics", response_model=MetricsResponse)
async def get_metrics(params: MetricsParams):
    """Calculate additional metrics on last trained model."""
    model, data = _get_model()
    if model is None:
        raise HTTPException(status_code=400, detail="No model trained yet.")

    try:
        y = data.get("y")
        preds = data.get("preds")

        if y is None or preds is None:
            raise ValueError("No predictions available")

        # Try using FedotIndustrial get_metrics
        try:
            probs = None
            if hasattr(model, "predict_proba"):
                X = data.get("X")
                probs = model.predict_proba(predict_data=(X, y))

            result = model.get_metrics(
                labels=preds,
                probs=probs,
                target=y.values if hasattr(y, "values") else y,
                metric_names=params.metric_names,
            )
            metrics = result if isinstance(result, dict) else {"result": str(result)}
        except Exception as e:
            logger.warning(f"FedotIndustrial get_metrics failed: {e}, using sklearn")
            metrics = _sklearn_metrics(
                data.get("problem", "classification"), y, preds, params.metric_names
            )

        return MetricsResponse(metrics=metrics, status="success")

    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        return MetricsResponse(metrics={}, status="error", error=str(e))


@app.post("/status")
async def get_status():
    return {
        "status": "ready",
        "service": "fedot-industrial",
        "version": "3.0.0",
        "model_loaded": _latest_session is not None,
        "active_sessions": len(_models),
        "supported_problems": list(OPERATIONS_BY_TASK.keys()),
        "endpoints": {
            "/health": "GET - Health check",
            "/mcp": "POST - Train and predict",
            "/explain": "POST - Explain model predictions",
            "/get_metrics": "POST - Calculate additional metrics",
            "/available_operations/{problem}": "GET - Available operations for task",
            "/params": "GET - Parameter schema",
        },
    }


# ============== Helpers ==============

def _calculate_metrics(problem, y, preds, industrial, X):
    """Calculate metrics based on problem type."""
    metrics = {}

    try:
        if problem in ("classification", "ts_classification", "anomaly_detection"):
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score as sk_f1

            try:
                if hasattr(industrial, "predict_proba"):
                    preds_proba = industrial.predict_proba(features=X)
                    if preds_proba.ndim > 1:
                        preds_proba = preds_proba[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y, preds_proba))
            except Exception:
                pass

            metrics["accuracy"] = float(accuracy_score(y, preds))
            metrics["f1"] = float(sk_f1(y, preds, zero_division=0))
            metrics["primary_score"] = metrics.get("roc_auc", metrics["accuracy"])

        else:  # regression, ts_regression, ts_forecasting
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

            y_arr = np.array(y).flatten()
            p_arr = np.array(preds).flatten()
            min_len = min(len(y_arr), len(p_arr))
            y_arr, p_arr = y_arr[:min_len], p_arr[:min_len]

            metrics["r2"] = float(r2_score(y_arr, p_arr))
            metrics["mse"] = float(mean_squared_error(y_arr, p_arr))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_arr, p_arr))
            metrics["primary_score"] = metrics["r2"]

    except Exception as e:
        logger.warning(f"Metric calculation error: {e}")
        metrics["primary_score"] = 0
        metrics["error"] = str(e)

    return metrics


def _sklearn_metrics(problem, y, preds, metric_names):
    """Fallback sklearn metrics."""
    from sklearn import metrics as sk_metrics
    result = {}
    y_arr = np.array(y).flatten()
    p_arr = np.array(preds).flatten()
    min_len = min(len(y_arr), len(p_arr))
    y_arr, p_arr = y_arr[:min_len], p_arr[:min_len]

    metric_map = {
        "f1": lambda: float(sk_metrics.f1_score(y_arr, p_arr, zero_division=0)),
        "accuracy": lambda: float(sk_metrics.accuracy_score(y_arr, p_arr)),
        "precision": lambda: float(sk_metrics.precision_score(y_arr, p_arr, zero_division=0)),
        "r2": lambda: float(sk_metrics.r2_score(y_arr, p_arr)),
        "mse": lambda: float(sk_metrics.mean_squared_error(y_arr, p_arr)),
        "rmse": lambda: float(np.sqrt(sk_metrics.mean_squared_error(y_arr, p_arr))),
        "mae": lambda: float(sk_metrics.mean_absolute_error(y_arr, p_arr)),
    }

    for name in metric_names:
        if name in metric_map:
            try:
                result[name] = metric_map[name]()
            except Exception as e:
                result[name] = None
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
