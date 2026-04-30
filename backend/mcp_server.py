"""
MCP Server for Industrial Learning Agent.

Single server exposing ALL tools: data profiling, sklearn baselines,
Fedot.Industrial (via proxy to fedot-server), analysis, config, reporting.

Run with: python mcp_server.py (stdio transport)
"""

import sys
import os

# Ensure backend directory is on path (for subprocess execution)
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error,
)
from sklearn.model_selection import train_test_split

from mcp.server.fastmcp import FastMCP

from data_profiler import DataProfiler
from agents.schemas import (
    AVAILABLE_OPERATIONS, INDUSTRIAL_STRATEGIES, METRICS_BY_TASK,
    ALL_PROBLEM_TYPES, TS_PROBLEM_TYPES, is_ts_task,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP-Server")

FEDOT_URL = os.environ.get("FEDOT_URL", os.environ.get("FEDOT_HOST", "http://fedot-server:8000"))

mcp = FastMCP("IndustrialLearningTools")


# ============== SKLEARN PIPELINES REGISTRY ==============

def _get_pipelines(task_type: str) -> dict:
    if task_type == "regression":
        return {
            "baseline_lr": ("LinearRegression", Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])),
            "ridge": ("Ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
            "svr": ("SVR", Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))])),
            "xgb_reg": ("XGBRegressor", Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(n_estimators=100, random_state=42))])),
        }
    elif task_type == "classification":
        return {
            "baseline_lr": ("LogisticRegression", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])),
            "svc": ("SVC", Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True))])),
            "xgb_clf": ("XGBClassifier", Pipeline([("scaler", StandardScaler()), ("model", XGBClassifier(n_estimators=100, random_state=42))])),
            "rf": ("RandomForest", Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))])),
        }
    return {}


# ================================================================
#                     DATA PROFILING TOOLS
# ================================================================

@mcp.tool()
def get_data_profile(csv_path: str, target_column: str, task_type: str = "classification") -> str:
    """Profile a CSV dataset: statistics, issues, recommendations. Returns JSON with n_samples, n_features, issues, recommendations."""
    try:
        df = pd.read_csv(csv_path)
        y = df[target_column] if target_column in df.columns else None
        X = df.drop(columns=[target_column]) if target_column in df.columns else df
        profile = DataProfiler.profile(X=X, y=y, task_type=task_type)
        profile["is_time_series"] = is_ts_task(task_type)
        return json.dumps(profile, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================
#                     BASELINE TOOLS
# ================================================================

@mcp.tool()
def get_available_baselines(task_type: str) -> str:
    """Get available sklearn baseline pipelines for classification or regression. Returns JSON with pipeline names and model types. Not available for time series tasks."""
    if is_ts_task(task_type):
        return json.dumps({"note": f"Sklearn baselines not available for '{task_type}'. Use Fedot.Industrial via call_fedot.", "pipelines": {}})

    pipelines = _get_pipelines(task_type)
    info = {name: {"model": model_name, "steps": ["StandardScaler", model_name]} for name, (model_name, _) in pipelines.items()}
    return json.dumps({"task_type": task_type, "pipelines": info})


@mcp.tool()
def train_baseline(
    csv_path: str,
    target_column: str,
    pipeline_name: str,
    task_type: str = "classification",
    test_size: float = 0.2,
) -> str:
    """Train a sklearn baseline pipeline by name. Returns JSON with score and metrics (roc_auc/f1 for classification, r2/mse for regression)."""
    try:
        if is_ts_task(task_type):
            return json.dumps({"error": f"Baselines not available for '{task_type}'", "score": 0})

        pipelines = _get_pipelines(task_type)
        if pipeline_name not in pipelines:
            return json.dumps({"error": f"Unknown pipeline '{pipeline_name}'. Available: {list(pipelines.keys())}", "score": 0})

        df = pd.read_csv(csv_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Encode if needed
        if task_type == "classification" and y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        _, pipe = pipelines[pipeline_name]
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        metrics = {}
        if task_type == "classification":
            y_proba = y_pred
            if hasattr(pipe, "predict_proba"):
                try:
                    y_proba = pipe.predict_proba(X_val)[:, 1]
                except Exception:
                    pass
            score = float(roc_auc_score(y_val, y_proba))
            metrics = {
                "roc_auc": score,
                "precision": float(precision_score(y_val, y_pred, zero_division=0)),
                "recall": float(recall_score(y_val, y_pred, zero_division=0)),
                "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            }
        else:
            score = float(r2_score(y_val, y_pred))
            mse = float(mean_squared_error(y_val, y_pred))
            metrics = {"r2": score, "mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y_val, y_pred))}

        return json.dumps({"name": pipeline_name, "score": score, "metrics": metrics})

    except Exception as e:
        return json.dumps({"name": pipeline_name, "score": 0, "error": str(e)})


# ================================================================
#                  FEDOT.INDUSTRIAL TOOLS (proxy)
# ================================================================

@mcp.tool()
def get_available_operations(task_type: str) -> str:
    """Get available Fedot.Industrial operations (models, preprocessing, strategies, metrics) for a task type. Supports: classification, regression, ts_forecasting, ts_classification, ts_regression, anomaly_detection."""
    if task_type not in AVAILABLE_OPERATIONS:
        return json.dumps({"error": f"Unknown task type: {task_type}. Available: {ALL_PROBLEM_TYPES}"})

    ops = AVAILABLE_OPERATIONS[task_type]
    strategies_info = {}
    for s_name, s_info in INDUSTRIAL_STRATEGIES.items():
        if task_type in s_info["applicable"]:
            strategies_info[s_name] = {"description": s_info["description"], "params": s_info["params"]}

    return json.dumps({
        "task_type": task_type,
        "is_time_series": is_ts_task(task_type),
        "models": ops["models"],
        "preprocessing": ops["preprocessing"],
        "strategies": strategies_info,
        "metrics": METRICS_BY_TASK.get(task_type, []),
        "requires_forecast_length": task_type == "ts_forecasting",
    })


@mcp.tool()
def call_fedot(
    csv_path: str,
    target_column: str,
    problem: str = "classification",
    timeout: float = 5.0,
    preset: str = "auto",
    metric: Optional[str] = None,
    n_jobs: int = 2,
    strategy: Optional[str] = None,
    strategy_params: Optional[str] = None,
    task_params: Optional[str] = None,
    available_operations: Optional[str] = None,
    cv_folds: Optional[int] = None,
    pop_size: Optional[int] = None,
    max_depth: Optional[int] = None,
    with_tuning: Optional[bool] = None,
    use_meta_rules: Optional[bool] = None,
) -> str:
    """Call Fedot.Industrial for training. Supports ALL task types including time series. For ts_forecasting, task_params must be JSON with forecast_length. Returns JSON with score, metrics, config_used."""
    try:
        payload = {
            "datapath": csv_path,
            "target": target_column,
            "problem": problem,
            "timeout": timeout,
            "preset": preset,
            "n_jobs": n_jobs,
        }
        if metric:
            payload["metric"] = metric
        if strategy:
            payload["strategy"] = strategy
        if cv_folds:
            payload["cv_folds"] = cv_folds
        if pop_size:
            payload["pop_size"] = pop_size
        if max_depth:
            payload["max_depth"] = max_depth
        if with_tuning is not None:
            payload["with_tuning"] = with_tuning
        if use_meta_rules is not None:
            payload["use_meta_rules"] = use_meta_rules

        # Parse JSON string params
        if task_params:
            payload["task_params"] = json.loads(task_params) if isinstance(task_params, str) else task_params
        if strategy_params:
            payload["strategy_params"] = json.loads(strategy_params) if isinstance(strategy_params, str) else strategy_params
        if available_operations:
            payload["available_operations"] = json.loads(available_operations) if isinstance(available_operations, str) else available_operations

        logger.info(f"Calling Fedot: {FEDOT_URL}/mcp with {payload}")
        resp = requests.post(f"{FEDOT_URL}/mcp", json=payload, timeout=600)

        if resp.status_code == 200:
            result = resp.json()
            return json.dumps({
                "score": result.get("score", 0),
                "metrics": result.get("metrics", {}),
                "config_used": result.get("config_used", payload),
                "shape": result.get("shape"),
                "status": "success",
            })
        else:
            return json.dumps({"score": 0, "status": "error", "error": f"HTTP {resp.status_code}: {resp.text[:300]}"})

    except Exception as e:
        logger.error(f"call_fedot error: {e}")
        return json.dumps({"score": 0, "status": "error", "error": str(e)})


@mcp.tool()
def explain_model(method: str = "point", window: int = 5, samples: int = 1) -> str:
    """Explain the last trained Fedot.Industrial model. Methods: point, recurrence, shap, lime. Requires a prior call_fedot."""
    try:
        resp = requests.post(f"{FEDOT_URL}/explain", json={"method": method, "window": window, "samples": samples}, timeout=120)
        if resp.status_code == 200:
            return json.dumps(resp.json())
        else:
            return json.dumps({"error": f"HTTP {resp.status_code}", "status": "error"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


@mcp.tool()
def get_additional_metrics(metric_names: str) -> str:
    """Calculate additional metrics on last trained Fedot model. metric_names is comma-separated (e.g. 'rmse,mae,mape')."""
    try:
        names = [m.strip() for m in metric_names.split(",")]
        resp = requests.post(f"{FEDOT_URL}/get_metrics", json={"metric_names": names}, timeout=60)
        if resp.status_code == 200:
            return json.dumps(resp.json())
        else:
            return json.dumps({"error": f"HTTP {resp.status_code}", "status": "error"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "error"})


# ================================================================
#                     ANALYSIS TOOLS
# ================================================================

@mcp.tool()
def analyze_errors(baseline_results_json: str, fedot_score: float, task_type: str) -> str:
    """Analyze and compare model results. baseline_results_json is a JSON array of {name, score, metrics}. Returns comparison statistics."""
    try:
        baselines = json.loads(baseline_results_json) if isinstance(baseline_results_json, str) else baseline_results_json
        scores = [b["score"] for b in baselines if b.get("score", 0) > 0]
        failed = [b["name"] for b in baselines if b.get("error")]

        analysis = {
            "task_type": task_type,
            "is_ts": is_ts_task(task_type),
            "fedot_score": fedot_score,
        }

        if scores:
            best_bl = max(baselines, key=lambda b: b.get("score", 0))
            analysis["baseline_stats"] = {
                "mean": round(float(np.mean(scores)), 4),
                "std": round(float(np.std(scores)), 4),
                "min": round(float(min(scores)), 4),
                "max": round(float(max(scores)), 4),
                "count": len(scores),
            }
            analysis["best_baseline"] = {"name": best_bl["name"], "score": best_bl["score"], "metrics": best_bl.get("metrics", {})}

        if failed:
            analysis["failed_pipelines"] = failed

        return json.dumps(analysis)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_feature_importance(csv_path: str, target_column: str, model_name: str = "rf", task_type: str = "classification") -> str:
    """Get feature importance from a tree-based sklearn model. Only for classification/regression."""
    try:
        if is_ts_task(task_type):
            return json.dumps({"note": "Feature importance not available for TS. Use explain_model.", "available": False})

        df = pd.read_csv(csv_path)
        y = df[target_column]
        X = df.drop(columns=[target_column])

        if task_type == "classification" and y.dtype == "object":
            from sklearn.preprocessing import LabelEncoder
            y = pd.Series(LabelEncoder().fit_transform(y))

        pipelines = _get_pipelines(task_type)
        if model_name not in pipelines:
            return json.dumps({"error": f"Model '{model_name}' not found", "available": False})

        _, pipe = pipelines[model_name]
        pipe.fit(X, y)
        model = pipe.named_steps.get("model")

        if not hasattr(model, "feature_importances_"):
            return json.dumps({"note": f"Model '{model_name}' has no feature_importances_", "available": False})

        importances = model.feature_importances_
        imp_dict = {col: round(float(imp), 4) for col, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)[:10]}
        return json.dumps({"available": True, "model": model_name, "features": imp_dict})

    except Exception as e:
        return json.dumps({"error": str(e), "available": False})


@mcp.tool()
def compare_results(results_json: str) -> str:
    """Compare all model results (baselines + Fedot). results_json is JSON array of {name, score, type, metrics}. Returns best model, comparison text."""
    try:
        results = json.loads(results_json) if isinstance(results_json, str) else results_json
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        best = results[0] if results else {"name": "none", "score": 0}
        baselines = [r for r in results if r.get("type") == "baseline"]
        best_bl = max(baselines, key=lambda x: x.get("score", 0), default={"name": "none", "score": 0})
        fedot = next((r for r in results if r.get("type") == "automl"), {"name": "fedot", "score": 0})

        return json.dumps({
            "all_models": results,
            "best_overall": best["name"],
            "best_overall_score": best.get("score", 0),
            "best_baseline": best_bl["name"],
            "best_baseline_score": best_bl.get("score", 0),
            "fedot_score": fedot.get("score", 0),
            "architect_wins": best_bl.get("score", 0) > fedot.get("score", 0),
            "comparison_text": f"Best baseline: {best_bl['name']} ({best_bl.get('score', 0):.4f}) vs Fedot ({fedot.get('score', 0):.4f})",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================
#                     CONFIG TOOLS
# ================================================================

@mcp.tool()
def propose_fedot_config(
    task_type: str,
    preset: str = "auto",
    timeout: float = 5.0,
    metric: Optional[str] = None,
    n_jobs: int = 2,
    strategy: Optional[str] = None,
    task_params: Optional[str] = None,
    available_operations: Optional[str] = None,
    cv_folds: Optional[int] = None,
    pop_size: Optional[int] = None,
    max_depth: Optional[int] = None,
    with_tuning: Optional[bool] = None,
    use_meta_rules: Optional[bool] = None,
) -> str:
    """Propose a validated Fedot.Industrial configuration. For ts_forecasting, task_params MUST be JSON with forecast_length (e.g. '{"forecast_length": 14}')."""
    config = {"problem": task_type, "preset": preset, "timeout": timeout, "n_jobs": n_jobs}

    if metric:
        config["metric"] = metric
    if strategy:
        config["strategy"] = strategy
    if cv_folds:
        config["cv_folds"] = cv_folds
    if pop_size:
        config["pop_size"] = pop_size
    if max_depth:
        config["max_depth"] = max_depth
    if with_tuning is not None:
        config["with_tuning"] = with_tuning
    if use_meta_rules is not None:
        config["use_meta_rules"] = use_meta_rules

    if task_params:
        config["task_params"] = json.loads(task_params) if isinstance(task_params, str) else task_params
    if available_operations:
        config["available_operations"] = json.loads(available_operations) if isinstance(available_operations, str) else available_operations

    # Validate
    if task_type == "ts_forecasting" and not config.get("task_params", {}).get("forecast_length"):
        return json.dumps({"error": "ts_forecasting requires task_params with forecast_length", "status": "invalid"})

    return json.dumps({"fedot_config": config, "status": "proposed"})


@mcp.tool()
def mutate_fedot_config(current_config_json: str, changes_json: str) -> str:
    """Apply changes to an existing Fedot config. Both arguments are JSON strings."""
    try:
        config = json.loads(current_config_json) if isinstance(current_config_json, str) else current_config_json
        changes = json.loads(changes_json) if isinstance(changes_json, str) else changes_json

        applied = {}
        for key, value in changes.items():
            config[key] = value
            applied[key] = value

        return json.dumps({"applied_changes": applied, "new_config": config})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================
#                     REPORT TOOLS
# ================================================================

@mcp.tool()
def generate_report(iterations_json: str) -> str:
    """Compile iteration results into a structured report. iterations_json is a JSON array of iteration data."""
    try:
        iterations = json.loads(iterations_json) if isinstance(iterations_json, str) else iterations_json
        n = len(iterations)
        best_score = 0
        best_model = "N/A"
        summaries = []

        for it in iterations:
            eng = it.get("engineer", {})
            score = eng.get("best_score", 0)
            model = eng.get("best_model", "N/A")
            fedot_score = eng.get("fedot_result", {}).get("score", 0)
            summaries.append({
                "iteration": it.get("iteration", "?"),
                "best_score": score,
                "best_model": model,
                "fedot_score": fedot_score,
            })
            if score > best_score:
                best_score = score
                best_model = model

        return json.dumps({
            "n_iterations": n,
            "iteration_summaries": summaries,
            "best_overall_score": best_score,
            "best_overall_model": best_model,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")
