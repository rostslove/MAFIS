"""
MCP server hosting all tools for the MAFIS multi-agent system.

Tools call Fedot.Industrial directly (no HTTP proxy). The registry is loaded
in-process by mcp_client.py to avoid the Pydantic v2 dependency of the
official MCP SDK.
"""

import json
import logging
import os
import sys
from copy import deepcopy
from inspect import Parameter, signature
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

# Make backend importable when launched as subprocess
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_profiler import DataProfiler
from graph_engine import (
    OPERATIONS, METRICS_BY_TASK, SUPPORTED_TASKS, DEFAULT_GRAPHS,
    PipelineGraph, compute_metrics, get_operation_catalog, get_training_strategies,
    get_training_strategy_hints, is_ts_task,
    input_sample_count, load_input_data, slice_input_data, split_input_data,
)
from path_utils import normalize_csv_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP")


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Union:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _json_type(non_none[0]) if non_none else "string"
    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation in (str, Parameter.empty):
        return "string"
    return "object"


class LocalMCPRegistry:
    """Small MCP-like tool registry without the external MCP SDK dependency."""

    def __init__(self, name: str):
        self.name = name
        self._tools: Dict[str, Any] = {}

    def tool(self):
        def decorator(func):
            self._tools[func.__name__] = func
            return func
        return decorator

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name](**arguments)

    def list_tools(self) -> List[Dict[str, Any]]:
        tools = []
        for name, func in self._tools.items():
            sig = signature(func)
            properties = {}
            required = []
            for param_name, param in sig.parameters.items():
                properties[param_name] = {"type": _json_type(param.annotation)}
                if param.default is Parameter.empty:
                    required.append(param_name)

            tools.append({
                "name": name,
                "description": (func.__doc__ or "").strip(),
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })
        return tools

    def run(self, transport: str = "stdio") -> None:
        raise RuntimeError(
            "This project uses the local MCP adapter in-process to avoid the "
            "Pydantic v2 dependency conflict with Fedot.Industrial 0.5.0."
        )


mcp = LocalMCPRegistry("MAFISTools")


# ============== Trained-model store (single-process) ==============

_LAST: Dict[str, Any] = {"pipeline": None, "graph": None, "input_data": None, "predictions": None}


def _store_run(pipeline, graph, input_data, predictions) -> None:
    _LAST["pipeline"] = pipeline
    _LAST["graph"] = graph
    _LAST["input_data"] = input_data
    _LAST["predictions"] = predictions


def _predict_pipeline(pipeline, data, task_type: str) -> np.ndarray:
    if task_type in ("classification", "ts_classification"):
        try:
            return np.asarray(pipeline.predict(data, output_mode="labels").predict)
        except Exception:
            pass
    return np.asarray(pipeline.predict(data).predict)


def _task_problem(task_type: str) -> str:
    if task_type in ("classification", "ts_classification"):
        return "classification"
    if task_type in ("regression", "ts_regression"):
        return "regression"
    return task_type


def _strategy_data_type(task_type: str) -> str:
    return "time_series" if task_type in ("ts_classification", "ts_regression") else "table"


def _is_table_strategy(data_type: str) -> bool:
    return str(data_type or "").lower() in {"table", "tabular"}


def _default_strategy_operations(problem: str, data_type: str) -> List[str]:
    if not _is_table_strategy(data_type):
        return []
    if problem == "classification":
        preferred = ["rf", "xgboost", "logit", "dt", "lgbm", "catboost"]
        return [operation for operation in preferred if operation in OPERATIONS.get("classification", {}).get("models", [])]
    if problem == "regression":
        preferred = ["treg", "xgbreg", "ridge", "lasso", "dtreg", "lgbmreg", "sgdr", "catboostreg"]
        return [operation for operation in preferred if operation in OPERATIONS.get("regression", {}).get("models", [])]
    return []


def _load_strategy_arrays(csv_path: str, target_column: str, task_type: str) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    csv_path = normalize_csv_path(csv_path)
    df = pd.read_csv(csv_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in CSV")

    features_df = df.drop(columns=[target_column])
    encoded_df = pd.get_dummies(features_df, dummy_na=True)
    encoded_df = encoded_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    if encoded_df.shape[1] == 0:
        raise ValueError(
            "No feature columns found after removing the target column. "
            "Choose another target or upload a dataset with feature columns."
        )

    target_info = _target_info(csv_path, target_column, task_type)
    X = encoded_df.values.astype(float)
    y = df[target_column].values

    if task_type in ("classification", "ts_classification"):
        encoder = LabelEncoder()
        y = encoder.fit_transform(pd.Series(y).astype(str).values)
        target_info["fedot_receives_raw_target"] = False
        target_info["reference_encoded"] = True
        target_info["reference_encoding"] = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
    else:
        try:
            y = y.astype(float)
        except ValueError as exc:
            raise ValueError(
                "Regression target contains non-numeric values. "
                "Choose classification or convert the target to numbers."
            ) from exc
    return X, np.asarray(y), target_info


def _split_strategy_arrays(X: np.ndarray, y: np.ndarray, task_type: str, test_size: float):
    stratify = None
    if task_type in ("classification", "ts_classification"):
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) > 1 and int(counts.min()) >= 2:
            stratify = y
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=stratify)


def _make_strategy_input(features: np.ndarray, target: np.ndarray, task_type: str):
    from fedot.core.data.data import InputData
    from fedot.core.repository.dataset_types import DataTypesEnum
    from fedot.core.repository.tasks import Task, TaskTypesEnum

    problem = _task_problem(task_type)
    fedot_task = Task(TaskTypesEnum.classification if problem == "classification" else TaskTypesEnum.regression)
    X = np.asarray(features, dtype=float)
    data_type = DataTypesEnum.table
    # RAFEnsembler always builds image-like branch data sources internally.
    # Keep tabular tasks 2D so sklearn/FEDOT table models stay cheap; use a
    # singleton channel only for fixed-window time-series strategy runs.
    if task_type in ("ts_classification", "ts_regression") and X.ndim == 2:
        X = X.reshape(X.shape[0], 1, X.shape[1])
        data_type = DataTypesEnum.image
    y = np.asarray(target)
    if problem == "classification":
        y = y.reshape(-1, 1)
    else:
        y = y.astype(float).reshape(-1)
    return InputData(
        idx=np.arange(X.shape[0]),
        features=X,
        target=y,
        task=fedot_task,
        data_type=data_type,
    )


def _predict_raf_ensemble(solver, input_data, problem: str) -> np.ndarray:
    """Predict with RAFEnsembler's decomposed branches and head model.

    The upstream RAFEnsembler.predict path calls the decomposed pipeline with a
    single default data source, while its branch preprocessors were fitted on
    data_source_img/{i}. Fedot.Industrial's own strategy code predicts branches
    first and then feeds their outputs to the head; mirror that behavior here.
    """
    branch_nodes = list(getattr(solver.current_pipeline.root_node, "nodes_from", []) or [])
    if not branch_nodes:
        return np.asarray(solver.current_pipeline.predict(input_data, "labels").predict)

    branch_outputs = [np.asarray(branch.predict(input_data).predict) for branch in branch_nodes]
    if problem == "classification":
        branch_labels = []
        for output in branch_outputs:
            if output.ndim < 2:
                output = np.array([output, 1 - output]).T
            branch_labels.append(np.argmax(output, axis=1).reshape(-1, 1))
        head_features = np.hstack(branch_labels)
        output_mode = "labels"
    else:
        head_features = np.hstack([output.reshape(-1, 1) for output in branch_outputs])
        output_mode = "default"

    head_input = deepcopy(input_data)
    n_samples, n_channels = head_features.shape
    head_input.idx = np.arange(n_samples)
    head_input.features = head_features.reshape(n_samples, n_channels, 1)

    head_model = deepcopy(solver.current_pipeline.root_node)
    head_model.nodes_from = []
    return np.asarray(head_model.predict(head_input, output_mode).predict)


def _train_strategy_graph(
    graph: PipelineGraph,
    csv_path: str,
    target_column: str,
    primary_metric: Optional[str],
    test_size: float,
) -> Dict[str, Any]:
    if not graph.training_strategy:
        raise ValueError("Graph has no training_strategy")
    if graph.training_strategy.get("name") != "federated_automl":
        raise ValueError(f"Unsupported training strategy: {graph.training_strategy.get('name')}")
    if graph.task_type == "ts_forecasting":
        raise ValueError("federated_automl is not exposed for ts_forecasting")

    from fedot_ind.core.ensemble.random_automl_forest import RAFEnsembler

    ts = min(max(float(test_size) if test_size is not None else 0.2, 0.05), 0.5)
    X, y, target_info = _load_strategy_arrays(csv_path, target_column, graph.task_type)
    X_train, X_test, y_train, y_test = _split_strategy_arrays(X, y, graph.task_type, ts)
    train = _make_strategy_input(X_train, y_train, graph.task_type)
    test = _make_strategy_input(X_test, y_test, graph.task_type)

    strategy = graph.training_strategy
    params = dict(strategy.get("params", {}) or {})
    problem = _task_problem(graph.task_type)
    data_type = params.get("data_type") or _strategy_data_type(graph.task_type)
    params.update({
        "problem": problem,
        "data_type": data_type,
        "timeout": float(params.get("timeout", 10) or 10),
        "n_jobs": int(params.get("n_jobs", 1) or 1),
    })
    if not params.get("available_operations"):
        operations = _default_strategy_operations(problem, data_type)
        if operations:
            params["available_operations"] = operations
    if primary_metric:
        params["metric"] = primary_metric

    n_train = int(train.features.shape[0])
    requested_splits = int(params.pop("n_splits", 5) or 5)
    n_splits = max(2, min(requested_splits, n_train)) if n_train >= 2 else 1
    batch_size = max(1, int(np.ceil(n_train / n_splits)))
    report_params = dict(params)

    solver = RAFEnsembler(composing_params=dict(params), n_splits=n_splits, batch_size=batch_size)
    solver.fit(train)
    train_preds = _predict_raf_ensemble(solver, train, problem).reshape(-1)
    test_preds = _predict_raf_ensemble(solver, test, problem).reshape(-1)
    train_metrics = compute_metrics(graph.task_type, y_train, train_preds, primary_metric)
    test_metrics = compute_metrics(graph.task_type, y_test, test_preds, primary_metric)

    _store_run(solver, graph, train, test_preds)
    return {
        "score": test_metrics["primary_score"],
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_info": {"test_size": ts, "n_train": len(y_train), "n_test": len(y_test)},
        "graph": graph.to_dict(),
        "training_strategy": {
            "name": strategy.get("name"),
            "params": report_params,
            "n_splits": n_splits,
            "batch_size": batch_size,
            "head": "xgboost" if problem == "classification" else "treg",
        },
        "target_info": target_info,
        "training_notes": [
            "Fedot.Industrial federated_automl strategy was used instead of direct graph execution.",
            "The displayed graph is the approved architecture contract; RAFEnsembler internally trains AutoML branches on worker folds and joins them with a head model.",
            f"Strategy split: {n_splits} branch models, batch_size={batch_size}, head={'xgboost' if problem == 'classification' else 'treg'}.",
            (
                "data_type=table constrained available_operations to tabular FEDOT models."
                if _is_table_strategy(data_type)
                else "data_type=time_series allows Fedot.Industrial TS feature extraction and heavier industrial operations."
            ),
        ],
    }


def _target_info(csv_path: str, target_column: str, task_type: str) -> Dict[str, Any]:
    csv_path = normalize_csv_path(csv_path)
    df = pd.read_csv(csv_path, usecols=[target_column])
    y = df[target_column]
    values = y.dropna()
    info: Dict[str, Any] = {
        "column": target_column,
        "raw_dtype": str(y.dtype),
        "unique_values": int(values.nunique()),
        "sample_values": [str(v) for v in values.head(10).tolist()],
        "fedot_receives_raw_target": task_type in ("classification", "ts_classification"),
        "reference_encoded": False,
    }
    if task_type in ("classification", "ts_classification") and not np.issubdtype(y.dtype, np.number):
        encoder = LabelEncoder()
        encoder.fit(values.astype(str).values)
        info["reference_encoded"] = True
        info["reference_encoding"] = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
    return info


# ================================================================
#                          DATA / OPERATIONS
# ================================================================

@mcp.tool()
def get_data_profile(csv_path: str, target_column: str, task_type: str = "classification") -> str:
    """Profile a CSV: returns JSON with n_samples, n_features, issues, recommendations."""
    try:
        csv_path = normalize_csv_path(csv_path)
        df = pd.read_csv(csv_path)
        y = df[target_column] if target_column in df.columns else None
        X = df.drop(columns=[target_column]) if target_column in df.columns else df
        profile = DataProfiler.profile(X=X, y=y, task_type=task_type)
        profile["is_time_series"] = is_ts_task(task_type)
        return json.dumps(profile, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_available_operations(task_type: str) -> str:
    """Get atomic operations available for a task type. Returns models, preprocessing, metrics."""
    if task_type not in OPERATIONS:
        return json.dumps({"error": f"Unknown task '{task_type}'. Available: {SUPPORTED_TASKS}"})
    ops = OPERATIONS[task_type]
    return json.dumps({
        "task_type": task_type,
        "is_time_series": is_ts_task(task_type),
        "preprocessing": ops["preprocessing"],
        "models": ops["models"],
        "catalog": get_operation_catalog(task_type),
        "metrics": METRICS_BY_TASK.get(task_type, []),
        "training_strategies_catalog": get_training_strategies(task_type),
        "training_strategies": get_training_strategy_hints(task_type),
        "default_graph": DEFAULT_GRAPHS.get(task_type, []),
    })


# ================================================================
#                          GRAPH OPS (no training)
# ================================================================

@mcp.tool()
def propose_graph(graph_json: str) -> str:
    """Validate and register a pipeline graph. Input is JSON: {task_type, nodes:[{id, operation, params, inputs}]}."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        return json.dumps({
            "valid": ok,
            "message": msg,
            "graph": graph.to_dict(),
            "mermaid": graph.to_mermaid() if ok else "",
        })
    except Exception as e:
        return json.dumps({"valid": False, "message": str(e), "graph": None})


@mcp.tool()
def mutate_graph(graph_json: str, mutation_json: str) -> str:
    """Apply a mutation: type=add|remove|replace|set_params|connect plus details. Returns the new graph."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        mutation = json.loads(mutation_json)
        new_graph = graph.apply_mutation(mutation)
        ok, msg = new_graph.validate()
        return json.dumps({
            "valid": ok,
            "message": msg,
            "graph": new_graph.to_dict(),
            "mermaid": new_graph.to_mermaid() if ok else "",
        })
    except Exception as e:
        return json.dumps({"valid": False, "message": str(e)})


@mcp.tool()
def visualize_graph(graph_json: str) -> str:
    """Render a graph to Mermaid markup."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        return json.dumps({"mermaid": graph.to_mermaid()})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================
#                          GRAPH TRAIN / TUNE
# ================================================================

@mcp.tool()
def train_graph(
    graph_json: str,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
    test_size: float = 0.2,
) -> str:
    """Train a pipeline graph as-is on the data. Returns score and per-task metrics on the held-out test split."""
    try:
        csv_path = normalize_csv_path(csv_path)
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps({"score": 0, "error": f"Invalid graph: {msg}"})

        ts = float(test_size) if test_size is not None else 0.2
        ts = min(max(ts, 0.05), 0.5)

        if graph.uses_training_strategy():
            return json.dumps(_train_strategy_graph(graph, csv_path, target_column, primary_metric, ts))

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data, test_size=ts)

        pipeline = graph.to_fedot_pipeline()
        pipeline.fit(train)
        train_preds = _predict_pipeline(pipeline, train, graph.task_type)
        test_preds = _predict_pipeline(pipeline, val, graph.task_type)

        train_metrics = compute_metrics(graph.task_type, train.target, train_preds, primary_metric)
        test_metrics = compute_metrics(graph.task_type, val.target, test_preds, primary_metric)
        _store_run(pipeline, graph, input_data, test_preds)

        target_info = _target_info(csv_path, target_column, graph.task_type)
        training_notes = []
        if target_info.get("fedot_receives_raw_target") and target_info.get("reference_encoded"):
            training_notes.append(
                "Fedot graph received the raw string classification target; reference mapping is shown only for readable diagnostics."
            )

        return json.dumps({
            "score": test_metrics["primary_score"],
            "metrics": test_metrics,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "split_info": {
                "test_size": ts,
                "n_train": input_sample_count(train),
                "n_test": input_sample_count(val),
            },
            "graph": graph.to_dict(),
            "n_train": input_sample_count(train),
            "n_val": input_sample_count(val),
            "target_info": target_info,
            "training_notes": training_notes,
        })
    except Exception as e:
        logger.exception("train_graph failed")
        return json.dumps({"score": 0, "error": str(e)})


@mcp.tool()
def tune_graph_hyperparameters(
    graph_json: str,
    csv_path: str,
    target_column: str,
    iterations: int = 20,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
    test_size: float = 0.2,
) -> str:
    """Tune node hyperparameters of a graph using Fedot's PipelineTuner. Returns tuned graph + score."""
    try:
        from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder

        csv_path = normalize_csv_path(csv_path)
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps({"score": 0, "error": f"Invalid graph: {msg}"})

        if graph.uses_training_strategy():
            payload = _train_strategy_graph(graph, csv_path, target_column, primary_metric, test_size)
            return json.dumps(payload)

        if graph.task_type in ("classification", "ts_classification"):
            return json.dumps({
                "score": 0,
                "error": "tuning skipped for classification",
                "target_info": _target_info(csv_path, target_column, graph.task_type),
            })

        ts = float(test_size) if test_size is not None else 0.2
        ts = min(max(ts, 0.05), 0.5)

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data, test_size=ts)

        pipeline = graph.to_fedot_pipeline()

        try:
            tuner = TunerBuilder(input_data.task).with_iterations(iterations).build(train)
            pipeline = tuner.tune(pipeline)
        except Exception as e:
            logger.warning(f"Tuning failed, using untuned pipeline: {e}")

        pipeline.fit(train)
        train_preds = _predict_pipeline(pipeline, train, graph.task_type)
        test_preds = _predict_pipeline(pipeline, val, graph.task_type)
        train_metrics = compute_metrics(graph.task_type, train.target, train_preds, primary_metric)
        test_metrics = compute_metrics(graph.task_type, val.target, test_preds, primary_metric)
        metrics = test_metrics
        _store_run(pipeline, graph, input_data, test_preds)

        # Extract tuned params back per node
        tuned_nodes = []
        try:
            for fnode, gnode in zip(pipeline.nodes, graph.nodes):
                tuned_params = dict(getattr(fnode, "parameters", {}) or {})
                if tuned_params:
                    gnode.params = {**(gnode.params or {}), **tuned_params}
                tuned_nodes.append({"id": gnode.id, "operation": gnode.operation, "tuned_params": tuned_params})
        except Exception:
            pass

        target_info = _target_info(csv_path, target_column, graph.task_type)
        training_notes = []
        if target_info.get("fedot_receives_raw_target") and target_info.get("reference_encoded"):
            training_notes.append(
                "Fedot graph received the raw string classification target; reference mapping is shown only for readable diagnostics."
            )

        return json.dumps({
            "score": metrics["primary_score"],
            "metrics": metrics,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "split_info": {
                "test_size": ts,
                "n_train": input_sample_count(train),
                "n_test": input_sample_count(val),
            },
            "graph": graph.to_dict(),
            "tuned_nodes": tuned_nodes,
            "target_info": target_info,
            "training_notes": training_notes,
        })
    except Exception as e:
        logger.exception("tune_graph_hyperparameters failed")
        return json.dumps({"score": 0, "error": str(e)})


@mcp.tool()
def validate_graph(
    graph_json: str,
    csv_path: str,
    target_column: str,
    cv_folds: int = 3,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
) -> str:
    """Cross-validate the graph. Returns mean & std of the primary metric across folds."""
    try:
        csv_path = normalize_csv_path(csv_path)
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps({"error": f"Invalid graph: {msg}"})

        if graph.uses_training_strategy():
            return json.dumps({
                "skipped": "Cross-validation for training strategies is skipped to avoid repeatedly running nested AutoML ensembles.",
                "primary_metric": primary_metric,
                "primary_score_direction": "higher_is_better",
            })

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)

        # Simple CV: rotate validation cuts
        n = input_sample_count(input_data)
        scores: List[float] = []
        for fold in range(cv_folds):
            offset = fold / cv_folds
            cut_lo = int(n * offset)
            cut_hi = cut_lo + n // cv_folds
            mask = np.ones(n, dtype=bool)
            mask[cut_lo:cut_hi] = False

            train = slice_input_data(input_data, np.flatnonzero(mask))
            val = slice_input_data(input_data, np.flatnonzero(~mask))

            pipeline = graph.to_fedot_pipeline()
            pipeline.fit(train)
            preds = _predict_pipeline(pipeline, val, graph.task_type)
            m = compute_metrics(graph.task_type, val.target, preds, primary_metric)
            scores.append(m["primary_score"])

        return json.dumps({
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "fold_scores": scores,
            "cv_folds": cv_folds,
            "primary_metric": primary_metric,
            "primary_score_direction": "higher_is_better",
        })
    except Exception as e:
        logger.exception("validate_graph failed")
        return json.dumps({"error": str(e)})


# ================================================================
#                          ANALYSIS
# ================================================================

@mcp.tool()
def get_node_importance(
    graph_json: str,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
) -> str:
    """Estimate per-node importance via leave-one-out ablation (training-only, can be slow)."""
    try:
        csv_path = normalize_csv_path(csv_path)
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        if graph.uses_training_strategy():
            return json.dumps({
                "skipped": "Node ablation is unavailable for strategy graphs.",
                "node_importance": {},
                "diagnostics": [{
                    "agent": "Critic",
                    "kind": "strategy_node_importance_skipped",
                    "summary": "Strategy internals are generated by Fedot.Industrial, so node ablation on the display graph would be misleading.",
                    "recommendations": ["Use train/test metrics and history comparison to judge the strategy run."],
                    "recoverable": True,
                }],
            })
        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data)

        # Reference score: full graph
        full_pipe = graph.to_fedot_pipeline()
        full_pipe.fit(train)
        full_score = compute_metrics(
            graph.task_type,
            val.target,
            _predict_pipeline(full_pipe, val, graph.task_type),
            primary_metric,
        )["primary_score"]

        importances: Dict[str, float] = {}
        for node in graph.nodes:
            # Skip the root model - removing it breaks the pipeline
            if node.id == graph.root_id():
                continue
            try:
                ablated = graph.apply_mutation({"type": "remove", "node_id": node.id})
                ok, _ = ablated.validate()
                if not ok:
                    continue
                pipe = ablated.to_fedot_pipeline()
                pipe.fit(train)
                score = compute_metrics(
                    graph.task_type,
                    val.target,
                    _predict_pipeline(pipe, val, graph.task_type),
                    primary_metric,
                )["primary_score"]
                importances[node.id] = round(full_score - score, 4)
            except Exception as e:
                importances[node.id] = None

        return json.dumps({"full_score": full_score, "node_importance": importances})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def explain_graph(top_k: int = 10) -> str:
    """Explain the last trained graph: feature importance from the model node if available."""
    pipeline = _LAST.get("pipeline")
    graph = _LAST.get("graph")
    if pipeline is None or graph is None:
        return json.dumps({"error": "No trained graph yet. Call train_graph or tune_graph_hyperparameters first."})

    try:
        # Find the root (model) node and try to extract feature_importances_
        out: Dict[str, Any] = {"graph": graph.to_dict()}
        try:
            root_fn = pipeline.root_node
            operation = getattr(root_fn, "operation", None)
            fitted = getattr(operation, "fitted_operation", None) if operation else None
            if fitted is not None and hasattr(fitted, "feature_importances_"):
                imp = fitted.feature_importances_
                top = sorted(enumerate(imp), key=lambda x: x[1], reverse=True)[:top_k]
                out["feature_importance"] = {f"f{i}": round(float(v), 4) for i, v in top}
        except Exception as e:
            out["feature_importance_error"] = str(e)

        out["pipeline_structure"] = graph.to_mermaid()
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================
#                          REPORT
# ================================================================

@mcp.tool()
def generate_report(evaluations_json: str) -> str:
    """Compile evaluation data into a structured report (best score, summaries, mermaid of best graph)."""
    try:
        iterations = json.loads(evaluations_json)
        best_score = -1.0
        best_iter = None
        summaries: List[Dict[str, Any]] = []

        for it in iterations:
            eng = it.get("engineer", {})
            score = float(eng.get("graph_score", 0))
            metrics = eng.get("graph_metrics", {}) or {}
            summaries.append({
                "iteration": it.get("iteration", "?"),
                "graph_score": score,
                "primary_metric": metrics.get("primary_metric", ""),
                "primary_metric_value": metrics.get("primary_metric_value", score),
                "winner": it.get("critic", {}).get("winner", "?"),
                "stop": it.get("critic", {}).get("should_stop", False),
            })
            if score > best_score:
                best_score = score
                best_iter = it

        best_graph = (best_iter or {}).get("architect", {}).get("graph", {})
        best_mermaid = ""
        if best_graph:
            try:
                best_mermaid = PipelineGraph.from_dict(best_graph).to_mermaid()
            except Exception:
                pass

        return json.dumps({
            "n_evaluations": len(iterations),
            "n_iterations": len(iterations),
            "evaluation_summaries": summaries,
            "iteration_summaries": summaries,
            "best_score": best_score,
            "best_graph": best_graph,
            "best_graph_mermaid": best_mermaid,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================================================================

if __name__ == "__main__":
    print(json.dumps({"server": mcp.name, "tools": mcp.list_tools()}, ensure_ascii=False, indent=2))
