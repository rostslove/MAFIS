"""
MCP server hosting all tools for the GraphAutoML multi-agent system.

Tools call Fedot.Industrial directly (no HTTP proxy). The registry is loaded
in-process by mcp_client.py to avoid the Pydantic v2 dependency of the
official MCP SDK.
"""

import json
import logging
import os
import sys
from inspect import Parameter, signature
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

# Make backend importable when launched as subprocess
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from data_profiler import DataProfiler
from graph_engine import (
    OPERATIONS, METRICS_BY_TASK, SUPPORTED_TASKS, DEFAULT_GRAPHS,
    PipelineGraph, compute_metrics, diagnose_runtime_error, get_operation_catalog, is_ts_task,
    load_input_data, split_input_data,
)

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


mcp = LocalMCPRegistry("GraphAutoMLTools")


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


def _target_info(csv_path: str, target_column: str, task_type: str) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, usecols=[target_column])
    y = df[target_column]
    values = y.dropna()
    info: Dict[str, Any] = {
        "column": target_column,
        "raw_dtype": str(y.dtype),
        "unique_values": int(values.nunique()),
        "sample_values": [str(v) for v in values.head(10).tolist()],
        "fedot_receives_raw_target": task_type in ("classification", "ts_classification"),
        "baseline_encoded": False,
    }
    if task_type in ("classification", "ts_classification") and not np.issubdtype(y.dtype, np.number):
        encoder = LabelEncoder()
        encoder.fit(values.astype(str).values)
        info["baseline_encoded"] = True
        info["baseline_encoding"] = {str(label): int(code) for code, label in enumerate(encoder.classes_)}
    return info


def _failure_payload(error: Exception, task_type: str = "", graph: Optional[PipelineGraph] = None) -> Dict[str, Any]:
    diagnostic = diagnose_runtime_error(error, task_type=task_type, graph=graph)
    return {
        "score": 0,
        "error": str(error),
        "diagnostics": [diagnostic],
        "recommendations": diagnostic.get("recommendations", []),
    }


def _invalid_graph_payload(message: str, graph: PipelineGraph) -> Dict[str, Any]:
    diagnostic = diagnose_runtime_error(message, task_type=graph.task_type, graph=graph)
    return {
        "score": 0,
        "error": f"Invalid graph: {message}",
        "diagnostics": [diagnostic],
        "recommendations": diagnostic.get("recommendations", []),
    }


# ============== Sklearn baselines ==============

def _baselines(task_type: str) -> Dict[str, SkPipeline]:
    if task_type in ("classification", "ts_classification"):
        return {
            "logreg": SkPipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
            "rf": SkPipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=100, random_state=42))]),
            "xgb": SkPipeline([("scaler", StandardScaler()), ("model", XGBClassifier(n_estimators=100, random_state=42))]),
        }
    if task_type in ("regression", "ts_regression"):
        return {
            "ridge": SkPipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
            "rf": SkPipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(n_estimators=100, random_state=42))]),
            "xgb": SkPipeline([("scaler", StandardScaler()), ("model", XGBRegressor(n_estimators=100, random_state=42))]),
        }
    return {}


# ================================================================
#                          DATA / OPERATIONS
# ================================================================

@mcp.tool()
def get_data_profile(csv_path: str, target_column: str, task_type: str = "classification") -> str:
    """Profile a CSV: returns JSON with n_samples, n_features, issues, recommendations."""
    try:
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
        diagnostics = [] if ok else [diagnose_runtime_error(msg, task_type=graph.task_type, graph=graph)]
        return json.dumps({
            "valid": ok,
            "message": msg,
            "graph": graph.to_dict(),
            "mermaid": graph.to_mermaid() if ok else "",
            "diagnostics": diagnostics,
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
        diagnostics = [] if ok else [diagnose_runtime_error(msg, task_type=new_graph.task_type, graph=new_graph)]
        return json.dumps({
            "valid": ok,
            "message": msg,
            "graph": new_graph.to_dict(),
            "mermaid": new_graph.to_mermaid() if ok else "",
            "diagnostics": diagnostics,
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
#                          BASELINES
# ================================================================

@mcp.tool()
def get_baselines(task_type: str) -> str:
    """List sklearn baseline names for the task. Empty for ts_forecasting."""
    if task_type == "ts_forecasting":
        return json.dumps({"baselines": [], "note": "Baselines unavailable for ts_forecasting"})
    return json.dumps({"baselines": list(_baselines(task_type).keys())})


@mcp.tool()
def train_baseline(csv_path: str, target_column: str, baseline_name: str, task_type: str = "classification") -> str:
    """Train an sklearn baseline. Returns score and metrics."""
    try:
        if task_type == "ts_forecasting":
            return json.dumps({"name": baseline_name, "score": 0, "error": "Baselines unavailable for ts_forecasting"})

        baselines = _baselines(task_type)
        if baseline_name not in baselines:
            return json.dumps({"name": baseline_name, "score": 0, "error": f"Unknown baseline; available: {list(baselines.keys())}"})

        df = pd.read_csv(csv_path)
        y = df[target_column].values
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number]).values
        baseline_encoded = False
        if task_type in ("classification", "ts_classification") and not np.issubdtype(np.asarray(y).dtype, np.number):
            y = LabelEncoder().fit_transform(pd.Series(y).astype(str).values)
            baseline_encoded = True

        stratify = y if task_type in ("classification", "ts_classification") else None
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)

        pipe = baselines[baseline_name]
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        metrics = compute_metrics(task_type, y_va, preds)
        return json.dumps({
            "name": baseline_name,
            "score": metrics["primary_score"],
            "metrics": metrics,
            "target_info": _target_info(csv_path, target_column, task_type),
            "baseline_encoded_target": baseline_encoded,
        })
    except Exception as e:
        return json.dumps({"name": baseline_name, "score": 0, "error": str(e)})


# ================================================================
#                          GRAPH TRAIN / TUNE
# ================================================================

@mcp.tool()
def train_graph(graph_json: str, csv_path: str, target_column: str, forecast_length: Optional[int] = None) -> str:
    """Train a pipeline graph as-is on the data. Returns score and per-task metrics."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps(_invalid_graph_payload(msg, graph))

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data)

        pipeline = graph.to_fedot_pipeline()
        pipeline.fit(train)
        preds = _predict_pipeline(pipeline, val, graph.task_type)

        metrics = compute_metrics(graph.task_type, val.target, preds)
        _store_run(pipeline, graph, input_data, preds)

        target_info = _target_info(csv_path, target_column, graph.task_type)
        training_notes = []
        if target_info.get("fedot_receives_raw_target") and target_info.get("baseline_encoded"):
            training_notes.append(
                "Fedot graph received the raw string classification target; sklearn baselines encoded it only for baseline fitting."
            )

        return json.dumps({
            "score": metrics["primary_score"],
            "metrics": metrics,
            "graph": graph.to_dict(),
            "n_train": len(train.features),
            "n_val": len(val.features),
            "target_info": target_info,
            "training_notes": training_notes,
        })
    except Exception as e:
        logger.exception("train_graph failed")
        return json.dumps(_failure_payload(e, task_type=getattr(locals().get("graph", None), "task_type", ""), graph=locals().get("graph")))


@mcp.tool()
def tune_graph_hyperparameters(
    graph_json: str,
    csv_path: str,
    target_column: str,
    iterations: int = 20,
    forecast_length: Optional[int] = None,
) -> str:
    """Tune node hyperparameters of a graph using Fedot's PipelineTuner. Returns tuned graph + score."""
    try:
        from fedot.core.pipelines.tuning.tuner_builder import TunerBuilder

        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps(_invalid_graph_payload(msg, graph))

        if graph.task_type in ("classification", "ts_classification"):
            diagnostic = {
                "agent": "Engineer",
                "kind": "tuning_skipped",
                "summary": "Classification tuning skipped to avoid Fedot ROC-AUC metric tracebacks.",
                "technical_message": (
                    "Fedot.Industrial 0.5 tuner may evaluate binary ROC-AUC on a two-column probability matrix "
                    "and print repeated internal ValueError tracebacks. Training the graph as-is is cleaner."
                ),
                "recommendations": [
                    "Use train_graph for classification graphs.",
                    "Let Critic compare the trained graph with baselines and suggest a model replacement.",
                ],
                "recoverable": True,
            }
            return json.dumps({
                "score": 0,
                "error": "tuning skipped for classification",
                "diagnostics": [diagnostic],
                "recommendations": diagnostic["recommendations"],
                "target_info": _target_info(csv_path, target_column, graph.task_type),
            })

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data)

        pipeline = graph.to_fedot_pipeline()

        try:
            tuner = TunerBuilder(input_data.task).with_iterations(iterations).build(train)
            pipeline = tuner.tune(pipeline)
        except Exception as e:
            logger.warning(f"Tuning failed, using untuned pipeline: {e}")

        pipeline.fit(train)
        preds = _predict_pipeline(pipeline, val, graph.task_type)
        metrics = compute_metrics(graph.task_type, val.target, preds)
        _store_run(pipeline, graph, input_data, preds)

        # Extract tuned params back per node
        tuned_nodes = []
        try:
            for fnode, gnode in zip(pipeline.nodes, graph.nodes):
                tuned_nodes.append({"id": gnode.id, "operation": gnode.operation, "tuned_params": dict(getattr(fnode, "parameters", {}) or {})})
        except Exception:
            pass

        target_info = _target_info(csv_path, target_column, graph.task_type)
        training_notes = []
        if target_info.get("fedot_receives_raw_target") and target_info.get("baseline_encoded"):
            training_notes.append(
                "Fedot graph received the raw string classification target; sklearn baselines encoded it only for baseline fitting."
            )

        return json.dumps({
            "score": metrics["primary_score"],
            "metrics": metrics,
            "graph": graph.to_dict(),
            "tuned_nodes": tuned_nodes,
            "target_info": target_info,
            "training_notes": training_notes,
        })
    except Exception as e:
        logger.exception("tune_graph_hyperparameters failed")
        return json.dumps(_failure_payload(e, task_type=getattr(locals().get("graph", None), "task_type", ""), graph=locals().get("graph")))


@mcp.tool()
def validate_graph(graph_json: str, csv_path: str, target_column: str, cv_folds: int = 3, forecast_length: Optional[int] = None) -> str:
    """Cross-validate the graph. Returns mean & std of the primary metric across folds."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            payload = _invalid_graph_payload(msg, graph)
            return json.dumps({"error": payload["error"], "diagnostics": payload["diagnostics"], "recommendations": payload["recommendations"]})

        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)

        # Simple CV: rotate validation cuts
        n = len(input_data.features)
        scores: List[float] = []
        for fold in range(cv_folds):
            offset = fold / cv_folds
            cut_lo = int(n * offset)
            cut_hi = cut_lo + n // cv_folds
            mask = np.ones(n, dtype=bool)
            mask[cut_lo:cut_hi] = False

            from fedot.core.data.data import InputData
            train = InputData(
                idx=np.arange(mask.sum()), features=input_data.features[mask],
                target=input_data.target[mask], task=input_data.task, data_type=input_data.data_type,
            )
            val = InputData(
                idx=np.arange((~mask).sum()), features=input_data.features[~mask],
                target=input_data.target[~mask], task=input_data.task, data_type=input_data.data_type,
            )

            pipeline = graph.to_fedot_pipeline()
            pipeline.fit(train)
            preds = _predict_pipeline(pipeline, val, graph.task_type)
            m = compute_metrics(graph.task_type, val.target, preds)
            scores.append(m["primary_score"])

        return json.dumps({
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "fold_scores": scores,
            "cv_folds": cv_folds,
        })
    except Exception as e:
        logger.exception("validate_graph failed")
        payload = _failure_payload(e, task_type=getattr(locals().get("graph", None), "task_type", ""), graph=locals().get("graph"))
        return json.dumps({"error": payload["error"], "diagnostics": payload["diagnostics"], "recommendations": payload["recommendations"]})


# ================================================================
#                          ANALYSIS
# ================================================================

@mcp.tool()
def analyze_errors(baseline_results_json: str, graph_score: float, task_type: str) -> str:
    """Compare graph score against sklearn baselines. Returns winner + statistics."""
    try:
        baselines = json.loads(baseline_results_json)
        scores = [b["score"] for b in baselines if b.get("score", 0) > 0]
        out: Dict[str, Any] = {
            "task_type": task_type,
            "graph_score": graph_score,
            "n_baselines": len(baselines),
        }
        if scores:
            best = max(baselines, key=lambda b: b.get("score", 0))
            out["best_baseline"] = {"name": best["name"], "score": best["score"]}
            out["graph_beats_baselines"] = graph_score > best["score"]
            out["delta"] = round(graph_score - best["score"], 4)
            out["baseline_mean"] = round(float(np.mean(scores)), 4)
        else:
            out["graph_beats_baselines"] = True
        out["failed_baselines"] = [b["name"] for b in baselines if b.get("error")]
        return json.dumps(out)
    except Exception as e:
        payload = _failure_payload(e, task_type=getattr(locals().get("graph", None), "task_type", ""), graph=locals().get("graph"))
        return json.dumps({"error": payload["error"], "diagnostics": payload["diagnostics"], "recommendations": payload["recommendations"]})


@mcp.tool()
def get_node_importance(graph_json: str, csv_path: str, target_column: str, forecast_length: Optional[int] = None) -> str:
    """Estimate per-node importance via leave-one-out ablation (training-only, can be slow)."""
    try:
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, val = split_input_data(input_data)

        # Baseline: full graph
        full_pipe = graph.to_fedot_pipeline()
        full_pipe.fit(train)
        full_score = compute_metrics(
            graph.task_type,
            val.target,
            _predict_pipeline(full_pipe, val, graph.task_type),
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
def generate_report(iterations_json: str) -> str:
    """Compile per-iteration data into a structured report (best score, summaries, mermaid of best graph)."""
    try:
        iterations = json.loads(iterations_json)
        best_score = -1.0
        best_iter = None
        summaries: List[Dict[str, Any]] = []

        for it in iterations:
            eng = it.get("engineer", {})
            score = float(eng.get("graph_score", 0))
            summaries.append({
                "iteration": it.get("iteration", "?"),
                "graph_score": score,
                "best_baseline": eng.get("best_baseline_score", 0),
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
            "n_iterations": len(iterations),
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
