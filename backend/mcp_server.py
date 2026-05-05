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
from sklearn.preprocessing import LabelEncoder

from data_profiler import DataProfiler
from graph_engine import (
    OPERATIONS, METRICS_BY_TASK, SUPPORTED_TASKS, DEFAULT_GRAPHS, INDUSTRIAL_GRAPH_TEMPLATES,
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


def _graph_to_pipeline_builder(graph: PipelineGraph):
    """Convert a PipelineGraph into a Fedot PipelineBuilder usable as
    ``initial_assumption`` / ``model_to_tune`` for Fedot.Industrial."""
    from fedot.core.pipelines.pipeline_builder import PipelineBuilder

    builder = PipelineBuilder()
    nodes_by_id = {n.id: n for n in graph.nodes}
    root_id = graph.root_id()
    root = nodes_by_id[root_id]

    if not root.inputs:
        builder.add_node(
            operation_type=root.operation,
            branch_idx=0,
            params=dict(root.params) if root.params else None,
        )
        return builder

    # Each direct input of the root spawns its own branch; the chain back to a
    # leaf becomes the body of that branch.
    for branch_idx, head_input_id in enumerate(root.inputs):
        chain: List[Any] = []
        current_id = head_input_id
        while True:
            current = nodes_by_id[current_id]
            chain.insert(0, current)
            if not current.inputs:
                break
            if len(current.inputs) > 1:
                raise ValueError(
                    f"Node '{current.id}' has multiple inputs but is not the root; "
                    "fan-in joins are only supported at the root."
                )
            current_id = current.inputs[0]
        for node in chain:
            builder.add_node(
                operation_type=node.operation,
                branch_idx=branch_idx,
                params=dict(node.params) if node.params else None,
            )

    if len(root.inputs) == 1:
        builder.add_node(
            operation_type=root.operation,
            branch_idx=0,
            params=dict(root.params) if root.params else None,
        )
    else:
        builder.join_branches(
            operation_type=root.operation,
            params=dict(root.params) if root.params else None,
        )
    return builder


def _pipeline_to_graph_dict(pipeline, task_type: str) -> Dict[str, Any]:
    """Read a fitted Fedot Pipeline back into our PipelineGraph dict form so
    the UI can show the polished assumption graph after finetune."""
    if pipeline is None or not hasattr(pipeline, "root_node"):
        return {}
    nodes_dict: Dict[int, Dict[str, Any]] = {}

    def _safe_params(node) -> Dict[str, Any]:
        try:
            params = getattr(node, "parameters", None) or {}
        except Exception:
            params = {}
        return {str(k): v for k, v in params.items() if not callable(v)}

    def _walk(node) -> str:
        node_key = id(node)
        if node_key in nodes_dict:
            return nodes_dict[node_key]["id"]
        nid = f"node_{len(nodes_dict)}"
        # Reserve slot before recursion so we don't loop on cycles
        nodes_dict[node_key] = {"id": nid, "operation": "", "params": {}, "inputs": []}
        parents = list(getattr(node, "nodes_from", []) or [])
        parent_ids = [_walk(parent) for parent in parents]
        operation_obj = getattr(node, "operation", None)
        operation_name = (
            getattr(operation_obj, "operation_type", None)
            or getattr(node, "name", None)
            or ""
        )
        nodes_dict[node_key].update({
            "id": nid,
            "operation": str(operation_name),
            "params": _safe_params(node),
            "inputs": parent_ids,
        })
        return nid

    try:
        _walk(pipeline.root_node)
    except Exception:
        return {}

    return {
        "task_type": task_type,
        "nodes": list(nodes_dict.values()),
    }


def _build_api_config(
    task_type: str,
    forecast_length: Optional[int],
    primary_metric: Optional[str],
    industrial_strategy: str,
    industrial_strategy_params: Optional[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Any]:
    """Assemble the four-section api_config that FedotIndustrial expects."""
    from fedot_ind.core.repository.config_repository import (
        DEFAULT_AUTOML_LEARNING_CONFIG,
        DEFAULT_CLF_AUTOML_CONFIG,
        DEFAULT_REG_AUTOML_CONFIG,
        DEFAULT_TSF_AUTOML_CONFIG,
        DEFAULT_COMPUTE_CONFIG,
    )

    problem = _task_problem(task_type)
    fedot_problem = "ts_forecasting" if task_type == "ts_forecasting" else problem

    industrial_config: Dict[str, Any] = {"problem": fedot_problem}
    if task_type == "ts_forecasting":
        industrial_config["task_params"] = {"forecast_length": forecast_length or 14}

    strategy = (industrial_strategy or "tabular").strip().lower()
    if strategy == "tabular":
        industrial_config["strategy"] = "tabular"
    elif strategy in ("federated_automl", "sampling_strategy"):
        merged_params = {
            "problem": fedot_problem,
            "data_type": _strategy_data_type(task_type),
            "timeout": int(timeout),
        }
        if isinstance(industrial_strategy_params, dict):
            merged_params.update({
                str(k): v for k, v in industrial_strategy_params.items() if v is not None
            })
        industrial_config["learning_strategy"] = strategy
        industrial_config["strategy"] = strategy
        industrial_config["strategy_params"] = merged_params
    else:
        # Unknown strategy label - fall back to tabular path so the run still completes.
        industrial_config["strategy"] = "tabular"

    if task_type == "ts_forecasting":
        automl_config = dict(DEFAULT_TSF_AUTOML_CONFIG)
        automl_config["task_params"] = {"forecast_length": forecast_length or 14}
    elif problem == "classification":
        automl_config = dict(DEFAULT_CLF_AUTOML_CONFIG)
    else:
        automl_config = dict(DEFAULT_REG_AUTOML_CONFIG)

    learning_loss = primary_metric if primary_metric else (
        "rmse" if problem == "regression" or task_type == "ts_forecasting" else "accuracy"
    )
    learning_config = {
        "learning_strategy": "from_scratch",
        "learning_strategy_params": {**DEFAULT_AUTOML_LEARNING_CONFIG, "timeout": int(timeout)},
        "optimisation_loss": {"quality_loss": learning_loss},
    }

    return {
        "industrial_config": industrial_config,
        "automl_config": automl_config,
        "learning_config": learning_config,
        "compute_config": dict(DEFAULT_COMPUTE_CONFIG),
    }


def _input_to_tuple(data) -> tuple:
    return (np.asarray(data.features), np.asarray(data.target))


def _train_via_finetune(
    graph: PipelineGraph,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int],
    primary_metric: Optional[str],
    test_size: float,
    industrial_strategy: str,
    industrial_strategy_params: Optional[Dict[str, Any]],
    timeout: int,
) -> Dict[str, Any]:
    """Run the architect-approved graph as ``initial_assumption`` through
    Fedot.Industrial's finetune flow.

    The graph is treated as a hypothesis; FedotIndustrial.finetune polishes its
    hyperparameters and returns a fitted solver we then score on the held-out
    test split.
    """
    from fedot_ind.api.main import FedotIndustrial

    input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
    train, test = split_input_data(input_data, test_size=test_size)

    api_config = _build_api_config(
        task_type=graph.task_type,
        forecast_length=forecast_length,
        primary_metric=primary_metric,
        industrial_strategy=industrial_strategy,
        industrial_strategy_params=industrial_strategy_params,
        timeout=timeout,
    )

    industrial = FedotIndustrial(**api_config)
    builder = _graph_to_pipeline_builder(graph)
    industrial.finetune(
        train_data=_input_to_tuple(train),
        tuning_params={"tuning_iterations": 5},
        model_to_tune=builder,
        return_only_fitted=False,
    )

    fitted_pipeline = industrial.manager.solver
    test_preds = np.asarray(industrial.predict(_input_to_tuple(test))).reshape(-1)
    train_preds = np.asarray(industrial.predict(_input_to_tuple(train))).reshape(-1)
    industrial.shutdown()

    train_metrics = compute_metrics(graph.task_type, train.target, train_preds, primary_metric)
    test_metrics = compute_metrics(graph.task_type, test.target, test_preds, primary_metric)

    assumption_graph = _pipeline_to_graph_dict(fitted_pipeline, graph.task_type)
    try:
        assumption_mermaid = (
            PipelineGraph.from_dict(assumption_graph).to_mermaid() if assumption_graph else ""
        )
    except Exception:
        assumption_mermaid = ""

    _store_run(fitted_pipeline, graph, train, test_preds)
    target_info = _target_info(csv_path, target_column, graph.task_type)

    notes = [
        "Engineer ran Fedot.Industrial finetune over the architect's graph as the initial assumption.",
    ]
    if industrial_strategy and industrial_strategy != "tabular":
        notes.append(
            f"industrial_strategy='{industrial_strategy}' was attached to the run via industrial_config."
        )
    else:
        notes.append("industrial_strategy='tabular' - default Fedot context was used (no industrial strategy).")

    return {
        "score": test_metrics["primary_score"],
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_info": {
            "test_size": test_size,
            "n_train": input_sample_count(train),
            "n_test": input_sample_count(test),
        },
        "graph": graph.to_dict(),
        "assumption_graph": assumption_graph,
        "assumption_mermaid": assumption_mermaid,
        "industrial_strategy": industrial_strategy or "tabular",
        "industrial_strategy_params": industrial_strategy_params or {},
        "target_info": target_info,
        "training_notes": notes,
    }


def _train_direct(
    graph: PipelineGraph,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int],
    primary_metric: Optional[str],
    test_size: float,
) -> Dict[str, Any]:
    """Fallback path: fit the graph as-is without Fedot.Industrial finetune."""
    input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
    train, val = split_input_data(input_data, test_size=test_size)

    pipeline = graph.to_fedot_pipeline()
    pipeline.fit(train)
    train_preds = _predict_pipeline(pipeline, train, graph.task_type)
    test_preds = _predict_pipeline(pipeline, val, graph.task_type)

    train_metrics = compute_metrics(graph.task_type, train.target, train_preds, primary_metric)
    test_metrics = compute_metrics(graph.task_type, val.target, test_preds, primary_metric)
    _store_run(pipeline, graph, input_data, test_preds)

    target_info = _target_info(csv_path, target_column, graph.task_type)
    notes = ["Engineer used the direct-fit fallback (Fedot.Industrial finetune unavailable)."]
    if target_info.get("fedot_receives_raw_target") and target_info.get("reference_encoded"):
        notes.append(
            "Fedot graph received the raw string classification target; reference mapping is shown only for readable diagnostics."
        )

    return {
        "score": test_metrics["primary_score"],
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_info": {
            "test_size": test_size,
            "n_train": input_sample_count(train),
            "n_test": input_sample_count(val),
        },
        "graph": graph.to_dict(),
        "assumption_graph": graph.to_dict(),
        "assumption_mermaid": graph.to_mermaid(),
        "industrial_strategy": "tabular",
        "industrial_strategy_params": {},
        "target_info": target_info,
        "training_notes": notes,
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
    """Profile a CSV: returns factual dataset shape, target distribution and issues."""
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
        "industrial_strategies_catalog": get_training_strategies(task_type),
        "industrial_strategy_hints": get_training_strategy_hints(task_type),
        "default_graph": DEFAULT_GRAPHS.get(task_type, []),
        "industrial_templates": INDUSTRIAL_GRAPH_TEMPLATES.get(task_type, []),
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
    """Apply a graph-only mutation: type=add|remove|replace|connect plus details. Returns the new graph.

    Strategy mutations (set_strategy/clear_strategy) are no longer applied to the
    graph - they target DataContext.industrial_strategy at the orchestrator level.
    """
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
#                          GRAPH TRAIN
# ================================================================

@mcp.tool()
def train_graph(
    graph_json: str,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int] = None,
    primary_metric: Optional[str] = None,
    test_size: float = 0.2,
    industrial_strategy: str = "tabular",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
    finetune_timeout: int = 5,
) -> str:
    """Treat the graph as Fedot.Industrial's initial assumption and run finetune.

    Behavior:
    - Loads the CSV, splits into train/test.
    - Converts the graph into a PipelineBuilder usable as ``model_to_tune``.
    - Runs ``FedotIndustrial.finetune`` with ``industrial_strategy`` (when not
      ``tabular``) and the optional ``industrial_strategy_params`` attached to
      ``industrial_config``.
    - When finetune raises, attempts a direct fit so we can still expose
      baseline metrics, and returns ``finetune_error``/``fallback_used`` so the
      Engineer can decide whether node-skip recovery is needed.

    Returns score, metrics, the post-finetune assumption graph, and the
    industrial strategy actually used during the run.
    """
    try:
        csv_path = normalize_csv_path(csv_path)
        graph = PipelineGraph.from_dict(json.loads(graph_json))
        ok, msg = graph.validate()
        if not ok:
            return json.dumps({"score": 0, "error": f"Invalid graph: {msg}"})

        ts = float(test_size) if test_size is not None else 0.2
        ts = min(max(ts, 0.05), 0.5)
        timeout = max(1, int(finetune_timeout or 5))

        try:
            result = _train_via_finetune(
                graph=graph,
                csv_path=csv_path,
                target_column=target_column,
                forecast_length=forecast_length,
                primary_metric=primary_metric,
                test_size=ts,
                industrial_strategy=industrial_strategy or "tabular",
                industrial_strategy_params=industrial_strategy_params,
                timeout=timeout,
            )
            return json.dumps(result)
        except Exception as finetune_exc:
            logger.exception("Finetune flow failed; recording error and trying direct fit")
            finetune_error = repr(finetune_exc)
            try:
                fallback = _train_direct(
                    graph=graph,
                    csv_path=csv_path,
                    target_column=target_column,
                    forecast_length=forecast_length,
                    primary_metric=primary_metric,
                    test_size=ts,
                )
                fallback["training_notes"].insert(
                    0,
                    f"Fedot.Industrial finetune raised: {finetune_error}. Direct-fit fallback metrics were produced as a baseline.",
                )
                fallback["finetune_error"] = finetune_error
                fallback["fallback_used"] = "direct_fit"
                fallback["fallback_reason"] = "Fedot.Industrial finetune raised before producing a fitted solver."
                return json.dumps(fallback)
            except Exception as fallback_exc:
                logger.exception("Direct-fit fallback also failed")
                return json.dumps({
                    "score": 0,
                    "error": (
                        f"Fedot.Industrial finetune failed: {finetune_error}; "
                        f"direct-fit fallback also failed: {fallback_exc!r}."
                    ),
                    "finetune_error": finetune_error,
                    "fallback_error": repr(fallback_exc),
                })
    except Exception as e:
        logger.exception("train_graph failed")
        return json.dumps({"score": 0, "error": repr(e)})

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
        return json.dumps({"error": "No trained graph yet. Call train_graph first."})

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
