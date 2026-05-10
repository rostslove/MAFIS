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
import time
import traceback
from copy import deepcopy
from inspect import Parameter, signature
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

# Make backend importable when launched as subprocess
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, Normalizer, OneHotEncoder

from benchmarks import (
    DEFAULT_BENCHMARK_TARGET_COLUMN,
    load_benchmark_artifact_data,
    load_benchmark_artifact_metadata,
)
from data_profiler import DataProfiler
from graph_engine import (
    OPERATIONS, METRICS_BY_TASK, SUPPORTED_TASKS, DEFAULT_GRAPHS, INDUSTRIAL_GRAPH_TEMPLATES,
    PipelineGraph, compute_metrics, get_operation_catalog, get_training_strategies,
    get_training_strategy_hints, is_industrial_native_model, is_ts_task,
    industrial_tuple_sample_count, input_sample_count, load_industrial_tuple_data,
    load_input_data, make_tabular_input_data_like, slice_input_data,
    split_industrial_tuple_data, split_input_data,
)
from path_utils import normalize_csv_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MCP")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except (TypeError, ValueError):
        return default


TRAINING_PROGRESS_LOG_INTERVAL = max(
    5,
    _int_env("TRAINING_PROGRESS_LOG_INTERVAL", 30),
)


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

TRAIN_ONLY_OPS = {"resample", "class_decompose"}
DATA_BOUNDARY_PREPROCESSING_OPS = {"one_hot_encoding", "label_encoding", "cat_features", "normalization"}
EXECUTION_ADAPTER_OPS = TRAIN_ONLY_OPS | DATA_BOUNDARY_PREPROCESSING_OPS
CATBOOST_OPS = {"catboost", "catboostreg"}
LGBM_OPS = {"lgbm", "lgbmreg"}
N_JOBS_OPERATION_OPS = {
    "rf", "rfr", "xgboost", "xgbreg", "lgbm", "lgbmreg", "catboost", "catboostreg",
    "extra_trees", "extra_trees_reg", "isolation_forest_class", "isolation_forest_reg",
}


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


def _graph_operation_names(graph: PipelineGraph) -> List[str]:
    return [node.operation for node in graph.nodes]


def _industrial_native_model_ops(graph: PipelineGraph) -> List[str]:
    return [
        node.operation
        for node in graph.nodes
        if is_industrial_native_model(node.operation)
    ]


def _is_tuple_data(data: Any) -> bool:
    return isinstance(data, tuple) and len(data) == 2


def _target_values(data: Any) -> np.ndarray:
    if _is_tuple_data(data):
        return np.asarray(data[1])
    return np.asarray(data.target)


def _sample_count(data: Any) -> int:
    return industrial_tuple_sample_count(data) if _is_tuple_data(data) else input_sample_count(data)


def _slice_training_data(data: Any, indices: np.ndarray) -> Any:
    if _is_tuple_data(data):
        X, y = data
        return np.asarray(X)[indices], np.asarray(y)[indices]
    return slice_input_data(data, indices)


def _task_problem(task_type: str) -> str:
    if task_type in ("classification", "ts_classification"):
        return "classification"
    if task_type in ("regression", "ts_regression"):
        return "regression"
    return task_type


def _available_cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def _resolve_training_n_jobs(params: Optional[Dict[str, Any]] = None) -> int:
    raw_value = None
    if isinstance(params, dict):
        raw_value = params.get("n_jobs")
    if raw_value is None:
        raw_value = os.environ.get("FEDOT_N_JOBS") or os.environ.get("N_JOBS")
    try:
        requested = int(raw_value)
    except (TypeError, ValueError):
        requested = 0
    return _available_cpu_count() if requested <= 0 else max(1, requested)


def _graph_operation_summary(graph: PipelineGraph) -> str:
    return " -> ".join(node.operation for node in graph.nodes)


def _graph_with_runtime_model_params(graph: PipelineGraph, n_jobs: int) -> PipelineGraph:
    """Attach runtime execution params that upstream repositories default too low."""
    adapted = deepcopy(graph)
    for node in adapted.nodes:
        params = dict(node.params or {})
        if node.operation in N_JOBS_OPERATION_OPS:
            params["n_jobs"] = n_jobs
        if node.operation in LGBM_OPS:
            params["use_eval_set"] = False
            params["early_stopping_rounds"] = None
        node.params = params
    return adapted


def _runtime_model_param_adapters(graph: PipelineGraph) -> List[str]:
    adapters: List[str] = []
    if any(node.operation in N_JOBS_OPERATION_OPS for node in graph.nodes):
        adapters.append("n_jobs_runtime_params")
    if any(node.operation in LGBM_OPS for node in graph.nodes):
        adapters.append("lgbm_eval_set_sanitizer")
    return adapters


def _adapt_initial_assumption_graph(
    graph: PipelineGraph,
) -> tuple[PipelineGraph, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Move runtime-adapted operations out of the executable Fedot.Industrial graph."""
    raw_graph = graph.to_dict()
    nodes = raw_graph.get("nodes", []) or []
    removed = [node for node in nodes if node.get("operation") in EXECUTION_ADAPTER_OPS]
    if not removed:
        return graph, [], []

    data_boundary_nodes = [
        node for node in removed
        if node.get("operation") in DATA_BOUNDARY_PREPROCESSING_OPS
    ]
    train_only_nodes = [
        node for node in removed
        if node.get("operation") in TRAIN_ONLY_OPS
    ]

    removed_by_id = {node.get("id"): node for node in removed if node.get("id")}

    def _rewired_inputs(node_id: str, seen: Optional[set[str]] = None) -> List[str]:
        seen = set(seen or set())
        if node_id in seen:
            return []
        seen.add(node_id)
        removed_node = removed_by_id.get(node_id)
        if not removed_node:
            return [node_id]
        expanded: List[str] = []
        for parent_id in removed_node.get("inputs", []) or []:
            expanded.extend(_rewired_inputs(str(parent_id), seen))
        return expanded

    adapted_nodes = []
    for node in nodes:
        if node.get("id") in removed_by_id:
            continue
        inputs: List[str] = []
        for parent_id in node.get("inputs", []) or []:
            for rewired in _rewired_inputs(str(parent_id)):
                if rewired and rewired not in inputs:
                    inputs.append(rewired)
        adapted_nodes.append({**node, "inputs": inputs})

    adapted_graph = PipelineGraph.from_dict({**raw_graph, "nodes": adapted_nodes})
    return adapted_graph, data_boundary_nodes, train_only_nodes


def _catboost_search_space_adapter(graph: PipelineGraph) -> List[str]:
    """Avoid upstream CatBoost border_count/max_bin tuning conflict."""
    patched: List[str] = []
    operations = set(_graph_operation_names(graph))
    if not operations.intersection(CATBOOST_OPS):
        return patched
    try:
        from fedot_ind.core.tuning import search_space as industrial_search_space

        for operation in CATBOOST_OPS.intersection(operations):
            for repository_name in ("industrial_search_space", "default_fedot_operation_params"):
                repository = getattr(industrial_search_space, repository_name, {})
                space = repository.get(operation) if isinstance(repository, dict) else None
                if isinstance(space, dict) and "border_count" in space:
                    space.pop("border_count", None)
                    patched.append(f"{repository_name}.{operation}")
    except Exception as exc:
        logger.warning("CatBoost search-space adapter failed: %s", exc)
    return patched


def _resample_indices(y: np.ndarray, params: Optional[Dict[str, Any]] = None) -> np.ndarray:
    params = dict(params or {})
    y_arr = np.asarray(y)
    unique, counts = np.unique(y_arr, return_counts=True)
    if len(unique) != 2 or counts[0] == counts[1]:
        return np.arange(len(y_arr))

    minority_pos = int(np.argmin(counts))
    majority_pos = int(np.argmax(counts))
    minority_label = unique[minority_pos]
    majority_label = unique[majority_pos]
    minority_indices = np.flatnonzero(y_arr == minority_label)
    majority_indices = np.flatnonzero(y_arr == majority_label)

    try:
        ratio = float(params.get("balance_ratio", 1.0))
    except (TypeError, ValueError):
        ratio = 1.0
    ratio = min(max(ratio, 0.0), 1.0)
    balance = str(params.get("balance") or "expand_minority")
    replace = bool(params.get("replace", False))
    rng = np.random.default_rng(int(params.get("random_state", 42) or 42))

    if balance == "reduce_majority":
        desired_majority = int(round(len(majority_indices) - (len(majority_indices) - len(minority_indices)) * ratio))
        desired_majority = max(len(minority_indices), desired_majority)
        selected_majority = rng.choice(
            majority_indices,
            size=desired_majority,
            replace=replace and desired_majority > len(majority_indices),
        )
        selected = np.concatenate([minority_indices, selected_majority])
    else:
        desired_minority = int(round(len(minority_indices) + (len(majority_indices) - len(minority_indices)) * ratio))
        desired_minority = max(len(minority_indices), desired_minority)
        selected_minority = rng.choice(
            minority_indices,
            size=desired_minority,
            replace=True if desired_minority > len(minority_indices) else replace,
        )
        selected = np.concatenate([selected_minority, majority_indices])

    return rng.permutation(selected)


def _apply_train_only_operations(train_data: Any, train_only_nodes: List[Dict[str, Any]]) -> tuple[Any, List[str]]:
    notes: List[str] = []
    updated = train_data
    for node in train_only_nodes:
        operation = node.get("operation")
        if operation == "class_decompose":
            notes.append(
                f"Operation '{node.get('id')}' (class_decompose) was removed from the executable graph: "
                "FEDOT class decomposition requires both a model-prediction parent and a data parent, "
                "which is not supported by the current linear initial_assumption graph dialect."
            )
            continue
        if operation != "resample":
            continue
        before = _sample_count(updated)
        indices = _resample_indices(_target_values(updated), node.get("params") or {})
        updated = _slice_training_data(updated, indices)
        after = _sample_count(updated)
        notes.append(
            f"Operation '{node.get('id')}' (resample) was executed as a train-only data-boundary step "
            f"and removed from the executable initial_assumption graph ({before} -> {after} train samples)."
        )
    return updated, notes


def _input_metadata_value(input_data: Any, attr: str) -> Any:
    value = getattr(input_data, attr, None)
    supplementary = getattr(input_data, "supplementary_data", None)
    if value is None and supplementary is not None:
        value = getattr(supplementary, attr, None)
    return value


def _input_metadata_indices(input_data: Any, attr: str) -> np.ndarray:
    value = _input_metadata_value(input_data, attr)
    if value is None:
        return np.asarray([], dtype=int)
    return np.asarray(value, dtype=int)


def _input_feature_names(input_data: Any) -> List[str]:
    features = np.asarray(input_data.features)
    value = _input_metadata_value(input_data, "features_names")
    if value is not None and len(value) == features.shape[1]:
        return [str(item) for item in list(value)]
    return [f"feature_{idx}" for idx in range(features.shape[1])]


def _new_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _to_dense_array(values: Any) -> np.ndarray:
    if hasattr(values, "toarray"):
        return np.asarray(values.toarray())
    return np.asarray(values)


def _apply_one_hot_boundary(train_data: Any, test_data: Any) -> tuple[Any, Any, str]:
    categorical_idx = _input_metadata_indices(train_data, "categorical_idx")
    train_features = np.asarray(train_data.features)
    test_features = np.asarray(test_data.features)
    if train_features.ndim != 2 or test_features.ndim != 2 or categorical_idx.size == 0:
        return (
            train_data,
            test_data,
            "Operation one_hot_encoding was moved to the data boundary, but no categorical columns were available; no feature change was applied.",
        )

    feature_names = _input_feature_names(train_data)
    all_idx = np.arange(train_features.shape[1])
    categorical_set = set(int(idx) for idx in categorical_idx)
    numerical_idx = [int(idx) for idx in all_idx if int(idx) not in categorical_set]

    encoder = _new_one_hot_encoder()
    train_categorical = train_features[:, categorical_idx]
    test_categorical = test_features[:, categorical_idx]
    train_encoded = _to_dense_array(encoder.fit_transform(train_categorical))
    test_encoded = _to_dense_array(encoder.transform(test_categorical))

    parts_train = []
    parts_test = []
    transformed_names = [feature_names[idx] for idx in numerical_idx]
    if numerical_idx:
        parts_train.append(train_features[:, numerical_idx])
        parts_test.append(test_features[:, numerical_idx])

    try:
        encoded_names = [
            str(name)
            for name in encoder.get_feature_names_out([feature_names[idx] for idx in categorical_idx])
        ]
    except Exception:
        encoded_names = [f"one_hot_{idx}" for idx in range(train_encoded.shape[1])]

    parts_train.append(train_encoded)
    parts_test.append(test_encoded)
    transformed_names.extend(encoded_names)
    transformed_train = np.hstack(parts_train)
    transformed_test = np.hstack(parts_test)
    numerical_after = list(range(transformed_train.shape[1]))

    return (
        make_tabular_input_data_like(
            train_data,
            transformed_train,
            feature_names=transformed_names,
            categorical_idx=[],
            numerical_idx=numerical_after,
        ),
        make_tabular_input_data_like(
            test_data,
            transformed_test,
            feature_names=transformed_names,
            categorical_idx=[],
            numerical_idx=numerical_after,
        ),
        (
            "Operation one_hot_encoding was executed as a data-boundary transform "
            f"({len(categorical_idx)} categorical columns -> {train_encoded.shape[1]} encoded columns) "
            "and removed from the executable initial_assumption graph."
        ),
    )


def _apply_normalization_boundary(
    train_data: Any,
    test_data: Any,
    params: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Any, str]:
    params = dict(params or {})
    norm = str(params.get("norm") or "l2")
    if norm not in {"l1", "l2", "max"}:
        norm = "l2"

    train_features = np.asarray(train_data.features)
    test_features = np.asarray(test_data.features)
    if train_features.ndim != 2 or test_features.ndim != 2:
        return (
            train_data,
            test_data,
            f"Operation normalization was moved to the data boundary, but features were not 2D "
            f"(train_shape={train_features.shape}, test_shape={test_features.shape}); no feature change was applied.",
        )

    feature_names = _input_feature_names(train_data)
    categorical_idx = [int(idx) for idx in _input_metadata_indices(train_data, "categorical_idx")]
    numerical_idx = [int(idx) for idx in _input_metadata_indices(train_data, "numerical_idx")]
    if not numerical_idx:
        numerical_idx = [idx for idx in range(train_features.shape[1]) if idx not in set(categorical_idx)]
    if not numerical_idx:
        return (
            train_data,
            test_data,
            "Operation normalization was moved to the data boundary, but no numerical columns were available; no feature change was applied.",
        )

    train_normalized = np.asarray(train_features, dtype=float).copy()
    test_normalized = np.asarray(test_features, dtype=float).copy()
    normalizer = Normalizer(norm=norm)
    train_normalized[:, numerical_idx] = normalizer.fit_transform(train_normalized[:, numerical_idx])
    test_normalized[:, numerical_idx] = normalizer.transform(test_normalized[:, numerical_idx])

    return (
        make_tabular_input_data_like(
            train_data,
            train_normalized,
            feature_names=feature_names,
            categorical_idx=categorical_idx,
            numerical_idx=numerical_idx,
        ),
        make_tabular_input_data_like(
            test_data,
            test_normalized,
            feature_names=feature_names,
            categorical_idx=categorical_idx,
            numerical_idx=numerical_idx,
        ),
        (
            "Operation normalization was executed as a 2D table data-boundary transform "
            f"(norm={norm}, normalized_columns={len(numerical_idx)}) and removed from the executable initial_assumption graph."
        ),
    )


def _apply_data_boundary_preprocessing(
    train_data: Any,
    test_data: Any,
    preprocessing_nodes: List[Dict[str, Any]],
) -> tuple[Any, Any, List[str]]:
    notes: List[str] = []
    updated_train = train_data
    updated_test = test_data
    for node in preprocessing_nodes:
        operation = node.get("operation")
        if operation == "one_hot_encoding":
            updated_train, updated_test, note = _apply_one_hot_boundary(updated_train, updated_test)
            notes.append(f"Operation '{node.get('id')}' ({operation}) adapter: {note}")
        elif operation == "normalization":
            updated_train, updated_test, note = _apply_normalization_boundary(
                updated_train,
                updated_test,
                node.get("params") or {},
            )
            notes.append(f"Operation '{node.get('id')}' ({operation}) adapter: {note}")
        elif operation == "label_encoding":
            notes.append(
                f"Operation '{node.get('id')}' (label_encoding) was executed by the data loader's "
                "categorical factorization and removed from the executable initial_assumption graph."
            )
        elif operation == "cat_features":
            notes.append(
                f"Operation '{node.get('id')}' (cat_features) was treated as a categorical metadata marker: "
                "backend loaded FEDOT InputData with categorical_idx/numerical_idx metadata and removed the dummy upstream node."
            )
    return updated_train, updated_test, notes


def _load_finetune_train_test(
    graph: PipelineGraph,
    preprocessing_nodes: List[Dict[str, Any]],
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int],
    test_size: float,
) -> tuple[Any, Any, str]:
    if graph.task_type in ("classification", "regression") and preprocessing_nodes:
        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
        train, test = split_input_data(input_data, test_size=test_size)
        return train, test, "fedot_input_data_with_boundary_preprocessing"

    industrial_data = load_industrial_tuple_data(csv_path, target_column, graph.task_type, forecast_length)
    train, test = split_industrial_tuple_data(industrial_data, test_size=test_size)
    return train, test, "industrial_tuple"


def _record_training_event(
    events: List[Dict[str, Any]],
    stage: str,
    status: str,
    message: str,
    started_at: float,
    **details: Any,
) -> None:
    event = {
        "stage": stage,
        "status": status,
        "message": message,
        "elapsed_seconds": round(time.monotonic() - started_at, 1),
    }
    clean_details = {key: value for key, value in details.items() if value is not None}
    if clean_details:
        event["details"] = clean_details
    events.append(event)
    logger.info(
        "Training progress: stage=%s status=%s elapsed=%.1fs message=%s details=%s",
        stage,
        status,
        event["elapsed_seconds"],
        message,
        clean_details or {},
    )


def _start_training_heartbeat(run_label: str, stage: str) -> Event:
    stop = Event()
    started_at = time.monotonic()

    def _log_until_stopped() -> None:
        while not stop.wait(TRAINING_PROGRESS_LOG_INTERVAL):
            logger.info(
                "Training progress: %s still running stage=%s elapsed=%.1fs",
                run_label,
                stage,
                time.monotonic() - started_at,
            )

    Thread(target=_log_until_stopped, name=f"training-progress-{stage}", daemon=True).start()
    return stop


def _fedot_config_data_type(task_type: str) -> str:
    # Classification/regression CSVs stay feature-table data, but the default
    # strategy is still Fedot.Industrial unless the strategy name contains
    # "tabular" in upstream IndustrialConfig.
    return "time_series" if task_type in ("ts_classification", "ts_regression", "ts_forecasting") else "table"


def _fedot_repository_context(industrial_strategy: str) -> str:
    return "fedot_tabular" if "tabular" in str(industrial_strategy or "").lower() else "fedot_industrial"


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
            params=dict(root.params or {}),
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
                params=dict(node.params or {}),
            )

    if len(root.inputs) == 1:
        builder.add_node(
            operation_type=root.operation,
            branch_idx=0,
            params=dict(root.params or {}),
        )
    else:
        builder.join_branches(
            operation_type=root.operation,
            params=dict(root.params or {}),
        )
    return builder


def _pipeline_to_graph_dict(pipeline, task_type: str) -> Dict[str, Any]:
    """Read a fitted Fedot Pipeline back into our PipelineGraph dict form so
    the UI can show the polished assumption graph after finetune."""
    if pipeline is None or not hasattr(pipeline, "root_node"):
        return {}
    nodes_dict: Dict[int, Dict[str, Any]] = {}

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
            "params": {},
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
    n_jobs = _resolve_training_n_jobs(industrial_strategy_params)

    data_type = _fedot_config_data_type(task_type)
    industrial_config: Dict[str, Any] = {
        "problem": fedot_problem,
        "data_type": data_type,
        "strategy_params": {
            "problem": fedot_problem,
            "data_type": data_type,
        },
    }
    if task_type == "ts_forecasting":
        industrial_config["task_params"] = {"forecast_length": forecast_length or 14}

    requested_strategy = str(industrial_strategy or "default").strip().lower() or "default"
    if requested_strategy in ("federated_automl", "sampling_strategy"):
        merged_params = {
            "problem": fedot_problem,
            "data_type": data_type,
            "timeout": int(timeout),
        }
        if isinstance(industrial_strategy_params, dict):
            merged_params.update({
                str(k): v for k, v in industrial_strategy_params.items() if v is not None
            })
        merged_params["n_jobs"] = n_jobs
        industrial_config["learning_strategy"] = requested_strategy
        industrial_config["strategy"] = requested_strategy
        industrial_config["strategy_params"] = merged_params
    else:
        industrial_config["strategy"] = "default"

    logger.info(
        "Fedot.Industrial config: requested_strategy=%s, fedot_strategy=%s, data_type=%s, problem=%s, repository_context=%s, n_jobs=%s",
        requested_strategy,
        industrial_config.get("strategy", "default"),
        data_type,
        fedot_problem,
        _fedot_repository_context(industrial_config.get("strategy", "default")),
        n_jobs,
    )

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
        "learning_strategy_params": {
            **DEFAULT_AUTOML_LEARNING_CONFIG,
            "timeout": int(timeout),
            "n_jobs": n_jobs,
        },
        "optimisation_loss": {"quality_loss": learning_loss},
    }
    compute_config = dict(DEFAULT_COMPUTE_CONFIG)
    compute_config["n_jobs"] = n_jobs
    if isinstance(compute_config.get("distributed"), dict):
        compute_config["distributed"] = {
            **compute_config["distributed"],
            "threads_per_worker": n_jobs,
        }

    return {
        "industrial_config": industrial_config,
        "automl_config": automl_config,
        "learning_config": learning_config,
        "compute_config": compute_config,
    }


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
    fit_only: bool = False,
    fit_only_reason: str = "",
) -> Dict[str, Any]:
    """Run the architect-approved graph through Fedot.Industrial.

    The usual path treats the graph as a hypothesis and tunes it. ``fit_only``
    keeps Fedot.Industrial's data/backend lifecycle but asks it to only fit the
    supplied graph via the ``return_only_fitted`` path, without running a tuner.
    """
    from fedot_ind.api.main import FedotIndustrial

    run_started_at = time.monotonic()
    training_events: List[Dict[str, Any]] = []
    run_label = f"{graph.task_type}:{int(time.time())}"
    logger.info(
        "Training run started: label=%s task=%s graph=%s strategy=%s timeout=%s",
        run_label,
        graph.task_type,
        _graph_operation_summary(graph),
        industrial_strategy or "default",
        timeout,
    )
    executable_graph, preprocessing_nodes, train_only_nodes = _adapt_initial_assumption_graph(graph)
    train, test, data_boundary = _load_finetune_train_test(
        graph=executable_graph,
        preprocessing_nodes=preprocessing_nodes,
        csv_path=csv_path,
        target_column=target_column,
        forecast_length=forecast_length,
        test_size=test_size,
    )
    train, test, preprocessing_notes = _apply_data_boundary_preprocessing(train, test, preprocessing_nodes)
    train, train_only_notes = _apply_train_only_operations(train, train_only_nodes)
    _record_training_event(
        training_events,
        "data",
        "done",
        "Loaded CSV and created hold-out split.",
        run_started_at,
        boundary=data_boundary,
        train_samples=_sample_count(train),
        test_samples=_sample_count(test),
        test_size=test_size,
    )
    adapted_nodes = preprocessing_nodes + train_only_nodes
    if adapted_nodes:
        _record_training_event(
            training_events,
            "graph_adapter",
            "done",
            "Moved runtime-adapted operations out of the executable initial_assumption graph.",
            run_started_at,
            removed_operations=[node.get("operation") for node in adapted_nodes],
            executable_graph=_graph_operation_summary(executable_graph),
        )

    api_config = _build_api_config(
        task_type=executable_graph.task_type,
        forecast_length=forecast_length,
        primary_metric=primary_metric,
        industrial_strategy=industrial_strategy,
        industrial_strategy_params=industrial_strategy_params,
        timeout=timeout,
    )
    n_jobs = _resolve_training_n_jobs(industrial_strategy_params)
    executable_graph = _graph_with_runtime_model_params(executable_graph, n_jobs)
    model_param_adapters = _runtime_model_param_adapters(executable_graph)
    _record_training_event(
        training_events,
        "config",
        "done",
        "Prepared Fedot.Industrial API config.",
        run_started_at,
        data_type=_fedot_config_data_type(graph.task_type),
        repository_context=_fedot_repository_context(industrial_strategy or "default"),
        n_jobs=n_jobs,
    )
    if model_param_adapters:
        _record_training_event(
            training_events,
            "model_adapter",
            "done",
            "Applied model runtime parameter adapters.",
            run_started_at,
            adapters=model_param_adapters,
        )

    industrial = FedotIndustrial(**api_config)
    patched_search_space = [] if fit_only else _catboost_search_space_adapter(executable_graph)
    if patched_search_space:
        _record_training_event(
            training_events,
            "search_space",
            "done",
            "Sanitized CatBoost tuning search space to avoid border_count/max_bin conflicts.",
            run_started_at,
            operations=patched_search_space,
        )
    builder = _graph_to_pipeline_builder(executable_graph)
    try:
        if fit_only:
            fit_kwargs = {
                "tuning_params": {"tuning_iterations": 0, "n_jobs": n_jobs},
                "model_to_tune": builder,
                "return_only_fitted": True,
            }
            _record_training_event(
                training_events,
                "industrial_fit",
                "running",
                "Fedot.Industrial fit-only started.",
                run_started_at,
                n_jobs=n_jobs,
            )
            heartbeat_stop = _start_training_heartbeat(run_label, "industrial_fit")
            try:
                industrial.finetune(
                    train_data=train,
                    **fit_kwargs,
                )
            finally:
                heartbeat_stop.set()
            _record_training_event(
                training_events,
                "industrial_fit",
                "done",
                "Fedot.Industrial fit-only completed.",
                run_started_at,
            )
        else:
            finetune_kwargs = {
                "tuning_params": {"tuning_iterations": 5, "n_jobs": n_jobs},
                "model_to_tune": builder,
                "return_only_fitted": False,
            }
            _record_training_event(
                training_events,
                "finetune",
                "running",
                "Fedot.Industrial finetune started.",
                run_started_at,
                n_jobs=n_jobs,
                tuning_iterations=5,
            )
            heartbeat_stop = _start_training_heartbeat(run_label, "finetune")
            try:
                industrial.finetune(
                    train_data=train,
                    **finetune_kwargs,
                )
            finally:
                heartbeat_stop.set()
            _record_training_event(
                training_events,
                "finetune",
                "done",
                "Fedot.Industrial finetune completed.",
                run_started_at,
            )

        fitted_pipeline = industrial.manager.solver
        _record_training_event(
            training_events,
            "predict",
            "running",
            "Generating train and hold-out predictions.",
            run_started_at,
        )
        heartbeat_stop = _start_training_heartbeat(run_label, "predict")
        try:
            train_preds = np.asarray(industrial.predict(train)).reshape(-1)
            test_preds = np.asarray(industrial.predict(test)).reshape(-1)
        finally:
            heartbeat_stop.set()
        _record_training_event(
            training_events,
            "predict",
            "done",
            "Predictions generated.",
            run_started_at,
        )
    finally:
        try:
            industrial.shutdown()
        except Exception as shutdown_exc:
            logger.warning("FedotIndustrial shutdown failed: %s", shutdown_exc)

    train_metrics = compute_metrics(executable_graph.task_type, _target_values(train), train_preds, primary_metric)
    test_metrics = compute_metrics(executable_graph.task_type, _target_values(test), test_preds, primary_metric)
    _record_training_event(
        training_events,
        "metrics",
        "done",
        "Computed train and hold-out metrics.",
        run_started_at,
        primary_metric=test_metrics.get("primary_metric"),
        primary_metric_value=test_metrics.get("primary_metric_value"),
    )

    assumption_graph = _pipeline_to_graph_dict(fitted_pipeline, executable_graph.task_type)
    try:
        assumption_mermaid = (
            PipelineGraph.from_dict(assumption_graph).to_mermaid() if assumption_graph else ""
        )
    except Exception:
        assumption_mermaid = ""

    _store_run(fitted_pipeline, executable_graph, train, test_preds)
    target_info = _target_info(csv_path, target_column, executable_graph.task_type)

    if fit_only:
        notes = [
            "Engineer ran the graph through Fedot.Industrial fit-only lifecycle without hyperparameter tuning.",
            f"Fedot.Industrial fit ran with n_jobs={n_jobs}.",
        ]
        if fit_only_reason:
            notes.append(f"Finetune was skipped: {fit_only_reason}")
    else:
        notes = [
            "Engineer ran Fedot.Industrial finetune over the architect's graph as the initial assumption.",
            f"Fedot.Industrial tuning ran with n_jobs={n_jobs}.",
        ]
    notes.extend(preprocessing_notes)
    notes.extend(train_only_notes)
    if target_info.get("fedot_receives_encoded_target") and target_info.get("reference_encoded"):
        notes.append(
            "Classification target labels were encoded to integer class ids before Fedot.Industrial training; "
            "reference label mapping is shown in diagnostics."
        )
    if data_boundary == "fedot_input_data_with_boundary_preprocessing":
        notes.append(
            "Explicit categorical preprocessing node detected; backend used FEDOT InputData at the data boundary instead of executing the upstream encoder as a graph node."
        )
    if patched_search_space:
        notes.append(
            "CatBoost tuning search space was sanitized by removing border_count because upstream defaults already set max_bin."
        )
    if any((node.params or {}).get("n_jobs") == n_jobs for node in executable_graph.nodes):
        notes.append(f"Runtime model parameters set n_jobs={n_jobs} on supported model nodes.")
    if any(node.operation in LGBM_OPS for node in executable_graph.nodes):
        notes.append(
            "LGBM eval-set early stopping was disabled for this finetune path because upstream does not provide eval_set/eval_metric to LightGBM."
        )
    strategy_name = str(industrial_strategy or "default").strip().lower() or "default"
    data_type = _fedot_config_data_type(graph.task_type)
    repository_context = _fedot_repository_context(strategy_name)
    if strategy_name in ("federated_automl", "sampling_strategy"):
        notes.append(
            f"industrial_strategy='{strategy_name}' was attached to the run via industrial_config."
        )
    else:
        strategy_name = "default"
        notes.append(
            f"industrial_strategy='default' - Fedot.Industrial default strategy; "
            f"repository_context='{repository_context}', data_type='{data_type}'."
        )

    return {
        "score": test_metrics["primary_score"],
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_info": {
            "test_size": test_size,
            "n_train": _sample_count(train),
            "n_test": _sample_count(test),
            "data_boundary": data_boundary,
        },
        "graph": graph.to_dict(),
        "effective_graph": executable_graph.to_dict(),
        "assumption_graph": assumption_graph,
        "assumption_mermaid": assumption_mermaid,
        "industrial_strategy": strategy_name,
        "industrial_strategy_params": industrial_strategy_params or {},
        "industrial_data_type": data_type,
        "industrial_repository_context": repository_context,
        "fit_mode": "fedot_industrial_fit_no_tune" if fit_only else "fedot_industrial_finetune",
        "finetune_skipped": bool(fit_only),
        "finetune_skip_reason": fit_only_reason if fit_only else "",
        "target_info": target_info,
        "training_notes": notes,
        "training_log": training_events,
        "n_jobs": n_jobs,
    }


def _train_direct(
    graph: PipelineGraph,
    csv_path: str,
    target_column: str,
    forecast_length: Optional[int],
    primary_metric: Optional[str],
    test_size: float,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit the graph as-is without Fedot.Industrial finetune."""
    run_started_at = time.monotonic()
    training_events: List[Dict[str, Any]] = []
    resolved_n_jobs = n_jobs or _resolve_training_n_jobs()
    logger.info(
        "Direct baseline fit started: task=%s graph=%s n_jobs=%s",
        graph.task_type,
        _graph_operation_summary(graph),
        resolved_n_jobs,
    )
    executable_graph, preprocessing_nodes, train_only_nodes = _adapt_initial_assumption_graph(graph)
    executable_graph = _graph_with_runtime_model_params(executable_graph, resolved_n_jobs)
    input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)
    train, val = split_input_data(input_data, test_size=test_size)
    train, val, preprocessing_notes = _apply_data_boundary_preprocessing(train, val, preprocessing_nodes)
    train, train_only_notes = _apply_train_only_operations(train, train_only_nodes)
    _record_training_event(
        training_events,
        "data",
        "done",
        "Loaded CSV and created hold-out split for direct baseline.",
        run_started_at,
        train_samples=input_sample_count(train),
        test_samples=input_sample_count(val),
        test_size=test_size,
    )
    adapted_nodes = preprocessing_nodes + train_only_nodes
    if adapted_nodes:
        _record_training_event(
            training_events,
            "graph_adapter",
            "done",
            "Moved runtime-adapted operations out of the direct baseline graph.",
            run_started_at,
            removed_operations=[node.get("operation") for node in adapted_nodes],
            executable_graph=_graph_operation_summary(executable_graph),
        )

    pipeline = executable_graph.to_fedot_pipeline()
    _record_training_event(
        training_events,
        "direct_fit",
        "running",
        "Direct FEDOT baseline fit started.",
        run_started_at,
    )
    heartbeat_stop = _start_training_heartbeat(f"{graph.task_type}:{int(time.time())}", "direct_fit")
    try:
        pipeline.fit(train)
    finally:
        heartbeat_stop.set()
    _record_training_event(
        training_events,
        "direct_fit",
        "done",
        "Direct FEDOT baseline fit completed.",
        run_started_at,
    )
    train_preds = _predict_pipeline(pipeline, train, graph.task_type)
    test_preds = _predict_pipeline(pipeline, val, graph.task_type)

    train_metrics = compute_metrics(graph.task_type, train.target, train_preds, primary_metric)
    test_metrics = compute_metrics(graph.task_type, val.target, test_preds, primary_metric)
    _store_run(pipeline, executable_graph, input_data, test_preds)

    target_info = _target_info(csv_path, target_column, graph.task_type)
    notes = ["Engineer used the direct-fit fallback (Fedot.Industrial finetune unavailable)."]
    notes.extend(preprocessing_notes)
    notes.extend(train_only_notes)
    if any((node.params or {}).get("n_jobs") == resolved_n_jobs for node in executable_graph.nodes):
        notes.append(f"Runtime model parameters set n_jobs={resolved_n_jobs} on supported model nodes.")
    if any(node.operation in LGBM_OPS for node in executable_graph.nodes):
        notes.append(
            "LGBM eval-set early stopping was disabled for this fit path because no eval_set/eval_metric is supplied."
        )
    if target_info.get("fedot_receives_encoded_target") and target_info.get("reference_encoded"):
        notes.append(
            "Classification target labels were encoded to integer class ids before FEDOT training; "
            "reference label mapping is shown in diagnostics."
        )
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
        "effective_graph": executable_graph.to_dict(),
        "assumption_graph": executable_graph.to_dict(),
        "assumption_mermaid": executable_graph.to_mermaid(),
        "industrial_strategy": "default",
        "industrial_strategy_params": {},
        "industrial_data_type": _fedot_config_data_type(graph.task_type),
        "industrial_repository_context": _fedot_repository_context("default"),
        "target_info": target_info,
        "training_notes": notes,
        "training_log": training_events,
        "n_jobs": resolved_n_jobs,
    }


def _target_info(csv_path: str, target_column: str, task_type: str) -> Dict[str, Any]:
    csv_path = normalize_csv_path(csv_path)
    metadata = load_benchmark_artifact_metadata(csv_path)
    if metadata is not None:
        expected_target = metadata.get("target_column", DEFAULT_BENCHMARK_TARGET_COLUMN)
        if target_column != expected_target:
            raise ValueError(f"Target column '{target_column}' not in benchmark artifact; expected '{expected_target}'.")
        _, y, _ = load_benchmark_artifact_data(csv_path, mmap_mode="r")
        y_arr = np.asarray(y).astype(str)
        values = y_arr[y_arr != ""]
        unique_values = np.unique(values)
        info: Dict[str, Any] = {
            "column": target_column,
            "raw_dtype": str(y.dtype),
            "unique_values": int(len(unique_values)),
            "sample_values": [str(v) for v in values[:10].tolist()],
            "fedot_receives_raw_target": False,
            "fedot_receives_encoded_target": task_type in ("classification", "ts_classification"),
            "reference_encoded": False,
            "storage_format": metadata.get("storage_format"),
        }
        if task_type in ("classification", "ts_classification"):
            labels: List[str] = []
            groups = metadata.get("groups") if isinstance(metadata, dict) else None
            if isinstance(groups, list):
                labels.extend(str(item) for item in groups)
            group_labels = metadata.get("group_labels") if isinstance(metadata, dict) else None
            if isinstance(group_labels, dict):
                try:
                    ordered = [
                        value
                        for _, value in sorted(group_labels.items(), key=lambda item: int(item[0]))
                    ]
                except (TypeError, ValueError):
                    ordered = list(group_labels.values())
                labels.extend(str(item) for item in ordered)
            labels.extend(str(item) for item in unique_values if str(item) not in labels)
            labels = list(dict.fromkeys(labels))
            info["reference_encoded"] = True
            info["reference_encoding"] = {str(label): int(code) for code, label in enumerate(labels)}
        return info

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
        metadata = load_benchmark_artifact_metadata(csv_path)
        if metadata is not None:
            expected_target = metadata.get("target_column", DEFAULT_BENCHMARK_TARGET_COLUMN)
            if target_column != expected_target:
                raise ValueError(f"Target '{target_column}' not found; benchmark artifact target is '{expected_target}'")
            X, y, _ = load_benchmark_artifact_data(csv_path, mmap_mode="r")
            profile = DataProfiler.profile(X=X, y=y, task_type=task_type)
            profile["storage_format"] = metadata.get("storage_format")
            profile["benchmark_id"] = metadata.get("benchmark_id")
            profile["benchmark_name"] = metadata.get("benchmark_name")
        else:
            profile = DataProfiler.profile_csv(csv_path, target_column, task_type=task_type)
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
    """Validate and register a graph. Node params are preserved as fixed runtime hints."""
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
    """Apply a graph-only mutation: type=add|remove|replace|connect plus details."""
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
    industrial_strategy: str = "default",
    industrial_strategy_params: Optional[Dict[str, Any]] = None,
    finetune_timeout: int = 5,
    allow_direct_fallback: bool = False,
) -> str:
    """Treat the graph as Fedot.Industrial's initial assumption and run finetune.

    Behavior:
    - Loads the CSV, splits into train/test.
    - Converts the graph into a PipelineBuilder usable as ``model_to_tune``.
    - Runs ``FedotIndustrial.finetune`` with the task data_type attached to
      ``industrial_config``. ``industrial_strategy`` is only used for explicit
      strategy-specific execution such as ``federated_automl``.
    - Fedot.Industrial-native model operations use Fedot.Industrial fit-only
      execution: MAFIS skips the tuner but still trains and predicts through
      the Fedot.Industrial lifecycle.
    - When finetune raises, returns the Fedot.Industrial failure by default.
      ``allow_direct_fallback`` is an explicit diagnostic baseline mode, not
      the normal Engineer path.

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
        strategy_name = str(industrial_strategy or "default").strip().lower() or "default"
        if strategy_name not in ("default", "federated_automl", "sampling_strategy"):
            strategy_name = "default"
        data_type = _fedot_config_data_type(graph.task_type)
        repository_context = _fedot_repository_context(strategy_name)
        n_jobs = _resolve_training_n_jobs(industrial_strategy_params)
        industrial_model_ops = _industrial_native_model_ops(graph)

        if industrial_model_ops:
            logger.info(
                "Skipping Fedot.Industrial finetune for Industrial-native model(s): %s",
                ", ".join(industrial_model_ops),
            )
            skip_reason = (
                "Graph contains Fedot.Industrial-native model operation(s): "
                f"{', '.join(industrial_model_ops)}."
            )
            result = _train_via_finetune(
                graph=graph,
                csv_path=csv_path,
                target_column=target_column,
                forecast_length=forecast_length,
                primary_metric=primary_metric,
                test_size=ts,
                industrial_strategy=strategy_name,
                industrial_strategy_params=industrial_strategy_params,
                timeout=timeout,
                fit_only=True,
                fit_only_reason=skip_reason,
            )
            return json.dumps(result)

        try:
            result = _train_via_finetune(
                graph=graph,
                csv_path=csv_path,
                target_column=target_column,
                forecast_length=forecast_length,
                primary_metric=primary_metric,
                test_size=ts,
                industrial_strategy=strategy_name,
                industrial_strategy_params=industrial_strategy_params,
                timeout=timeout,
            )
            return json.dumps(result)
        except Exception as finetune_exc:
            logger.exception("Finetune flow failed")
            finetune_error = repr(finetune_exc)
            finetune_traceback = traceback.format_exc()
            if not allow_direct_fallback:
                return json.dumps({
                    "score": 0,
                    "metrics": {
                        "primary_score": 0.0,
                        "primary_metric": primary_metric or "",
                        "primary_metric_value": 0.0,
                        "primary_score_direction": "higher_is_better",
                    },
                    "finetune_error": finetune_error,
                    "finetune_traceback": finetune_traceback,
                    "industrial_strategy": strategy_name,
                    "industrial_strategy_params": industrial_strategy_params or {},
                    "industrial_data_type": data_type,
                    "industrial_repository_context": repository_context,
                    "n_jobs": n_jobs,
                    "fallback_skipped": True,
                    "training_notes": [
                        "Fedot.Industrial finetune failed; direct-fit fallback is deferred until structural recovery attempts are exhausted."
                    ],
                })
            logger.info("Trying direct-fit fallback after finetune failure")
            try:
                fallback = _train_direct(
                    graph=graph,
                    csv_path=csv_path,
                    target_column=target_column,
                    forecast_length=forecast_length,
                    primary_metric=primary_metric,
                    test_size=ts,
                    n_jobs=n_jobs,
                )
                fallback["training_notes"].insert(
                    0,
                    f"Fedot.Industrial finetune raised: {finetune_error}. Direct-fit fallback metrics were produced as a baseline.",
                )
                fallback["finetune_error"] = finetune_error
                fallback["finetune_traceback"] = finetune_traceback
                fallback["fallback_used"] = "direct_fit"
                fallback["fallback_reason"] = "Fedot.Industrial finetune raised before producing a fitted solver."
                fallback["industrial_strategy"] = strategy_name
                fallback["industrial_strategy_params"] = industrial_strategy_params or {}
                fallback["industrial_data_type"] = data_type
                fallback["industrial_repository_context"] = repository_context
                return json.dumps(fallback)
            except Exception as fallback_exc:
                logger.exception("Direct-fit fallback also failed")
                fallback_traceback = traceback.format_exc()
                return json.dumps({
                    "score": 0,
                    "error": (
                        f"Fedot.Industrial finetune failed: {finetune_error}; "
                        f"direct-fit fallback also failed: {fallback_exc!r}."
                    ),
                    "finetune_error": finetune_error,
                    "finetune_traceback": finetune_traceback,
                    "industrial_strategy": strategy_name,
                    "industrial_strategy_params": industrial_strategy_params or {},
                    "industrial_data_type": data_type,
                    "industrial_repository_context": repository_context,
                    "n_jobs": n_jobs,
                    "fallback_error": repr(fallback_exc),
                    "fallback_traceback": fallback_traceback,
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

        n_jobs = _resolve_training_n_jobs()
        executable_graph, preprocessing_nodes, train_only_nodes = _adapt_initial_assumption_graph(graph)
        executable_graph = _graph_with_runtime_model_params(executable_graph, n_jobs)
        input_data = load_input_data(csv_path, target_column, graph.task_type, forecast_length)

        # Simple CV: rotate validation cuts
        n = input_sample_count(input_data)
        scores: List[float] = []
        adapter_notes: List[str] = []
        for fold in range(cv_folds):
            offset = fold / cv_folds
            cut_lo = int(n * offset)
            cut_hi = cut_lo + n // cv_folds
            mask = np.ones(n, dtype=bool)
            mask[cut_lo:cut_hi] = False

            train = slice_input_data(input_data, np.flatnonzero(mask))
            val = slice_input_data(input_data, np.flatnonzero(~mask))
            train, val, preprocessing_notes = _apply_data_boundary_preprocessing(train, val, preprocessing_nodes)
            train, train_only_notes = _apply_train_only_operations(train, train_only_nodes)
            if fold == 0:
                adapter_notes.extend(preprocessing_notes)
                adapter_notes.extend(train_only_notes)

            pipeline = executable_graph.to_fedot_pipeline()
            pipeline.fit(train)
            preds = _predict_pipeline(pipeline, val, graph.task_type)
            m = compute_metrics(graph.task_type, _target_values(val), preds, primary_metric)
            scores.append(m["primary_score"])

        return json.dumps({
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "fold_scores": scores,
            "cv_folds": cv_folds,
            "primary_metric": primary_metric,
            "primary_score_direction": "higher_is_better",
            "effective_graph": executable_graph.to_dict(),
            "adapter_notes": adapter_notes,
            "n_jobs": n_jobs,
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
        n_jobs = _resolve_training_n_jobs()

        def _score_graph(candidate: PipelineGraph) -> float:
            executable_graph, preprocessing_nodes, train_only_nodes = _adapt_initial_assumption_graph(candidate)
            executable_graph = _graph_with_runtime_model_params(executable_graph, n_jobs)
            candidate_train, candidate_val, _ = _apply_data_boundary_preprocessing(
                train,
                val,
                preprocessing_nodes,
            )
            candidate_train, _ = _apply_train_only_operations(candidate_train, train_only_nodes)
            pipe = executable_graph.to_fedot_pipeline()
            pipe.fit(candidate_train)
            return compute_metrics(
                candidate.task_type,
                _target_values(candidate_val),
                _predict_pipeline(pipe, candidate_val, candidate.task_type),
                primary_metric,
            )["primary_score"]

        # Reference score: full graph
        full_score = _score_graph(graph)

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
                score = _score_graph(ablated)
                importances[node.id] = round(full_score - score, 4)
            except Exception as e:
                importances[node.id] = None

        effective_graph, _, _ = _adapt_initial_assumption_graph(graph)
        effective_graph = _graph_with_runtime_model_params(effective_graph, n_jobs)
        return json.dumps({
            "full_score": full_score,
            "node_importance": importances,
            "effective_graph": effective_graph.to_dict(),
        })
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
