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


# Atomic operations available per task type. Used for validation and proposal.
OPERATIONS: Dict[str, Dict[str, List[str]]] = {
    "classification": {
        # Fedot.Industrial 0.5 routes common preprocessing operations through
        # industrial multidimensional strategies. On ordinary tabular CSV data
        # they can receive 1D column slices and fail inside sklearn. Keep the
        # default tabular graph model-only; sklearn baselines still provide a
        # sanity check with scaling.
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
        {"id": "feat", "operation": "quantile_extractor", "params": {}, "inputs": []},
        {"id": "model", "operation": "industrial_stat_clf", "params": {}, "inputs": ["feat"]},
    ],
    "ts_regression": [
        {"id": "feat", "operation": "quantile_extractor", "params": {}, "inputs": []},
        {"id": "model", "operation": "industrial_stat_reg", "params": {}, "inputs": ["feat"]},
    ],
    "ts_forecasting": [
        {"id": "smooth", "operation": "smoothing", "params": {}, "inputs": []},
        {"id": "model", "operation": "ar", "params": {}, "inputs": ["smooth"]},
    ],
}


OPERATION_DESCRIPTIONS: Dict[str, str] = {
    "rf": "Random forest classifier; robust baseline for tabular classification.",
    "xgboost": "Gradient boosting classifier; useful for nonlinear tabular patterns.",
    "logit": "Linear logistic model; good for small or mostly linear classification tasks.",
    "knn": "Nearest-neighbor classifier; sensitive to feature scale.",
    "lgbm": "LightGBM classifier; fast gradient boosting when available.",
    "mlp": "Neural network classifier; needs enough samples.",
    "dt": "Decision tree classifier; interpretable but can overfit.",
    "ridge": "Regularized linear regressor; stable default for tabular regression.",
    "lasso": "Sparse linear regressor; can suppress weak features.",
    "xgbreg": "Gradient boosting regressor; strong nonlinear tabular baseline.",
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


def get_operation_catalog(task_type: str) -> Dict[str, List[Dict[str, str]]]:
    ops = OPERATIONS.get(task_type, {})
    return {
        group: [
            {"operation": name, "description": OPERATION_DESCRIPTIONS.get(name, "")}
            for name in names
        ]
        for group, names in ops.items()
    }


def diagnose_runtime_error(
    error: Any,
    task_type: str = "",
    graph: Optional["PipelineGraph"] = None,
) -> Dict[str, Any]:
    """Convert low-level Fedot/sklearn exceptions into user-facing guidance."""
    message = str(error or "")
    lower = message.lower()
    graph_dict = graph.to_dict() if graph else {}
    nodes = graph_dict.get("nodes", [])
    known_preprocessing_ops = {
        op
        for task_ops in OPERATIONS.values()
        for op in task_ops.get("preprocessing", [])
    }
    known_preprocessing_ops.update({"scaling", "normalization", "simple_imputation", "pca", "kernel_pca"})
    preprocessing_nodes = [
        n for n in nodes
        if n.get("operation") in known_preprocessing_ops
    ]

    diagnostic: Dict[str, Any] = {
        "agent": "system",
        "kind": "runtime_error",
        "summary": message[:500],
        "technical_message": message[:2000],
        "recommendations": [],
        "recoverable": True,
    }

    if "expected 2d array" in lower and "got 1d array" in lower:
        diagnostic.update({
            "kind": "data_shape_or_preprocessing_error",
            "summary": (
                "Fedot received a one-dimensional feature vector where the selected "
                "operation expected a two-dimensional feature matrix."
            ),
            "recommendations": [
                "For ordinary tabular CSV tasks, remove preprocessing nodes such as scaling/normalization/PCA and train a model node directly.",
                "If the data is a signal or time series, choose one of the ts_* task types and provide numeric sequence features.",
                "Check that the CSV has numeric feature columns besides the target column.",
            ],
        })
        if preprocessing_nodes:
            diagnostic["problem_nodes"] = [
                {"id": n.get("id"), "operation": n.get("operation")}
                for n in preprocessing_nodes
            ]
            diagnostic["suggested_mutations"] = [
                {"type": "remove", "node_id": n.get("id")}
                for n in preprocessing_nodes[:2]
            ]
        return diagnostic

    if "operation" in lower and "not allowed" in lower:
        diagnostic.update({
            "kind": "invalid_graph_operation",
            "summary": "The graph contains an operation that is not exposed for the selected task.",
            "recommendations": [
                "Remove the unsupported node or replace it with a model listed in the operation catalog.",
                "For tabular classification/regression, use a direct model-only graph.",
                "If the operation is a signal transform, switch to the matching ts_* task type.",
            ],
        })
        if preprocessing_nodes:
            diagnostic["problem_nodes"] = [
                {"id": n.get("id"), "operation": n.get("operation")}
                for n in preprocessing_nodes
            ]
            diagnostic["suggested_mutations"] = [
                {"type": "remove", "node_id": n.get("id")}
                for n in preprocessing_nodes[:2]
            ]
        return diagnostic

    if "0 feature" in lower or "at least one array or dtype is required" in lower:
        diagnostic.update({
            "kind": "no_numeric_features",
            "summary": "No numeric feature columns were found after removing the target column.",
            "recommendations": [
                "Convert categorical feature columns to numeric values before upload.",
                "Choose the correct target column; it must not be the only numeric column.",
            ],
        })
        return diagnostic

    if "target column" in lower and "not" in lower:
        diagnostic.update({
            "kind": "target_column_error",
            "summary": "The selected target column was not found in the uploaded CSV.",
            "recommendations": ["Select an existing target column in the sidebar and run again."],
            "recoverable": False,
        })
        return diagnostic

    if "could not convert" in lower or "invalid literal" in lower:
        diagnostic.update({
            "kind": "non_numeric_data",
            "summary": "A model received non-numeric values that were not encoded.",
            "recommendations": [
                "Encode categorical feature columns before upload.",
                "Remove text/id columns that are not useful model features.",
            ],
        })
        return diagnostic

    diagnostic["recommendations"] = [
        "Try a simpler graph with only the model node.",
        "Check the selected task type and target column.",
        "Review the technical error text if the dataset uses a custom format.",
    ]
    return diagnostic


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
        return cls(task_type=d.get("task_type", "classification"), nodes=nodes)

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
                    return False, f"Node '{n.id}' references missing input '{inp}'"

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
        """Return a new graph with the mutation applied."""
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
                    n.operation = new_op
                    if "params" in mutation:
                        n.params = mutation["params"]

        elif op == "set_params":
            nid = mutation["node_id"]
            params = mutation.get("params", {})
            for n in nodes:
                if n.id == nid:
                    n.params = {**n.params, **params}

        elif op == "connect":
            nid = mutation["node_id"]
            new_input = mutation["input_id"]
            for n in nodes:
                if n.id == nid and new_input not in n.inputs:
                    n.inputs.append(new_input)

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
    X = numeric_df.values

    if task_type in ("classification", "ts_classification"):
        if y.dtype == object:
            from sklearn.preprocessing import LabelEncoder
            y = LabelEncoder().fit_transform(y)
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

    return InputData(
        idx=np.arange(len(X)),
        features=X.astype(float),
        target=y,
        task=task,
        data_type=DataTypesEnum.table,
    )


def split_input_data(input_data: InputData, test_size: float = 0.2) -> Tuple[InputData, InputData]:
    """Train/val split for InputData."""
    n = len(input_data.features)
    cut = int(n * (1 - test_size))

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
    train = InputData(
        idx=np.arange(len(train_idx)), features=input_data.features[train_idx],
        target=input_data.target[train_idx], task=input_data.task, data_type=input_data.data_type,
    )
    val = InputData(
        idx=np.arange(len(val_idx)), features=input_data.features[val_idx],
        target=input_data.target[val_idx], task=input_data.task, data_type=input_data.data_type,
    )
    return train, val


# ============== Metrics ==============

def compute_metrics(task_type: str, y_true, y_pred) -> Dict[str, float]:
    """Compute standard metrics for the task. Returns {metric: value, primary_score: float}."""
    from sklearn import metrics as skm

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    out: Dict[str, float] = {}
    try:
        if task_type in ("classification", "ts_classification"):
            out["accuracy"] = float(skm.accuracy_score(y_true, y_pred))
            out["f1"] = float(skm.f1_score(y_true, y_pred, average="weighted", zero_division=0))
            out["precision"] = float(skm.precision_score(y_true, y_pred, average="weighted", zero_division=0))
            try:
                out["roc_auc"] = float(skm.roc_auc_score(y_true, y_pred))
            except Exception:
                pass
            out["primary_score"] = out.get("roc_auc", out["accuracy"])
        else:
            out["r2"] = float(skm.r2_score(y_true, y_pred))
            out["mse"] = float(skm.mean_squared_error(y_true, y_pred))
            out["rmse"] = float(np.sqrt(out["mse"]))
            out["mae"] = float(skm.mean_absolute_error(y_true, y_pred))
            out["primary_score"] = out["r2"]
    except Exception as e:
        out["primary_score"] = 0.0
        out["error"] = str(e)
    return out
