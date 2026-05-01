"""M4 benchmark helpers for the local GraphAutoML backend.

The benchmark follows the M4 classification utility used in the Fedot.Industrial
examples: each time series becomes one sample, and the target is the M4 frequency
group. Data is downloaded through ``datasetsforecast.m4.M4.download`` and then
read from the wide CSV files without melting them into a huge long dataframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split

from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum

from graph_engine import PipelineGraph, compute_metrics
from path_utils import data_dir


M4_GROUPS: Tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")


def _group_list(groups: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not groups:
        return M4_GROUPS
    selected = tuple(group for group in groups if group in M4_GROUPS)
    return selected or M4_GROUPS


def m4_classification_profile(
    n_per_group: int = 100,
    window_length: int = 50,
    test_size: float = 0.3,
    groups: Optional[Sequence[str]] = None,
    standardize: bool = True,
) -> Dict[str, Any]:
    selected_groups = _group_list(groups)
    n_rows = max(1, int(n_per_group))
    length = max(8, int(window_length))
    n_samples = n_rows * len(selected_groups)
    return {
        "benchmark": "M4",
        "benchmark_mode": "frequency-group ts_classification",
        "requires_architect_graph": True,
        "n_samples": n_samples,
        "n_features": length,
        "feature_shape": [n_samples, length],
        "target": "frequency_group",
        "target_type": "multiclass",
        "classes": list(selected_groups),
        "class_balance": {group: n_rows for group in selected_groups},
        "test_size": min(max(float(test_size), 0.05), 0.5),
        "standardize_each_series": bool(standardize),
        "is_time_series": True,
        "issues": [],
        "recommendations": [
            "Use ts_classification operations only.",
            "Prefer a feature extraction node followed by an industrial classifier.",
            "The graph should classify M4 frequency groups from fixed-length series windows.",
        ],
    }


def _m4_cache_root() -> Path:
    root = data_dir()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _m4_csv_path(group: str) -> Path:
    return _m4_cache_root() / "m4" / "datasets" / f"{group}-train.csv"


def _download_group_if_missing(group: str) -> None:
    csv_path = _m4_csv_path(group)
    if csv_path.exists():
        return

    from datasetsforecast.m4 import M4

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    M4.download(directory=str(_m4_cache_root()), group=group)


def _read_wide_group_csv(group: str, max_rows: int) -> np.ndarray:
    _download_group_if_missing(group)
    csv_path = _m4_csv_path(group)
    df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    id_col = df.columns[0]
    return df.drop(columns=[id_col]).to_numpy(dtype=np.float32)


def _prepare_series(row: np.ndarray, window_length: int, standardize: bool) -> np.ndarray:
    series = row[~np.isnan(row)].astype(np.float32, copy=False)
    if series.size == 0:
        series = np.zeros(1, dtype=np.float32)

    if standardize and series.size > 1:
        std = float(series.std()) or 1.0
        series = (series - float(series.mean())) / std

    if series.size < window_length:
        pad = np.zeros(window_length - series.size, dtype=np.float32)
        return np.concatenate([pad, series])
    return series[-window_length:]


def load_m4_classification(
    n_per_group: int = 100,
    window_length: int = 50,
    test_size: float = 0.3,
    random_state: int = 42,
    standardize: bool = True,
    groups: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    selected_groups = _group_list(groups)
    n_rows = max(1, int(n_per_group))
    length = max(8, int(window_length))

    x_all: List[np.ndarray] = []
    y_all: List[int] = []
    source_files: List[Dict[str, Any]] = []

    for label, group in enumerate(selected_groups):
        values = _read_wide_group_csv(group, max_rows=n_rows)
        for row in values:
            x_all.append(_prepare_series(row, length, standardize))
            y_all.append(label)

        csv_path = _m4_csv_path(group)
        source_files.append({
            "group": group,
            "path": str(csv_path),
            "exists": csv_path.exists(),
            "size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2) if csv_path.exists() else 0,
            "rows_read": int(values.shape[0]),
        })

    if not x_all:
        raise ValueError("No M4 rows were loaded. Check the selected groups and n_per_group.")

    x = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)

    stratify = y if len(np.unique(y)) > 1 and min(np.bincount(y)) > 1 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=min(max(float(test_size), 0.05), 0.5),
        random_state=int(random_state),
        stratify=stratify,
    )

    meta = {
        "groups": list(selected_groups),
        "group_labels": {str(idx): group for idx, group in enumerate(selected_groups)},
        "window_length": length,
        "n_per_group": n_rows,
        "standardize": bool(standardize),
        "source_files": source_files,
    }
    return (x_train, y_train), (x_test, y_test), meta


def _to_input_data(features: np.ndarray, target: np.ndarray) -> InputData:
    return InputData(
        idx=np.arange(len(features)),
        features=features.astype(float),
        target=target,
        task=Task(TaskTypesEnum.classification),
        data_type=DataTypesEnum.table,
    )


def _predict_labels(pipeline, data: InputData) -> np.ndarray:
    try:
        return np.asarray(pipeline.predict(data, output_mode="labels").predict)
    except Exception:
        return np.asarray(pipeline.predict(data).predict)


def _class_report(y_true: np.ndarray, y_pred: np.ndarray, groups: Sequence[str]) -> List[Dict[str, Any]]:
    labels = list(range(len(groups)))
    report = skm.classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(groups),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for group in groups:
        item = report.get(group, {}) or {}
        rows.append({
            "group": group,
            "precision": item.get("precision", 0.0),
            "recall": item.get("recall", 0.0),
            "f1": item.get("f1-score", 0.0),
            "support": item.get("support", 0.0),
        })
    return rows


def _group_counts(y_train: np.ndarray, y_test: np.ndarray, groups: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for idx, group in enumerate(groups):
        rows.append({
            "group": group,
            "train": int(np.sum(y_train == idx)),
            "test": int(np.sum(y_test == idx)),
        })
    return rows


def _validate_graph(graph: Optional[Dict[str, Any]]) -> PipelineGraph:
    if not graph:
        raise ValueError("M4 benchmark requires an Architect-approved ts_classification graph.")
    if graph.get("task_type") != "ts_classification":
        raise ValueError("M4 benchmark graph must use task_type='ts_classification'.")
    pipeline_graph = PipelineGraph.from_dict(graph)
    ok, message = pipeline_graph.validate()
    if not ok:
        raise ValueError(f"Invalid M4 benchmark graph: {message}")
    return pipeline_graph


def run_m4_classification_benchmark(
    graph: Optional[Dict[str, Any]] = None,
    n_per_group: int = 100,
    window_length: int = 50,
    test_size: float = 0.3,
    primary_metric: str = "f1",
    random_state: int = 42,
    standardize: bool = True,
    groups: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    pipeline_graph = _validate_graph(graph)
    (x_train, y_train), (x_test, y_test), meta = load_m4_classification(
        n_per_group=n_per_group,
        window_length=window_length,
        test_size=test_size,
        random_state=random_state,
        standardize=standardize,
        groups=groups,
    )

    train_data = _to_input_data(x_train, y_train)
    test_data = _to_input_data(x_test, y_test)

    pipeline = pipeline_graph.to_fedot_pipeline()
    pipeline.fit(train_data)

    train_preds = _predict_labels(pipeline, train_data)
    test_preds = _predict_labels(pipeline, test_data)
    train_metrics = compute_metrics("ts_classification", y_train, train_preds, primary_metric)
    test_metrics = compute_metrics("ts_classification", y_test, test_preds, primary_metric)

    groups_used = meta["groups"]
    return {
        "benchmark": "M4",
        "mode": "frequency-group ts_classification",
        "task_type": "ts_classification",
        "score": test_metrics.get("primary_score", 0.0),
        "metrics": test_metrics,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "split_info": {
            "test_size": min(max(float(test_size), 0.05), 0.5),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "dataset": meta,
        "graph": pipeline_graph.to_dict(),
        "class_report": _class_report(y_test, test_preds, groups_used),
        "group_counts": _group_counts(y_train, y_test, groups_used),
    }
