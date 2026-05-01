"""M4 benchmark data loader.

This module only loads M4 series, reshapes them into a fixed-window
classification problem (label = frequency group), and saves the result as a
CSV under the shared data dir. The rest of the pipeline (Architect, Engineer,
Critic, Evaluate) reuses the standard CSV flow exactly as it does for any
user-uploaded dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from path_utils import data_dir


M4_GROUPS: Tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
M4_TARGET_COLUMN = "frequency_group"
M4_TASK_TYPE = "ts_classification"


def _group_list(groups: Optional[Sequence[str]]) -> Tuple[str, ...]:
    if not groups:
        return M4_GROUPS
    selected = tuple(group for group in groups if group in M4_GROUPS)
    return selected or M4_GROUPS


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


def _load_m4_matrix(
    n_per_group: int,
    window_length: int,
    standardize: bool,
    groups: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    x_all: List[np.ndarray] = []
    y_all: List[int] = []
    source_files: List[Dict[str, Any]] = []

    for label, group in enumerate(groups):
        values = _read_wide_group_csv(group, max_rows=n_per_group)
        for row in values:
            x_all.append(_prepare_series(row, window_length, standardize))
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
    return x, y, source_files


def prepare_m4_dataset_csv(
    n_per_group: int = 100,
    window_length: int = 50,
    standardize: bool = True,
    groups: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Download M4 if needed, build a classification CSV, return its descriptor.

    The CSV has ``window_length`` numeric feature columns named ``f_0 .. f_{N-1}``
    and a string ``frequency_group`` target column. Each row is one M4 series
    truncated/zero-padded to the window. Standard CSV flow takes over from here.
    """
    selected_groups = _group_list(groups)
    n_rows = max(1, int(n_per_group))
    length = max(8, int(window_length))

    x, y, source_files = _load_m4_matrix(n_rows, length, bool(standardize), selected_groups)

    feature_cols = [f"f_{i}" for i in range(length)]
    df = pd.DataFrame(x, columns=feature_cols)
    df[M4_TARGET_COLUMN] = [selected_groups[label] for label in y]

    cache = _m4_cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        groups_tag = "-".join(selected_groups)
        filename = f"m4_{groups_tag}_n{n_rows}_w{length}_{timestamp}.csv"
    csv_path = cache / filename
    df.to_csv(csv_path, index=False)

    class_balance = {group: int(np.sum(y == idx)) for idx, group in enumerate(selected_groups)}

    return {
        "csv_path": str(csv_path),
        "csv_filename": csv_path.name,
        "target_column": M4_TARGET_COLUMN,
        "task_type": M4_TASK_TYPE,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "feature_columns": feature_cols,
        "groups": list(selected_groups),
        "group_labels": {str(idx): group for idx, group in enumerate(selected_groups)},
        "window_length": length,
        "n_per_group": n_rows,
        "standardize": bool(standardize),
        "class_balance": class_balance,
        "source_files": source_files,
    }
