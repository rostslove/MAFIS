"""M4 benchmark data loader.

This module only loads M4 series, reshapes them into a fixed-length
classification problem (label = frequency group), and saves the result as a
CSV under the shared data dir. The rest of the pipeline (Architect, Engineer,
Critic, Evaluate) reuses the standard CSV flow exactly as it does for any
user-uploaded dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from path_utils import data_dir


M4_GROUPS: Tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
M4_TARGET_COLUMN = "frequency_group"
M4_TASK_TYPE = "ts_classification"
M4_SOURCE_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"


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


def _m4_train_url(group: str) -> str:
    return f"{M4_SOURCE_URL}/Train/{group}-train.csv"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "industrial-learning-agent/1.0"})
    try:
        with urlopen(request, timeout=120) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        tmp_path.replace(destination)
    except (OSError, URLError) as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Could not download M4 file {url}: {exc}") from exc


def _download_group_if_missing(group: str) -> None:
    csv_path = _m4_csv_path(group)
    if csv_path.exists():
        if csv_path.stat().st_size > 0:
            return
        csv_path.unlink()

    _download_file(_m4_train_url(group), csv_path)


def _read_wide_group_csv(group: str, max_rows: Optional[int]) -> np.ndarray:
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
    n_per_group: Optional[int],
    window_length: Optional[int],
    standardize: bool,
    groups: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    series_all: List[np.ndarray] = []
    y_all: List[int] = []
    source_files: List[Dict[str, Any]] = []
    series_lengths: List[int] = []

    for label, group in enumerate(groups):
        values = _read_wide_group_csv(group, max_rows=n_per_group)
        for row in values:
            series = row[~np.isnan(row)].astype(np.float32, copy=False)
            if series.size == 0:
                series = np.zeros(1, dtype=np.float32)
            series_all.append(series)
            series_lengths.append(int(series.size))
            y_all.append(label)

        csv_path = _m4_csv_path(group)
        source_files.append({
            "group": group,
            "path": str(csv_path),
            "exists": csv_path.exists(),
            "size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2) if csv_path.exists() else 0,
            "rows_read": int(values.shape[0]),
        })

    if not series_all:
        raise ValueError("No M4 rows were loaded. Check the selected groups and n_per_group.")

    if window_length is None:
        effective_window_length = max(series_lengths)
    else:
        effective_window_length = max(8, int(window_length))

    x_all = [
        _prepare_series(series, effective_window_length, standardize)
        for series in series_all
    ]
    x = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    length_info = {
        "window_length": int(effective_window_length),
        "window_length_mode": "full_history" if window_length is None else "fixed",
        "min_series_length": int(np.min(series_lengths)),
        "max_series_length": int(np.max(series_lengths)),
        "mean_series_length": round(float(np.mean(series_lengths)), 2),
    }
    return x, y, source_files, length_info


def prepare_m4_dataset_csv(
    n_per_group: Optional[int] = 100,
    window_length: Optional[int] = 50,
    standardize: bool = True,
    groups: Optional[Sequence[str]] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Download M4 if needed, build a classification CSV, return its descriptor.

    The CSV has numeric feature columns named ``f_0 .. f_{N-1}`` and a string
    ``frequency_group`` target column. Each row is one M4 series truncated or
    zero-padded to the selected fixed length. If ``window_length`` is ``None``,
    the fixed length is the longest available loaded series, so no loaded series
    is truncated.
    """
    selected_groups = _group_list(groups)
    n_rows = None if n_per_group is None or int(n_per_group) <= 0 else max(1, int(n_per_group))
    requested_window = None if window_length is None or int(window_length) <= 0 else max(8, int(window_length))

    x, y, source_files, length_info = _load_m4_matrix(
        n_rows,
        requested_window,
        bool(standardize),
        selected_groups,
    )
    length = int(length_info["window_length"])

    feature_cols = [f"f_{i}" for i in range(length)]
    df = pd.DataFrame(x, columns=feature_cols)
    df[M4_TARGET_COLUMN] = [selected_groups[label] for label in y]

    cache = _m4_cache_root()
    cache.mkdir(parents=True, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        groups_tag = "-".join(selected_groups)
        n_tag = "all" if n_rows is None else str(n_rows)
        w_tag = "full" if requested_window is None else str(length)
        filename = f"m4_{groups_tag}_n{n_tag}_w{w_tag}_{timestamp}.csv"
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
        "window_length_mode": length_info["window_length_mode"],
        "requested_window_length": requested_window,
        "n_per_group": n_rows,
        "all_samples": n_rows is None,
        "standardize": bool(standardize),
        "min_series_length": length_info["min_series_length"],
        "max_series_length": length_info["max_series_length"],
        "mean_series_length": length_info["mean_series_length"],
        "class_balance": class_balance,
        "source_files": source_files,
    }
