"""M4 benchmark data loader.

This module loads M4 series, reshapes them into a fixed-length classification
problem (label = frequency group), and stores the result as a lightweight
manifest plus ``.npy`` arrays under the shared data dir. A tiny preview CSV is
created for the UI; training still reads the full matrix through the manifest.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from utils.path_utils import data_dir


M4_GROUPS: Tuple[str, ...] = ("Daily", "Weekly", "Monthly", "Quarterly", "Yearly")
M4_TARGET_COLUMN = "frequency_group"
M4_TASK_TYPE = "ts_classification"
M4_SOURCE_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
M4_ARTIFACT_KIND = "mafis_m4_numpy_v1"
M4_PREVIEW_ROWS = 10
M4_PREVIEW_COLS = 10
M4_CHUNK_ROWS = max(64, int(os.getenv("M4_CHUNK_ROWS", "512")))


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


def _iter_wide_group_chunks(
    group: str,
    max_rows: Optional[int],
    chunksize: int = M4_CHUNK_ROWS,
):
    _download_group_if_missing(group)
    csv_path = _m4_csv_path(group)
    reader = pd.read_csv(
        csv_path,
        nrows=max_rows,
        chunksize=chunksize,
        low_memory=False,
    )
    for chunk in reader:
        id_col = chunk.columns[0]
        yield chunk.drop(columns=[id_col]).to_numpy(dtype=np.float32, copy=True)


def _series_lengths(values: np.ndarray) -> np.ndarray:
    lengths = np.count_nonzero(~np.isnan(values), axis=1).astype(np.int64, copy=False)
    lengths[lengths == 0] = 1
    return lengths


def _prepare_series_batch(values: np.ndarray, window_length: int, standardize: bool) -> np.ndarray:
    return np.stack([
        _prepare_series(row, window_length, standardize)
        for row in values
    ]).astype(np.float32, copy=False)


def _scan_m4_groups(
    n_per_group: Optional[int],
    groups: Sequence[str],
) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any], Dict[str, int]]:
    total_rows = 0
    min_len: Optional[int] = None
    max_len = 0
    sum_len = 0
    source_files: List[Dict[str, Any]] = []
    class_balance: Dict[str, int] = {}

    for group in groups:
        rows_read = 0
        for values in _iter_wide_group_chunks(group, max_rows=n_per_group):
            lengths = _series_lengths(values)
            rows_read += int(values.shape[0])
            total_rows += int(values.shape[0])
            batch_min = int(lengths.min()) if lengths.size else 0
            batch_max = int(lengths.max()) if lengths.size else 0
            min_len = batch_min if min_len is None else min(min_len, batch_min)
            max_len = max(max_len, batch_max)
            sum_len += int(lengths.sum())

        csv_path = _m4_csv_path(group)
        class_balance[group] = rows_read
        source_files.append({
            "group": group,
            "path": str(csv_path),
            "exists": csv_path.exists(),
            "size_mb": round(csv_path.stat().st_size / (1024 * 1024), 2) if csv_path.exists() else 0,
            "rows_read": rows_read,
        })

    if total_rows == 0:
        raise ValueError("No M4 rows were loaded. Check the selected groups and n_per_group.")

    length_info = {
        "min_series_length": int(min_len or 1),
        "max_series_length": int(max_len or 1),
        "mean_series_length": round(float(sum_len / total_rows), 2),
    }
    return total_rows, source_files, length_info, class_balance


def _resolve_artifact_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.exists():
        return raw
    candidates = [
        _m4_cache_root() / raw.name,
        _m4_cache_root() / "m4" / "artifacts" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return raw


def load_m4_artifact_metadata(path: str | Path) -> Optional[Dict[str, Any]]:
    manifest_path = _resolve_artifact_path(path)
    if manifest_path.suffix.lower() != ".json" or not manifest_path.exists():
        return None
    try:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metadata.get("kind") != M4_ARTIFACT_KIND:
        return None
    metadata["_manifest_path"] = str(manifest_path)
    return metadata


def is_m4_artifact(path: str | Path) -> bool:
    return load_m4_artifact_metadata(path) is not None


def _artifact_member_path(metadata: Dict[str, Any], key: str) -> Path:
    manifest_path = Path(metadata.get("_manifest_path") or metadata.get("artifact_path") or "")
    base_dir = manifest_path.parent if manifest_path else _m4_cache_root()
    value = Path(str(metadata[key]))
    if value.exists():
        return value
    return base_dir / value.name


def load_m4_artifact_data(
    path: str | Path,
    mmap_mode: Optional[str] = "r",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    metadata = load_m4_artifact_metadata(path)
    if metadata is None:
        raise ValueError(f"Not an M4 optimized artifact manifest: {path}")
    x_path = _artifact_member_path(metadata, "x_path")
    y_path = _artifact_member_path(metadata, "y_path")
    X = np.load(x_path, mmap_mode=mmap_mode)
    y = np.load(y_path, mmap_mode=mmap_mode)
    return X, y, metadata


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
    """Download M4 if needed, build an optimized classification artifact.

    The artifact stores a float32 ``.npy`` feature matrix, a target ``.npy``,
    a JSON manifest, and a tiny preview CSV. Each row is one M4 series truncated
    or zero-padded to the selected fixed length. If ``window_length`` is
    ``None``, the fixed length is the longest available loaded series, so no
    loaded series is truncated.
    """
    selected_groups = _group_list(groups)
    n_rows = None if n_per_group is None or int(n_per_group) <= 0 else max(1, int(n_per_group))
    requested_window = None if window_length is None or int(window_length) <= 0 else max(8, int(window_length))

    total_rows, source_files, scan_info, class_balance = _scan_m4_groups(
        n_rows,
        selected_groups,
    )
    length = int(scan_info["max_series_length"] if requested_window is None else requested_window)

    cache = _m4_cache_root() / "m4" / "artifacts"
    cache.mkdir(parents=True, exist_ok=True)
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        groups_tag = "-".join(selected_groups)
        n_tag = "all" if n_rows is None else str(n_rows)
        w_tag = "full" if requested_window is None else str(length)
        filename = f"m4_{groups_tag}_n{n_tag}_w{w_tag}_{timestamp}"
    stem = Path(filename).stem
    manifest_path = cache / f"{stem}.json"
    x_path = cache / f"{stem}_X.npy"
    y_path = cache / f"{stem}_y.npy"
    preview_path = cache / f"{stem}_preview.csv"

    x_mem = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(total_rows, length),
    )
    y_mem = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype="<U32",
        shape=(total_rows,),
    )

    # Shuffle once while writing so the train/test split can use contiguous
    # views instead of copying several gigabytes with fancy indexing.
    output_order = np.random.default_rng(42).permutation(total_rows)
    source_offset = 0
    for group in selected_groups:
        for values in _iter_wide_group_chunks(group, max_rows=n_rows):
            batch = _prepare_series_batch(values, length, bool(standardize))
            batch_size = int(batch.shape[0])
            output_idx = output_order[source_offset:source_offset + batch_size]
            x_mem[output_idx] = batch
            y_mem[output_idx] = group
            source_offset += batch_size

    x_mem.flush()
    y_mem.flush()

    preview_width = min(M4_PREVIEW_COLS, length)
    preview_cols = [f"f_{i}" for i in range(preview_width)]
    preview_df = pd.DataFrame(
        np.asarray(x_mem[:M4_PREVIEW_ROWS, :preview_width]),
        columns=preview_cols,
    )
    preview_df[M4_TARGET_COLUMN] = np.asarray(y_mem[:M4_PREVIEW_ROWS]).astype(str)
    preview_df.to_csv(preview_path, index=False)

    feature_cols = [f"f_{i}" for i in range(length)]
    metadata = {
        "kind": M4_ARTIFACT_KIND,
        "storage_format": M4_ARTIFACT_KIND,
        "artifact_path": str(manifest_path),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "preview_csv_path": str(preview_path),
        "target_column": M4_TARGET_COLUMN,
        "task_type": M4_TASK_TYPE,
        "n_samples": int(total_rows),
        "n_features": int(length),
        "dtype": "float32",
        "feature_columns_preview": feature_cols[:100],
        "groups": list(selected_groups),
        "group_labels": {str(idx): group for idx, group in enumerate(selected_groups)},
        "window_length": length,
        "window_length_mode": "full_history" if requested_window is None else "fixed",
        "requested_window_length": requested_window,
        "n_per_group": n_rows,
        "all_samples": n_rows is None,
        "standardize": bool(standardize),
        "min_series_length": scan_info["min_series_length"],
        "max_series_length": scan_info["max_series_length"],
        "mean_series_length": scan_info["mean_series_length"],
        "class_balance": class_balance,
        "source_files": source_files,
        "shuffled": True,
    }
    manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "csv_path": str(manifest_path),
        "artifact_path": str(manifest_path),
        "csv_filename": manifest_path.name,
        "preview_csv_path": str(preview_path),
        "target_column": M4_TARGET_COLUMN,
        "task_type": M4_TASK_TYPE,
        "n_samples": int(total_rows),
        "n_features": int(length),
        "feature_columns": feature_cols[:100],
        "feature_columns_preview": feature_cols[:100],
        "feature_columns_truncated": len(feature_cols) > 100,
        "groups": list(selected_groups),
        "group_labels": {str(idx): group for idx, group in enumerate(selected_groups)},
        "window_length": length,
        "window_length_mode": "full_history" if requested_window is None else "fixed",
        "requested_window_length": requested_window,
        "n_per_group": n_rows,
        "all_samples": n_rows is None,
        "standardize": bool(standardize),
        "min_series_length": scan_info["min_series_length"],
        "max_series_length": scan_info["max_series_length"],
        "mean_series_length": scan_info["mean_series_length"],
        "class_balance": class_balance,
        "source_files": source_files,
        "storage_format": M4_ARTIFACT_KIND,
    }
