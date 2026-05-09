import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from enum import Enum


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


DEFAULT_PROFILE_SAMPLE_ROWS = max(100, _int_env("DATA_PROFILE_SAMPLE_ROWS", 5000))
DEFAULT_PROFILE_SAMPLE_COLUMNS = max(20, _int_env("DATA_PROFILE_SAMPLE_COLUMNS", 200))
DEFAULT_TARGET_CHUNK_ROWS = max(1000, _int_env("DATA_PROFILE_TARGET_CHUNK_ROWS", 50000))


def _evenly_spaced(items: Sequence[Any], limit: int) -> List[Any]:
    values = list(items)
    if limit <= 0 or len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit, dtype=int)
    unique_indices = []
    seen = set()
    for idx in indices:
        idx_int = int(idx)
        if idx_int not in seen:
            unique_indices.append(idx_int)
            seen.add(idx_int)
    return [values[idx] for idx in unique_indices]


def _sample_indices(n_rows: int, sample_rows: int) -> np.ndarray:
    if n_rows <= 0:
        return np.asarray([], dtype=int)
    if n_rows <= sample_rows:
        return np.arange(n_rows, dtype=int)
    return np.linspace(0, n_rows - 1, sample_rows, dtype=int)


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _clean_issues(issues: List[str]) -> List[str]:
    return [issue for issue in issues if issue and issue != "none"]


def _apply_shape_issues(profile: Dict[str, Any], n_samples: int, n_features: int) -> None:
    shape_prefixes = (
        "small_dataset",
        "large_dataset",
        "high_dimensionality",
        "wide_feature_matrix",
    )
    issues = [
        issue
        for issue in _clean_issues(list(profile.get("issues", []) or []))
        if not any(str(issue).startswith(prefix) for prefix in shape_prefixes)
    ]
    if n_samples < 100:
        issues.append("small_dataset")
    elif n_samples >= 50000:
        issues.append("large_dataset")
    if n_features > n_samples:
        issues.append("high_dimensionality")
    if n_features > 1000:
        issues.append("wide_feature_matrix")
    profile["issues"] = issues if issues else ["none"]


def _apply_target_issues(profile: Dict[str, Any]) -> None:
    issues = [
        issue
        for issue in _clean_issues(list(profile.get("issues", []) or []))
        if not str(issue).startswith("class_imbalance")
    ]
    if profile.get("is_imbalanced"):
        issues.append(f"class_imbalance (ratio={profile.get('imbalance_ratio', 'N/A')})")
    profile["issues"] = issues if issues else ["none"]


class DataProfiler:

    @staticmethod
    def profile(
        X,
        y=None,
        task_type: str = None,
        sample_rows: int = DEFAULT_PROFILE_SAMPLE_ROWS,
        sample_columns: int = DEFAULT_PROFILE_SAMPLE_COLUMNS,
    ) -> Dict[str, Any]:
        """Profile a dataset without materializing large derived arrays.

        For large/wide matrices, expensive statistics are computed on an
        evenly-spaced sample of rows and columns. The reported shape still
        describes the original dataset.
        """
        sample_rows = max(1, int(sample_rows or DEFAULT_PROFILE_SAMPLE_ROWS))
        sample_columns = max(1, int(sample_columns or DEFAULT_PROFILE_SAMPLE_COLUMNS))

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
            categorical_columns = list(X.select_dtypes(exclude=[np.number]).columns)
            n_samples = len(X)
            n_features = len(feature_names)
            sampled_numeric_columns = _evenly_spaced(numeric_columns, sample_columns)
            row_idx = _sample_indices(n_samples, sample_rows)
            if sampled_numeric_columns:
                X_arr = X.iloc[row_idx][sampled_numeric_columns].to_numpy(dtype=float, copy=True)
            else:
                X_arr = np.empty((len(row_idx), 0))
        else:
            raw = X
            raw_shape = getattr(raw, "shape", None)
            if raw_shape is None:
                raw = np.asarray(raw)
                raw_shape = raw.shape
            if len(raw_shape) == 1:
                n_samples, n_features = int(raw_shape[0]), 1
                raw_2d = np.asarray(raw).reshape(-1, 1)
            else:
                n_samples, n_features = int(raw_shape[0]), int(raw_shape[1])
                raw_2d = raw
            feature_names = [f"feature_{i}" for i in range(n_features)]
            numeric_columns = feature_names
            categorical_columns = []
            sampled_col_indices = [
                int(name.rsplit("_", 1)[-1])
                for name in _evenly_spaced(feature_names, sample_columns)
            ]
            row_idx = _sample_indices(n_samples, sample_rows)
            if len(row_idx) and sampled_col_indices:
                X_arr = np.asarray(raw_2d[np.ix_(row_idx, sampled_col_indices)], dtype=float)
            else:
                X_arr = np.empty((len(row_idx), 0))
            sampled_numeric_columns = [feature_names[idx] for idx in sampled_col_indices]

        if isinstance(y, pd.Series):
            y_arr = y.values
        elif y is not None:
            y_arr = np.asarray(y)
        else:
            y_arr = None

        profile = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names[:20],  # first 20 for display
            "numeric_feature_names": numeric_columns[:30],
            "categorical_feature_names": categorical_columns[:30],
            "sample_feature_ratio": round(n_samples / max(n_features, 1), 1),
        }

        sampled_rows, sampled_features = X_arr.shape
        has_missing = np.isnan(X_arr).any() if X_arr.size > 0 else False
        missing_ratio = float(np.isnan(X_arr).sum() / max(X_arr.size, 1))
        if X_arr.size:
            with np.errstate(invalid="ignore", divide="ignore"):
                std_values = np.nanstd(X_arr, axis=0)
                mean_values = np.nanmean(X_arr, axis=0)
            mean_of_means = float(np.nanmean(np.abs(mean_values))) if mean_values.size > 0 else 0
            mean_std_ratio = float(np.nanmean(std_values) / mean_of_means) if mean_of_means != 0 else 0
        else:
            std_values = np.asarray([])
            mean_values = np.asarray([])
            mean_std_ratio = 0.0

        feature_stats = {
            "numeric_count": n_features,
            "sampled_rows": int(sampled_rows),
            "sampled_columns": int(sampled_features),
            "sampled_numeric_columns": sampled_numeric_columns[:30],
            "mean_std_ratio": round(mean_std_ratio, 3),
            "has_missing": bool(has_missing),
            "missing_ratio": round(missing_ratio, 4),
        }

        if X_arr.size > 0 and n_features > 0:
            with np.errstate(invalid="ignore", divide="ignore"):
                centered = X_arr - mean_values
                std_safe = np.where(std_values > 0, std_values, 1.0)
                z = centered / std_safe
                skew = np.nanmean(z ** 3, axis=0)
                kurt = np.nanmean(z ** 4, axis=0) - 3
            feature_stats["skewness_median"] = round(float(np.nanmedian(np.abs(skew))), 3)
            feature_stats["kurtosis_median"] = round(float(np.nanmedian(kurt)), 3)

            try:
                outlier_mask = np.abs(z) > 3
                feature_stats["outlier_ratio"] = round(float(np.nanmean(outlier_mask)), 4)
            except Exception:
                feature_stats["outlier_ratio"] = 0.0

            if sampled_features > 1 and sampled_rows > 1:
                try:
                    corr = np.corrcoef(X_arr, rowvar=False)
                    abs_corr = np.abs(corr)
                    np.fill_diagonal(abs_corr, np.nan)
                    feature_stats["max_abs_correlation"] = round(float(np.nanmax(abs_corr)), 3)
                    feature_stats["mean_abs_correlation"] = round(float(np.nanmean(abs_corr)), 3)
                except Exception:
                    pass

        if isinstance(X, pd.DataFrame):
            non_numeric = X.select_dtypes(exclude=[np.number])
            if not non_numeric.empty:
                cardinalities = {col: int(non_numeric[col].nunique()) for col in non_numeric.columns}
                feature_stats["categorical_count"] = len(cardinalities)
                feature_stats["max_categorical_cardinality"] = max(cardinalities.values()) if cardinalities else 0
                feature_stats["high_cardinality_categoricals"] = sum(1 for c in cardinalities.values() if c > 50)
                feature_stats["categorical_cardinalities"] = dict(list(cardinalities.items())[:30])

        profile["feature_stats"] = feature_stats

        # Target statistics
        if y_arr is not None:
            unique_values = len(np.unique(y_arr))
            inferred_type = task_type or DataProfiler._infer_task_type(y_arr)

            target_stats = {
                "unique_values": unique_values,
                "type": inferred_type,
            }

            if unique_values < 20:
                vals, counts = np.unique(y_arr, return_counts=True)
                target_stats["distribution"] = {str(v): int(c) for v, c in zip(vals, counts)}

            profile["target_stats"] = target_stats

            # Check imbalance for classification
            if inferred_type in ("classification", "ts_classification") and unique_values < 20:
                try:
                    vals, counts = np.unique(y_arr, return_counts=True)
                    ratio = float(counts.min() / counts.max())
                    profile["is_imbalanced"] = ratio < 0.7
                    profile["imbalance_ratio"] = round(ratio, 3)
                except Exception:
                    profile["is_imbalanced"] = False

        # Detect issues
        issues = []
        if has_missing:
            issues.append(f"missing_values ({missing_ratio:.1%})")
        if profile.get("is_imbalanced"):
            issues.append(f"class_imbalance (ratio={profile.get('imbalance_ratio', 'N/A')})")
        if mean_std_ratio > 3:
            issues.append("feature_scaling_needed")
        if profile.get("feature_stats", {}).get("skewness_median", 0) > 1.5:
            issues.append("skewed_features")
        if profile.get("feature_stats", {}).get("max_abs_correlation", 0) > 0.95:
            issues.append("multicollinearity")
        if profile.get("feature_stats", {}).get("high_cardinality_categoricals", 0) > 0:
            issues.append("high_cardinality_categoricals")
        if profile.get("feature_stats", {}).get("outlier_ratio", 0) > 0.05:
            issues.append("frequent_outliers")

        profile["issues"] = issues if issues else ["none"]
        _apply_shape_issues(profile, n_samples, n_features)

        return profile

    @staticmethod
    def _infer_task_type(y) -> str:
        unique_count = len(np.unique(y))
        return TaskType.CLASSIFICATION.value if unique_count < 20 else TaskType.REGRESSION.value

    @staticmethod
    def profile_csv(
        csv_path: str | Path,
        target_column: str,
        task_type: Optional[str] = None,
        sample_rows: int = DEFAULT_PROFILE_SAMPLE_ROWS,
        sample_columns: int = DEFAULT_PROFILE_SAMPLE_COLUMNS,
    ) -> Dict[str, Any]:
        """Profile a CSV by reading only a bounded feature sample.

        This avoids loading benchmark-sized wide CSVs into backend memory while
        preserving the exact row/column shape and target distribution whenever
        the target cardinality is small enough to track.
        """
        path = Path(csv_path)
        header = pd.read_csv(path, nrows=0)
        columns = [str(column) for column in header.columns]
        if target_column not in columns:
            raise ValueError(f"Target '{target_column}' not found")

        n_samples = _csv_row_count(path)
        feature_columns = [column for column in columns if column != target_column]
        selected_features = _evenly_spaced(feature_columns, int(sample_columns or DEFAULT_PROFILE_SAMPLE_COLUMNS))
        sample_usecols = selected_features + [target_column]
        sample_df = pd.read_csv(
            path,
            usecols=sample_usecols,
            nrows=max(1, int(sample_rows or DEFAULT_PROFILE_SAMPLE_ROWS)),
            low_memory=False,
        )
        sample_x = sample_df[selected_features] if selected_features else pd.DataFrame(index=sample_df.index)
        sample_y = sample_df[target_column] if target_column in sample_df.columns else None
        profile = DataProfiler.profile(
            X=sample_x,
            y=sample_y,
            task_type=task_type,
            sample_rows=sample_rows,
            sample_columns=sample_columns,
        )
        profile.update({
            "n_samples": int(n_samples),
            "n_features": int(len(feature_columns)),
            "feature_names": feature_columns[:20],
            "sample_feature_ratio": round(n_samples / max(len(feature_columns), 1), 1),
            "profile_mode": "csv_sample",
        })
        profile["feature_stats"] = {
            **dict(profile.get("feature_stats", {}) or {}),
            "numeric_count": int(len(feature_columns)),
            "sampled_rows": int(len(sample_df)),
            "sampled_columns": int(len(selected_features)),
        }

        target_profile = DataProfiler._profile_csv_target(path, target_column, task_type)
        if target_profile:
            profile.update(target_profile)

        _apply_target_issues(profile)
        _apply_shape_issues(profile, n_samples, len(feature_columns))
        return profile

    @staticmethod
    def _profile_csv_target(
        csv_path: Path,
        target_column: str,
        task_type: Optional[str],
    ) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        overflow_count = 0
        max_tracked = 1000
        sample_values: List[str] = []

        for chunk in pd.read_csv(
            csv_path,
            usecols=[target_column],
            chunksize=DEFAULT_TARGET_CHUNK_ROWS,
            low_memory=False,
        ):
            series = chunk[target_column]
            if len(sample_values) < 10:
                sample_values.extend(str(value) for value in series.dropna().head(10 - len(sample_values)).tolist())
            value_counts = series.astype("string").fillna("__missing__").value_counts(dropna=False)
            for raw_value, raw_count in value_counts.items():
                key = str(raw_value)
                count = int(raw_count)
                if key in counts or len(counts) < max_tracked:
                    counts[key] = counts.get(key, 0) + count
                else:
                    overflow_count += count

        unique_values = len(counts) + (1 if overflow_count else 0)
        inferred_type = task_type or (TaskType.CLASSIFICATION.value if unique_values < 20 else TaskType.REGRESSION.value)
        target_stats: Dict[str, Any] = {
            "unique_values": int(unique_values),
            "type": inferred_type,
            "sample_values": sample_values,
        }
        exact_small_distribution = not overflow_count and unique_values < 20
        if exact_small_distribution:
            target_stats["distribution"] = dict(counts)

        result: Dict[str, Any] = {"target_stats": target_stats}
        if inferred_type in ("classification", "ts_classification") and exact_small_distribution and counts:
            count_values = np.asarray(list(counts.values()), dtype=float)
            ratio = float(count_values.min() / count_values.max()) if count_values.max() else 1.0
            result["is_imbalanced"] = ratio < 0.7
            result["imbalance_ratio"] = round(ratio, 3)
        return result
