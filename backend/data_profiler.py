from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"


PROFILE_ROWS = 5000
PROFILE_COLUMNS = 200


def _pick(total: int, limit: int) -> np.ndarray:
    if total <= 0:
        return np.asarray([], dtype=int)
    if total <= limit:
        return np.arange(total, dtype=int)
    return np.linspace(0, total - 1, limit, dtype=int)


def _shape_issues(n_samples: int, n_features: int) -> list[str]:
    issues: list[str] = []
    if n_samples < 100:
        issues.append("small_dataset")
    elif n_samples >= 50000:
        issues.append("large_dataset")
    if n_features > n_samples:
        issues.append("high_dimensionality")
    if n_features > 1000:
        issues.append("wide_feature_matrix")
    return issues


def _csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


class DataProfiler:

    @staticmethod
    def profile(
        X,
        y=None,
        task_type: Optional[str] = None,
        *,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        feature_names: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Profile a bounded sample while preserving original dataset shape."""
        if isinstance(X, pd.DataFrame):
            all_features = feature_names or list(X.columns)
            numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
            categorical_columns = list(X.select_dtypes(exclude=[np.number]).columns)
            selected = [numeric_columns[int(idx)] for idx in _pick(len(numeric_columns), PROFILE_COLUMNS)]
            rows = _pick(len(X), PROFILE_ROWS)
            X_arr = X.iloc[rows][selected].to_numpy(dtype=float, copy=True) if selected else np.empty((len(rows), 0))
            sampled_numeric = selected[:30]
        else:
            raw = np.asarray(X)
            raw = raw.reshape(-1, 1) if raw.ndim == 1 else raw
            all_features = feature_names or [f"feature_{i}" for i in range(raw.shape[1])]
            numeric_columns = all_features
            categorical_columns: list[str] = []
            row_idx, col_idx = _pick(raw.shape[0], PROFILE_ROWS), _pick(raw.shape[1], PROFILE_COLUMNS)
            X_arr = np.asarray(raw[np.ix_(row_idx, col_idx)], dtype=float) if row_idx.size and col_idx.size else np.empty((0, 0))
            sampled_numeric = [all_features[int(idx)] for idx in col_idx[:30]]

        actual_samples = int(n_samples if n_samples is not None else getattr(X, "shape", (len(X),))[0])
        actual_features = int(n_features if n_features is not None else len(all_features))
        y_arr = np.asarray(y) if y is not None else None

        has_missing = bool(np.isnan(X_arr).any()) if X_arr.size else False
        missing_ratio = float(np.isnan(X_arr).mean()) if X_arr.size else 0.0
        means = np.nanmean(X_arr, axis=0) if X_arr.size else np.asarray([])
        stds = np.nanstd(X_arr, axis=0) if X_arr.size else np.asarray([])
        mean_abs = float(np.nanmean(np.abs(means))) if means.size else 0.0
        mean_std_ratio = float(np.nanmean(stds) / mean_abs) if mean_abs else 0.0

        feature_stats: Dict[str, Any] = {
            "numeric_count": actual_features,
            "sampled_rows": int(X_arr.shape[0]),
            "sampled_columns": int(X_arr.shape[1]),
            "sampled_numeric_columns": sampled_numeric,
            "mean_std_ratio": round(mean_std_ratio, 3),
            "has_missing": has_missing,
            "missing_ratio": round(missing_ratio, 4),
        }
        if X_arr.size:
            z = (X_arr - means) / np.where(stds > 0, stds, 1.0)
            feature_stats.update({
                "skewness_median": round(float(np.nanmedian(np.abs(np.nanmean(z ** 3, axis=0)))), 3),
                "kurtosis_median": round(float(np.nanmedian(np.nanmean(z ** 4, axis=0) - 3)), 3),
                "outlier_ratio": round(float(np.nanmean(np.abs(z) > 3)), 4),
            })
            if X_arr.shape[0] > 1 and X_arr.shape[1] > 1:
                try:
                    corr = np.abs(np.corrcoef(X_arr, rowvar=False))
                    np.fill_diagonal(corr, np.nan)
                    feature_stats["max_abs_correlation"] = round(float(np.nanmax(corr)), 3)
                    feature_stats["mean_abs_correlation"] = round(float(np.nanmean(corr)), 3)
                except Exception:
                    pass

        if isinstance(X, pd.DataFrame) and categorical_columns:
            cards = {col: int(X[col].nunique()) for col in categorical_columns}
            feature_stats.update({
                "categorical_count": len(cards),
                "max_categorical_cardinality": max(cards.values()) if cards else 0,
                "high_cardinality_categoricals": sum(value > 50 for value in cards.values()),
                "categorical_cardinalities": dict(list(cards.items())[:30]),
            })

        profile: Dict[str, Any] = {
            "n_samples": actual_samples,
            "n_features": actual_features,
            "feature_names": all_features[:20],
            "numeric_feature_names": numeric_columns[:30],
            "categorical_feature_names": categorical_columns[:30],
            "sample_feature_ratio": round(actual_samples / max(actual_features, 1), 1),
            "feature_stats": feature_stats,
        }

        if y_arr is not None:
            target = pd.Series(y_arr).astype("string").fillna("__missing__")
            counts = target.value_counts(dropna=False)
            target_type = task_type or DataProfiler._infer_task_type(y_arr)
            profile["target_stats"] = {"unique_values": int(len(counts)), "type": target_type}
            if len(counts) < 20:
                profile["target_stats"]["distribution"] = {str(k): int(v) for k, v in counts.items()}
                if target_type in ("classification", "ts_classification") and len(counts):
                    ratio = float(counts.min() / counts.max()) if counts.max() else 1.0
                    profile["is_imbalanced"] = ratio < 0.7
                    profile["imbalance_ratio"] = round(ratio, 3)

        issues = _shape_issues(actual_samples, actual_features)
        if has_missing:
            issues.append(f"missing_values ({missing_ratio:.1%})")
        if profile.get("is_imbalanced"):
            issues.append(f"class_imbalance (ratio={profile.get('imbalance_ratio', 'N/A')})")
        if mean_std_ratio > 3:
            issues.append("feature_scaling_needed")
        if feature_stats.get("skewness_median", 0) > 1.5:
            issues.append("skewed_features")
        if feature_stats.get("max_abs_correlation", 0) > 0.95:
            issues.append("multicollinearity")
        if feature_stats.get("high_cardinality_categoricals", 0) > 0:
            issues.append("high_cardinality_categoricals")
        if feature_stats.get("outlier_ratio", 0) > 0.05:
            issues.append("frequent_outliers")
        profile["issues"] = issues if issues else ["none"]
        return profile

    @staticmethod
    def profile_csv(csv_path: str | Path, target_column: str, task_type: Optional[str] = None) -> Dict[str, Any]:
        path = Path(csv_path)
        columns = [str(column) for column in pd.read_csv(path, nrows=0).columns]
        if target_column not in columns:
            raise ValueError(f"Target '{target_column}' not found")

        row_count = _csv_rows(path)
        features = [column for column in columns if column != target_column]
        selected = [features[int(idx)] for idx in _pick(len(features), PROFILE_COLUMNS)]
        sample = (
            pd.read_csv(path, usecols=selected, nrows=PROFILE_ROWS, low_memory=False)
            if selected
            else pd.DataFrame(index=range(min(row_count, PROFILE_ROWS)))
        )
        target = pd.read_csv(path, usecols=[target_column], low_memory=False)[target_column]
        return DataProfiler.profile(
            sample,
            target,
            task_type,
            n_samples=row_count,
            n_features=len(features),
            feature_names=features,
        )

    @staticmethod
    def _infer_task_type(y) -> str:
        return TaskType.CLASSIFICATION.value if pd.Series(y).nunique(dropna=False) < 20 else TaskType.REGRESSION.value
