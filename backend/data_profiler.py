from enum import Enum
from typing import Any, Dict

import numpy as np
import pandas as pd


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"


class DataProfiler:

    @staticmethod
    def profile(
        X,
        y=None,
        task_type: str = None,
    ) -> Dict[str, Any]:
        """Profile dataset, accepting both numpy arrays and pandas DataFrames."""
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
            numeric_columns = list(X.select_dtypes(include=[np.number]).columns)
            categorical_columns = list(X.select_dtypes(exclude=[np.number]).columns)
            X_arr = X[numeric_columns].values if numeric_columns else np.empty((len(X), 0))
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            numeric_columns = feature_names
            categorical_columns = []
            X_arr = np.asarray(X, dtype=float)

        if isinstance(y, pd.Series):
            y_arr = y.values
        elif y is not None:
            y_arr = np.asarray(y)
        else:
            y_arr = None

        n_samples, n_features = X_arr.shape

        profile = {
            "n_samples": n_samples,
            "n_features": n_features,
            "feature_names": feature_names[:20],  # first 20 for display
            "numeric_feature_names": numeric_columns[:30],
            "categorical_feature_names": categorical_columns[:30],
            "sample_feature_ratio": round(n_samples / max(n_features, 1), 1),
        }

        # Feature statistics
        has_missing = np.isnan(X_arr).any() if X_arr.size > 0 else False
        missing_ratio = float(np.isnan(X_arr).sum() / max(X_arr.size, 1))
        std_values = np.nanstd(X_arr, axis=0)
        mean_values = np.nanmean(X_arr, axis=0)
        mean_of_means = float(np.nanmean(np.abs(mean_values))) if mean_values.size > 0 else 0
        mean_std_ratio = float(np.nanmean(std_values) / mean_of_means) if mean_of_means != 0 else 0

        feature_stats = {
            "numeric_count": n_features,
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

            if n_features > 1 and n_samples > 1:
                try:
                    sample = X_arr if n_samples <= 5000 else X_arr[
                        np.random.default_rng(0).choice(n_samples, 5000, replace=False)
                    ]
                    corr = np.corrcoef(sample, rowvar=False)
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
        if n_samples < 100:
            issues.append("small_dataset")
        elif n_samples > 100000:
            issues.append("large_dataset")
        if n_features > n_samples:
            issues.append("high_dimensionality")
        if profile.get("feature_stats", {}).get("skewness_median", 0) > 1.5:
            issues.append("skewed_features")
        if profile.get("feature_stats", {}).get("max_abs_correlation", 0) > 0.95:
            issues.append("multicollinearity")
        if profile.get("feature_stats", {}).get("high_cardinality_categoricals", 0) > 0:
            issues.append("high_cardinality_categoricals")
        if profile.get("feature_stats", {}).get("outlier_ratio", 0) > 0.05:
            issues.append("frequent_outliers")

        profile["issues"] = issues if issues else ["none"]

        return profile

    @staticmethod
    def _infer_task_type(y) -> str:
        unique_count = len(np.unique(y))
        return TaskType.CLASSIFICATION.value if unique_count < 20 else TaskType.REGRESSION.value
