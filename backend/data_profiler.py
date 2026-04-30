import numpy as np
import pandas as pd
from typing import Dict, Any, List
from enum import Enum


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
            X_arr = X.select_dtypes(include=[np.number]).values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
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
            "sample_feature_ratio": round(n_samples / max(n_features, 1), 1),
        }

        # Feature statistics
        has_missing = np.isnan(X_arr).any() if X_arr.size > 0 else False
        missing_ratio = float(np.isnan(X_arr).sum() / max(X_arr.size, 1))
        std_values = np.nanstd(X_arr, axis=0)
        mean_values = np.nanmean(X_arr, axis=0)
        mean_of_means = float(np.nanmean(np.abs(mean_values))) if mean_values.size > 0 else 0
        mean_std_ratio = float(np.nanmean(std_values) / mean_of_means) if mean_of_means != 0 else 0

        profile["feature_stats"] = {
            "numeric_count": n_features,
            "mean_std_ratio": round(mean_std_ratio, 3),
            "has_missing": bool(has_missing),
            "missing_ratio": round(missing_ratio, 4),
        }

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
            if inferred_type == "classification" and unique_values < 20:
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

        profile["issues"] = issues if issues else ["none"]
        profile["recommendations"] = DataProfiler._generate_recommendations(profile)

        return profile

    @staticmethod
    def _infer_task_type(y) -> str:
        unique_count = len(np.unique(y))
        return TaskType.CLASSIFICATION.value if unique_count < 20 else TaskType.REGRESSION.value

    @staticmethod
    def _generate_recommendations(profile: Dict) -> List[str]:
        recommendations = []
        issues = profile.get("issues", [])

        for issue in issues:
            if "missing_values" in issue:
                recommendations.append("Handle missing values (imputation or removal)")
            if "class_imbalance" in issue:
                recommendations.append("Use SMOTE or class_weight balancing")
            if "feature_scaling" in issue:
                recommendations.append("Apply StandardScaler or RobustScaler")
            if "small_dataset" in issue:
                recommendations.append("Use regularization, avoid complex models")
            if "large_dataset" in issue:
                recommendations.append("Consider sampling or distributed training")
            if "high_dimensionality" in issue:
                recommendations.append("Apply PCA or feature selection")

        if profile.get("sample_feature_ratio", 0) < 10:
            recommendations.append("Low sample-to-feature ratio: prefer simpler models")

        return recommendations
