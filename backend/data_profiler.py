import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from enum import Enum

class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TS_FORECASTING = "ts_forecasting"

class DataProfiler:
    
    @staticmethod
    def profile(
        X: np.ndarray,
        y: np.ndarray = None,
        task_type: str = None
    ) -> Dict[str, Any]:
        
        profile = {
            "n_samples": X.shape,
            "n_features": X.shape,
            "sample_feature_ratio": X.shape / X.shape,
        }
        
        profile["feature_stats"] = {
            "numeric_count": X.shape,
            "mean_std_ratio": np.mean(np.std(X, axis=0)) / np.mean(np.mean(X, axis=0)) if np.mean(np.mean(X, axis=0)) != 0 else 0,
            "has_missing": np.isnan(X).any(),
            "missing_ratio": np.isnan(X).sum() / X.size
        }
        
        if y is not None:
            profile["target_stats"] = {
                "unique_values": len(np.unique(y)),
                "type": task_type or DataProfiler._infer_task_type(y),
                "distribution": dict(zip(*np.unique(y, return_counts=True))) if len(np.unique(y)) < 20 else {}
            }
            
            if len(np.unique(y)) < 20:
                counts = np.bincount(y.astype(int))
                ratio = np.min(counts) / np.max(counts)
                profile["is_imbalanced"] = ratio < 0.7
            
            profile["target_type"] = profile["target_stats"]["type"]
        
        issues = []
        if np.isnan(X).any():
            issues.append("missing_values")
        if y is not None and profile.get("is_imbalanced"):
            issues.append("class_imbalance")
        if profile["feature_stats"]["mean_std_ratio"] > 3:
            issues.append("feature_scaling_needed")
        
        profile["issues"] = issues
        
        profile["recommendations"] = DataProfiler._generate_recommendations(profile)
        
        return profile
    
    @staticmethod
    def _infer_task_type(y: np.ndarray) -> str:
        unique_count = len(np.unique(y))
        return TaskType.CLASSIFICATION.value if unique_count < 20 else TaskType.REGRESSION.value
    
    @staticmethod
    def _generate_recommendations(profile: Dict) -> list:
        recommendations = []

        if "missing_values" in profile.get("issues", []):
            recommendations.append("Handle missing values (imputation/removal)")
        
        if "class_imbalance" in profile.get("issues", []):
            recommendations.append("Use SMOTE or class_weight balancing")
        
        if "feature_scaling_needed" in profile.get("issues", []):
            recommendations.append("Apply StandardScaler or RobustScaler")
        
        if profile.get("sample_feature_ratio", 0) < 10:
            recommendations.append("Low sample-to-feature ratio, use regularization")
        
        return recommendations
