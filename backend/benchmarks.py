"""Benchmark registry and dataset loaders.

The UI and API talk to this module through benchmark ids. Individual benchmark
adapters can stay specialized internally while the product surface remains a
catalog that can grow beyond the first dataset.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from m4_benchmark import (
    M4_ARTIFACT_KIND,
    M4_GROUPS,
    M4_TARGET_COLUMN,
    M4_TASK_TYPE,
    is_m4_artifact,
    load_m4_artifact_data,
    load_m4_artifact_metadata,
    prepare_m4_dataset_csv,
)


M4_BENCHMARK_ID = "m4_frequency_classification"
DEFAULT_BENCHMARK_ID = M4_BENCHMARK_ID
DEFAULT_BENCHMARK_TARGET_COLUMN = M4_TARGET_COLUMN


def get_benchmark_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        M4_BENCHMARK_ID: {
            "id": M4_BENCHMARK_ID,
            "name": "M4 frequency-group classification",
            "description": (
                "Loads train series from the public M4 source and saves an optimized "
                "time-series classification artifact."
            ),
            "task_type": M4_TASK_TYPE,
            "target_column": M4_TARGET_COLUMN,
            "default_primary_metric": "f1",
            "groups": list(M4_GROUPS),
            "default_window_length": 50,
            "default_n_per_group": 100,
            "options": {
                "groups": {
                    "label": "Frequency groups",
                    "type": "multiselect",
                    "options": list(M4_GROUPS),
                    "default": list(M4_GROUPS),
                    "minimum_selected": 2,
                    "help": "At least two groups are recommended for this classification benchmark.",
                },
                "all_samples": {
                    "label": "Use all available series",
                    "type": "checkbox",
                    "default": False,
                    "help": "Load every available series from each selected group instead of limiting rows per group.",
                    "warning": "Loading all samples can create a much larger artifact and make Industrial feature extraction slower.",
                },
                "full_history": {
                    "label": "Use full available history",
                    "type": "checkbox",
                    "default": False,
                    "help": "Use the longest loaded series as the window length and left-pad shorter series.",
                    "warning": "Full history can create a much wider artifact and substantially increase training memory.",
                },
                "n_per_group": {
                    "label": "Series per group",
                    "type": "number",
                    "default": 100,
                    "min": 10,
                    "max": 1000,
                    "step": 10,
                    "disabled_when": "all_samples",
                },
                "window_length": {
                    "label": "Window length",
                    "type": "number",
                    "default": 50,
                    "min": 8,
                    "max": 500,
                    "step": 5,
                    "disabled_when": "full_history",
                },
                "standardize": {
                    "label": "Standardize each series",
                    "type": "checkbox",
                    "default": True,
                },
            },
        }
    }


def prepare_benchmark_dataset(benchmark_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    catalog = get_benchmark_catalog()
    if benchmark_id not in catalog:
        available = ", ".join(catalog)
        raise ValueError(f"Unknown benchmark '{benchmark_id}'. Available benchmarks: {available}")

    if benchmark_id == M4_BENCHMARK_ID:
        n_per_group_value = payload.get("n_per_group", 100)
        window_length_value = payload.get("window_length", 50)
        n_per_group = None if payload.get("all_samples") or n_per_group_value is None else int(n_per_group_value or 100)
        window_length = None if payload.get("full_history") or window_length_value is None else int(window_length_value or 50)
        result = prepare_m4_dataset_csv(
            n_per_group=n_per_group,
            window_length=window_length,
            standardize=bool(payload.get("standardize", True)),
            groups=payload.get("groups") or None,
        )
        result["benchmark_id"] = benchmark_id
        result["benchmark_name"] = catalog[benchmark_id]["name"]
        return result

    raise ValueError(f"Benchmark '{benchmark_id}' has no loader.")


def load_benchmark_artifact_metadata(path: str) -> Optional[Dict[str, Any]]:
    metadata = load_m4_artifact_metadata(path)
    if metadata is not None:
        metadata.setdefault("benchmark_id", M4_BENCHMARK_ID)
        metadata.setdefault("benchmark_name", get_benchmark_catalog()[M4_BENCHMARK_ID]["name"])
    return metadata


def is_benchmark_artifact(path: str) -> bool:
    return is_m4_artifact(path)


def load_benchmark_artifact_data(
    path: str,
    mmap_mode: Optional[str] = "r",
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    metadata = load_benchmark_artifact_metadata(path)
    if metadata is None:
        raise ValueError(f"Not a registered benchmark artifact manifest: {path}")
    if metadata.get("kind") == M4_ARTIFACT_KIND:
        X, y, loaded_metadata = load_m4_artifact_data(path, mmap_mode=mmap_mode)
        loaded_metadata.setdefault("benchmark_id", M4_BENCHMARK_ID)
        loaded_metadata.setdefault("benchmark_name", get_benchmark_catalog()[M4_BENCHMARK_ID]["name"])
        return X, y, loaded_metadata
    raise ValueError(f"Benchmark artifact kind is not supported: {metadata.get('kind')}")
