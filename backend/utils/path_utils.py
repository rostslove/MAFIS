"""Path helpers shared by backend services running on host or in Docker."""

from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTAINER_DATA_DIR = Path("/app/data")
PROJECT_DATA_DIR = PROJECT_ROOT / "data"


def data_dir() -> Path:
    if CONTAINER_DATA_DIR.exists():
        return CONTAINER_DATA_DIR
    PROJECT_DATA_DIR.mkdir(exist_ok=True)
    return PROJECT_DATA_DIR


def normalize_csv_path(csv_path: str) -> str:
    """Map host/frontend CSV paths to the backend-visible data volume path."""
    if not csv_path:
        return csv_path

    raw = Path(csv_path)
    if raw.exists():
        return str(raw)

    candidates: List[Path] = [
        data_dir() / raw.name,
        CONTAINER_DATA_DIR / raw.name,
        PROJECT_DATA_DIR / raw.name,
    ]

    parts = list(raw.parts)
    if "data" in parts:
        idx = parts.index("data")
        rel_after_data = Path(*parts[idx + 1:]) if idx + 1 < len(parts) else Path(raw.name)
        candidates.extend([
            data_dir() / rel_after_data,
            CONTAINER_DATA_DIR / rel_after_data,
            PROJECT_DATA_DIR / rel_after_data,
        ])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def describe_missing_csv(original_path: str) -> str:
    normalized = normalize_csv_path(original_path)
    return (
        f"CSV file is not visible to backend. Received path: {original_path}. "
        f"Backend tried: {normalized}. If Docker is used, upload files into the shared ./data volume."
    )
