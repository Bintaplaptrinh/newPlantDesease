from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.json_io import write_json
from src.utils.paths import project_paths


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _short_label(full: str) -> str:
    # "Tomato___Bacterial_spot" -> "Tomato Bacterial spot"
    return full.replace("___", " ").replace("_", " ").strip()


@dataclass(frozen=True)
class WebArtifactsPaths:
    base_dir: Path

    @staticmethod
    def default() -> "WebArtifactsPaths":
        p = project_paths()
        return WebArtifactsPaths(base_dir=p.data_dir / "web")

    @property
    def classes_json(self) -> Path:
        return self.base_dir / "classes.json"

    @property
    def dataset_stats_json(self) -> Path:
        return self.base_dir / "dataset_stats.json"

    @property
    def class_distribution_json(self) -> Path:
        return self.base_dir / "class_distribution.json"

    @property
    def models_json(self) -> Path:
        return self.base_dir / "models.json"

    def model_dir(self, model_id: str) -> Path:
        return self.base_dir / "models" / model_id

    def model_confusion_json(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "confusion_matrix.json"

    def model_training_history_json(self, model_id: str) -> Path:
        return self.model_dir(model_id) / "training_history.json"

    @property
    def roc_micro_json(self) -> Path:
        return self.base_dir / "roc_micro.json"


def write_classes(class_labels: List[str], *, paths: Optional[WebArtifactsPaths] = None) -> None:
    paths = paths or WebArtifactsPaths.default()
    payload = {
        "generatedAt": _utc_now_iso(),
        "labels": class_labels,
        "shortLabels": [_short_label(c) for c in class_labels],
        "numClasses": len(class_labels),
    }
    write_json(paths.classes_json, payload)


def write_dataset_stats(
    *,
    total_images: int,
    train_images: int,
    validation_images: int,
    test_images: int = 0,
    num_classes: int,
    extra: Optional[Dict[str, Any]] = None,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()
    payload: Dict[str, Any] = {
        "generatedAt": _utc_now_iso(),
        "totalImages": int(total_images),
        "trainImages": int(train_images),
        "validationImages": int(validation_images),
        "testImages": int(test_images),
        "numClasses": int(num_classes),
    }
    if extra:
        payload.update(extra)
    write_json(paths.dataset_stats_json, payload)


def write_class_distribution(
    class_labels: List[str],
    counts: List[int] | np.ndarray,
    *,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()
    counts_arr = np.asarray(counts).astype(int)

    items = []
    for label, count in zip(class_labels, counts_arr.tolist()):
        items.append({"label": label, "shortLabel": _short_label(label), "count": int(count)})

    payload = {
        "generatedAt": _utc_now_iso(),
        "items": items,
        "totalImages": int(counts_arr.sum()),
        "numClasses": len(class_labels),
    }
    write_json(paths.class_distribution_json, payload)


def write_models_index(
    models: List[Dict[str, Any]],
    *,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()

    existing_models_by_id: Dict[str, Dict[str, Any]] = {}
    if paths.models_json.exists():
        try:
            with paths.models_json.open("r", encoding="utf-8") as f:
                existing_payload = json.load(f)
            existing_models = existing_payload.get("models")
            if isinstance(existing_models, list):
                for m in existing_models:
                    if isinstance(m, dict) and isinstance(m.get("id"), str):
                        existing_models_by_id[m["id"]] = m
        except Exception:
            # If file is corrupted, fall back to writing the provided index.
            existing_models_by_id = {}

    merged_models: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    for m in models:
        if not isinstance(m, dict) or not isinstance(m.get("id"), str):
            continue
        mid = m["id"]
        seen_ids.add(mid)
        prev = existing_models_by_id.get(mid, {})

        merged = {**prev, **m}
        # Never overwrite real values with placeholders.
        if m.get("metrics") is None and prev.get("metrics") is not None:
            merged["metrics"] = prev.get("metrics")
        if m.get("hyperparameters") is None and prev.get("hyperparameters") is not None:
            merged["hyperparameters"] = prev.get("hyperparameters")

        merged_models.append(merged)

    # Preserve any extra models that existed previously.
    for mid, prev in existing_models_by_id.items():
        if mid not in seen_ids:
            merged_models.append(prev)

    payload = {
        "generatedAt": _utc_now_iso(),
        "models": merged_models,
    }
    write_json(paths.models_json, payload)


def write_confusion_matrix(
    model_id: str,
    labels: List[str],
    matrix: List[List[int]] | np.ndarray,
    *,
    normalized: bool = False,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()
    mat = np.asarray(matrix)
    payload = {
        "generatedAt": _utc_now_iso(),
        "modelId": model_id,
        "labels": labels,
        "matrix": mat.astype(int).tolist(),
        "normalized": bool(normalized),
    }
    write_json(paths.model_confusion_json(model_id), payload)


def write_training_history(
    model_id: str,
    history_rows: List[Dict[str, Any]],
    *,
    time_taken: Optional[str] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()
    payload = {
        "generatedAt": _utc_now_iso(),
        "modelId": model_id,
        "timeTaken": time_taken,
        "hyperparameters": hyperparameters,
        "history": history_rows,
    }
    write_json(paths.model_training_history_json(model_id), payload)


def write_roc_micro(
    points: List[Dict[str, float]],
    aucs: Dict[str, Dict[str, float]],
    *,
    paths: Optional[WebArtifactsPaths] = None,
) -> None:
    paths = paths or WebArtifactsPaths.default()
    payload = {
        "generatedAt": _utc_now_iso(),
        "points": points,
        "aucs": aucs,
    }
    write_json(paths.roc_micro_json, payload)
