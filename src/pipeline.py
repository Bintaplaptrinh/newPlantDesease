from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

import numpy as np

from src.preprocessing import (
    DatasetPaths,
    default_exclude_old_injection_classes,
    download_kaggle_datasets,
    materialize_data_folder,
    prepare_data,
)
from src.web_artifacts import (
    WebArtifactsPaths,
    write_class_distribution,
    write_classes,
    write_dataset_stats,
    write_models_index,
)


def _count_labels_from_concat_dataset(ds) -> np.ndarray:
    # ds could be ImageFolder or ConcatDataset; iterate targets through samples
    # We avoid loading actual image tensors.
    counts = None

    def ensure(n: int):
        nonlocal counts
        if counts is None:
            counts = np.zeros((n,), dtype=int)

    # ImageFolder
    if hasattr(ds, "targets") and hasattr(ds, "classes"):
        ensure(len(ds.classes))
        for y in ds.targets:
            counts[int(y)] += 1
        return counts

    # ConcatDataset / Subset / Remap
    for _, y in ds:
        if counts is None:
            # y is already new index; we don't know num classes here, caller should handle
            raise RuntimeError("Unable to infer num_classes from iterable dataset")
        counts[int(y)] += 1

    raise RuntimeError("Unsupported dataset type for counting")


def prepare_and_export_web_data(
    *,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 8,
    data_root: str = "data",
    exclude_old_injection_classes: Optional[set[str]] = None,
    paths: Optional[WebArtifactsPaths] = None,
) -> Dict[str, Any]:
    """Download + merge datasets, then export dataset + placeholder model JSON for the web.

    This does NOT train. It only prepares dataset loaders and exports:
    - data/web/classes.json
    - data/web/dataset_stats.json
    - data/web/class_distribution.json
    - data/web/models.json (model registry only; metrics can be added later)
    """

    paths = paths or WebArtifactsPaths.default()
    exclude_old_injection_classes = exclude_old_injection_classes or default_exclude_old_injection_classes()

    pack = prepare_data(image_size=image_size, batch_size=batch_size, num_workers=num_workers)

    # ensure kaggle downloads are visible under data/
    ds_paths: DatasetPaths = pack["paths"]
    materialize_info = materialize_data_folder(ds_paths, data_root=data_root)

    class_names: List[str] = list(pack["class_names"])
    num_classes = int(pack["num_classes"])

    train_base_ds = pack["train_base_ds"]
    valid_base_ds = pack["valid_base_ds"]

    # counts based on the NEW (main) dataset; this is stable and does not require iterating over transforms
    train_counts = _count_labels_from_concat_dataset(train_base_ds)
    val_counts = _count_labels_from_concat_dataset(valid_base_ds)
    total_counts = train_counts + val_counts

    write_classes(class_names, paths=paths)
    write_dataset_stats(
        total_images=int(total_counts.sum()),
        train_images=int(train_counts.sum()),
        validation_images=int(val_counts.sum()),
        test_images=0,
        num_classes=num_classes,
        extra={
            "excludeOldInjectionClasses": sorted(list(exclude_old_injection_classes)),
            "datasetPaths": asdict(ds_paths),
            "materialize": materialize_info,
        },
        paths=paths,
    )
    write_class_distribution(class_names, total_counts, paths=paths)

    # minimal model registry; metrics/hyperparams can be filled by training/evaluation run elsewhere.
    model_index = [
        {
            "id": "mobilenet",
            "name": "MobileNetV3 Large",
            "pthPath": "src/model/mobilenet_v3.pth",
            "metrics": None,
            "hyperparameters": None,
        },
        {
            "id": "resnet18",
            "name": "ResNet18",
            "pthPath": "src/model/resnet.pth",
            "metrics": None,
            "hyperparameters": None,
        },
        {
            "id": "efficientnet",
            "name": "EfficientNet-B0",
            "pthPath": "src/model/efficientnet.pth",
            "metrics": None,
            "hyperparameters": None,
        },
    ]
    write_models_index(model_index, paths=paths)

    return {
        "num_classes": num_classes,
        "class_names": class_names,
        "train_images": int(train_counts.sum()),
        "validation_images": int(val_counts.sum()),
        "materialize": materialize_info,
        "web_artifacts_dir": str(paths.base_dir),
    }


def ensure_web_placeholders(*, paths: Optional[WebArtifactsPaths] = None) -> None:
    """Create empty placeholder JSON for artifacts that depend on evaluation/training.

    Web reads these and shows empty/"not available" states instead of demo data.
    """

    from src.utils.json_io import write_json

    paths = paths or WebArtifactsPaths.default()

    # Safety: do not overwrite real artifacts if they already exist.
    if not paths.roc_micro_json.exists():
        write_json(paths.roc_micro_json, {"generatedAt": None, "points": [], "aucs": {}})

    for model_id in ["mobilenet", "resnet18", "efficientnet"]:
        cm_path = paths.model_confusion_json(model_id)
        if not cm_path.exists():
            write_json(cm_path, {"generatedAt": None, "modelId": model_id, "labels": [], "matrix": [], "normalized": False})

        th_path = paths.model_training_history_json(model_id)
        if not th_path.exists():
            write_json(th_path, {"generatedAt": None, "modelId": model_id, "timeTaken": None, "hyperparameters": None, "history": []})


def export_web_from_trained_models(*args, **kwargs):
    # Kept for backward compatibility if you later implement evaluation-based exports.
    raise NotImplementedError(
        "Use a training/evaluation machine to compute metrics and write JSON into data/web. "
        "This repo currently exports dataset-level JSON + placeholders only."
    )
