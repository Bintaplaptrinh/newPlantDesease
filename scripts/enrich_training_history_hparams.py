"""Enrich exported web artifacts with training hyperparameters.

This project stores per-model training history at:
  data/web/models/<modelId>/training_history.json

The web UI expects optional fields:
  - timeTaken (string)
  - hyperparameters (object)

Older artifacts may miss `hyperparameters`. This script fills them in using
defaults derived from the training pipeline code in `src/model/*` and
`src/preprocessing.py`.

Safe to re-run: it only writes `hyperparameters` when missing/null, unless
`--force` is provided.

Usage:
  python scripts/enrich_training_history_hparams.py
  python scripts/enrich_training_history_hparams.py --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


AUGMENTATIONS = [
    "RandomResizedCrop(scale=0.6..1.0, ratio=0.75..1.3333)",
    "RandomHorizontalFlip(p=0.5)",
    "RandAugment(num_ops=2, magnitude=9) [if available]",
    "ColorJitter(brightness=0.4, contrast=0.4, saturation=0.25, hue=0.02) (p=0.8)",
    "GaussianBlur(kernel_size=3, sigma=0.1..2.0) (p=0.3)",
    "RandomPerspective(distortion_scale=0.25, p=0.2)",
    "Normalize(ImageNet mean/std)",
    "RandomErasing(p=0.25, scale=0.02..0.15, ratio=0.3..3.3)",
]


DEFAULTS_BY_MODEL: Dict[str, Dict[str, Any]] = {
    "mobilenet": {
        "model": "MobileNetV3 Large",
        "optimizer": "AdamW",
        "learningRate": 1e-3,
        "batchSize": 64,
        "epochs": 30,
        "imageSize": "224x224",
        "dropout": 0.2,
        "l2Regularization": 1e-4,
        "augmentation": AUGMENTATIONS,
        "labelSmoothing": 0.1,
        "scheduler": "CosineAnnealingLR(T_max=epochs, eta_min=1e-5)",
        "notes": "Train classifier only; optional AdaBN + TTA flip in validation.",
    },
    "resnet18": {
        "model": "ResNet18",
        "optimizer": "AdamW",
        "learningRate": 1e-3,
        "batchSize": 64,
        "epochs": 30,
        "imageSize": "224x224",
        "dropout": None,
        "l2Regularization": 1e-4,
        "augmentation": AUGMENTATIONS,
        "labelSmoothing": 0.1,
        "scheduler": "CosineAnnealingLR(T_max=epochs, eta_min=1e-5)",
        "notes": "Train fc only; optional AdaBN + TTA flip in validation.",
    },
    "efficientnet": {
        "model": "EfficientNet-B0",
        "optimizer": "AdamW",
        "learningRate": 1e-3,
        "batchSize": 64,
        "epochs": 30,
        "imageSize": "224x224",
        "dropout": 0.2,
        "l2Regularization": 1e-4,
        "augmentation": AUGMENTATIONS,
        "labelSmoothing": 0.1,
        "scheduler": "CosineAnnealingLR(T_max=epochs, eta_min=1e-5)",
        "notes": "Train classifier only; optional AdaBN + TTA flip in validation.",
    },
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def enrich_file(path: Path, *, force: bool) -> bool:
    payload = load_json(path)
    model_id = payload.get("modelId") or path.parent.name

    defaults = DEFAULTS_BY_MODEL.get(model_id)
    if not defaults:
        return False

    existing = payload.get("hyperparameters")
    if existing and not force:
        return False

    payload["modelId"] = model_id
    if not payload.get("generatedAt"):
        payload["generatedAt"] = utc_now_iso()

    # Keep timeTaken if already present; don't fabricate.
    if "timeTaken" not in payload:
        payload["timeTaken"] = None

    payload["hyperparameters"] = defaults

    save_json(path, payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing hyperparameters in training_history.json",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root (defaults to repo root)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    base = root / "data" / "web" / "models"
    if not base.exists():
        raise SystemExit(f"Missing directory: {base}")

    changed = 0
    scanned = 0

    for p in base.glob("*/training_history.json"):
        scanned += 1
        if enrich_file(p, force=args.force):
            changed += 1

    print(f"Scanned {scanned} files; updated {changed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
