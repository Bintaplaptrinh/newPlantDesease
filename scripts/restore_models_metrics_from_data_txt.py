"""Restore `data/web/models.json` metrics from a backup text file.

Why:
- `src/pipeline.ensure_web_placeholders()` and registry export can overwrite
  previously computed model metrics.

This script extracts overall metrics from `data.txt` (classification report)
and combines them with `training_history.json` (timeTaken, best valAccuracy)
plus model parameter counts from `.pth`.

Usage:
  python scripts/restore_models_metrics_from_data_txt.py --backup data.txt

Notes:
- Confusion matrices cannot be reconstructed from classification report alone.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


MODEL_SECTIONS = {
    "mobilenet": {"header": "===== MobileNetV3 ====="},
    "resnet18": {"header": "===== ResNet18 ====="},
    "efficientnet": {"header": "===== EfficientNetB0 ====="},
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_block(text: str, header: str) -> Optional[str]:
    idx = text.find(header)
    if idx < 0:
        return None
    # heuristic: next model header or end
    next_idxs = [
        text.find(h["header"], idx + len(header))
        for h in MODEL_SECTIONS.values()
        if h["header"] != header
    ]
    next_idxs = [i for i in next_idxs if i >= 0]
    end = min(next_idxs) if next_idxs else len(text)
    return text[idx:end]


def parse_overall_and_weighted(block: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    # overall accuracy
    m_acc = re.search(r"Overall accuracy:\s*([0-9.]+)", block)
    acc = float(m_acc.group(1)) if m_acc else None

    # weighted avg line: precision recall f1-score support
    m_w = re.search(r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)", block)
    if not m_w:
        return acc, None, None, None

    precision = float(m_w.group(1))
    recall = float(m_w.group(2))
    f1 = float(m_w.group(3))
    return acc, precision, recall, f1


def best_val_accuracy(training_history_path: Path) -> Tuple[Optional[float], Optional[int], Optional[str], Optional[Dict[str, Any]]]:
    if not training_history_path.exists():
        return None, None, None, None

    payload = load_json(training_history_path)
    hist = payload.get("history") or []
    vals = [row.get("valAcc") for row in hist if isinstance(row, dict) and isinstance(row.get("valAcc"), (int, float))]
    best = max(vals) if vals else None
    epochs = len(hist) if isinstance(hist, list) else None
    time_taken = payload.get("timeTaken")
    hyper = payload.get("hyperparameters")
    return best, epochs, time_taken, hyper


def format_params(n: int) -> str:
    # human-friendly like 5.28M
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def try_count_parameters(model_path: Path) -> Optional[str]:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not model_path.exists():
        return None

    try:
        m = torch.load(model_path, weights_only=False, map_location="cpu")
        # model saved as nn.Module
        total = 0
        for p in m.parameters():
            total += int(p.numel())
        return format_params(total)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=str, required=True, help="Path to backup text file (e.g. data.txt)")
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="Project root")
    args = parser.parse_args()

    root = Path(args.root)
    backup_path = (root / args.backup) if not Path(args.backup).is_absolute() else Path(args.backup)

    if not backup_path.exists():
        raise SystemExit(f"Backup file not found: {backup_path}")

    web_dir = root / "data" / "web"
    models_json_path = web_dir / "models.json"
    if not models_json_path.exists():
        raise SystemExit(f"Missing models.json: {models_json_path}")

    text = backup_path.read_text(encoding="utf-8", errors="ignore")
    models_index = load_json(models_json_path)

    models = models_index.get("models") or []
    if not isinstance(models, list):
        raise SystemExit("models.json has invalid format")

    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if model_id not in MODEL_SECTIONS:
            continue

        block = find_block(text, MODEL_SECTIONS[model_id]["header"])
        if not block:
            continue

        acc, prec, rec, f1 = parse_overall_and_weighted(block)

        th_path = web_dir / "models" / model_id / "training_history.json"
        best_val, epochs, time_taken, hyper = best_val_accuracy(th_path)

        # parameter count from pth path in models.json
        pth_rel = model.get("pthPath")
        params_str = None
        if isinstance(pth_rel, str) and pth_rel:
            params_str = try_count_parameters((root / pth_rel).resolve())

        metrics: Dict[str, Any] = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}
        if acc is not None:
            metrics["accuracy"] = acc
        if prec is not None:
            metrics["precision"] = prec
        if rec is not None:
            metrics["recall"] = rec
        if f1 is not None:
            metrics["f1Score"] = f1
        if best_val is not None:
            metrics["valAccuracy"] = best_val
        if epochs is not None:
            metrics["epochs"] = epochs
        if isinstance(time_taken, str) and time_taken:
            metrics["trainTime"] = time_taken
        if params_str is not None:
            metrics["params"] = params_str

        model["metrics"] = metrics or None

        # also fill hyperparameters in models.json for completeness
        if model.get("hyperparameters") in (None, {}):
            if isinstance(hyper, dict):
                model["hyperparameters"] = hyper

    models_index["generatedAt"] = utc_now_iso()
    save_json(models_json_path, models_index)

    print(f"Restored metrics into: {models_json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
