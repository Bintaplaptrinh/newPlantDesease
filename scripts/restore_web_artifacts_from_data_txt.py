"""Restore web artifacts from a backup text file (data.txt).

Restores:
- data/web/roc_micro.json (from the ROC tables in the backup)
- data/web/models/<id>/training_history.json for any model whose `history` is empty

This is designed to recover from accidental placeholder overwrites.

Usage:
  python scripts/restore_web_artifacts_from_data_txt.py --backup data.txt
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


EPOCH_RE = re.compile(
    r"^Epoch\s*\[(\d+)\/(\d+)\]\s*\|\s*Train Loss:\s*([0-9.]+)\s*\|\s*Train Acc:\s*([0-9.]+)%\s*\|\s*Val Loss:\s*([0-9.]+)\s*\|\s*Val Acc:\s*([0-9.]+)%\s*\|\s*LR:\s*([0-9.eE+-]+)\s*$"
)


def classify_marker(line: str) -> Optional[str]:
    l = line.strip().lower()
    if not (l.startswith("***") and l.endswith("***")):
        return None

    # normalize
    if "resnet" in l:
        return "resnet18"
    if "mobil" in l:
        return "mobilenet"
    if "eficient" in l or "efficient" in l:
        return "efficientnet"
    return None


def parse_training_histories(text: str) -> Dict[str, List[Dict[str, Any]]]:
    current: Optional[str] = None
    rows_by_model: Dict[str, List[Dict[str, Any]]] = {"mobilenet": [], "resnet18": [], "efficientnet": []}

    for raw in text.splitlines():
        marker = classify_marker(raw)
        if marker:
            current = marker
            continue

        if current is None:
            continue

        m = EPOCH_RE.match(raw.strip())
        if not m:
            continue

        epoch = int(m.group(1))
        train_loss = float(m.group(3))
        train_acc = float(m.group(4)) / 100.0
        val_loss = float(m.group(5))
        val_acc = float(m.group(6)) / 100.0
        lr = float(m.group(7))

        rows_by_model[current].append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "valLoss": val_loss,
                "trainAcc": train_acc,
                "valAcc": val_acc,
                "lr": lr,
            }
        )

    return rows_by_model


AUC_RE = re.compile(r"micro AUC\s*=\s*([0-9.]+),\s*macro AUC\s*=\s*([0-9.]+)")
TABLE_ROW_RE = re.compile(r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*$")


def parse_roc(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]]]:
    # AUC block
    aucs: Dict[str, Dict[str, float]] = {}
    # Map names used in backup -> ids used in JSON
    name_to_id = {
        "mobilenetv3": "mobilenet",
        "resnet18": "resnet18",
        "efficientnetb0": "efficientnet",
    }

    lines = text.splitlines()

    # Find and parse AUC lines following model names
    for i, line in enumerate(lines):
        key = line.strip().lower().replace(" ", "")
        if key in name_to_id and i + 1 < len(lines):
            m = AUC_RE.search(lines[i + 1])
            if m:
                aucs[name_to_id[key]] = {"micro": float(m.group(1)), "macro": float(m.group(2))}

    # Parse FPR/TPR tables
    table_id: Optional[str] = None
    tables: Dict[str, Dict[float, float]] = {"mobilenet": {}, "resnet18": {}, "efficientnet": {}}

    def table_marker_to_id(line_: str) -> Optional[str]:
        l = line_.strip().lower()
        if l.startswith("***") and l.endswith("***"):
            core = l.strip("*").strip().replace(" ", "")
            if "mobil" in core:
                return "mobilenet"
            if "resnet" in core:
                return "resnet18"
            if "efficient" in core or "eficient" in core:
                return "efficientnet"
        return None

    for line in lines:
        m_id = table_marker_to_id(line)
        if m_id:
            table_id = m_id
            continue
        if table_id is None:
            continue
        m = TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        fpr = float(m.group(1))
        tpr = float(m.group(2))
        tables[table_id][fpr] = tpr

    # merge by fpr
    all_fprs = sorted({*tables["mobilenet"].keys(), *tables["resnet18"].keys(), *tables["efficientnet"].keys()})
    points: List[Dict[str, Any]] = []
    for fpr in all_fprs:
        points.append(
            {
                "fpr": fpr,
                "mobilenet": tables["mobilenet"].get(fpr),
                "resnet18": tables["resnet18"].get(fpr),
                "efficientnet": tables["efficientnet"].get(fpr),
            }
        )

    return points, aucs


def restore_training_history_if_empty(path: Path, *, model_id: str, rows: List[Dict[str, Any]]) -> bool:
    if not path.exists():
        return False
    payload = load_json(path)
    hist = payload.get("history")
    if isinstance(hist, list) and len(hist) > 0:
        return False

    payload["generatedAt"] = utc_now_iso()
    payload["modelId"] = model_id
    payload["history"] = rows

    # keep other keys if present
    if "timeTaken" not in payload:
        payload["timeTaken"] = None
    if "hyperparameters" not in payload:
        payload["hyperparameters"] = None

    save_json(path, payload)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", type=str, required=True)
    parser.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.root)
    backup_path = (root / args.backup) if not Path(args.backup).is_absolute() else Path(args.backup)
    if not backup_path.exists():
        raise SystemExit(f"Backup file not found: {backup_path}")

    text = backup_path.read_text(encoding="utf-8", errors="ignore")

    web_dir = root / "data" / "web"

    # Restore ROC
    points, aucs = parse_roc(text)
    roc_path = web_dir / "roc_micro.json"
    if points:
        save_json(roc_path, {"generatedAt": utc_now_iso(), "points": points, "aucs": aucs})
        print(f"Restored ROC into: {roc_path}")
    else:
        print("No ROC points found in backup; roc_micro.json not changed")

    # Restore training histories when empty
    rows_by_model = parse_training_histories(text)
    restored = 0
    for model_id, rows in rows_by_model.items():
        if not rows:
            continue
        th_path = web_dir / "models" / model_id / "training_history.json"
        if restore_training_history_if_empty(th_path, model_id=model_id, rows=rows):
            restored += 1
            print(f"Restored training history: {th_path}")

    print(f"Training histories restored: {restored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
