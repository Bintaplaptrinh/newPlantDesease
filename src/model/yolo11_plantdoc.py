from __future__ import annotations

# script finetune yolo trên dataset plantdoc
# file này yêu cầu ultralytics + yaml, và dataset yaml đúng đường dẫn

import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def _best_effort_fix_data_yaml(data_yaml: Path) -> Path:

    # cố gắng sửa data.yaml để ultralytics đọc được (đường dẫn train/val/test)

    dataset_root = data_yaml.parent.resolve()

    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    def resolve_split(value: str) -> Path:
        # resolve path tương đối theo thư mục yaml, và normalize '..'
        return (dataset_root / Path(value)).resolve()

    def pick_existing(candidates: list[str]) -> str | None:
        for c in candidates:
            if c is None:
                continue
            p = resolve_split(c)
            if p.exists():
                # ưu tiên lưu path tương đối theo dataset_root
                try:
                    return str(p.relative_to(dataset_root)).replace("\\", "/")
                except Exception:
                    return str(p).replace("\\", "/")
        return None

    # đảm bảo base path là dataset_root
    cfg["path"] = str(dataset_root).replace("\\", "/")

    # folder train phải tồn tại
    train_value = cfg.get("train", "train/images")
    train_fixed = pick_existing([
        str(train_value),
        str(Path(str(train_value)).as_posix()).lstrip("../"),
        "train/images",
        "train",
    ])
    if not train_fixed:
        raise FileNotFoundError(
            f"Could not resolve train images path from {data_yaml}. "
            f"Expected something like '{dataset_root / 'train/images'}'."
        )
    cfg["train"] = train_fixed

    # val: ưu tiên valid/ hoặc val/, nếu không có thì fallback test/images
    val_value = cfg.get("val")
    val_fixed = pick_existing([
        str(val_value) if isinstance(val_value, str) else None,
        "valid/images",
        "val/images",
        "validation/images",
        "test/images",  # fallback khi không có split valid
    ])
    if not val_fixed:
        raise FileNotFoundError(
            f"Could not resolve val images path from {data_yaml}. "
            f"Tried valid/val/test under '{dataset_root}'."
        )
    cfg["val"] = val_fixed

    # test split là tuỳ chọn
    test_value = cfg.get("test")
    test_fixed = pick_existing([
        str(test_value) if isinstance(test_value, str) else None,
        "test/images",
    ])
    if test_fixed and test_fixed != cfg.get("val"):
        cfg["test"] = test_fixed
    else:
        cfg.pop("test", None)

    fixed_yaml = dataset_root / "data_ultralytics_fixed.yaml"
    with fixed_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return fixed_yaml


def main() -> int:
    # train yolo và copy weight best/last ra file out
    out = "yolo11n_finetune.pt"

    data_yaml = Path("PlantDoc-1/data.yaml").expanduser().resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    fixed_yaml = _best_effort_fix_data_yaml(data_yaml)

    model = YOLO("yolo11n.pt")

    train_results = model.train(
        data=str(fixed_yaml),
        epochs=50,
        imgsz=640,
        batch=16,
        device= "0", # dùng "0" nếu muốn chạy gpu
        lr0=0.01,
        patience=20,
        verbose=True,  # in progress ra console
        plots=False,   # hạn chế artifact không cần thiết
        save=True,
        val=True,
        pretrained=True,
        deterministic=True,
        seed=42,
        workers=8,
    )

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    src = best_pt if best_pt.exists() else last_pt
    if not src.exists():
        raise FileNotFoundError(f"Could not find saved weights at: {best_pt} or {last_pt}")

    out_path = Path(out).resolve()
    shutil.copy2(src, out_path)
    print(f"Saved finetuned weights: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
