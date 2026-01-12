from __future__ import annotations

# cấu hình cho flask server: đường dẫn artifact web, model pth, và model yolo (nếu có)

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.utils.paths import project_paths


@dataclass(frozen=True)
class ServerConfig:
    # config chạy server (đường dẫn data/web và đường dẫn model)
    data_web_dir: Path
    models: Dict[str, Path]
    yolo_model_path: Optional[Path] = None
    default_image_size: int = 224
    yolo_detection_size: int = 640  # size input chuẩn của yolo

    @staticmethod
    def default() -> "ServerConfig":
        p = project_paths()
        data_web_dir = p.data_dir / "web"

        models = {
            "mobilenet": p.root / "src" / "model" / "mobilenet_v3.pth",
            "resnet18": p.root / "src" / "model" / "resnet.pth",
            "efficientnet": p.root / "src" / "model" / "efficientnet.pth",
        }

        yolo_path = p.root / "src" / "model" / "yolo11n_finetune.pt"

        return ServerConfig(
            data_web_dir=data_web_dir, 
            models=models,
            yolo_model_path=yolo_path if yolo_path.exists() else None
        )


def safe_join(base: Path, relative: str) -> Path:
    # nối path an toàn, tránh path traversal (..)
    target = (base / relative).resolve()
    base_resolved = base.resolve()
    if base_resolved not in target.parents and target != base_resolved:
        raise ValueError("Invalid path")
    return target


def detect_device() -> str:
    # auto chọn device để infer (ưu tiên cuda nếu có)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
