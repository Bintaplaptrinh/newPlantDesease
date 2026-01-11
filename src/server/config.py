from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from src.utils.paths import project_paths


@dataclass(frozen=True)
class ServerConfig:
    data_web_dir: Path
    models: Dict[str, Path]
    default_image_size: int = 224

    @staticmethod
    def default() -> "ServerConfig":
        p = project_paths()
        data_web_dir = p.data_dir / "web"

        models = {
            "mobilenet": p.root / "src" / "model" / "mobilenet_v3.pth",
            "resnet18": p.root / "src" / "model" / "resnet.pth",
            "efficientnet": p.root / "src" / "model" / "efficientnet.pth",
        }

        return ServerConfig(data_web_dir=data_web_dir, models=models)


def safe_join(base: Path, relative: str) -> Path:
    # Prevent path traversal
    target = (base / relative).resolve()
    base_resolved = base.resolve()
    if base_resolved not in target.parents and target != base_resolved:
        raise ValueError("Invalid path")
    return target


def detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
