from __future__ import annotations

# tiện ích xác định các path gốc của project

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    web_dir: Path

    @staticmethod
    def detect() -> "ProjectPaths":
        # src/utils/paths.py -> src/utils -> src -> project root
        root = Path(__file__).resolve().parents[2]
        return ProjectPaths(
            root=root,
            data_dir=root / "data",
            web_dir=root / "web" / "NewPlantDesease",
        )


def project_paths() -> ProjectPaths:
    # hàm wrapper để lấy ProjectPaths
    return ProjectPaths.detect()
