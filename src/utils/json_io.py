from __future__ import annotations

# tiện ích đọc/ghi json (utf-8, không escape unicode)

import json
from pathlib import Path
from typing import Any


def read_json(path: str | Path) -> Any:
    # đọc json từ file
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    # ghi json ra file, tự tạo thư mục cha nếu chưa có
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=indent)
