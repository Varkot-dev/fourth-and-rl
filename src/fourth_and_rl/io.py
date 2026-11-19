"""Small helpers for writing results that the README is allowed to quote."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False, default=float)
        f.write("\n")
    return path


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
