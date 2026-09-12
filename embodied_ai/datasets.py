from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import pandas as pd

def export_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, default=str) for row in records) + ("\n" if records else ""), encoding="utf-8")
    return path

def export_parquet(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path, index=False)
    return path

def export_csv(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return path
