# src/io_utils.py
import os
import json
import gzip
from typing import Iterable, Iterator, Dict, Any

def _open_text(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode.replace("b", "").replace("t", "t"), encoding="utf-8")
    return open(path, mode, encoding="utf-8", newline="")

def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream JSON Lines from a file (.jsonl or .jsonl.gz).
    Skips blank lines and lines starting with '#'.
    """
    with _open_text(path, "rt") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError as e:
                # surface which line broke to make debugging easy
                raise ValueError(f"Invalid JSON on line {ln} of {path}: {e.msg}") from e

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dicts to JSON Lines (.jsonl or .jsonl.gz).
    Creates parent directories if needed.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with _open_text(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
