from __future__ import annotations

import json
import os
import hashlib
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass
class Timer:
    name: str
    t0: float = 0.0
    elapsed_s: float = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.elapsed_s = time.time() - self.t0


def ensure_dir(p: str | Path) -> str:
    p = str(p)
    os.makedirs(p, exist_ok=True)
    return p


def now_run_id() -> str:
    # compact timestamp-based run id
    return time.strftime("%Y%m%d_%H%M%S")


def sha256_file(path: str, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json(path: str, data: Dict[str, Any], indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def disk_free_bytes(path: str | Path) -> int:
    usage = shutil.disk_usage(str(path))
    return int(usage.free)


def require_free_space(path: str | Path, need_bytes: int) -> Tuple[bool, str]:
    free_b = disk_free_bytes(path)
    ok = free_b >= need_bytes
    msg = f"free={free_b/1e9:.2f}GB need={need_bytes/1e9:.2f}GB"
    return ok, msg


class SimpleLogger:
    """
    Simple line-based logger writing both stdout and run.log.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(Path(log_path).parent)

    def log(self, msg: str) -> None:
        line = msg.rstrip("\n")
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
