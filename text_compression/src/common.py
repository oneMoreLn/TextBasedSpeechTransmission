import argparse
import json
import os
import sys
import time
from typing import Optional


def prepare_log_path(log_file: str, log_dir: str, prefix: str) -> str:
    path = log_file
    if not path:
        try:
            os.makedirs(log_dir, exist_ok=True)
            auto = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            path = os.path.join(log_dir, auto)
        except Exception:
            path = ""
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": "run",
                    "ts": time.time(),
                    "cwd": os.getcwd(),
                    "argv": sys.argv,
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass
    return path
