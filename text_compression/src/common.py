import argparse
import json
import os
import sys
import time
from typing import Optional


def prepare_log_path(log_file: str, log_dir: str, prefix: str) -> str:
    """
    Determine a log file path and ensure its directory exists.
    - If log_file is empty: create auto-named file under log_dir.
    - If log_file is a bare filename (no directory): place it under log_dir.
    - If log_file includes a directory or is absolute: use as-is and create parent dirs.
    """
    path = ""
    try:
        # Ensure base log_dir exists for cases using it
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        if not log_file:
            auto = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
            path = os.path.join(log_dir or ".", auto)
        else:
            # If user passed only a filename, put it under log_dir
            if not os.path.isabs(log_file) and not os.path.dirname(log_file):
                path = os.path.join(log_dir or ".", log_file)
            else:
                path = log_file
            # Ensure parent dir exists for custom paths
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

        # Write run header
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "type": "run",
                "ts": time.time(),
                "cwd": os.getcwd(),
                "argv": sys.argv,
            }, ensure_ascii=False) + "\n")
    except Exception:
        # Fall back to no logging on error
        return ""
    return path
