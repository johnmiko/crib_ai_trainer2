"""Project-wide logging setup.

Python auto-imports sitecustomize when it is on sys.path. This ensures
all scripts/tests log to text/log_file.log unless CRIB_LOG_FILE is set.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _configure_logging() -> None:
    override = os.getenv("CRIB_LOG_FILE")
    if override is None or override.strip() == "":
        log_path = Path(__file__).resolve().parent / "text" / "log_file.log"
    else:
        log_path = Path(override).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    has_file = False
    has_stream = False
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path:
            has_file = True
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) in {sys.stdout, sys.stderr}:
            has_stream = True
    if not has_file:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        root.addHandler(file_handler)
    if not has_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        root.addHandler(stream_handler)


_configure_logging()
