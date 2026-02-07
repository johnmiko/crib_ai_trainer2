"""Project-wide logging setup (scripts entrypoint).

This mirrors the repo-level sitecustomize.py so that running
`python scripts/<file>.py` still initializes logging.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path


def _configure_logging() -> None:
    override = os.getenv("CRIB_LOG_FILE")
    if override is None or override.strip() == "":
        log_path = Path(__file__).resolve().parent.parent / "text" / "log_file.log"
    else:
        log_path = Path(override).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    already = False
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and Path(getattr(handler, "baseFilename", "")) == log_path:
            already = True
            break
    if not already:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


_configure_logging()
