from __future__ import annotations

from pathlib import Path


def read_queue_models(queue_models: str | None, queue_file: str | None) -> list[str]:
    if queue_models:
        return [p.strip() for p in queue_models.split(",") if p.strip()]
    if queue_file:
        qpath = Path(queue_file)
        if not qpath.exists():
            raise SystemExit(f"--queue_file not found: {qpath}")
        return [line.strip() for line in qpath.read_text(encoding="utf-8").splitlines() if line.strip()]
    return []


def resolve_model_dir(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists() and (path / "model_meta.json").exists():
        return path
    if path.exists():
        run_dirs = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
        if run_dirs:
            latest = max(int(p.name) for p in run_dirs)
            return path / f"{latest:03d}"
    parent = path.parent
    if parent.exists() and (parent / "model_meta.json").exists():
        return parent
    return path
