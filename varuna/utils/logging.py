"""Logging and experiment tracking helpers."""
from __future__ import annotations

import json
import logging
from pathlib import Path


def setup_logger(name: str = "varuna", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


class MetricsLogger:
    """Small JSON-line logger for incremental training metrics."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.fp = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")

    def log(self, step: int, **metrics) -> None:
        row = {"step": step, **{k: float(v) for k, v in metrics.items()}}
        self.fp.write(json.dumps(row) + "\n")
        self.fp.flush()

    def close(self) -> None:
        self.fp.close()