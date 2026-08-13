"""Checkpoint manager for training resumption and best-model tracking."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Checkpoint:
    epoch: int
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict | None
    metrics: dict
    best_metric: float
    is_best: bool


class CheckpointManager:
    """Manages periodic and best-model checkpoints with optional resume."""

    def __init__(
        self,
        run_dir: str | Path,
        monitor: str = "val_loss",
        mode: str = "min",
        save_every: int = 5,
        max_keep: int = 3,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.save_every = save_every
        self.max_keep = max_keep
        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_path = self.run_dir / "best.pt"
        self.last_path = self.run_dir / "last.pt"
        self.history_path = self.run_dir / "checkpoint_history.json"
        self._history: list[dict] = []

    def _is_better(self, current: float) -> bool:
        if self.mode == "min":
            return current < self.best_metric
        return current > self.best_metric

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        metrics: dict,
    ) -> list[Path]:
        """Save checkpoint for current epoch; return list of saved paths."""
        current = metrics.get(self.monitor, float("inf"))
        is_best = self._is_better(current)
        if is_best:
            self.best_metric = current

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_metric": self.best_metric,
        }

        saved = []
        # periodic checkpoint
        if epoch % self.save_every == 0:
            p = self.run_dir / f"epoch_{epoch:04d}.pt"
            torch.save(state, p)
            saved.append(p)
            self._history.append({"epoch": epoch, "path": str(p), "is_best": is_best, "metrics": metrics})

        # always save 'last'
        torch.save(state, self.last_path)
        if self.last_path not in saved:
            saved.append(self.last_path)

        # best model
        if is_best:
            torch.save(state, self.best_path)
            if self.best_path not in saved:
                saved.append(self.best_path)

        # cleanup old periodic checkpoints
        periodic = sorted(self.run_dir.glob("epoch_*.pt"))
        for old in periodic[:-self.max_keep]:
            old.unlink(missing_ok=True)

        # save history
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2)

        return saved

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> tuple[int, dict]:
        """Load the 'last' checkpoint if it exists; return (start_epoch, metrics)."""
        if not self.last_path.exists():
            return 0, {}

        state = torch.load(self.last_path, map_location="cpu")
        model.load_state_dict(state["model"])
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        self.best_metric = state.get("best_metric", self.best_metric)
        return state["epoch"] + 1, state.get("metrics", {})

    def load_best(self, model: torch.nn.Module) -> dict:
        """Load the best checkpoint; return metrics."""
        if not self.best_path.exists():
            return {}
        state = torch.load(self.best_path, map_location="cpu")
        model.load_state_dict(state["model"])
        return state.get("metrics", {})