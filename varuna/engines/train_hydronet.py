"""HydroNet training + conformal calibration with checkpointing."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from varuna.uncertainty.conformal import ConformalRegressor
from varuna.utils.checkpoint import CheckpointManager
from varuna.utils.logging import setup_logger

logger = setup_logger("varuna.train.hydronet")


def train_hydronet(
    model: torch.nn.Module,
    X: torch.Tensor,
    edge_index: torch.Tensor,
    t_span: torch.Tensor,
    Y: torch.Tensor,
    epochs: int = 300,
    lr: float = 5e-3,
    device: str = "cpu",
    log_interval: int = 50,
    log_dir: str = "runs/hydronet",
    resume: bool = True,
) -> torch.nn.Module:
    """X: [T, N, F] features; Y: [T, N, 1] discharge targets."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    edge_index = edge_index.to(device)

    ckpt = CheckpointManager(log_dir, monitor="loss", mode="min", save_every=50)
    start_epoch = 0
    if resume:
        start_epoch, _ = ckpt.load_latest(model, opt, None)

    for epoch in range(start_epoch, epochs):
        opt.zero_grad()
        mu, logvar, _ = model(X.to(device), edge_index, t_span.to(device))
        var = torch.exp(logvar).clamp_min(1e-6)
        nll = (logvar + (Y.to(device) - mu) ** 2 / var).mean() / 2.0
        loss = nll + 0.1 * F.mse_loss(mu, Y.to(device))
        loss.backward()
        opt.step()
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            logger.info("hydronet epoch %d | loss %.5f", epoch + 1, loss.item())

        ckpt.save(epoch, model, opt, None, {"loss": loss.item()})
    return model


@torch.no_grad()
def predict_discharge(model: torch.nn.Module, X: torch.Tensor, edge_index: torch.Tensor, t_span: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, logvar, _ = model(X, edge_index, t_span)
    return mu.detach().cpu().numpy(), torch.exp(0.5 * logvar).detach().cpu().numpy(), X


def calibrate_discharge(
    model: torch.nn.Module,
    X_cal: torch.Tensor,
    edge_index: torch.Tensor,
    t_span: torch.Tensor,
    Y_cal: torch.Tensor,
    alpha: float = 0.10,
) -> ConformalRegressor:
    """Calibrate conformal intervals on held-out data (never the train split)."""
    mu, _, _ = predict_discharge(model, X_cal, edge_index, t_span)
    reg = ConformalRegressor(alpha=alpha)
    q = reg.calibrate(mu.ravel(), Y_cal.numpy().ravel())
    logger.info("Conformal quantile q=%.4f (alpha=%.2f)", q, alpha)
    return reg