"""RainCast training loop with checkpointing."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from varuna.uncertainty.losses import raincast_loss
from varuna.utils.checkpoint import CheckpointManager
from varuna.utils.logging import MetricsLogger, setup_logger

logger = setup_logger("varuna.train.raincast")


def train_raincast(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "runs/raincast",
    resume: bool = True,
) -> torch.nn.Module:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    logger.info("RainCast training on %s", device)
    ml = MetricsLogger(log_dir)

    ckpt = CheckpointManager(log_dir, monitor="val_loss", mode="min", save_every=5)
    start_epoch = 0
    if resume:
        start_epoch, _ = ckpt.load_latest(model, opt, sched)

    best = float("inf")
    for epoch in range(start_epoch, epochs):
        model.train()
        run_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            mu, logvar = model(x)
            out = raincast_loss(mu, logvar, y)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += out["loss"].item()
        sched.step()

        val_loss, val_crps = evaluate_raincast(model, val_loader, device)
        ml.log(epoch, train_loss=run_loss / max(len(train_loader), 1), val_loss=val_loss, val_crps=val_crps)
        ckpt.save(epoch, model, opt, sched, {"val_loss": val_loss, "val_crps": val_crps})

        if val_loss < best:
            best = val_loss
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("epoch %d | train %.4f | val %.4f | crps %.4f", epoch, run_loss / max(len(train_loader), 1), val_loss, val_crps)
    ml.close()
    return model


@torch.no_grad()
def evaluate_raincast(model: torch.nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    tot_loss, tot_crps, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        mu, logvar = model(x)
        out = raincast_loss(mu, logvar, y)
        tot_loss += out["loss"].item()
        tot_crps += out["crps"].item()
        n += 1
    return tot_loss / max(n, 1), tot_crps / max(n, 1)