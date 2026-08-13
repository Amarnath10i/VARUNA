"""FloodMap self-supervised pretraining + few-shot fine-tuning with checkpointing."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from varuna.utils.checkpoint import CheckpointManager
from varuna.utils.logging import MetricsLogger, setup_logger

logger = setup_logger("varuna.train.floodmap")


def pretrain_floodmap(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 80,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "runs/floodmap",
    resume: bool = True,
) -> torch.nn.Module:
    """Self-supervised cross-modal masked reconstruction."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ml = MetricsLogger(log_dir)

    ckpt = CheckpointManager(log_dir, monitor="pretrain_loss", mode="min", save_every=10)
    start_epoch = 0
    if resume:
        start_epoch, _ = ckpt.load_latest(model, opt, None)

    for epoch in range(start_epoch, epochs):
        model.train()
        tot = 0.0
        for sar, opt_img in loader:
            sar, opt_img = sar.to(device), opt_img.to(device)
            opt.zero_grad()
            out = model.pretrain_loss(sar, opt_img)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += out["loss"].item()
        ml.log(epoch, pretrain_loss=tot / max(len(loader), 1))
        ckpt.save(epoch, model, opt, None, {"pretrain_loss": tot / max(len(loader), 1)})

        if (epoch + 1) % 10 == 0:
            logger.info("floodmap pretrain epoch %d | recon loss %.5f", epoch + 1, tot / max(len(loader), 1))
    ml.close()
    return model


def finetune_flood_head(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    log_dir: str = "runs/floodmap",
    resume: bool = True,
) -> torch.nn.Module:
    """Few-shot supervised fine-tuning of the flood head (<50 labelled masks)."""
    model.to(device)
    # freeze the pretrained backbone
    for p in model.sar_enc.parameters():
        p.requires_grad = False
    for p in model.opt_enc.parameters():
        p.requires_grad = False
    for p in model.fuse.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.flood_head.parameters(), lr=lr)

    ckpt = CheckpointManager(log_dir, monitor="finetune_loss", mode="min", save_every=5)
    start_epoch = 0
    if resume:
        start_epoch, _ = ckpt.load_latest(model, opt, None)

    for epoch in range(start_epoch, epochs):
        model.train()
        tot = 0.0
        for sar, opt_img, mask in loader:
            sar, opt_img, mask = sar.to(device), opt_img.to(device), mask.to(device)
            opt.zero_grad()
            prob = model.flood_prob(sar, opt_img)
            loss = F.binary_cross_entropy(prob, mask)
            loss.backward()
            opt.step()
            tot += loss.item()
        ckpt.save(epoch, model, opt, None, {"finetune_loss": tot / max(len(loader), 1)})

        if (epoch + 1) % 5 == 0:
            logger.info("floodmap finetune epoch %d | bce %.5f", epoch + 1, tot / max(len(loader), 1))
    return model