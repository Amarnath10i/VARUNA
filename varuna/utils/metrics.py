"""Metric utilities used for evaluation of every VARUNA component."""
from __future__ import annotations

import numpy as np


def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency (hydrology standard)."""
    obs, sim = np.asarray(obs), np.asarray(sim)
    denom = np.sum((obs - obs.mean()) ** 2)
    if denom == 0:
        return float("-inf")
    return float(1.0 - np.sum((obs - sim) ** 2) / denom)


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta efficiency."""
    obs, sim = np.asarray(obs), np.asarray(sim)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / (np.std(obs) + 1e-9)
    beta = sim.mean() / (obs.mean() + 1e-9)
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(obs) - np.asarray(sim)) ** 2)))


def crps_empirical(obs: np.ndarray, samples: np.ndarray) -> float:
    """Empirical CRPS from an ensemble of forecast samples."""
    obs = np.asarray(obs).ravel()
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[None, :]
    m = samples.shape[1]
    total = 0.0
    for o, s in zip(obs, samples):
        s = np.sort(s)
        mean_abs = s.mean()
        sum_pairs = 0.0
        for i in range(m):
            for j in range(m):
                sum_pairs += abs(s[i] - s[j])
        crps = np.mean(np.abs(s - o)) - sum_pairs / (2 * m * m)
        total += crps
    return float(total / len(obs))


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else float("nan")


def hit_rate(prob: np.ndarray, mask: np.ndarray, threshold: float) -> float:
    """Fraction of true-flood pixels caught above a probability threshold."""
    pred = prob >= threshold
    pos = mask > 0
    denom = pos.sum()
    return float(np.logical_and(pred, pos).sum() / denom) if denom > 0 else float("nan")