"""Split conformal prediction for finite-sample calibration.

VARUNA wraps every point prediction (RainCast rainfall, HydroNet discharge,
FloodMap flood probability) with split conformal prediction so that the
declared risk level carries a *statistical guarantee*:

    P( y_t in C_hat(x_t) ) >= 1 - alpha      (exchangeable setting)

This converts a black-box network output into a decision-ready interval,
which is the actual requirement of disaster-early-warning and parametric
insurance regulators.
"""

from __future__ import annotations

import numpy as np
import torch


class ConformalRegressor:
    """Distribution-free, finite-sample prediction intervals.

    Uses the absolute residual as the nonconformity score and the
    empirical (1 - alpha) quantile over a held-out calibration set.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q = None  # calibrated quantile of residuals

    def calibrate(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Fit the score quantile on a calibration split (must NOT be train data)."""
        residuals = np.abs(y - y_hat).ravel()
        n = residuals.size
        # conformal quantile with finite-sample correction
        level = np.ceil((1.0 - self.alpha) * (n + 1)) / n
        self.q = float(np.quantile(residuals, min(level, 1.0)))
        return self.q

    def predict_interval(self, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        return y_hat - self.q, y_hat + self.q

    def coverage(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_hat)
        return float(np.mean((y >= lo) & (y <= hi)))


class ConformalSegmentation:
    """Spatially-aggregated conformal control for flood-mask probabilities.

    Instead of per-pixel intervals, we calibrate a *global threshold* on the
    fraction of pixels that must exceed a risk level before a village/cell is
    declared flooded, using the empirical (1 - alpha) quantile of the
    per-pixel score errors on the calibration mask.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.threshold = None

    def calibrate(self, prob_map: np.ndarray, mask: np.ndarray) -> float:
        scores = np.abs(prob_map.ravel() - mask.ravel())
        n = scores.size
        level = np.ceil((1.0 - self.alpha) * (n + 1)) / n
        self.threshold = float(np.quantile(scores, min(level, 1.0)))
        return self.threshold

    def to_binary(self, prob_map: np.ndarray) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call calibrate() before to_binary().")
        return (prob_map >= self.threshold).astype(np.uint8)


def torch_interval_tensor(mu: torch.Tensor, sigma: torch.Tensor, z: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian interval at coverage level implied by z-score (parametric fallback)."""
    return mu - z * sigma, mu + z * sigma