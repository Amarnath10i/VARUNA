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
import pandas as pd
import torch


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score: the order statistic the
    finite-sample guarantee is about (interpolating between scores, NumPy's
    default, can undercover). Infinite when n is too small for the level."""
    s = np.asarray(scores, float).ravel()
    s = s[np.isfinite(s)]
    k = int(np.ceil((len(s) + 1) * (1.0 - alpha)))
    if len(s) == 0 or k > len(s):
        return float("inf")
    return float(np.partition(s, k - 1)[k - 1])


class ConformalRegressor:
    """Distribution-free, finite-sample prediction intervals.

    Uses the absolute residual as the nonconformity score and the
    empirical (1 - alpha) quantile over a held-out calibration set.
    One width for every prediction: on rivers, where errors grow with flow
    and differ by orders of magnitude between catchments, this covers on
    average and fails exactly in floods (see reports/uncertainty.md);
    GroupedLogConformal is the one to use for discharge.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        self.alpha = alpha
        self.q = None  # calibrated quantile of residuals

    def calibrate(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        """Fit the score quantile on a calibration split (must NOT be train data)."""
        self.q = conformal_quantile(np.abs(np.asarray(y) - np.asarray(y_hat)), self.alpha)
        return self.q

    def predict_interval(self, y_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        return y_hat - self.q, y_hat + self.q

    def coverage(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        lo, hi = self.predict_interval(y_hat)
        return float(np.mean((y >= lo) & (y <= hi)))


class GroupedLogConformal:
    """Split conformal for discharge: scores in log space, quantiles per group.

    - Log space: flow errors scale with flow, so |log(y + e) - log(y_hat + e)|
      is closer to identically distributed than |y - y_hat|. Intervals become
      multiplicative, never negative, and wider on the side floods go.
    - Groups (Mondrian conformal; Vovk et al. 2005): a separate quantile per
      group, e.g. per catchment or per flow regime, gives coverage within each
      group rather than only on average. Groups with fewer than `min_size`
      calibration points use the pooled quantile.
    `offset` (e) keeps dry days finite; pass it per point (e.g. a fraction of
    each catchment's mean flow) or as one number.
    """

    def __init__(self, alpha: float = 0.10, min_size: int = 100) -> None:
        self.alpha = alpha
        self.min_size = min_size
        self.q: dict = {}
        self.q_all: float | None = None

    @staticmethod
    def _score(y_hat, y, offset):
        return np.abs(np.log(np.asarray(y) + offset) - np.log(np.asarray(y_hat) + offset))

    def calibrate(self, y_hat, y, groups=None, offset=1e-3) -> GroupedLogConformal:
        s = self._score(y_hat, y, offset)
        self.q_all = conformal_quantile(s, self.alpha)
        if groups is not None:
            frame = pd.DataFrame({"s": s, "g": np.asarray(groups)})
            frame = frame[np.isfinite(frame["s"])]
            self.q = {
                g: conformal_quantile(v["s"].to_numpy(), self.alpha)
                for g, v in frame.groupby("g")
                if len(v) >= self.min_size
            }
        return self

    def quantiles(self, n: int, groups=None) -> np.ndarray:
        if self.q_all is None:
            raise RuntimeError("Call calibrate() before predict_interval().")
        if groups is None:
            return np.full(n, self.q_all)
        return pd.Series(np.asarray(groups)).map(self.q).fillna(self.q_all).to_numpy(float)

    def predict_interval(self, y_hat, groups=None, offset=1e-3, q=None):
        y_hat = np.asarray(y_hat, float)
        q = self.quantiles(len(y_hat), groups) if q is None else q
        base = np.log(np.clip(y_hat, 0, None) + offset)
        lo = np.clip(np.exp(base - q) - offset, 0, None)
        return lo, np.exp(base + q) - offset


def adaptive_quantiles(cal_scores, scores, alpha: float = 0.10, gamma: float = 0.005,
                       delay: int = 1) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Adaptive conformal inference (Gibbs & Candes, NeurIPS 2021) along one
    series. The miscoverage level moves with the misses seen so far,
    alpha_t+1 = alpha_t + gamma (alpha - miss_t), so long-run coverage holds
    even when the river drifts away from the calibration years. A forecast
    `delay` days ahead is checked `delay` days after it was issued, so step t
    only learns from steps up to t - delay.

    `scores[t]` is the realized nonconformity score (NaN if unobserved).
    Returns the score quantile used at each step and whether it missed."""
    cal = np.sort(np.asarray(cal_scores, float)[np.isfinite(cal_scores)])
    n = len(cal)
    scores = np.asarray(scores, float)
    q = np.empty(len(scores))
    miss = np.full(len(scores), np.nan)
    level = alpha
    for t in range(len(scores)):
        j = t - delay
        if j >= 0 and np.isfinite(miss[j]):
            level += gamma * (alpha - miss[j])
        k = int(np.ceil((n + 1) * (1.0 - level)))
        q[t] = np.inf if k > n else (-np.inf if k < 1 else cal[k - 1])
        if np.isfinite(scores[t]):
            miss[t] = float(scores[t] > q[t])
    return q, miss


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