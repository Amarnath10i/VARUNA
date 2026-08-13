"""Lightweight visualisation helpers (matplotlib optional)."""
from __future__ import annotations

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False


def plot_forecast_with_interval(
    t: np.ndarray,
    obs: np.ndarray,
    mu: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    save_path: str,
    title: str = "Discharge forecast with conformal interval",
) -> str | None:
    if not _HAS_MPL:
        return None
    plt.figure(figsize=(10, 4))
    plt.plot(t, obs, "k-", label="Observed")
    plt.plot(t, mu, "b--", label="Forecast mean")
    plt.fill_between(t, lo, hi, color="b", alpha=0.2, label="90% conformal interval")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path


def plot_flood_map(prob: np.ndarray, mask: np.ndarray | None, save_path: str) -> str | None:
    if not _HAS_MPL:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    im0 = axes[0].imshow(prob, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Predicted flood prob.")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    if mask is not None:
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Ground truth (if provided)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    return save_path