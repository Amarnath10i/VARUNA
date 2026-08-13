"""Distributional loss functions for uncertainty-aware forecasting.

Novelty note
-----------
Classical neural operators (FNO, DeepONet) are trained with plain MSE and
output point predictions, which are useless for early-warning thresholds.
VARUNA trains RainCast with a *CRPS-regularised spectral Gaussian
negative log-likelihood*: the network emits (mu, sigma) at every output
pixel and the sigma field is parametrised in the spectral domain so that
per-wavenumber uncertainty is modelled, not just per-pixel variance.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood.

    Args:
        mu: predicted mean            [B, C, H, W]
        logvar: predicted log-variance
        y: ground truth

    Returns:
        Scalar NLL (mean over batch/channels/spatial).
    """
    var = torch.exp(logvar).clamp_min(1e-6)
    return (logvar + (y - mu) ** 2 / var).mean() / 2.0


def crps_gaussian(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Closed-form Continuous Ranked Probability Score for a Gaussian forecast.

    CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ],  z = (y-mu)/sigma

    CRPS is the strictly-proper score for probabilistic forecasts; it jointly
    rewards calibration and sharpness, which is what an early-warning system
    needs at the decision threshold.
    """
    z = (y - mu) / sigma.clamp_min(1e-6)
    normal = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
    phi = torch.exp(normal.log_prob(z))
    Phi = normal.cdf(z)
    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / torch.sqrt(torch.tensor(torch.pi, device=sigma.device)))
    return crps.mean()


def raincast_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    y: torch.Tensor,
    crps_weight: float = 1.0,
    nll_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined loss for RainCast.

    Returns a dict so training logs can track each term separately.
    """
    sigma = torch.exp(0.5 * logvar)
    nll = gaussian_nll(mu, logvar, y)
    crps = crps_gaussian(mu, sigma, y)
    mse = F.mse_loss(mu, y)
    total = nll_weight * nll + crps_weight * crps + 0.1 * mse
    return {"loss": total, "nll": nll, "crps": crps, "mse": mse}


def quantile_loss(pred: torch.Tensor, y: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball (quantile) loss for arbitrary quantile regression."""
    err = y - pred
    return torch.where(err >= 0, tau * err, (tau - 1) * err).mean()