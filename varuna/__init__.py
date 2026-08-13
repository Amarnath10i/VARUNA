"""VARUNA - Uncertainty-aware neural-operator flood early-warning system.

VARUNA couples three novel learning components into one operational loop:

1. RainCast     : uncertainty-aware Fourier Neural Operator (FNO) for
                  sub-seasonal extreme-rainfall forecasting with a
                  calibrated spectral predictive distribution.
2. HydroNet     : a Graph-Neural-ODE that propagates rainfall through the
                  river network and emits discharge with per-step
                  conformal prediction intervals.
3. FloodMap     : a cross-modal SAR <-> optical masked autoencoder that
                  maps floods in near-real-time from Sentinel-1 with
                  few-shot supervision.

Plus a RiskEngine that converts calibrated forecasts into parametric
insurance triggers and village-level early warnings.
"""

__version__ = "0.1.0"

from varuna.uncertainty import losses
from varuna.uncertainty import conformal

__all__ = ["losses", "conformal", "__version__"]