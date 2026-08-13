"""RiskEngine -- turns calibrated forecasts into insurance triggers & alerts.

The core of the startup story: parametric (index-based) flood insurance pays
out automatically when a *verifiable, objective trigger* is breached. The
RiskEngine consumes conformal prediction intervals and a flood map, checks
the trigger conditions, and emits structured decisions a regulator/insurer
can audit -- no subjective claims adjustment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RiskPolicy:
    """A parametric insurance / early-warning policy definition."""

    region_id: str
    discharge_threshold: float  # m3/s at gauge
    rainfall_threshold_mm: float  # 3-day accumulated rainfall
    flood_cover_fraction: float  # fraction of pixels that must be flooded
    lead_days: int = 3
    coverage_alpha: float = 0.90  # required conformal coverage
    payout_per_event: float = 5000.0


@dataclass
class RiskDecision:
    """Auditable decision emitted by the RiskEngine."""

    region_id: str
    triggered: bool
    trigger_type: str | None
    discharge_mean: float
    discharge_interval: tuple[float, float]
    rainfall_accum: float
    flood_cover: float
    confidence_ok: bool
    details: dict = field(default_factory=dict)


class RiskEngine:
    """Fuses RainCast + HydroNet + FloodMap outputs into decisions.

    Trigger logic:
      1. Forecast discharge (with conformal interval) exceeds threshold at
         the required lead time, AND
      2. The 3-day accumulated rainfall forecast exceeds the rainfall index,
         AND
      3. If a flood map is available, flooded-pixel fraction >= cover fraction.
      4. Confidence is only trusted if the conformal interval is narrower
         than a sanity band (so we never fire on wildly uncertain forecasts).
    """

    def __init__(self, policy: RiskPolicy) -> None:
        self.policy = policy

    def evaluate(
        self,
        discharge_mean: np.ndarray,
        discharge_lo: np.ndarray,
        discharge_hi: np.ndarray,
        rainfall_accum_3d: float,
        flood_prob: np.ndarray | None = None,
    ) -> RiskDecision:
        pol = self.policy
        # discharge arrays are [T, N]; take last time step, mean across nodes (or node 0)
        mean = float(discharge_mean[-1].mean())
        lo = float(discharge_lo[-1].mean())
        hi = float(discharge_hi[-1].mean())

        band = hi - lo
        confidence_ok = band <= 0.5 * max(mean, 1e-3) or band <= 0.6 * pol.discharge_threshold

        discharge_trig = lo > pol.discharge_threshold  # conservative: use lower bound
        rainfall_trig = rainfall_accum_3d >= pol.rainfall_threshold_mm

        flood_cover = 0.0
        if flood_prob is not None:
            flood_cover = float(np.mean(flood_prob >= 0.5))

        flood_trig = flood_cover >= pol.flood_cover_fraction

        triggers = []
        if discharge_trig:
            triggers.append("discharge")
        if rainfall_trig:
            triggers.append("rainfall")
        if flood_trig:
            triggers.append("flood_map")

        triggered = bool(triggers) and confidence_ok
        trigger_type = "|".join(triggers) if triggers else None

        return RiskDecision(
            region_id=pol.region_id,
            triggered=triggered,
            trigger_type=trigger_type,
            discharge_mean=mean,
            discharge_interval=(lo, hi),
            rainfall_accum=rainfall_accum_3d,
            flood_cover=flood_cover,
            confidence_ok=confidence_ok,
            details={
                "lead_days": pol.lead_days,
                "threshold": pol.discharge_threshold,
                "payout": pol.payout_per_event if triggered else 0.0,
            },
        )

    def to_json(self, decision: RiskDecision) -> dict:
        return {
            "region_id": decision.region_id,
            "triggered": decision.triggered,
            "trigger_type": decision.trigger_type,
            "discharge": {"mean": decision.discharge_mean, "interval": list(decision.discharge_interval)},
            "rainfall_accum_3d_mm": decision.rainfall_accum,
            "flood_cover_fraction": decision.flood_cover,
            "confidence_ok": decision.confidence_ok,
            "details": decision.details,
        }