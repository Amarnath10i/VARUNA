"""Smoke tests that validate the full stack without heavy training or network.

Run: python -m pytest -q   (or)   python tests/test_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varuna.engines.risk_engine import RiskEngine, RiskPolicy
from varuna.uncertainty.conformal import ConformalRegressor, ConformalSegmentation
from varuna.uncertainty.losses import crps_gaussian, gaussian_nll, raincast_loss
from varuna.utils.metrics import iou, kge, nse

HAS_TORCH = True
try:
    import torch

    from varuna.models.floodmap import FloodMap
    from varuna.models.hydronet import HydroNet
    from varuna.models.raincast import RainCast
except Exception:
    HAS_TORCH = False


def test_metrics():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([1.1, 2.0, 2.8, 4.2])
    assert abs(nse(obs, sim) - 0.982) < 0.01
    assert kge(obs, sim) < 1.0
    a = np.array([[1, 1, 0], [0, 1, 0]])
    b = np.array([[1, 0, 0], [0, 1, 1]])
    assert abs(iou(a, b) - 0.5) < 1e-6


def test_crps_loss():
    mu = torch.zeros(2, 1, 4, 4) if HAS_TORCH else None
    if mu is None:
        return
    sigma = torch.ones_like(mu)
    y = torch.zeros_like(mu)
    val = crps_gaussian(mu, sigma, y).item()
    # perfect Gaussian forecast at its mean: sigma*(2*phi(0) - 1/sqrt(pi)) with sigma=1
    expected = 2.0 / np.sqrt(2 * np.pi) - 1.0 / np.sqrt(np.pi)
    assert abs(val - expected) < 1e-4


def test_conformal_coverage():
    rng = np.random.default_rng(0)
    mu = rng.normal(0, 1, 500)
    y = mu + rng.normal(0, 1, 500)
    reg = ConformalRegressor(alpha=0.10)
    reg.calibrate(mu[:250], y[:250])
    cov = reg.coverage(mu[250:], y[250:])
    assert cov >= 0.85
    seg = ConformalSegmentation(alpha=0.10)
    p = rng.random((10, 10))
    m = (p > 0.5).astype(np.uint8)
    seg.calibrate(p, m)
    assert seg.to_binary(p).shape == (10, 10)


def test_risk_engine():
    pol = RiskPolicy(region_id="test", discharge_threshold=100.0, rainfall_threshold_mm=50.0, flood_cover_fraction=0.2)
    eng = RiskEngine(pol)
    mean = np.array([120.0])
    lo = np.array([110.0])
    hi = np.array([130.0])
    d = eng.evaluate(mean, lo, hi, rainfall_accum_3d=60.0, flood_prob=np.full((10, 10), 0.7), exceed_prob=0.4)
    assert d.triggered is True
    assert "discharge" in d.trigger_type
    assert d.details["payout"] == pol.payout_per_event
    quiet = eng.evaluate(mean, lo, hi, rainfall_accum_3d=10.0, exceed_prob=0.05)
    assert quiet.triggered is False and quiet.details["payout"] == 0.0

    legacy = RiskEngine(RiskPolicy("test", 100.0, 50.0, 0.2, trigger_rule="lower_bound"))
    assert legacy.evaluate(mean, lo, hi, rainfall_accum_3d=10.0).triggered is True
    wide = legacy.evaluate(mean, np.array([60.0]), np.array([400.0]), rainfall_accum_3d=10.0)
    assert wide.triggered is False  # the lower bound misses the threshold


def test_models_forward():
    if not HAS_TORCH:
        return
    torch.manual_seed(0)

    rc = RainCast(in_channels=5, hidden_channels=32, n_blocks=2, forecast_horizon=3)
    x = torch.randn(2, 5, 32, 32)
    mu, logvar = rc.predict_distribution(x)
    assert mu.shape == (2, 3, 1, 32, 32)
    out = raincast_loss(mu, logvar, torch.randn_like(mu))
    assert torch.isfinite(out["loss"])

    hn = HydroNet(num_nodes=6, node_feat_dim=8, hidden_dim=16)
    # Model expects total features = dyn_dim + static_dim where dyn_dim = (hidden+node_feat) - total
    # With hidden=16, node_feat=8: total must be 16 (so dyn_dim=8, static=8)
    feat = torch.randn(5, 6, 16)  # [T, N, dyn+static]
    ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 2, 3, 4, 5, 0]], dtype=torch.long)
    ts = torch.linspace(0, 1, 5)
    mu2, lv2, _ = hn(feat, ei, ts)
    assert mu2.shape == (5, 6, 1)

    fm = FloodMap(img_size=32, embed_dim=48, depth=2, nhead=4, mask_ratio=0.5)
    sar = torch.randn(1, 2, 32, 32)
    opt_img = torch.randn(1, 3, 32, 32)
    outm = fm.pretrain_loss(sar, opt_img)
    assert torch.isfinite(outm["loss"])
    prob = fm.flood_prob(sar, opt_img)
    assert prob.shape == (1, 1, 32, 32)


def test_live_pipeline_refuses_to_decide_without_real_inputs():
    import json

    from varuna.pipeline.forecast import ForecastPipeline

    cfg = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    pipe = ForecastPipeline(cfg, run_dir="runs/test_cycle")
    assert pipe.run() == []
    out = json.loads((pipe.run_dir / "latest_cycle.json").read_text())
    assert out["status"] == "no_decision" and out["decisions"] == []
    assert any("benchmarked" in r for r in out["reasons"])
    assert out["feeds"]["gfs"]["placeholder"]  # GRIB2 is never decoded


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("All smoke tests passed.")