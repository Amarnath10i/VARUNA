"""Conformal intervals: the finite-sample quantile, coverage per group, and
adaptive coverage under drift."""

import numpy as np
import pytest

from varuna.uncertainty.conformal import (
    ConformalRegressor,
    GroupedLogConformal,
    adaptive_quantiles,
    conformal_quantile,
)


def test_quantile_is_the_right_order_statistic():
    s = np.arange(1.0, 10.0)  # n = 9, alpha = 0.1 -> ceil(10 * 0.9) = 9th smallest
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s[:8], 0.1) == np.inf  # too few points for 90%
    assert conformal_quantile(np.arange(1.0, 101.0), 0.1) == 91.0


def test_split_conformal_covers_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cover = []
    for _ in range(200):
        mu = rng.normal(size=60)
        y = mu + rng.standard_t(3, size=60)
        reg = ConformalRegressor(0.1)
        reg.calibrate(mu[:30], y[:30])
        cover.append(reg.coverage(mu[30:], y[30:]))
    assert np.mean(cover) >= 0.89  # guarantee: >= 0.9 in expectation


def heteroscedastic(rng, n):
    """Two rivers: errors proportional to flow, flows 100x apart."""
    g = rng.integers(0, 2, n)
    flow = np.where(g == 0, 1.0, 100.0) * rng.lognormal(0, 0.5, n)
    y = flow * rng.lognormal(0, 0.3, n)
    return g, flow, y


def test_one_width_fails_the_small_river_grouped_log_does_not():
    rng = np.random.default_rng(1)
    g, f, y = heteroscedastic(rng, 20000)
    gt, ft, yt = heteroscedastic(rng, 20000)
    glob = ConformalRegressor(0.1)
    glob.calibrate(f, y)
    lo, hi = glob.predict_interval(ft)
    inside = (yt >= lo) & (yt <= hi)
    assert inside[gt == 0].mean() > 0.99  # far too wide for the small river
    assert inside[gt == 1].mean() < 0.85  # too narrow for the big one

    grp = GroupedLogConformal(0.1).calibrate(f, y, groups=g)
    lo, hi = grp.predict_interval(ft, groups=gt)
    inside = (yt >= lo) & (yt <= hi)
    for k in (0, 1):
        assert inside[gt == k].mean() == pytest.approx(0.9, abs=0.015)
    assert (lo >= 0).all()


def test_small_groups_fall_back_to_the_pooled_quantile():
    rng = np.random.default_rng(2)
    f = rng.lognormal(size=500)
    y = f * rng.lognormal(0, 0.3, 500)
    groups = np.r_[np.zeros(450), np.ones(50)]
    c = GroupedLogConformal(0.1, min_size=100).calibrate(f, y, groups=groups)
    assert 1 not in c.q and 0 in c.q
    assert c.quantiles(1, groups=[1])[0] == c.q_all


def test_adaptive_conformal_recovers_coverage_after_drift():
    rng = np.random.default_rng(3)
    cal = np.abs(rng.normal(size=2000))
    later = np.abs(rng.normal(scale=2.0, size=6000))  # errors double after calibration
    fixed_miss = later > conformal_quantile(cal, 0.1)
    q, miss = adaptive_quantiles(cal, later, alpha=0.1, gamma=0.01, delay=3)
    assert fixed_miss.mean() > 0.3
    assert miss[1000:].mean() == pytest.approx(0.1, abs=0.02)


def test_adaptive_conformal_waits_for_outcomes():
    cal = np.linspace(0, 1, 101)
    scores = np.r_[np.full(5, 5.0), np.zeros(5)]  # every early step misses
    q, _ = adaptive_quantiles(cal, scores, alpha=0.1, gamma=0.05, delay=3)
    assert (q[:3] == q[0]).all()  # nothing is known until step 3
    assert q[3] > q[0]


def test_adaptive_conformal_skips_unobserved_steps():
    cal = np.linspace(0, 1, 101)
    q, miss = adaptive_quantiles(cal, np.full(10, np.nan), delay=1)
    assert np.isnan(miss).all() and (q == q[0]).all()
