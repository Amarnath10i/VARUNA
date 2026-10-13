"""Flood-event scoring, regimes and flood probabilities on hand-made rivers."""

import numpy as np
import pandas as pd
import pytest

from varuna.bench import uq


def stats(threshold=10.0):
    return pd.DataFrame(
        {"q50": [1.0], "q90": [4.0], "q99": [threshold], "threshold": [threshold],
         "offset": [0.01], "mean": [2.0]},
        index=pd.Index([7], name="gauge"),
    )  # fmt: skip


def test_onsets_need_a_seen_rise():
    obs = np.array([1, 1, 20, 20, 1, np.nan, 20, 1], float)
    pred = np.array([1, 1, 15, 1, 1, 1, 15, 1], float)
    # the second event starts after a missing day: its rise wasn't seen
    assert uq.onset_warnings(obs, pred, 10) == (1, 1)
    assert uq.onset_warnings(obs, np.ones(8), 10) == (1, 0)


def test_persistence_never_warns_of_an_onset():
    rng = np.random.default_rng(0)
    obs = rng.gamma(0.5, 4.0, 2000)
    persistence = np.r_[np.nan, obs[:-1]]
    n, warned = uq.onset_warnings(obs, np.nan_to_num(persistence, nan=-1), 10)
    assert n > 20 and warned == 0


def test_regimes_follow_the_catchments_quantiles():
    r = uq.regime(np.array([0.5, 2.0, 5.0, 50.0]), np.full(4, 7), stats())
    assert r.tolist() == [0, 1, 2, 3]


def calibration(rng, n=5000):
    pred = rng.lognormal(0.5, 1.2, n)
    return pd.DataFrame({"gauge": 7, "q_sim": pred, "q": pred * rng.lognormal(0, 0.4, n)})


def test_flood_probability_rises_with_the_forecast_and_is_calibrated():
    rng = np.random.default_rng(1)
    cal, test = calibration(rng, 20000), calibration(rng, 100000)
    p = uq.exceedance_prob(cal, test, "q_sim", stats())
    pred = test["q_sim"].to_numpy()
    reg = uq.regime(pred, test["gauge"].to_numpy(), stats())
    for k in np.unique(reg):  # each regime has its own error distribution
        order = np.argsort(pred[reg == k])
        assert (np.diff(p[reg == k][order]) >= 0).all()
    flood = test["q"].to_numpy() > 10
    for lo, hi in ((0.1, 0.3), (0.3, 0.7), (0.7, 0.95)):
        m = (p >= lo) & (p < hi)
        assert flood[m].mean() == pytest.approx(p[m].mean(), abs=0.02)


def test_flood_probability_extremes():
    rng = np.random.default_rng(2)
    cal = calibration(rng)
    x = pd.DataFrame({"gauge": [7, 7], "q_sim": [0.01, 1000.0]})
    p = uq.exceedance_prob(cal, x, "q_sim", stats())
    assert p[0] < 0.01 and p[1] > 0.99
