import numpy as np
import pandas as pd
import pytest

from varuna.bench.metrics import (
    bootstrap_median,
    event_scores,
    events,
    kge,
    nse,
    peak_bias,
    pooled_event_skill,
)


def test_nse_reference_points():
    o = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert nse(o, o) == 1.0
    assert nse(o, np.full(5, 2.5)) == pytest.approx(0.0)  # the observed mean
    assert nse(o, o + 1) == pytest.approx(0.2)  # 1 - 4 / 5
    assert nse(o, o + 2) < 0


def test_kge_parts():
    o = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    total, r, alpha, beta = kge(o, 2 * o)
    assert r == pytest.approx(1.0) and alpha == pytest.approx(2.0) and beta == pytest.approx(2.0)
    assert total == pytest.approx(1 - np.sqrt(2.0))
    assert kge(o, o)[0] == pytest.approx(1.0)


def test_peak_bias_sees_only_floods():
    o = np.r_[np.ones(98), 50.0, 60.0]
    s = o.copy()
    s[:98] = 5.0  # wrong everywhere except the two peaks
    assert peak_bias(o, s) == pytest.approx(0.0)
    s[-2:] /= 2
    assert peak_bias(o, s) == pytest.approx(-50.0)


def test_events_are_runs():
    assert events(np.array([0, 1, 1, 0, 1, 0, 0, 1], bool)) == [(1, 2), (4, 4), (7, 7)]
    assert events(np.zeros(4, bool)) == []


def test_event_scores_hits_misses_false_alarms():
    obs = np.zeros(30)
    sim = np.zeros(30)
    obs[[3, 4]] = 10  # event 1: simulated a day late -> hit with 1-day slack
    sim[5] = 10
    obs[15] = 10  # event 2: missed
    sim[25] = 10  # false alarm
    r = event_scores(obs, sim, threshold=5)
    assert r == {"events": 2, "hits": 1, "misses": 1, "false_alarms": 1}
    assert event_scores(obs, sim, threshold=5, window=0)["hits"] == 0


def test_unobserved_days_are_neither_events_nor_false_alarms():
    obs = np.full(10, np.nan)
    sim = np.full(10, 9.0)
    assert event_scores(obs, sim, threshold=5) == {
        "events": 0,
        "hits": 0,
        "misses": 0,
        "false_alarms": 0,
    }


def test_pooled_skill():
    c = pd.DataFrame({"events": [2, 1], "hits": [1, 1], "misses": [1, 0], "false_alarms": [0, 2]})
    s = pooled_event_skill(c)
    assert s["POD"] == pytest.approx(2 / 3)
    assert s["FAR"] == pytest.approx(2 / 4)
    assert s["CSI"] == pytest.approx(2 / 5)


def test_bootstrap_median_brackets_point():
    v = np.random.default_rng(0).normal(0.6, 0.2, 200)
    m, lo, hi = bootstrap_median(v)
    assert lo < m < hi and hi - lo < 0.1
