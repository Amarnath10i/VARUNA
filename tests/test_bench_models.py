"""The learned models and baselines on synthetic catchments: what they may
see (training years only), and the pieces they are built from."""

import numpy as np
import pandas as pd
import pytest
import torch

from varuna.bench import lgbm, lstm
from varuna.bench.conceptual import climatology
from varuna.bench.data import FORCING, STATIC, Camels, period_mask


def synthetic(n_gauges=3, seed=0) -> Camels:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1980-01-01", "2020-12-31")
    rows = []
    for g in range(n_gauges):
        p = np.where(rng.random(len(dates)) < 0.3, rng.gamma(0.8, 10, len(dates)), 0.0)
        q = np.zeros(len(dates))
        store = 0.0
        for t, pt in enumerate(p):  # a linear reservoir
            store = 0.9 * store + pt
            q[t] = 0.05 * (g + 1) * store
        f = {c: rng.normal(size=len(dates)) for c in FORCING}
        f["prcp"] = p
        rows.append(pd.DataFrame({"gauge": 100 + g, "date": dates, **f, "q": q}))
    daily = pd.concat(rows, ignore_index=True)
    daily.loc[daily.sample(frac=0.2, random_state=seed).index, "q"] = np.nan
    static = pd.DataFrame(rng.random((n_gauges, len(STATIC))) + 1, columns=STATIC,
                          index=pd.Index(100 + np.arange(n_gauges), name="gauge"))  # fmt: skip
    return Camels(daily, static, pd.DataFrame(index=static.index))


def scramble_outside_training(c: Camels) -> Camels:
    d = c.daily.copy()
    out = ~period_mask(d["date"], "train")
    d.loc[out, "q"] = np.random.default_rng(1).gamma(1, 50, out.sum())
    return Camels(d, c.static, c.names)


def test_lstm_never_trains_on_validation_or_test_flows():
    c = synthetic()
    weights = []
    for cc in (c, scramble_outside_training(c)):
        a = lstm.build(cc)
        net, _ = lstm.train(
            a, seed=0, epochs=1, windows_per_epoch=16, batch=8, hidden=8, device="cpu"
        )
        weights.append([p.detach().clone() for p in net.parameters()])
    for w0, w1 in zip(*weights, strict=True):
        assert torch.equal(w0, w1)


def test_lstm_standardizes_with_training_years_only():
    c = synthetic()
    a, b = lstm.build(c), lstm.build(scramble_outside_training(c))
    assert a.y_mean == b.y_mean and a.y_std == b.y_std


def test_training_windows_predict_inside_the_training_years():
    a = lstm.build(synthetic())
    w = lstm._windows(a, np.arange(3))
    first = a.dates[w[:, 1] + lstm.WARMUP]
    last = a.dates[w[:, 1] + lstm.WARMUP + lstm.HORIZON - 1]
    assert first.min() >= pd.Timestamp("1991-01-01")
    assert last.max() <= pd.Timestamp("2015-12-31")


def test_predict_fills_exactly_the_requested_years():
    a = lstm.build(synthetic())
    net = lstm.Net(len(FORCING), len(STATIC), hidden=4)
    out = lstm.predict([net], a, [2017], device="cpu")
    year = a.dates.year == 2017
    assert np.isfinite(out[:, year]).all() and np.isnan(out[:, ~year]).all()
    assert (out[:, year] >= 0).all()


def test_predict_does_not_depend_on_later_forcing():
    a = lstm.build(synthetic())
    net = lstm.Net(len(FORCING), len(STATIC), hidden=4)
    before = lstm.predict([net], a, [2017], device="cpu")
    a.x[:, a.year_start(2018) :] = 99.0
    after = lstm.predict([net], a, [2017], device="cpu")
    np.testing.assert_array_equal(before, after)


def test_antecedent_precipitation_index_half_life():
    p = np.zeros(40)
    p[0] = 1.0
    api = lgbm._api(p, 10)
    assert api[10] == pytest.approx(0.5) and api[20] == pytest.approx(0.25)


def test_climatology_uses_training_years_only():
    c = synthetic()
    a = climatology(c.daily)
    b = climatology(scramble_outside_training(c).daily)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy())
    assert a.notna().all()


def test_lgbm_features_look_only_backwards():
    c = synthetic(n_gauges=1)
    f0 = lgbm.features(c)
    d = c.daily.copy()
    d.loc[d["date"] >= "2018-01-01", "prcp"] = 500.0
    f1 = lgbm.features(Camels(d, c.static, c.names))
    before = f0["date"] < "2018-01-01"
    cols = [k for k in f0.columns if k not in ("gauge", "date", "q")]
    pd.testing.assert_frame_equal(f0.loc[before, cols], f1.loc[before, cols])


@pytest.mark.parametrize("lead", [1, 3])
def test_forecast_features_stop_at_the_issue_day(lead):
    from varuna.bench import forecast

    c = synthetic(n_gauges=1)
    base = lgbm.features(c)
    f0 = forecast.features(base, lead)
    target = 9000
    changed = base.copy()
    changed.loc[target - lead + 1 :, "q"] = 1e6  # flows after the issue day
    f1 = forecast.features(changed, lead)
    cols = [k for k in f0.columns if k.startswith("q_obs")]
    pd.testing.assert_series_equal(f0.loc[target, cols], f1.loc[target, cols])
    assert forecast.persistence(changed, lead)[target] == base.loc[target - lead, "q"] or np.isnan(
        base.loc[target - lead, "q"]
    )
