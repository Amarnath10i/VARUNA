"""Regional gradient-boosted trees on hand-made catchment memory.

The LSTM learns how long rain is remembered; this baseline is given it:
recent daily rain, rainfall totals over 3 days to a year, and antecedent
precipitation indices (exponentially decaying rain, half-lives of 3 to 90
days), plus evaporative demand, the season and the static attributes. Like
the LSTM it sees no observed flow, and it is fitted with the same NSE*
weighting, so the comparison isolates learned versus engineered memory.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from .data import STATIC, Camels, period_mask

SUMS = [3, 7, 15, 30, 60, 90, 180, 365]
HALF_LIVES = [3, 10, 30, 90]


def _api(p: np.ndarray, half_life: float) -> np.ndarray:
    k = 0.5 ** (1.0 / half_life)
    out = np.empty_like(p)
    acc = 0.0
    for i, v in enumerate(p):
        acc = k * acc + v
        out[i] = acc
    return out


def features(c: Camels) -> pd.DataFrame:
    d = c.daily.sort_values(["gauge", "date"]).reset_index(drop=True)
    g = d.groupby("gauge", sort=False)
    f = pd.DataFrame({"gauge": d["gauge"], "date": d["date"]})
    for lag in range(7):
        f[f"p_lag{lag}"] = g["prcp"].shift(lag)
    for w in SUMS:
        f[f"p_sum{w}"] = g["prcp"].transform(lambda s, w=w: s.rolling(w, min_periods=1).sum())
    for h in HALF_LIVES:
        f[f"api{h}"] = g["prcp"].transform(
            lambda s, h=h: pd.Series(_api(s.to_numpy(float), h), s.index)
        )
    for w in (7, 30, 90, 365):
        f[f"pet_mean{w}"] = g["pet_gleam"].transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()
        )
    for col in ("tmax", "tmin", "srad_sw", "rel_hum", "wind"):
        f[col] = d[col]
    doy = d["date"].dt.dayofyear
    f["doy_sin"], f["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    st = c.static[STATIC].reindex(d["gauge"]).reset_index(drop=True)
    f = pd.concat([f, st.add_prefix("s_")], axis=1)
    f["q"] = d["q"]
    return f


def fit_predict(f: pd.DataFrame, train_gauges, predict_gauges, seed: int = 0) -> pd.Series:
    """Fit on the training years of `train_gauges` (early stopping on their
    validation years), predict every day of `predict_gauges`."""
    cols = [c for c in f.columns if c not in ("gauge", "date", "q")]
    tr = f["gauge"].isin(train_gauges) & f["q"].notna()
    fit = tr & period_mask(f["date"], "train")
    val = tr & period_mask(f["date"], "valid")
    sd = f[fit].groupby("gauge")["q"].std()
    w = 1.0 / (f["gauge"].map(sd) / f.loc[fit, "q"].std() + 0.1) ** 2
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "seed": seed,
        "verbose": -1,
        "num_threads": 12,
    }
    dtr = lgb.Dataset(f.loc[fit, cols], f.loc[fit, "q"], weight=w[fit])
    dva = lgb.Dataset(f.loc[val, cols], f.loc[val, "q"], weight=w[val], reference=dtr)
    booster = lgb.train(params, dtr, 3000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])  # fmt: skip
    out = pd.Series(np.nan, index=f.index)
    m = f["gauge"].isin(predict_gauges)
    out[m] = np.clip(booster.predict(f.loc[m, cols], num_iteration=booster.best_iteration), 0, None)
    return out
