"""Flow 1-3 days ahead, using the flow observed on the day the forecast is issued.

Simulation models (GR4J, the LSTM) see only weather. An early-warning system
also knows today's river level, and at short leads that is most of the
signal, so every forecast here is compared with persistence (the flow `lead`
days later equals today's).

The forecaster is LightGBM on the engineered rainfall memory of `lgbm.py`
plus the observed flow up to the issue day. Rain between the issue day and
the target day is the observed rain: a perfect rainfall forecast, so these
are upper bounds on what a system fed by a numerical weather forecast would
reach. Trained on 1991-2015, early-stopped on 1980-1990, scored on 2016-2020.
"""

from __future__ import annotations

import pandas as pd

LEADS = (1, 2, 3)


def features(base: pd.DataFrame, lead: int) -> pd.DataFrame:
    """`base` from `lgbm.features`, sorted by gauge and date, plus what is
    observed on the issue day, `lead` days before the target day."""
    f = base.copy()
    g = f.groupby("gauge", sort=False)["q"]
    for k in range(3):
        f[f"q_obs_lag{k}"] = g.shift(lead + k)
    f["q_obs_mean7"] = g.transform(lambda s: s.shift(lead).rolling(7, min_periods=1).mean())
    f["q_obs_min30"] = g.transform(lambda s: s.shift(lead).rolling(30, min_periods=1).min())
    f["q_obs_max30"] = g.transform(lambda s: s.shift(lead).rolling(30, min_periods=1).max())
    f["rain_since_issue"] = f.groupby("gauge", sort=False)["p_lag0"].transform(
        lambda s: s.rolling(lead, min_periods=1).sum()
    )
    return f


def persistence(base: pd.DataFrame, lead: int) -> pd.Series:
    return base.groupby("gauge", sort=False)["q"].shift(lead)
