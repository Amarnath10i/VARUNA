"""Flood events, prediction intervals and VARUNA's trigger rule, on real rivers.

- A flood day is a day above the catchment's 99th percentile of observed
  training-period flow (about 3.7 days a year). Events are runs of flood
  days. Onset warning: was the first day of an event forecast above the
  threshold (for a lead-h forecast, issued h days before)? Persistence can
  never warn of an onset, by construction.
- Intervals are calibrated on 1981-1990 predictions (out of sample for every
  model; those years only chose the early-stopping epoch) and checked on
  2016-2020.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from varuna.uncertainty.conformal import (
    ConformalRegressor,
    GroupedLogConformal,
    adaptive_quantiles,
)

from .data import Camels, period_mask
from .metrics import bootstrap_event_skill, event_scores, events

ALPHA = 0.10
REGIMES = (0.5, 0.9, 0.99)


def gauge_stats(c: Camels) -> pd.DataFrame:
    """Per catchment: flood threshold, regime cut points, log offset."""
    d = c.daily[period_mask(c.daily["date"], "train")]
    g = d.groupby("gauge")["q"]
    out = g.quantile(list(REGIMES)).unstack()
    out.columns = [f"q{int(100 * r)}" for r in REGIMES]
    out["threshold"] = out["q99"]
    out["offset"] = np.maximum(0.01 * g.mean(), 1e-3)
    out["mean"] = g.mean()
    return out


def regime(pred: np.ndarray, gauges: np.ndarray, stats: pd.DataFrame) -> np.ndarray:
    """0: below the catchment's median flow ... 3: above its 99th percentile."""
    cuts = stats.loc[gauges, ["q50", "q90", "q99"]].to_numpy()
    return (np.asarray(pred)[:, None] > cuts).sum(axis=1)


def onset_warnings(obs, pred, threshold) -> tuple[int, int]:
    """(event onsets, onsets with the forecast for that day above threshold)."""
    obs, pred = np.asarray(obs, float), np.asarray(pred, float)
    ev = events(np.isfinite(obs) & (obs > threshold))
    starts = [s for s, _ in ev if s > 0 and np.isfinite(obs[s - 1])]  # a seen rise
    return len(starts), int(sum(pred[s] > threshold for s in starts))


def flood_skill(d: pd.DataFrame, models: dict[str, str], stats: pd.DataFrame) -> pd.DataFrame:
    """Event POD/FAR/CSI pooled over catchments, and onset warnings."""
    rows = []
    for col, label in models.items():
        counts, onsets, warned = [], 0, 0
        for g, x in d.groupby("gauge"):
            thr = stats.at[g, "threshold"]
            o, p = x["q"].to_numpy(), x[col].to_numpy()
            p = np.where(np.isfinite(p), p, -np.inf)
            counts.append(event_scores(o, p, thr, window=0))
            n, w = onset_warnings(o, p, thr)
            onsets, warned = onsets + n, warned + w
        sk = bootstrap_event_skill(pd.DataFrame(counts))
        rows.append({
            "model": label,
            "events": sk["events"],
            "POD (95% CI)": "{:.2f} ({:.2f}-{:.2f})".format(*sk["POD"]),
            "FAR (95% CI)": "{:.2f} ({:.2f}-{:.2f})".format(*sk["FAR"]),
            "CSI": f"{sk['CSI'][0]:.2f}",
            "onsets warned": f"{warned / max(onsets, 1):.0%} of {onsets}",
        })  # fmt: skip
    return pd.DataFrame(rows)


def intervals(cal: pd.DataFrame, test: pd.DataFrame, col: str, stats: pd.DataFrame,
              lead: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:  # fmt: skip
    """90% intervals for `test[col]` from each method, calibrated on `cal`."""
    out = {}
    glob = ConformalRegressor(ALPHA)
    glob.calibrate(cal[col].to_numpy(), cal["q"].to_numpy())
    lo, hi = glob.predict_interval(test[col].to_numpy())
    out["one width for all (current VARUNA)"] = (np.clip(lo, 0, None), hi)

    off_c = stats.loc[cal["gauge"], "offset"].to_numpy()
    off_t = stats.loc[test["gauge"], "offset"].to_numpy()
    per_g = GroupedLogConformal(ALPHA).calibrate(cal[col], cal["q"], cal["gauge"], off_c)
    out["log scale, per catchment"] = per_g.predict_interval(test[col], test["gauge"], off_t)

    rc = regime(cal[col].to_numpy(), cal["gauge"].to_numpy(), stats)
    rt = regime(test[col].to_numpy(), test["gauge"].to_numpy(), stats)
    per_r = GroupedLogConformal(ALPHA).calibrate(cal[col], cal["q"], rc, off_c)
    out["log scale, per flow regime"] = per_r.predict_interval(test[col], rt, off_t)

    # adaptive: per catchment, starting from its own calibration scores
    q_t = np.full(len(test), np.nan)
    cal_s = GroupedLogConformal._score(cal[col], cal["q"], off_c)
    test_s = GroupedLogConformal._score(test[col], test["q"], off_t)
    cal_g, test_g = cal["gauge"].to_numpy(), test["gauge"].to_numpy()
    for g in np.unique(test_g):
        m = np.flatnonzero(test_g == g)  # test is sorted by date within gauge
        q_t[m], _ = adaptive_quantiles(cal_s[cal_g == g], test_s[m], ALPHA, 0.005, lead)
    q_t = np.where(np.isfinite(q_t), q_t, per_g.quantiles(len(test), test["gauge"]))
    out["log scale, per catchment, adaptive"] = per_g.predict_interval(
        test[col], None, off_t, q=q_t
    )
    return out


def coverage_table(test: pd.DataFrame, ints: dict, stats: pd.DataFrame) -> pd.DataFrame:
    y = test["q"].to_numpy()
    seen = np.isfinite(y)
    thr = stats.loc[test["gauge"], "threshold"].to_numpy()
    low = stats.loc[test["gauge"], "q50"].to_numpy()
    mean = stats.loc[test["gauge"], "mean"].to_numpy()
    rows = []
    for name, (lo, hi) in ints.items():
        inside = (y >= lo) & (y <= hi)
        per_g = pd.Series(inside[seen]).groupby(test["gauge"].to_numpy()[seen]).mean()
        flood = seen & (y > thr)
        rows.append({
            "intervals": name,
            "coverage": f"{inside[seen].mean():.1%}",
            "on flood days": f"{inside[flood].mean():.1%}",
            "below median flow": f"{inside[seen & (y < low)].mean():.1%}",
            "catchments within 85-95%": f"{per_g.between(0.85, 0.95).mean():.0%}",
            "median width / mean flow": f"{np.median(((hi - lo) / mean)[seen]):.2f}",
            "width on flood days": f"{np.median(((hi - lo) / mean)[flood]):.1f}",
        })  # fmt: skip
    return pd.DataFrame(rows)


def trigger_table(test: pd.DataFrame, col: str, ints: dict, stats: pd.DataFrame,
                  methods: list[str]) -> pd.DataFrame:  # fmt: skip
    """VARUNA's discharge trigger (fire when the interval's lower bound clears
    the threshold) against firing on the point forecast or the upper bound."""
    years = test["date"].dt.year.nunique()
    rows = []
    for name in methods:
        lo, hi = ints[name]
        for rule, signal in (("lower bound", lo), ("point forecast", test[col].to_numpy()),
                             ("upper bound", hi)):  # fmt: skip
            if rule == "point forecast" and name != methods[0]:
                continue
            t = test.assign(_s=signal)
            sk = flood_skill(t, {"_s": rule}, stats).iloc[0]
            fired = sum(
                len(events(np.isfinite(x["q"]) & (x["_s"] > stats.at[g, "threshold"])))
                for g, x in t.groupby("gauge")
            )
            rows.append({
                "fires when": rule,
                "intervals": "-" if rule == "point forecast" else name,
                "POD": sk["POD (95% CI)"].split(" ")[0],
                "FAR": sk["FAR (95% CI)"].split(" ")[0],
                "onsets warned": sk["onsets warned"],
                "triggers per catchment-year": f"{fired / (t['gauge'].nunique() * years):.2f}",
            })  # fmt: skip
    return pd.DataFrame(rows)
