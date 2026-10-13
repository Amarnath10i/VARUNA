"""Streamflow metrics, flood-event scores and uncertainty across catchments.

Every metric is computed per catchment on days with an observation, then
summarized by the median over catchments, as in the hydrology benchmarks
(Kratzert et al. 2019; Newman et al. 2017). Intervals come from a bootstrap
over catchments, because days within a catchment are strongly correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _valid(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    m = np.isfinite(obs) & np.isfinite(sim)
    return obs[m], sim[m]


def nse(obs, sim) -> float:
    """Nash-Sutcliffe efficiency: 1 is perfect, 0 is no better than the mean
    of the observations over the same days."""
    o, s = _valid(obs, sim)
    if len(o) < 2 or o.var() == 0:
        return np.nan
    return float(1.0 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2))


def kge(obs, sim) -> tuple[float, float, float, float]:
    """Kling-Gupta efficiency (Gupta et al. 2009) and its parts: correlation,
    variability ratio, bias ratio."""
    o, s = _valid(obs, sim)
    if len(o) < 2 or o.std() == 0 or o.mean() == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(o, s)[0, 1] if s.std() > 0 else 0.0
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)), r, alpha, beta


def peak_bias(obs, sim, top: float = 0.02) -> float:
    """Percent bias of flow volume on the highest `top` fraction of observed
    days (FHV, Yilmaz et al. 2008): negative means floods are underestimated."""
    o, s = _valid(obs, sim)
    if len(o) < 50:
        return np.nan
    idx = np.argsort(o)[::-1][: max(1, int(top * len(o)))]
    return float(100 * (s[idx].sum() - o[idx].sum()) / o[idx].sum())


def events(exceed: np.ndarray) -> list[tuple[int, int]]:
    """Runs of consecutive True as (first, last) index pairs."""
    x = np.concatenate([[False], np.asarray(exceed, bool), [False]])
    d = np.diff(x.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1, strict=True))


def event_scores(obs, sim, threshold: float, window: int = 1) -> dict[str, int]:
    """Flood events as runs of days above `threshold`. An observed event is
    hit when the simulation crosses the threshold within `window` days of it;
    a simulated event that overlaps no observed event (with the same slack)
    is a false alarm. Days with no observation are neither."""
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    seen = np.isfinite(obs)
    o_ev = events(seen & (obs > threshold))
    s_ev = events(seen & (sim > threshold))

    def near(a, b):
        return a[0] - window <= b[1] and b[0] - window <= a[1]

    hits = sum(any(near(e, s) for s in s_ev) for e in o_ev)
    false = sum(not any(near(s, e) for e in o_ev) for s in s_ev)
    return {"events": len(o_ev), "hits": hits, "misses": len(o_ev) - hits, "false_alarms": false}


def pooled_event_skill(counts: pd.DataFrame) -> dict[str, float]:
    """POD, FAR and CSI from event counts summed over catchments."""
    h, m, f = counts["hits"].sum(), counts["misses"].sum(), counts["false_alarms"].sum()
    return {
        "events": int(counts["events"].sum()),
        "POD": h / max(h + m, 1),
        "FAR": f / max(h + f, 1),
        "CSI": h / max(h + m + f, 1),
    }


def bootstrap_median(values, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Median over catchments with a 95% percentile-bootstrap interval."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    rng = np.random.default_rng(seed)
    meds = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return float(np.median(v)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def bootstrap_event_skill(counts: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Pooled POD/FAR/CSI with intervals from resampling catchments."""
    point = pooled_event_skill(counts)
    rng = np.random.default_rng(seed)
    arr = counts[["hits", "misses", "false_alarms"]].to_numpy(float)
    draws = arr[rng.integers(0, len(arr), (n, len(arr)))].sum(axis=1)
    h, m, f = draws.T
    sk = {
        "POD": h / np.maximum(h + m, 1),
        "FAR": f / np.maximum(h + f, 1),
        "CSI": h / np.maximum(h + m + f, 1),
    }
    out = {"events": point["events"]}
    for k, d in sk.items():
        out[k] = (point[k], float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
    return out


def per_catchment(df: pd.DataFrame, sims: list[str], obs: str = "q") -> pd.DataFrame:
    """NSE, KGE and peak bias for every catchment and simulation column."""
    rows = []
    for g, d in df.groupby("gauge", sort=True):
        o = d[obs].to_numpy()
        if np.isfinite(o).sum() < 365:  # at least a year of observations
            continue
        for s in sims:
            k = kge(o, d[s].to_numpy())
            rows.append(
                {
                    "gauge": g,
                    "model": s,
                    "NSE": nse(o, d[s].to_numpy()),
                    "KGE": k[0],
                    "r": k[1],
                    "alpha": k[2],
                    "beta": k[3],
                    "FHV": peak_bias(o, d[s].to_numpy()),
                    "days": int(np.isfinite(o).sum()),
                }
            )
    return pd.DataFrame(rows)
