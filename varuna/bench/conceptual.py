"""Baselines the learned models have to beat, on the same days:

- climatology: each catchment's mean flow for the day of the year (training
  years), which in monsoon catchments already explains much of the variance;
- GR4J calibrated on each catchment's own 1991-2015 flows;
- the best of 47 conceptual models (MARRMoT, Mangukiya et al., "Benchmarking
  Conceptual Rainfall-Runoff Models in Peninsular India", Zenodo 18323832),
  chosen per catchment on 2010-2015.

The 47-model record does not state its calibration years. Recomputing NSE
on 2010-2020 reproduces its published validation NSE (median over 47 models
within 0.01) for most catchments, so its calibration ends before 2010 and its
2016-2020 flows are out of sample. Its published metrics include our test
years, so the per-catchment model choice uses 2010-2015 only.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import gr4j
from .data import CACHE, ROOT, Camels, period_mask
from .metrics import nse

log = logging.getLogger(__name__)

MARRMOT = ROOT / "data" / "raw" / "rr_benchmark" / "x"
SELECT = ("2010-01-01", "2015-12-31")


def climatology(daily: pd.DataFrame, half_window: int = 15) -> pd.Series:
    """Mean training-period flow for each gauge and day of year, smoothed
    with a circular +-15-day window."""
    train = daily[period_mask(daily["date"], "train")]
    doy = train["date"].dt.dayofyear.clip(upper=365)
    table = (
        train.groupby([train["gauge"], doy])["q"].mean().unstack().reindex(columns=range(1, 366))
    )
    vals = table.to_numpy()
    k = 2 * half_window + 1
    padded = np.concatenate([vals[:, -half_window:], vals, vals[:, :half_window]], axis=1)
    sums = np.nancumsum(np.c_[np.zeros(len(vals)), padded], axis=1)
    counts = np.cumsum(np.c_[np.zeros(len(vals)), np.isfinite(padded)], axis=1)
    smooth = (sums[:, k:] - sums[:, :-k]) / np.maximum(counts[:, k:] - counts[:, :-k], 1)
    table = pd.DataFrame(smooth, index=table.index, columns=table.columns)
    d = daily["date"].dt.dayofyear.clip(upper=365).to_numpy()
    pos = table.index.get_indexer(daily["gauge"])
    return pd.Series(table.to_numpy()[pos, d - 1], index=daily.index, name="q_clim")


def _fit_one(args):
    g, p, e, q, fit = args
    params, score = gr4j.calibrate(p, e, q, fit, seed=int(g) % 1000, maxiter=100)
    return g, params, score, gr4j.simulate(p, e, *params)[0]


def calibrate_gr4j(c: Camels, workers: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Per-catchment GR4J on the training years; the simulation runs from
    1980, so 1980-1990 forcing warms the stores without using their flows."""
    d = c.daily
    fit_all = period_mask(d["date"], "train")
    jobs = []
    for g, idx in d.groupby("gauge").indices.items():
        sub = d.iloc[idx]
        jobs.append(
            (
                g,
                sub["prcp"].to_numpy(np.float64),
                sub["pet_gleam"].to_numpy(np.float64),
                sub["q"].to_numpy(np.float64),
                fit_all[idx],
            )
        )
    sim = pd.Series(np.nan, index=d.index, name="q_gr4j")
    rows = []
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    with ProcessPoolExecutor(workers) as ex:
        for g, params, score, q in ex.map(_fit_one, jobs, chunksize=4):
            sim.iloc[d.groupby("gauge").indices[g]] = q
            rows.append({"gauge": g, "x1": params[0], "x2": params[1], "x3": params[2],
                         "x4": params[3], "train_nse": score})  # fmt: skip
    return pd.DataFrame(rows).set_index("gauge").sort_index(), sim


def marrmot(c: Camels, root: Path = MARRMOT) -> tuple[pd.Series, pd.DataFrame]:
    """Per-catchment best of the 47 conceptual models by NSE on 2010-2015."""
    d = c.daily
    out = pd.Series(np.nan, index=d.index, name="q_marrmot")
    rows = []
    groups = d.groupby("gauge").indices
    for f in sorted((root / "simulated_streamflow").glob("*.csv")):
        g = int(f.stem)
        if g not in groups:
            continue
        sim = pd.read_csv(f)
        sim.index = pd.to_datetime(sim[["year", "month", "day"]])
        models = [k for k in sim.columns if k.startswith("m_")]
        sub = d.iloc[groups[g]].set_index("date")
        sim = sim.reindex(sub.index)
        sel = (sub.index >= SELECT[0]) & (sub.index <= SELECT[1])
        obs = sub["q"].to_numpy()[sel]
        scores = {k: nse(obs, sim[k].to_numpy()[sel]) for k in models}
        valid = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not valid:
            continue
        best = max(valid, key=valid.get)
        out.iloc[groups[g]] = sim[best].to_numpy()
        rows.append({"gauge": g, "model": best, "select_nse": valid[best], **scores})
    return out, pd.DataFrame(rows).set_index("gauge")


def run(c: Camels) -> pd.DataFrame:
    """Simulations of every baseline, cached alongside the data."""
    path = CACHE / "sims_conceptual.parquet"
    if path.exists():
        return pd.read_parquet(path)
    d = c.daily
    params, q_gr4j = calibrate_gr4j(c)
    params.to_parquet(CACHE / "gr4j_params.parquet")
    q_mm, choice = marrmot(c)
    choice.to_parquet(CACHE / "marrmot_selection.parquet")
    sims = pd.DataFrame({"gauge": d["gauge"], "date": d["date"], "q_clim": climatology(d),
                         "q_gr4j": q_gr4j, "q_marrmot": q_mm})  # fmt: skip
    sims.to_parquet(path, index=False)
    log.info("GR4J median train NSE %.3f; 47-model benchmark covers %d catchments",
             params["train_nse"].median(), len(choice))  # fmt: skip
    return sims
