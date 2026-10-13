"""Scores every model on the test years and writes the comparison tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CACHE, Camels, period_mask, water_balance_outliers
from .metrics import bootstrap_median, per_catchment

MODELS = {
    "q_clim": "Climatology (day-of-year mean flow)",
    "q_gr4j": "GR4J, calibrated per catchment",
    "q_marrmot": "Best of 47 conceptual models, chosen per catchment",
    "q_lstm_pub": "CAMELS-IND regional LSTM (published predictions)",
    "q_lgbm": "LightGBM on engineered rainfall memory",
    "q_lstm_nostatic": "Regional LSTM without catchment attributes",
    "q_lstm_single": "Regional LSTM, one seed",
    "q_lstm": "**Regional LSTM, 3-seed ensemble**",
}
SIMS = ("sims_conceptual", "sims_lstm", "sims_lgbm", "sims_ungauged", "sims_forecast")


def with_sims(c: Camels) -> pd.DataFrame:
    d = c.daily[["gauge", "date", "prcp", "q", "q_lstm_pub"]]
    for name in SIMS:
        path = CACHE / f"{name}.parquet"
        if path.exists():
            d = d.merge(pd.read_parquet(path), on=["gauge", "date"], how="left")
    return d


def test_scores(c: Camels, d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Per-catchment test-period metrics, water-balance outliers excluded."""
    bad = water_balance_outliers(c)
    t = d[period_mask(d["date"], "test") & ~d["gauge"].isin(bad)]
    return per_catchment(t, models)


def summary(scores: pd.DataFrame, models: list[str], labels=MODELS) -> pd.DataFrame:
    rows = []
    for m in models:
        s = scores[scores["model"] == m].dropna(subset=["NSE"])
        if s.empty:
            continue
        med, lo, hi = bootstrap_median(s["NSE"])
        rows.append({
            "model": labels.get(m, m),
            "catchments": len(s),
            "median NSE (95% CI)": f"{med:.3f} ({lo:.3f} to {hi:.3f})",
            "median KGE": f"{s['KGE'].median():.3f}",
            "NSE > 0": f"{(s['NSE'] > 0).mean():.0%}",
            "peak-flow bias (FHV)": f"{s['FHV'].median():+.0f}%",
        })  # fmt: skip
    return pd.DataFrame(rows)


def paired(scores: pd.DataFrame, ref: str, others: list[str], labels=MODELS) -> pd.DataFrame:
    """Median per-catchment NSE gain of `ref` over each other model, on the
    catchments both cover."""
    w = scores.pivot(index="gauge", columns="model", values="NSE")
    rows = []
    for m in others:
        both = w[[ref, m]].dropna()
        if both.empty:
            continue
        diff = both[ref] - both[m]
        med, lo, hi = bootstrap_median(diff)
        rows.append({
            "versus": labels.get(m, m),
            "catchments": len(both),
            "median NSE gain (95% CI)": f"{med:+.3f} ({lo:+.3f} to {hi:+.3f})",
            "catchments improved": f"{(diff > 0).mean():.0%}",
        })  # fmt: skip
    return pd.DataFrame(rows)


def by_group(c: Camels, scores: pd.DataFrame, models: list[str], labels=MODELS) -> pd.DataFrame:
    """Median NSE by reservoir regulation and by aridity."""
    st = c.static
    ri = st["reservoir_index"]
    reg = pd.Series("unknown", index=st.index)
    reg[ri < 0.1] = "reservoir index < 0.1"
    reg[(ri >= 0.1) & (ri <= 1)] = "0.1 to 1"
    reg[ri > 1] = "> 1 (storage exceeds a year's flow)"
    q = st["aridity_p_pet"].quantile([1 / 3, 2 / 3]).to_numpy()
    ar = pd.Series(f"wet (P/PET > {q[1]:.2f})", index=st.index)
    ar[st["aridity_p_pet"] <= q[1]] = "middle"
    ar[st["aridity_p_pet"] <= q[0]] = f"dry (P/PET <= {q[0]:.2f})"
    w = scores.pivot(index="gauge", columns="model", values="NSE")[models]
    rows = []
    for name, groups in (("regulation", reg), ("aridity", ar)):
        keys = groups.reindex(w.index)
        order = list(dict.fromkeys([*groups.sort_values().unique()]))
        for gname in sorted(order, key=_rank):
            idx = keys.index[keys == gname]
            sub = w.loc[idx]
            rows.append({"split": name, "group": gname, "catchments": len(sub),
                         **{labels.get(m, m): f"{sub[m].median():.3f}" for m in models}})  # fmt: skip
    return pd.DataFrame(rows)


def _rank(name: str) -> int:
    for i, k in enumerate(
        ("reservoir index <", "0.1 to 1", "> 1", "unknown", "dry", "middle", "wet")
    ):
        if name.startswith(k):
            return i
    return 99


def outliers(c: Camels, d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    bad = water_balance_outliers(c)
    t = d[period_mask(d["date"], "test") & d["gauge"].isin(bad)]
    s = per_catchment(t, models)
    if s.empty:
        return s
    w = s.pivot(index="gauge", columns="model", values="NSE").round(3)
    w.columns = [MODELS.get(m, m).strip("*") for m in w.columns]
    ratio = (
        d[d["gauge"].isin(bad)].groupby("gauge").apply(lambda x: x["q"].mean() / x["prcp"].mean())
    )
    w.insert(0, "runoff / rainfall", ratio.round(2))
    return w


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "_none_"


def available(d: pd.DataFrame, models) -> list[str]:
    return [m for m in models if m in d.columns and np.isfinite(d[m]).any()]
