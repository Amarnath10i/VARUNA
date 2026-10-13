"""VARUNA's decision chain on a past day, with real inputs and benchmarked models.

    python -m varuna.pipeline.hindcast 2018-08-16 15024 15021

For each CAMELS-IND gauge: the regional LSTM ensemble (trained by
`varuna-bench lstm`) simulates flow from the catchment's IMD forcing; a
conformal predictive distribution, built from the model's errors on
1981-1990, turns it into a 90% interval and a probability of exceeding the
gauge's flood threshold; the RiskEngine decides with the cut-off the backtest
chose (`varuna-bench conformal`). The observed flow, when the gauge recorded
one, is attached so every decision can be audited.

The simulation uses the day's observed rain, i.e. a perfect rainfall
forecast: this replays what the model makes of the rain, not what a live
system fed by a weather forecast would have known.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import numpy as np
import pandas as pd

from varuna.ai.gemini import GeminiRiskBrief
from varuna.bench import lstm, uq
from varuna.bench.data import CACHE, period_mask
from varuna.bench.data import load as load_camels
from varuna.bench.experiments import RUNS
from varuna.engines.risk_engine import RiskEngine, RiskPolicy
from varuna.uncertainty.conformal import GroupedLogConformal


@lru_cache(maxsize=1)
def _context():
    import torch

    c = load_camels()
    a = lstm.build(c)  # the same standardization the ensemble was trained with
    paths = sorted(RUNS.glob("lstm_seed*.pt"))
    if not paths:
        raise FileNotFoundError("no trained ensemble in runs/bench: run `varuna-bench lstm`")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for p in paths:
        net = lstm.Net(a.x.shape[-1], a.s.shape[-1]).to(device)
        net.load_state_dict(torch.load(p, map_location=device))
        models.append(net)
    stats = uq.gauge_stats(c)
    sims = pd.read_parquet(CACHE / "sims_lstm.parquet", columns=["gauge", "date", "q_lstm"])
    d = c.daily[["gauge", "date", "q", "prcp"]].merge(sims, on=["gauge", "date"])
    cal = d[period_mask(d["date"], "valid") & d["q"].notna() & d["q_lstm"].notna()]
    cut = json.loads((RUNS / "trigger_cutoffs.json").read_text())["q_lstm"]
    rain3 = d.groupby("gauge")["prcp"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    d["rain3"] = rain3
    rain_thr = d[period_mask(d["date"], "train")].groupby("gauge")["rain3"].quantile(0.99)
    return c, a, models, device, stats, cal, cut, d, rain_thr


def hindcast(date: str, gauges: list[int], llm_brief: bool = False) -> dict:
    c, a, models, device, stats, cal, cut, d, rain_thr = _context()
    day = pd.Timestamp(date)
    idx = np.flatnonzero(np.isin(a.gauges, gauges))
    sim = lstm.predict(models, a, [day.year], idx, device)[:, a.dates.get_loc(day)]
    regimes_cal = uq.regime(cal["q_lstm"].to_numpy(), cal["gauge"].to_numpy(), stats)
    conf = GroupedLogConformal(0.10).calibrate(
        cal["q_lstm"], cal["q"], regimes_cal, stats.loc[cal["gauge"], "offset"].to_numpy()
    )
    decisions = []
    for g, q_sim in zip(a.gauges[idx], sim, strict=True):
        x = pd.DataFrame({"gauge": [g], "q_lstm": [q_sim]})
        p = float(uq.exceedance_prob(cal, x, "q_lstm", stats)[0])
        reg = uq.regime(np.array([q_sim]), np.array([g]), stats)
        lo, hi = conf.predict_interval([q_sim], reg, stats.at[g, "offset"])
        area = c.static.at[g, "cwc_area"]
        to_m3s = area / 86.4  # mm/day over the catchment -> m3/s
        row = d[(d["gauge"] == g) & (d["date"] == day)].iloc[0]
        policy = RiskPolicy(
            region_id=f"{g} {c.names.at[g, 'cwc_site_name']} ({c.names.at[g, 'cwc_river']})",
            discharge_threshold=float(stats.at[g, "threshold"] * to_m3s),
            rainfall_threshold_mm=float(rain_thr[g]),
            flood_cover_fraction=1.0,
            lead_days=0,
            min_probability=cut,
        )
        engine = RiskEngine(policy)
        dec = engine.evaluate(
            np.array([[q_sim * to_m3s]]), np.array([[lo[0] * to_m3s]]), np.array([[hi[0] * to_m3s]]),
            rainfall_accum_3d=float(row["rain3"]), exceed_prob=p,
        )  # fmt: skip
        out = engine.to_json(dec)
        observed = row["q"]
        out["observed"] = {
            "discharge": None if pd.isna(observed) else float(observed * to_m3s),
            "flood": None if pd.isna(observed) else bool(observed > stats.at[g, "threshold"]),
        }
        decisions.append(out)
    brief = (GeminiRiskBrief().generate if llm_brief else GeminiRiskBrief._template_fallback)(
        decisions
    )
    return {"date": date, "mode": "hindcast", "decisions": decisions, "brief": brief}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("date")
    ap.add_argument("gauges", nargs="+", type=int)
    ap.add_argument("--llm-brief", action="store_true", help="write the brief with Gemini")
    args = ap.parse_args()
    print(json.dumps(hindcast(args.date, args.gauges, args.llm_brief), indent=2))


if __name__ == "__main__":
    main()
