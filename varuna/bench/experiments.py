"""The experiments behind the README's tables; `varuna-bench <name>` runs `run_<name>`."""

from __future__ import annotations

import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from . import conceptual
from .data import CACHE, REPORTS, ROOT, load, period_mask, water_balance_outliers

log = logging.getLogger(__name__)
RUNS = ROOT / "runs" / "bench"


def run_conceptual(args) -> None:
    conceptual.run(load())


def _frame(a, pred, idx, name: str) -> pd.DataFrame:
    """(gauge, date, name) rows from a (gauges, days) prediction array."""
    return pd.DataFrame({
        "gauge": np.repeat(a.gauges[idx], len(a.dates)),
        "date": np.tile(a.dates, len(idx)),
        name: pred.reshape(-1),
    })  # fmt: skip


def _ensemble(a, train_idx, predict_idx, seeds, epochs, tag, static=True):
    """Train `seeds` LSTMs on `train_idx` catchments and predict 1981-2020 for
    `predict_idx`; returns the ensemble mean, each member's prediction and the
    training curves."""
    import torch

    from . import lstm

    b = a if static else replace(a, s=np.zeros_like(a.s))
    RUNS.mkdir(parents=True, exist_ok=True)
    models, history = [], []
    for k in range(seeds):
        net, h = lstm.train(b, train_idx, seed=k, epochs=epochs)
        torch.save(net.state_dict(), RUNS / f"{tag}_seed{k}.pt")
        models.append(net)
        history += [{**r, "seed": k, "run": tag} for r in h]
    years = range(1981, 2021)
    members = [lstm.predict([m], b, years, predict_idx) for m in models]
    return np.mean(members, axis=0), members, history


def run_lstm(args) -> None:
    from . import lstm

    c = load()
    a = lstm.build(c)
    keep = np.flatnonzero(~np.isin(a.gauges, water_balance_outliers(c)))
    allg = np.arange(len(a.gauges))
    frames, history = [], []
    mean, members, h = _ensemble(a, keep, allg, args.seeds, args.epochs, "lstm")
    history += h
    frames.append(_frame(a, mean, allg, "q_lstm"))
    frames.append(_frame(a, members[0], allg, "q_lstm_single"))
    mean, _, h = _ensemble(a, keep, allg, args.seeds, args.epochs, "lstm_nostatic", static=False)
    history += h
    frames.append(_frame(a, mean, allg, "q_lstm_nostatic"))
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["gauge", "date"])
    out.to_parquet(CACHE / "sims_lstm.parquet", index=False)
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(history).to_csv(REPORTS / "lstm_training.csv", index=False)


def run_lgbm(args) -> None:
    from . import lgbm

    c = load()
    f = lgbm.features(c)
    keep = [g for g in c.gauges if g not in water_balance_outliers(c)]
    q = lgbm.fit_predict(f, keep, c.gauges)
    out = pd.DataFrame({"gauge": f["gauge"], "date": f["date"], "q_lgbm": q.to_numpy()})
    out.to_parquet(CACHE / "sims_lgbm.parquet", index=False)


def _ungauged_section(c, d) -> list[str]:
    from . import evaluate as ev

    models = ev.available(d, ev.UNGAUGED)
    if "q_lstm_ug_basin" not in models:
        return []
    scores = ev.test_scores(c, d, models)
    ref = "q_lstm_ug_basin"
    others = [m for m in models if m != ref]
    return [
        "## Ungauged catchments",
        "",
        "Each catchment predicted by models that never saw its flows: 5 folds of "
        "whole river basins (Godavari alone is one fold), so no test gauge shares a "
        "river network with a training gauge. Random 5-fold over gauges, the usual "
        "protocol, is shown for comparison: nested gauges can then sit on both sides.",
        "",
        ev.md(ev.summary(scores, models, labels=ev.UNGAUGED)),
        "",
        ev.md(ev.paired(scores, ref, others, labels=ev.UNGAUGED)),
        "",
    ]


def run_evaluate(args) -> None:
    from . import evaluate as ev

    c = load()
    d = ev.with_sims(c)
    models = ev.available(d, ev.MODELS)
    scores = ev.test_scores(c, d, models)
    REPORTS.mkdir(exist_ok=True)
    scores.to_csv(REPORTS / "streamflow_per_catchment.csv", index=False)
    no_mm = [m for m in models if m != "q_marrmot"]
    covered = scores.loc[(scores["model"] == "q_marrmot") & scores["NSE"].notna(), "gauge"]
    sub = scores[scores["gauge"].isin(covered)]
    ref = "q_lstm" if "q_lstm" in models else "q_lstm_pub"
    bad = water_balance_outliers(c)
    text = [
        "# Streamflow simulation, 2016-2020",
        "",
        "Daily flow from forcing alone (no observed flow as input), scored on the "
        "test years 2016-2020, which no model was fitted or selected on. One row per "
        "model; medians over catchments with at least a year of observed flow in the "
        f"test years; intervals from 2,000 bootstrap resamples of catchments. "
        f"Catchments {', '.join(map(str, bad))} are excluded: their recorded runoff "
        "exceeds 1.5 times their rainfall (see the last table).",
        "",
        "## All catchments",
        "",
        ev.md(ev.summary(scores, no_mm)),
        "",
        "## Catchments covered by the 47-model conceptual benchmark",
        "",
        ev.md(ev.summary(sub, models)),
        "",
        f"## {ev.MODELS[ref].strip('*')} against each model, catchment by catchment",
        "",
        ev.md(ev.paired(scores, ref, [m for m in models if m != ref])),
        "",
        f"## By regulation and climate: {ev.MODELS[ref].strip('*')}, paired with each model",
        "",
        ev.md(
            ev.by_group(
                c,
                scores,
                ref,
                [m for m in ("q_gr4j", "q_lgbm", "q_lstm_pub") if m in models and m != ref],
            )
        ),
        "",
        *_ungauged_section(c, d),
        "## Excluded: runoff above 1.5 times rainfall (NSE)",
        "",
        ev.outliers(c, d, models).reset_index().to_markdown(index=False),
        "",
    ]
    (REPORTS / "streamflow.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def folds(c, gauges, scheme: str, k: int = 5, seed: int = 0) -> pd.Series:
    """Fold of each gauge: random, or whole river basins held out together,
    so no test gauge shares a river network with a training gauge."""
    from sklearn.model_selection import GroupKFold

    gauges = np.asarray(gauges)
    out = pd.Series(-1, index=gauges)
    if scheme == "random":
        perm = np.random.default_rng(seed).permutation(len(gauges))
        out.iloc[perm] = np.arange(len(gauges)) % k
    else:
        basin = c.names["river_basin"].reindex(gauges).to_numpy()
        for f, (_, test) in enumerate(GroupKFold(k).split(gauges, groups=basin)):
            out.iloc[test] = f
    return out


def donor_gr4j(c, target, donors, k: int = 5) -> pd.Series:
    """GR4J for `target` with the parameters of the k training catchments
    most similar in standardized attributes, simulations averaged."""
    from . import gr4j
    from .lstm import static_features

    params = pd.read_parquet(CACHE / "gr4j_params.parquet")
    st = static_features(c.static)
    st = ((st - st.loc[donors].mean()) / st.loc[donors].std()).fillna(0.0)
    dist = np.sqrt(((st.loc[donors] - st.loc[target]) ** 2).sum(axis=1))
    near = dist.nsmallest(k).index
    d = c.daily[c.daily["gauge"] == target]
    p, e = d["prcp"].to_numpy(float), d["pet_gleam"].to_numpy(float)
    sims = [gr4j.simulate(p, e, *params.loc[g, ["x1", "x2", "x3", "x4"]])[0] for g in near]
    return pd.Series(np.mean(sims, axis=0), index=d.index)


def run_ungauged(args) -> None:
    from . import lgbm, lstm

    c = load()
    keep = np.array([g for g in c.gauges if g not in water_balance_outliers(c)])
    f_lgbm = lgbm.features(c)
    out = c.daily[["gauge", "date"]].copy()
    for scheme in ("basin", "random"):
        fold = folds(c, keep, scheme)
        out[f"q_gr4j_donor_{scheme}"] = np.nan
        out[f"q_lgbm_ug_{scheme}"] = np.nan
        lstm_frames = []
        for f in range(fold.max() + 1):
            test, train = fold.index[fold == f], fold.index[fold != f]
            log.info("%s fold %d: %d test catchments %s", scheme, f, len(test),
                     sorted(c.names["river_basin"].reindex(test).unique()))  # fmt: skip
            for g in test:
                s = donor_gr4j(c, g, train)
                out.loc[s.index, f"q_gr4j_donor_{scheme}"] = s.to_numpy()
            q = lgbm.fit_predict(f_lgbm, train, test, seed=f)
            m = f_lgbm["gauge"].isin(test).to_numpy()
            key = pd.MultiIndex.from_frame(f_lgbm.loc[m, ["gauge", "date"]])
            pos = pd.MultiIndex.from_frame(out[["gauge", "date"]]).get_indexer(key)
            out.iloc[pos, out.columns.get_loc(f"q_lgbm_ug_{scheme}")] = q[m].to_numpy()
            a = lstm.build(c, train_gauges=train)
            tr, te = (
                np.flatnonzero(np.isin(a.gauges, train)),
                np.flatnonzero(np.isin(a.gauges, test)),
            )
            mean, _, _ = _ensemble(a, tr, te, args.seeds, args.epochs, f"ug_{scheme}_f{f}")
            lstm_frames.append(_frame(a, mean, te, f"q_lstm_ug_{scheme}"))
        out = out.merge(pd.concat(lstm_frames), on=["gauge", "date"], how="left")
        out.to_parquet(CACHE / "sims_ungauged.parquet", index=False)


def run_forecast(args) -> None:
    from . import forecast, lgbm

    c = load()
    base = lgbm.features(c)
    keep = [g for g in c.gauges if g not in water_balance_outliers(c)]
    out = base[["gauge", "date"]].copy()
    for lead in forecast.LEADS:
        out[f"q_persist_{lead}"] = forecast.persistence(base, lead).to_numpy()
        f = forecast.features(base, lead)
        out[f"q_fc_{lead}"] = lgbm.fit_predict(f, keep, c.gauges, seed=lead).to_numpy()
        log.info("lead %d done", lead)
    out.to_parquet(CACHE / "sims_forecast.parquet", index=False)


def _test_frame(c, d):
    t = d[period_mask(d["date"], "test") & ~d["gauge"].isin(water_balance_outliers(c))]
    return t.sort_values(["gauge", "date"]).reset_index(drop=True)


def run_floods(args) -> None:
    from . import evaluate as ev
    from . import uq

    c = load()
    d = ev.with_sims(c)
    stats = uq.gauge_stats(c)
    t = _test_frame(c, d)
    sim = {m: ev.MODELS[m].strip("*") for m in ("q_clim", "q_gr4j", "q_lstm_pub", "q_lgbm", "q_lstm")
           if m in t}  # fmt: skip
    fc = {}
    for h in (1, 2, 3):
        fc[f"q_persist_{h}"] = f"Persistence, {h} day ahead"
        fc[f"q_fc_{h}"] = f"LightGBM with observed flow, {h} day ahead"
    text = [
        "# Flood events, 2016-2020",
        "",
        "A flood day is a day above the catchment's 99th percentile of observed "
        "1991-2015 flow (about 3.7 days a year); an event is a run of flood days. "
        "An observed event is detected if the model is above the threshold on any "
        "of its days; a model event that overlaps no observed event is a false "
        "alarm. *Onsets warned*: the model was above the threshold on the first day "
        "of the event, i.e. for a forecast, before the river got there. Counts are "
        "pooled over catchments; intervals resample catchments.",
        "",
        "## Simulation from weather alone",
        "",
        ev.md(uq.flood_skill(t, sim, stats)),
        "",
        "## Forecasts that also know today's flow",
        "",
        "Rain between the issue day and the target day is the observed rain (a "
        "perfect rainfall forecast), so these are upper bounds.",
        "",
        ev.md(uq.flood_skill(t, {k: v for k, v in fc.items() if k in t}, stats)),
        "",
    ]
    (REPORTS / "floods.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def run_conformal(args) -> None:
    from . import evaluate as ev
    from . import uq

    c = load()
    d = ev.with_sims(c)
    d = d[~d["gauge"].isin(water_balance_outliers(c))].sort_values(["gauge", "date"])
    stats = uq.gauge_stats(c)
    text = [
        "# Prediction intervals and the flood trigger",
        "",
        "90% split-conformal intervals calibrated on 1981-1990 predictions and "
        "checked on 2016-2020. *On flood days*: days above the catchment's 99th "
        "percentile, the days a warning or an insurance trigger is about. Widths "
        "are relative to the catchment's mean flow.",
        "",
    ]
    methods = ["one width for all (current VARUNA)", "log scale, per flow regime",
               "log scale, per catchment, adaptive"]  # fmt: skip
    cuts = {}
    targets = [("q_lstm", 1, "Regional LSTM simulation")]
    targets += [(f"q_fc_{h}", h, f"Forecast {h} day ahead") for h in (1, 3)]
    for col, lead, label in targets:
        if col not in d:
            continue
        x = d[np.isfinite(d[col])]
        cal = x[period_mask(x["date"], "valid") & x["q"].notna()]
        test = x[period_mask(x["date"], "test")].reset_index(drop=True)
        ints = uq.intervals(cal, test, col, stats, lead)
        cal_days = x[period_mask(x["date"], "valid")].reset_index(drop=True)
        cuts[col] = uq.tune_probability(
            cal_days, uq.exceedance_prob(cal, cal_days, col, stats), stats
        )
        prob = uq.exceedance_prob(cal, test, col, stats)
        text += [
            f"## {label}",
            "",
            ev.md(uq.coverage_table(test, ints, stats)),
            "",
            f"### Trigger backtest, {label.lower()}",
            "",
            ev.md(uq.trigger_table(test, col, ints, stats, methods, prob, cuts[col])),
            "",
        ]
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "trigger_cutoffs.json").write_text(json.dumps(cuts, indent=2), encoding="utf-8")
    (REPORTS / "uncertainty.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


KERALA = [15024, 15021, 15012, 15014, 15005, 15026, 15011]


def run_kerala(args) -> None:
    """The August 2018 Kerala floods, replayed through the hindcast."""
    from varuna.pipeline.hindcast import hindcast

    days = pd.date_range("2018-08-06", "2018-08-24")
    rows = []
    for day in days:
        for dec in hindcast(str(day.date()), KERALA)["decisions"]:
            gauge = int(dec["region_id"].split()[0])
            rows.append({
                "gauge": gauge, "site": dec["region_id"].split(" ", 1)[1], "date": day,
                "rain_3d": dec["rainfall_accum_3d_mm"], "sim": dec["discharge"]["mean"],
                "lo": dec["discharge"]["interval"][0], "hi": dec["discharge"]["interval"][1],
                "p_flood": dec["details"]["exceed_prob"], "threshold": dec["details"]["threshold"],
                "fired": "discharge" in (dec["trigger_type"] or ""),
                "obs": dec["observed"]["discharge"],
            })  # fmt: skip
    k = pd.DataFrame(rows)
    k.to_csv(REPORTS / "kerala_2018.csv", index=False)
    summary = []
    for _, x in k.groupby("gauge", sort=False):
        seen = x["obs"].notna()
        flood = seen & (x["obs"] > x["threshold"])
        peak = x.loc[x["obs"].idxmax()] if seen.any() else None
        first_fire = x.loc[x["fired"], "date"].min()
        summary.append({
            "gauge": x["site"].iloc[0],
            "silent on 14-20 Aug": f"{(~seen & x['date'].between('2018-08-14', '2018-08-20')).sum()} of 7 days",
            "observed flood days": int(flood.sum()),
            "trigger fired on them": f"{int((flood & x['fired']).sum())}",
            "fired on observed non-flood days": int((seen & ~flood & x["fired"]).sum()),
            "first trigger": "-" if pd.isna(first_fire) else f"{first_fire:%d %b}",
            "highest reading": "-" if peak is None else f"{peak['obs']:,.0f} m3/s on {peak['date']:%d %b}",
            "simulated peak": f"{x['sim'].max():,.0f} m3/s on {x.loc[x['sim'].idxmax(), 'date']:%d %b}",
        })  # fmt: skip
    p = k[k["gauge"] == KERALA[0]]
    detail = pd.DataFrame({
        "date": p["date"].dt.strftime("%d %b"),
        "3-day rain (mm)": p["rain_3d"].round(0).astype(int),
        "observed (m3/s)": p["obs"].map(lambda v: "no reading" if pd.isna(v) else f"{v:,.0f}"),
        "simulated (m3/s)": p["sim"].map("{:,.0f}".format),
        "90% interval": [f"{a:,.0f}-{b:,.0f}" for a, b in zip(p["lo"], p["hi"], strict=True)],
        "P(flood)": p["p_flood"].map("{:.2f}".format),
        "trigger": p["fired"].map({True: "fires", False: ""}),
    })  # fmt: skip
    _kerala_figure(k)
    thr = p["threshold"].iloc[0]
    text = [
        "# Kerala, August 2018",
        "",
        "The worst flooding in Kerala in a century peaked on 15-17 August 2018. "
        "This replays VARUNA's decision chain (`python -m varuna.pipeline.hindcast`) "
        "for each day and every CAMELS-IND gauge in the affected rivers: the "
        "regional LSTM simulates flow from the day's rain, the conformal predictive "
        "distribution gives a 90% interval and P(flood), and the RiskEngine fires "
        "when P(flood) reaches the cut-off chosen on 1981-1990. Flood threshold: "
        "the gauge's 99th percentile of 1991-2015 flow. None of 2018 was used for "
        "training or tuning; the rain is the observed rain (a perfect forecast).",
        "",
        "![Kerala 2018](figures/kerala_2018.png)",
        "",
        "## Each gauge, 6-24 August",
        "",
        pd.DataFrame(summary).to_markdown(index=False),
        "",
        f"## {p['site'].iloc[0]}, the one gauge that recorded the peak (threshold {thr:,.0f} m3/s)",
        "",
        detail.to_markdown(index=False),
        "",
    ]
    (REPORTS / "kerala_2018.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def _kerala_figure(k: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ink, ink2, muted, grid, base = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
    sim_c, obs_c, crit = "#2a78d6", "#eb6834", "#d03b3b"
    show = KERALA[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True, facecolor="#fcfcfb")
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    for ax, g in zip(axes.flat, show, strict=True):
        x = k[k["gauge"] == g]
        ax.set_facecolor("#fcfcfb")
        silent = x["obs"].isna().to_numpy()
        for day, quiet in zip(x["date"], silent, strict=True):
            if quiet:
                ax.axvspan(day - pd.Timedelta(hours=12), day + pd.Timedelta(hours=12),
                           facecolor="none", edgecolor=grid, hatch="//", linewidth=0)  # fmt: skip
        ax.fill_between(x["date"], x["lo"], x["hi"], color=sim_c, alpha=0.14, linewidth=0)
        ax.plot(x["date"], x["sim"], color=sim_c, lw=2)
        ax.plot(x["date"], x["obs"], color=obs_c, lw=2, marker="o", ms=5)
        ax.axhline(x["threshold"].iloc[0], color=muted, lw=1.2, ls=(0, (4, 3)))
        top = ax.get_ylim()[1]
        fired = x[x["fired"]]
        ax.plot(
            fired["date"], np.full(len(fired), top * 0.97), ls="none", marker="v", ms=7, color=crit
        )
        ax.set_ylim(0, top)
        ax.set_title(x["site"].iloc[0], loc="left", fontsize=10.5, color=ink)
        ax.grid(axis="y", color=grid, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(base)
        ax.tick_params(colors=muted, labelsize=8.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for ax in axes[:, 0]:
        ax.set_ylabel("discharge (m³/s)", color=ink2, fontsize=9)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=obs_c, lw=2, marker="o", ms=5, label="observed"),
        Line2D([], [], color=sim_c, lw=2, label="LSTM simulation"),
        Patch(facecolor=sim_c, alpha=0.14, label="90% interval"),
        Line2D([], [], color=muted, lw=1.2, ls=(0, (4, 3)), label="flood threshold"),
        Patch(facecolor="none", edgecolor=grid, hatch="//", label="gauge silent"),
        Line2D([], [], ls="none", marker="v", ms=7, color=crit, label="trigger fired"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False, fontsize=9,
               labelcolor=ink2, bbox_to_anchor=(0.5, 0.995))  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS / "figures" / "kerala_2018.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
