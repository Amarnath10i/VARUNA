"""CAMELS-IND: daily forcing, streamflow and attributes for Peninsular
Indian catchments, 1980-2020 (Mangukiya et al., ESSD 2025); 242 catchments
in the release used here.

Everything is put into one long table (gauge, date) with flow in mm/day, so
catchments of very different sizes are comparable, and kept in a parquet cache.

The evaluation protocol is the one the dataset's own regional LSTM used
(train 1991-2015, validate 1980-1990, test 2016-2020), so its published
predictions, shipped with the data, are a fair baseline on the same days.

Static attributes are restricted to what is known without the streamflow
record: CAMELS-IND's "hydrological signatures" (mean flow, runoff ratio, flow
percentiles, ...) are computed from 1980-2020 flows, test years included, and
would leak the answer into the inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "camels_ind"
CACHE = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

PERIODS = {
    "valid": ("1980-01-01", "1990-12-31"),
    "train": ("1991-01-01", "2015-12-31"),
    "test": ("2016-01-01", "2020-12-31"),
}
FORCING = ["prcp", "tmax", "tmin", "srad_lw", "srad_sw", "wind", "rel_hum", "pet_gleam"]
# known before any streamflow is observed; no hydrological signatures
STATIC = [
    "cwc_area", "elev_mean", "slope_mean", "dpsbar",
    "p_mean", "pet_mean", "aridity_p_pet", "p_monthly_variability", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur", "tmax_mean",
    "trees_frac", "crops_frac", "built_area_frac", "water_frac", "lai_mean", "lai_diff",
    "soil_depth", "soil_conductivity_top", "soil_awc_top", "sand_frac_top", "clay_frac_top",
    "geol_porosity", "geol_permeability",
    "num_dams", "reservoir_index",
]  # fmt: skip


@dataclass
class Camels:
    daily: pd.DataFrame  # gauge, date, FORCING..., q (mm/day, NaN if missing), q_lstm_pub
    static: pd.DataFrame  # indexed by gauge, STATIC columns
    names: pd.DataFrame  # indexed by gauge: site name, river, basin

    @property
    def gauges(self) -> np.ndarray:
        return self.static.index.to_numpy()


def _rename(cols):
    return [c.split("(")[0] for c in cols]


def _wide_flow(path: Path) -> pd.DataFrame:
    w = pd.read_csv(path)
    dates = pd.to_datetime(w[["year", "month", "day"]])
    w = w.drop(columns=["year", "month", "day"]).set_index(pd.DatetimeIndex(dates, name="date"))
    long = w.stack(future_stack=True).rename("q_m3s").reset_index()
    long.columns = ["date", "gauge", "q_m3s"]
    long["gauge"] = long["gauge"].astype(int)
    return long


def load() -> Camels:
    files = {k: CACHE / f"camels_ind_{k}.parquet" for k in ("daily", "static", "names")}
    if all(f.exists() for f in files.values()):
        return Camels(*(pd.read_parquet(files[k]) for k in ("daily", "static", "names")))

    attrs = {}
    for name in ("name", "topo", "clim", "land", "soil", "geol", "anth"):
        attrs[name] = pd.read_csv(RAW / "attributes_csv" / f"camels_ind_{name}.csv").set_index(
            "gauge_id"
        )
    static = pd.concat([attrs[k] for k in ("topo", "clim", "land", "soil", "geol", "anth")], axis=1)
    static = static.loc[:, ~static.columns.duplicated()][STATIC].apply(
        pd.to_numeric, errors="coerce"
    )
    names = attrs["name"][["cwc_site_name", "river_basin", "cwc_river", "flow_availability"]]
    gauges = sorted(int(p.stem) for p in (RAW / "catchment_mean_forcings").glob("*.csv"))
    static, names = static.reindex(gauges), names.reindex(gauges)
    static.index.name = names.index.name = "gauge"

    frames = []
    for g in gauges:
        f = pd.read_csv(RAW / "catchment_mean_forcings" / f"{g:05d}.csv")
        f.columns = _rename(f.columns)
        f["date"] = pd.to_datetime(f[["year", "month", "day"]])
        f["gauge"] = g
        frames.append(f[["gauge", "date", *FORCING]])
    daily = pd.concat(frames, ignore_index=True)

    obs = _wide_flow(RAW / "streamflow_timeseries" / "streamflow_observed.csv")
    pub = _wide_flow(RAW / "streamflow_timeseries" / "lstm_pred_streamflow.csv").rename(
        columns={"q_m3s": "q_pub_m3s"}
    )
    daily = daily.merge(obs, on=["gauge", "date"], how="left").merge(
        pub, on=["gauge", "date"], how="left"
    )
    # m3/s -> mm/day over the drainage area: 86.4 / area_km2
    to_mm = 86.4 / static["cwc_area"].reindex(daily["gauge"]).to_numpy()
    daily["q"] = (daily.pop("q_m3s") * to_mm).astype(np.float32)
    daily["q_lstm_pub"] = (daily.pop("q_pub_m3s") * to_mm).astype(np.float32)
    daily.loc[daily["q"] < 0, "q"] = np.nan  # negative discharge is a recording error
    daily["q_raw"] = daily["q"]
    daily.loc[spikes(daily), "q"] = np.nan
    # 0 degC is a fill value in the gridded temperature (tmin reads 0 in every
    # catchment on 2019-01-11, among ~300 such days; partly filled cells on
    # 2019-05-07 leave tmax below tmin): interpolate within gauge
    bad_max = (daily["tmax"] == 0) | (daily["tmax"] < daily["tmin"])
    daily["tmax"] = daily["tmax"].mask(bad_max)
    daily["tmin"] = daily["tmin"].mask(daily["tmin"] == 0)
    for c in ("tmax", "tmin"):
        daily[c] = daily.groupby("gauge")[c].transform(
            lambda s: s.interpolate(limit_direction="both")
        )
    for c in FORCING:
        daily[c] = daily[c].astype(np.float32)

    CACHE.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(files["daily"], index=False)
    static.to_parquet(files["static"])
    names.to_parquet(files["names"])
    log.info("CAMELS-IND: %d catchments, %d days, %.0f%% of flow days observed",
             len(gauges), daily["date"].nunique(), 100 * daily["q"].notna().mean())  # fmt: skip
    return Camels(daily, static, names)


def spikes(daily: pd.DataFrame, jump: float = 10.0, high: float = 5.0) -> np.ndarray:
    """Isolated one-day spikes: more than `jump` times both neighbouring days
    and `high` times the gauge's 99th percentile. A real flood rises and falls
    over several days; e.g. gauge 14001 reads 4.7, 11381.3, 7.1 mm/day on
    18-20 Oct 2020, which no catchment can do."""
    q = daily["q"]
    g = daily["gauge"]
    prev, nxt = q.groupby(g).shift(1), q.groupby(g).shift(-1)
    neighbours = np.fmax(prev, nxt)
    p99 = q.groupby(g).transform(lambda s: s.quantile(0.99))
    return ((q > jump * neighbours) & (q > high * p99)).to_numpy()


def water_balance_outliers(c: Camels, max_ratio: float = 1.5) -> list[int]:
    """Catchments whose long-term runoff exceeds 1.5x their rainfall: a wrong
    drainage area, inter-basin transfers or badly underestimated rain. Results
    are reported with and without them."""
    d = c.daily
    g = d.groupby("gauge").agg(q=("q", "mean"), p=("prcp", "mean"))
    return sorted(g.index[(g["q"] / g["p"]) > max_ratio].tolist())


def period_mask(dates: pd.Series, name: str) -> np.ndarray:
    lo, hi = PERIODS[name]
    return ((dates >= lo) & (dates <= hi)).to_numpy()
