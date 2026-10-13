"""VARUNA results dashboard: the CAMELS-IND benchmark, the Kerala 2018 replay and the uncertainty study.

    streamlit run app.py

Reads only the committed files in reports/ (pandas + matplotlib), so it runs
without the datasets or a GPU.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

REPORTS = Path(__file__).resolve().parent / "reports"
MODELS = {
    "q_clim": "Climatology (day-of-year mean)",
    "q_gr4j": "GR4J, calibrated per catchment",
    "q_marrmot": "Best of 47 conceptual models (MARRMoT)",
    "q_lstm_pub": "CAMELS-IND published LSTM",
    "q_lgbm": "LightGBM, engineered rainfall memory",
    "q_lstm_nostatic": "Regional LSTM, no catchment attributes",
    "q_lstm_single": "Regional LSTM, single seed",
    "q_lstm": "Regional LSTM, 3-seed ensemble",
}

st.set_page_config(page_title="VARUNA", page_icon="🌊", layout="wide")
st.title("VARUNA: flood early warning for Indian rivers")
st.caption("Results from the committed reports. Every number is on 2016-2020, which no model was fitted, tuned or "
           "selected on. Reproduce them with the commands in the README.")

tab_bench, tab_kerala, tab_unc, tab_train = st.tabs(
    ["Streamflow benchmark", "Kerala 2018 replay", "Floods and uncertainty", "LSTM training"])


@st.cache_data
def load(name):
    return pd.read_csv(REPORTS / name)


with tab_bench:
    d = load("streamflow_per_catchment.csv")
    d["model_name"] = d["model"].map(MODELS)
    st.markdown(f"Daily flow simulated from weather alone in **{d.gauge.nunique()} catchments** of Peninsular "
                "India (CAMELS-IND). NSE = 1 is perfect, 0 is no better than the catchment's mean flow.")
    chosen = st.multiselect("Models", list(MODELS.values()),
                            default=[MODELS[m] for m in ("q_clim", "q_gr4j", "q_lstm_pub", "q_lgbm", "q_lstm")])
    summary = (d.groupby("model_name")
                 .agg(median_NSE=("NSE", "median"), median_KGE=("KGE", "median"),
                      peak_flow_bias_pct=("FHV", "median"), catchments_NSE_above_0_5=("NSE", lambda x: (x > 0.5).mean()))
                 .loc[[m for m in MODELS.values() if m in chosen]].round(3))
    st.dataframe(summary, use_container_width=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for name in chosen:
        x = np.sort(d.loc[d.model_name == name, "NSE"].clip(-1, 1))
        ax.plot(x, np.linspace(0, 1, len(x)), label=name, lw=2 if "ensemble" in name else 1.2)
    ax.set(xlabel="NSE (clipped at -1)", ylabel="fraction of catchments", title="Distribution over catchments")
    ax.axvline(0, color="0.7", lw=0.8)
    ax.legend(fontsize=8)
    st.pyplot(fig)
    gauge = st.selectbox("Look at one catchment", sorted(d.gauge.unique()))
    st.dataframe(d[d.gauge == gauge].set_index("model_name")[["NSE", "KGE", "FHV", "days"]].round(3),
                 use_container_width=True)
    st.caption("FHV: bias of the top 2% of flows (%). Every model underestimates floods.")

with tab_kerala:
    k = load("kerala_2018.csv")
    k["date"] = pd.to_datetime(k["date"])
    st.markdown("The decision chain replayed day by day through the August 2018 floods (peak 15-17 August): the "
                "regional LSTM simulates flow from the day's rain, conformal calibration gives a 90% interval and "
                "P(flood), and the trigger fires when P(flood) passes the cut-off chosen on 1981-1990.")
    site = st.selectbox("Gauge", list(k.site.unique()), index=list(k.site.unique()).index(
        "Pulamanthole (Bharathapuzha/Pulanthodu)") if "Pulamanthole (Bharathapuzha/Pulanthodu)" in set(k.site) else 0)
    g = k[k.site == site]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.fill_between(g.date, g.lo, g.hi, alpha=0.2, label="90% interval")
    ax1.plot(g.date, g.sim, label="simulated", lw=2)
    ax1.plot(g.date, g.obs, "ko", ms=4, label="observed")
    ax1.axhline(g.threshold.iloc[0], color="C3", ls="--", label="flood threshold (99th percentile)")
    fired = g[g.fired]
    ax1.plot(fired.date, fired.sim, "v", color="C3", ms=9, label="trigger fired")
    ax1.set(ylabel="flow (m³/s)", yscale="log", title=site)
    ax1.legend(fontsize=8, ncol=3)
    ax2.bar(g.date, g.rain_3d, color="C0", alpha=0.6)
    ax2.set(ylabel="3-day rain (mm)")
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    md = (REPORTS / "kerala_2018.md").read_text(encoding="utf-8")
    with st.expander("Full Kerala 2018 report"):
        st.markdown(md.replace("![Kerala 2018](figures/kerala_2018.png)", ""))

with tab_unc:
    for name in ("floods.md", "uncertainty.md"):
        p = REPORTS / name
        if p.exists():
            st.markdown(p.read_text(encoding="utf-8"))

with tab_train:
    t = load("lstm_training.csv")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for (run, seed), g in t.groupby(["run", "seed"]):
        ax.plot(g.epoch, g.valid_nse, label=f"{run}, seed {seed}", color="C0" if run == "lstm" else "C1", alpha=0.8)
    ax.set(xlabel="epoch", ylabel="validation NSE",
           title="Regional LSTM with and without the 28 catchment attributes")
    ax.legend(fontsize=8)
    st.pyplot(fig)
