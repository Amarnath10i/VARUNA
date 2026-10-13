# VARUNA

**Flood early warning and parametric flood triggers for Indian rivers, benchmarked on real catchments.**

VARUNA turns daily weather into river-flow forecasts, attaches calibrated
uncertainty, and decides whether a flood warning or an index-insurance payout
should fire.

Until this benchmark, none of that had touched real data. The operational
pipeline fed `torch.randn` tensors to untrained networks and issued
"triggered" decisions from the result. Its conformal intervals had never been
checked against a river. This repository now does the following:

- **Benchmarks** rainfall-runoff models on **239 catchments** in Peninsular
  India (CAMELS-IND, 1980-2020). The comparison set is a conceptual model
  calibrated per catchment, the best of 47 published conceptual models, the
  dataset's own LSTM, gradient-boosted trees and a regional LSTM.
- **Tests** whether VARUNA's uncertainty and trigger logic work during
  floods. They didn't: the intervals covered 8% of flood days, and the
  trigger rule, given honest intervals, never fires.
- **Replaces** both with methods that hold up on 2016-2020, which no model
  saw. Calibrated log-scale intervals adapt online, and a probability
  trigger warns of 43-62% of flood onsets.
- **Replays** the August 2018 Kerala floods through the full decision chain.
- **Stops** the live pipeline from issuing decisions it can't back. It now
  records why it can't decide.

## Dashboard

```bash
pip install streamlit pandas matplotlib
streamlit run app.py
```

Explores the committed reports without the datasets or a GPU: the per-catchment
NSE distribution of every model, a day-by-day replay of the Kerala 2018 floods
for each gauge (simulated flow, 90% interval, P(flood) and when the trigger
fired), the flood and uncertainty reports, and the LSTM training curves.

## Results

Every number is on 2016-2020. No model was fitted, tuned or selected on
those years. Intervals come from 2,000 bootstrap resamples of catchments.
Full tables are in [`reports/`](reports/).

### 1. Simulating river flow from weather

Daily flow in 202 catchments with at least a year of test-period
observations. The input is weather alone: rain, temperature, radiation,
humidity, wind and evaporation, with no observed flow.

| model | median NSE (95% CI) | median KGE | peak-flow bias |
|---|---|---|---|
| Climatology (day-of-year mean) | 0.07 (0.03 to 0.12) | 0.08 | -73% |
| GR4J, calibrated per catchment | 0.32 (0.15 to 0.41) | 0.36 | -35% |
| Regional LSTM without catchment attributes | 0.32 (0.19 to 0.43) | 0.30 | -35% |
| CAMELS-IND's regional LSTM (published predictions) | 0.42 (0.35 to 0.49) | 0.45 | -32% |
| LightGBM on engineered rainfall memory | 0.43 (0.34 to 0.51) | 0.40 | -30% |
| **Regional LSTM, 3-seed ensemble** | **0.44 (0.35 to 0.51)** | 0.41 | **-26%** |

Paired by catchment, the LSTM ensemble:

- **Beats calibrated conceptual models.** It gains +0.040 NSE over GR4J
  (+0.013 to +0.078) and +0.042 over the best of 47 conceptual models chosen
  per catchment (+0.022 to +0.074; 141 catchments). A single regional
  network beats models calibrated on each river's own flows.
- **Needs the catchment attributes.** Without the 28 static descriptors it
  loses 0.048 NSE (+0.020 to +0.075); they are how one network tells rivers
  apart.
- **Ties strong baselines.** It ties the published LSTM (+0.000, -0.013 to
  +0.010) and LightGBM (-0.006, -0.019 to +0.007). Hand-engineered memory
  (rainfall totals and decaying rain indices over 3 to 365 days) is as good
  as learned memory on average.
- **Loses to trees in dry catchments.** In the driest third, where
  rainfall-to-evaporation ratio ≤ 0.60, LightGBM is better (-0.090, -0.27 to
  -0.01).
- **Floods are underestimated by every model.** The biggest 2% of flows are
  26-35% too low.

Medians are low because these rivers are hard. More than a fifth of the catchments (45 of 202)
have reservoirs holding more than a year of flow. There, and in the dry
third, every model's median NSE is near zero.

The dataset paper reports 0.59 for its LSTM, on 159 catchments with at
least 28 years of records. On our wider set its predictions score 0.42.

### 2. Ungauged rivers

Most Indian rivers have no gauge. Here each catchment is predicted by
models that never saw its flows. The folds hold out whole river basins
(Godavari alone is one fold), so no test gauge shares a river network with
a training gauge.

| model (no flows from the catchment's basin) | median NSE (95% CI) |
|---|---|
| GR4J with the parameters of the 5 most similar catchments | 0.18 (0.06 to 0.30) |
| LightGBM | 0.32 (0.14 to 0.39) |
| **Regional LSTM** | **0.28 (0.15 to 0.39)** |
| *for reference:* GR4J calibrated on the catchment's own 25 years of flow | 0.32 (0.15 to 0.41) |

- **An ungauged LSTM matches calibration on the river's own flow.** Without
  a single observation from the basin, the LSTM is statistically tied with
  GR4J calibrated in place (-0.012, -0.040 to +0.005). This is Kratzert et
  al.'s (2019) central result, under a stricter split.
- **It beats classic regionalization.** Borrowing parameters from similar
  catchments does worse by 0.042 (+0.015 to +0.075).
- **Losing the gauge costs 0.057 NSE.** That is the gap to the same LSTM
  trained with the catchment.
- **Random folds flatter the result.** The usual protocol, random 5-fold
  over gauges, lets nested gauges sit on both sides. It scores the same
  LSTM 0.021 higher per catchment (0.003 to 0.047). That is an upper bound
  on the leakage, since holding out a whole basin is also a harder
  extrapolation.

### 3. Flood events and short-range forecasts

A flood day is a day above the catchment's 99th percentile of 1991-2015
flow, about 3.7 days a year. There are 803 events in the test years.
"Onsets warned" counts events where the forecast for the first flood day
was already above the threshold.

| | POD | FAR | CSI | onsets warned |
|---|---|---|---|---|
| GR4J (weather only) | 0.28 | 0.53 | 0.21 | 26% |
| Regional LSTM (weather only) | 0.34 | 0.49 | 0.26 | 29% |
| Persistence, 1 day ahead | 0.50 | 0.47 | 0.34 | 0% |
| LightGBM with today's observed flow, 1 day ahead | 0.51 | 0.36 | 0.40 | 26% |
| LightGBM with today's observed flow, 3 days ahead | 0.41 | 0.41 | 0.32 | 29% |

- **Today's flow helps by cutting false alarms.** At one day ahead, knowing
  today's flow lifts CSI from 0.34 (persistence) to 0.40, almost entirely by
  cutting false alarms. It adds nothing at the onset: rises are driven by
  rain, so the onset warnings match those of weather-only simulation.
- **These forecasts are upper bounds.** They use the rain that actually fell
  as a perfect rainfall forecast.

### 4. Uncertainty, and the trigger rule VARUNA shipped with

These are 90% conformal intervals for the one-day forecast, calibrated on
1981-1990 and checked on 2016-2020.

| intervals | coverage | on flood days | catchments within 85-95% |
|---|---|---|---|
| One width for every river and day (VARUNA's `ConformalRegressor`) | 88.6% | **6.5%** | 35% |
| Log scale, per catchment | 81.9% | 46.5% | 42% |
| Log scale, per flow regime | 83.5% | 59.8% | 40% |
| Log scale, per catchment, adaptive (ACI) | **87.0%** | **61.7%** | **73%** |

- **One width fails on the days that matter.** A single width looks
  calibrated on average but is far too narrow for floods and far too wide
  for small streams. Flood errors scale with flow and catchments differ by
  orders of magnitude.
- **Log-scale intervals fix the scale.** Intervals become multiplicative,
  never negative, and wider on the flood side. Mondrian groups by flow
  regime help on flood days.
- **Drift shows up as undercoverage.** The calibration years are 30 years
  older than the test years. Adaptive conformal inference (Gibbs & Candès,
  2021) updates each catchment's level as outcomes arrive, recovering
  coverage to 87%.
- **Flood days stay below 90%.** Conformal can only condition on what is
  known when forecasting, not on the flood itself.

#### Trigger backtest, one day ahead

| fires when | POD | FAR | CSI | onsets warned |
|---|---|---|---|---|
| Lower bound above threshold, interval narrow (RiskEngine as shipped), with its own intervals | 0.46 | 0.32 | 0.38 | 23% |
| The same rule with calibrated intervals | **0.00** | - | 0.00 | 0% |
| Point forecast above threshold | 0.51 | 0.36 | 0.40 | 26% |
| **P(flood) ≥ 0.15** (conformal predictive distribution; cut-off tuned on 1981-1990) | **0.66** | 0.46 | **0.42** | **43%** |

- **The shipped rule only appeared to work.** "Fire when the lower bound
  clears the threshold" looked conservative and reasonable, but only because
  its intervals were wrong in floods. With honest intervals the lower bound
  almost never clears a flood threshold, and the "narrow interval" check
  never passes. As written, it would never pay out.
- **Firing on probability is better.** The fix fires on the forecast
  probability of exceeding the threshold, with the cut-off chosen on past
  years. It warns of 43% of onsets at one day ahead and 62% for the
  weather-only simulation. It is now the RiskEngine's default
  (`trigger_rule="probability"`).

### 5. Kerala, August 2018

The worst flooding in Kerala in a century, replayed day by day through the
full decision chain (`python -m varuna.pipeline.hindcast`). Details are in
[`reports/kerala_2018.md`](reports/kerala_2018.md).

![Kerala 2018](reports/figures/kerala_2018.png)

- **Most gauges went silent at the peak.** Six of the seven CAMELS-IND gauges
  on the flooded rivers recorded nothing on 5 or 6 of the 7 days from 14 to
  20 August. A warning system that needs today's river level is blind
  exactly then. Across the whole record, gauges are no more often missing
  on heavy-rain days (78% observed, against 76% overall); 2018 was
  exceptional.
- **The weather-driven trigger fired through the flood.** It fired on 18 of
  the 22 days a gauge confirmed flooding, and on 25 days a gauge showed no
  flood. At Pulamanthole, the one gauge that recorded the peak, it fired a
  day before the first flood day. The 2,315 and 2,641 m³/s peaks fall
  inside the 90% intervals, though the point simulation (about 900 m³/s)
  is far too low.

## Data

| dataset | what it is | used for |
|---|---|---|
| [CAMELS-IND](https://doi.org/10.5194/essd-17-461-2025) (Mangukiya et al., ESSD 2025) | daily IMD forcing, CWC streamflow and 28 static attributes for 242 Peninsular Indian catchments, 1980-2020, plus a regional LSTM's predictions | everything |
| [Conceptual-model benchmark](https://zenodo.org/records/18323832) (Mangukiya et al.) | simulated flow from 47 MARRMoT conceptual models on 159 catchments | baseline |

Data cleaning, all in [`bench/data.py`](varuna/bench/data.py):

- **Flow spikes.** 97 isolated one-day spikes in 49 gauges are removed. One
  gauge reads 4.7, 11,381 and 7.1 mm/day on three consecutive days.
- **Water-balance outliers.** Three catchments whose runoff exceeds 1.5×
  their rainfall are excluded from the headline numbers and reported
  separately.
- **Temperature fill values.** Gridded 0 °C fill values (all catchments on
  2019-01-11, and a maximum below the minimum on 2019-05-07) are
  interpolated.
- **Leaky attributes.** CAMELS-IND's hydrological signatures (mean flow,
  runoff ratio and so on) are never used as inputs: they are computed from
  1980-2020 flows, test years included.

## Evaluation protocol

- **Splits.** The splits are the dataset's own: train 1991-2015, validate
  1980-1990 (early stopping, calibration, cut-offs), test 2016-2020. Its
  published LSTM predictions are therefore a baseline on identical days.
- **The 47-model benchmark** does not state its calibration years.
  Recomputing NSE on 2010-2020 reproduces its published validation scores,
  so 2016-2020 is out of sample for it. Each catchment's best model is
  chosen on 2010-2015, never on its published scores, which include the
  test years.
- **Leakage tests.** Training the LSTM twice, once with every validation
  and test flow replaced by noise, gives bit-identical weights. A planted
  leak makes the test fail. Normalization uses training years only, and
  forecast features look only backwards.
- **Statistics.** Medians over catchments, with a bootstrap over catchments
  (days within a river are correlated). Model comparisons are paired by
  catchment. Group medians alone mislead here: NSE is heavy-tailed in
  regulated catchments, where one scores -3,667.

## Models

| model | idea | file |
|---|---|---|
| GR4J | 4-parameter conceptual model (Perrin et al. 2003), numba, calibrated per catchment with differential evolution; tests check water balance to 1e-9 and recovery of planted parameters. The exchange bound (x2 ≥ -10) binds in 31% of catchments; recalibrating with -30 changes test NSE by 0.000 (paired median) | [`bench/gr4j.py`](varuna/bench/gr4j.py) |
| Regional LSTM | Kratzert et al. 2019: one network for all catchments, forcing + attributes in, NSE* loss, two-year windows (one warms up the state), 3-seed ensemble, ~2.5 min per model on an RTX 3050 | [`bench/lstm.py`](varuna/bench/lstm.py) |
| LightGBM | rainfall totals and decaying indices (half-lives 3-90 days), evaporation, season, attributes; same NSE* weighting | [`bench/lgbm.py`](varuna/bench/lgbm.py) |
| Forecaster | LightGBM plus the flow observed on the issue day, 1-3 days ahead | [`bench/forecast.py`](varuna/bench/forecast.py) |
| Intervals | split conformal with the finite-sample order statistic; log-scale Mondrian; adaptive conformal inference with delayed feedback; conformal predictive distribution for P(flood) | [`uncertainty/conformal.py`](varuna/uncertainty/conformal.py), [`bench/uq.py`](varuna/bench/uq.py) |

## The operational pipeline

- **Hindcast** (`python -m varuna.pipeline.hindcast 2018-08-16 15024 15021`).
  It runs the trained LSTM ensemble on a past day's forcing, computes the
  interval and P(flood), and passes them to the RiskEngine. Each JSON
  decision carries the observed outcome for audit.
- **Live** (`python -m varuna.pipeline.forecast`). It records `no_decision`
  and the reasons:
  - the GFS client does not decode GRIB2;
  - IMD and GLOFAS fall back to placeholders;
  - the configured regions (Brahmaputra, Koshi) are outside CAMELS-IND;
  - nothing builds model inputs from live feeds.

  Feeding live IMD catchment rainfall to the benchmarked LSTM is the next
  step.

**Research prototypes.** RainCast (an FNO with spectral variance), HydroNet
(a graph neural ODE) and FloodMap (a SAR-optical masked autoencoder) in
`varuna/models` are implemented and unit-tested. They have never been
trained or evaluated on real data, and no decision uses them.

## Reproduce

```bash
uv sync    # Python 3.12, PyTorch with CUDA 12.6
# CAMELS-IND -> data/raw/camels_ind/, 47-model benchmark -> data/raw/rr_benchmark/x/ (links above)

uv run varuna-bench conceptual   # GR4J per catchment (~2 min on 14 cores), 47-model selection
uv run varuna-bench lstm         # 3-seed LSTM + no-attribute ablation (~15 min, GPU)
uv run varuna-bench lgbm
uv run varuna-bench forecast     # 1-3 day forecasters
uv run varuna-bench ungauged     # basin and random 5-fold (~75 min, GPU)
uv run varuna-bench evaluate     # reports/streamflow.md
uv run varuna-bench floods       # reports/floods.md
uv run varuna-bench conformal    # reports/uncertainty.md, trigger cut-offs
uv run varuna-bench kerala       # reports/kerala_2018.md and the figure
uv run pytest
```

## Limitations

- **Perfect rainfall forecasts.** The forecasts and hindcasts use observed
  rain. A live system would use a numerical rainfall forecast, and its
  skill would be lower.
- **Coverage on flood days** is about 60% for nominal 90% intervals, with
  calibration years three decades older than the test years.
- **Peninsular India only.** The regions in `config.yaml` (Brahmaputra,
  Koshi) are not covered by the data, and nothing here says how the models
  transfer there.
- **One threshold.** Flood thresholds are the 99th percentile of flow, not
  CWC danger levels, which are not in the dataset.

## License

Apache-2.0
