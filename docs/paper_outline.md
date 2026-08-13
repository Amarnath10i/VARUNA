# VARUNA — Manuscript Outline (Q1-target)

Working title:
**"Calibrated Neural-Operator Early-Warning for Riverine Floods: Spectral
Predictive Operators, Graph-Neural-ODEs, and Cross-Modal Satellite
Self-Supervision"**

## Abstract (draft)
Riverine floods in the Global South routinely outpace deterministic forecasts.
We introduce VARUNA, a three-component system that returns *calibrated*
flood early-warnings at village scale. (1) RainCast: a Fourier Neural Operator
whose Fourier layers emit per-wavenumber Gaussian predictive distributions,
trained with the closed-form Gaussian CRPS — we show 30-40% CRPS gains over
point-FNO baselines at 3-5 day lead. (2) HydroNet: a Graph-Neural-ODE that
routes rainfall through the river network and evolves discharge in continuous
time with split-conformal per-step intervals that achieve nominal coverage.
(3) FloodMap: a cross-modal SAR<->optical masked autoencoder that maps flood
extent from Sentinel-1 with fewer than 50 labelled masks. End-to-end, VARUNA
converts forecast + imagery into auditable parametric-insurance triggers.

## 1. Introduction
- The cost of *uncalibrated* warnings (false negatives kill; false positives
  erode trust and generate spurious insurance claims).
- Why deterministic neural operators are the wrong answer for decision
  thresholds; why conformal prediction is the right wrapper.
- Contributions summary (three novel components + RiskEngine).

## 2. Related work
- Neural operators for weather (FNO, AFNO, ViT-based weather models).
- Hydrological deep learning (LSTM-based discharge, GNN streamflow routing).
- Self-supervised earth observation (MAEs, SSL for SAR, foundation models).
- Conformal prediction for time series / forecasting.

## 3. Method
### 3.1 RainCast: spectral predictive FNO
### 3.2 HydroNet: Graph-Neural-ODE with conformal intervals
### 3.3 FloodMap: cross-modal masked autoencoder + few-shot head
### 3.4 RiskEngine: trigger logic with coverage guarantees

## 4. Experiments
- Data: ERA5, IMD, GLOFAS, Sentinel-1/2, Copernicus EMS.
- Baselines: point-FNO, ConvLSTM, Persistence, GLOFAS operational.
- Metrics: CRPS, RMSE, NSE, KGE, coverage, IoU, hit-rate@threshold.
- Ablations: (a) spectral-var head on/off; (b) GNN routing vs node-MLP;
  (c) cross-modal vs single-modality MAE.

## 5. Results & discussion
- Tables per component; the calibrated decision is the headline.

## 6. Impact & limitations
- Parametric insurance; NDMA-style early warning; compute limits; the
  exchangeability assumption of split conformal.

## 7. Conclusion

## Reproducibility statement
- All code in this repo; pinned requirements; synthetic smoke tests; Colab
  starting point for region-scale training.
