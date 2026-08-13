# VARUNA

**Uncertainty-aware neural-operator flood early-warning system**

VARUNA couples three novel learning components into a single operational loop
that turns public satellite + reanalysis data into **auditable, calibrated**
flood early-warnings and **parametric-insurance triggers** for monsoon-prone
South Asia.

```
Live forcing (GFS / IMD / GLOFAS / Sentinel-NRT)
        │
        ▼
┌─────────────────┐    ┌─────────────────┐    ┌──────────────────────────┐
│  RainCast       │───▶│  HydroNet       │───▶│  RiskEngine              │
│  FNO + spectral │    │  Graph-Neural-  │    │  conformal intervals +    │
│  predictive     │    │  ODE discharge  │    │  parametric trigger logic │
│  distribution   │    │  forecasting    │    │                          │
└─────────────────┘    └─────────────────┘    └──────────────────────────┘
        │                       │                         ▲
        └──────▶ FloodMap (cross-modal SAR↔optical MAE, few-shot floods) ─┘
```

## Why this is a research project (not a demo repo)

1. **RainCast — spectral predictive distributions.** Classic FNOs output point
   fields. RainCast attaches a *spectral log-variance* branch to every Fourier
   block and trains on the closed-form Gaussian CRPS + spectral Gaussian NLL
   (`varuna/uncertainty/losses.py`). Uncertainty is modelled per-wavenumber —
   exactly where a point operator fails (high-frequency storm cells).
2. **HydroNet — Graph-Neural-ODE.** A GNN routes rainfall through the river
   network and a dependency-free Midpoint ODE integrator evolves discharge in
   continuous time, matching flood-wave travel physics. Split conformal
   prediction (`varuna/uncertainty/conformal.py`) wraps the output so every
   lead step carries a finite-sample 90% interval.
3. **FloodMap — cross-modal masked autoencoder.** A ViT-Tiny autoencoder
   reconstructs *both* Sentinel-1 (SAR) and Sentinel-2 (optical) tokens from
   the union of unmasked tokens, forcing a modality-consistent flood latent.
   Pretrains on unlabelled tiles, then a flood head fine-tunes with `<50`
   labelled masks (Copernicus EMS / Global Flood Database).

## Repository layout

```
varuna/
├── models/          # RainCast (FNO+UQ), HydroNet (GNN+ODE), FloodMap (MAE)
├── uncertainty/     # conformal prediction, CRPS / Gaussian NLL losses
├── data/            # ERA5, GFS, IMD, GLOFAS, Sentinel-NRT clients (live APIs)
├── engines/         # training loops + RiskEngine (parametric trigger logic)
├── pipeline/        # end-to-end operational forecast cycle
└── utils/           # NSE/KGE/CRPS/IoU metrics, logging, viz
config/config.yaml   # model + uncertainty + monitored-region policies
scripts/             # data downloaders & entrypoints
tests/               # smoke tests (torch-free optional path)
```

## Quick start

```bash
# 1. install (Python 3.10+)
python -m venv .venv
.venv\Scripts\activate          # windows
pip install -e .

# 2. run one full forecast cycle (works offline; live APIs degrade gracefully)
python -m varuna.pipeline.forecast
cat runs/forecast/latest_cycle.json

# 3. smoke tests
python -m pytest -q
```

## Real-time data

| Feed          | Role                              | Refresh        | Access                |
|---------------|-----------------------------------|----------------|-----------------------|
| NOAA GFS      | live rainfall forecast           | 4x / day       | open (no key)         |
| IMD           | India nowcast / rainfall         | near-real-time | open                  |
| ECMWF GLOFAS  | live river-discharge forecast    | daily          | Copernicus open       |
| Sentinel-1/2  | near-real-time flood mapping     | per-pass       | CDSE OAuth2 (env keys)|
| ERA5          | historical training data         | ~5-day latency | cdsapi (CDS account)  |

Set `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` for live Sentinel access, and a
`~/.cdsapirc` for ERA5. Every client returns placeholders when offline so the
pipeline never hard-crashes.

## Reproducing a benchmark

1. `python scripts/download_era5.py` — pull rainfall reanalysis for your region.
2. Train RainCast: `python scripts/train_all.py --component raincast`
3. Train HydroNet and calibrate conformal intervals:
   `python scripts/train_all.py --component hydronet`
4. Evaluate against GLOFAS baselines with NSE / KGE / CRPS / RMSE
   (`varuna/utils/metrics.py`).

## Startup story

The RiskEngine emits a JSON decision per monitored region that is fully
auditable (thresholds, conformal bounds, flood cover). That is exactly what
parametric-flood-insurance underwriters and state disaster authorities need:
an objective, trigger-based payout engine with guaranteed coverage, instead of
subjective claims adjustment.

## Roadmap

- [x] Novel architectures (RainCast, HydroNet, FloodMap) implemented
- [x] Conformal UQ + CRPS training objectives
- [x] Live data clients (GFS, IMD, GLOFAS, Sentinel-NRT, ERA5)
- [x] RiskEngine + end-to-end pipeline
- [ ] Full ERA5 + Sentinel training on a region (compute note: starts on free
      Colab TPU/T4 budget; tiles are cropped to one state)
- [ ] Benchmark vs GLOFAS / Copernicus EMS on a historical monsoon event
- [ ] Web dashboard + alerting (SMS/WhatsApp/API)

## License

Apache-2.0
