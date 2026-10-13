"""Live operational forecast cycle for VARUNA.

Intended chain: live forcing (GFS/IMD) -> discharge forecast with conformal
intervals -> RiskEngine decision per monitored region -> plain-language brief.

A decision is only emitted from real inputs and a model benchmarked on real
rivers. The live path has neither yet:

- the GFS client does not decode GRIB2, and IMD/GLOFAS fall back to
  placeholder zeros when unreachable;
- RainCast, HydroNet and FloodMap (varuna/models) have never been trained or
  evaluated on real data, and nothing builds their inputs from live feeds;
- the monitored regions in config.yaml (Brahmaputra, Koshi) are outside
  CAMELS-IND, where the benchmarked model was trained.

Until those exist, a live cycle records *why* it cannot decide rather than
turning random tensors into flood triggers, which is what it used to do.
Decisions on real data, from the benchmarked regional LSTM and calibrated
intervals: `python -m varuna.pipeline.hindcast` (see reports/).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from varuna.data.sources import GFSPrecipitationAPI, GloFASAPI, IMDNowcastAPI
from varuna.utils.logging import setup_logger

logger = setup_logger("varuna.pipeline")

NOT_BUILT = [
    "no discharge model is benchmarked on live inputs: the regional LSTM is "
    "trained on CAMELS-IND gauges and needs their catchment-mean IMD forcing "
    "(use varuna.pipeline.hindcast); RainCast/HydroNet/FloodMap are untrained",
]


class ForecastPipeline:
    def __init__(self, config_path: str | Path, run_dir: str = "runs/forecast") -> None:
        with open(config_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.feeds = {"gfs": GFSPrecipitationAPI(), "imd": IMDNowcastAPI(), "glofas": GloFASAPI()}

    def readiness(self, samples: dict) -> list[str]:
        """Everything that stands between this cycle and a real decision."""
        reasons = [f"{name}: no data (placeholder)" for name, s in samples.items() if s.placeholder]
        return reasons + NOT_BUILT

    def run(self) -> list[dict]:
        """One live cycle: fetch feeds, and decide only if every input is real
        and the model is benchmarked; otherwise record why not."""
        samples = {
            "gfs": self.feeds["gfs"].fetch(),
            "imd": self.feeds["imd"].fetch_rainfall(),
            "glofas": self.feeds["glofas"].fetch(),
        }
        reasons = self.readiness(samples)
        out = {
            "issued": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "no_decision" if reasons else "ready",
            "reasons": reasons,
            "feeds": {k: {"time": s.time, "placeholder": s.placeholder} for k, s in samples.items()},
            "regions": [r["region_id"] for r in self.cfg.get("regions", [])],
            "decisions": [],
        }
        with open(self.run_dir / "latest_cycle.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        for r in reasons:
            logger.error("No decision: %s", r)
        return []


if __name__ == "__main__":
    ForecastPipeline("config/config.yaml").run()
