"""End-to-end operational forecast loop for VARUNA.

Chain: live forcing (GFS/IMD) -> RainCast rainfall distribution
    -> HydroNet discharge with conformal intervals
    -> FloodMap flood probability (if SAR/optical tiles available)
    -> RiskEngine decision for each monitored region
    -> Gemini plain-language brief.

Designed to be scheduled (cron / GitHub Actions) every 6h.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from varuna.ai.gemini import GeminiRiskBrief, load_env_file
from varuna.data.sources import GFSPrecipitationAPI, GloFASAPI, IMDNowcastAPI, SentinelNRTClient
from varuna.engines.risk_engine import RiskEngine, RiskPolicy
from varuna.models.floodmap import FloodMap
from varuna.models.hydronet import HydroNet
from varuna.models.raincast import RainCast
from varuna.uncertainty.conformal import ConformalRegressor, ConformalSegmentation
from varuna.utils.logging import setup_logger

logger = setup_logger("varuna.pipeline")


class ForecastPipeline:
    def __init__(self, config_path: str | Path, run_dir: str = "runs/forecast") -> None:
        self.cfg = self._load(config_path)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._build()

    def _load(self, path: str | Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build(self) -> None:
        m = self.cfg["model"]
        self.raincast = RainCast(
            in_channels=m["raincast"]["in_channels"],
            hidden_channels=m["raincast"]["hidden"],
            n_blocks=m["raincast"]["n_blocks"],
            forecast_horizon=m["raincast"]["horizon"],
        )
        self.hydronet = HydroNet(
            num_nodes=m["hydronet"]["num_nodes"],
            node_feat_dim=m["hydronet"]["node_feat_dim"],
            hidden_dim=m["hydronet"]["hidden"],
        )
        self.floodmap = FloodMap(
            img_size=m["floodmap"]["img_size"],
            embed_dim=m["floodmap"]["embed_dim"],
            depth=m["floodmap"]["depth"],
        )
        self.raincast_cal = ConformalRegressor(alpha=self.cfg["uncertainty"]["alpha"])
        self.hydronet_cal = ConformalRegressor(alpha=self.cfg["uncertainty"]["alpha"])
        self.flood_cal = ConformalSegmentation(alpha=self.cfg["uncertainty"]["alpha"])

        self.live_gfs = GFSPrecipitationAPI()
        self.live_imd = IMDNowcastAPI()
        self.live_glofas = GloFASAPI()
        self.live_sentinel = SentinelNRTClient()

        # Load .env for Gemini key
        load_env_file()
        self.gemini = GeminiRiskBrief()

    def run(self) -> list[dict]:
        """Execute one full operational cycle and persist the decisions."""
        logger.info("Starting VARUNA forecast cycle (device=%s)", self.device)

        # 1) live forcing
        gfs = self.live_gfs.fetch()
        imd = self.live_imd.fetch_rainfall()
        glofas = self.live_glofas.fetch()
        logger.info("Live feeds: gfs=%s imd=%s glofas=%s", gfs.time, imd.time, glofas.time)

        # 2) RainCast distributional forecast (placeholder input grid)
        H = W = 64
        x = torch.randn(1, self.raincast.in_channels, H, W)
        mu, sigma = self.raincast.predict_distribution(x)
        rainfall_accum_3d = float(mu.sum().item()) if mu.numel() else 0.0

        # 3) HydroNet discharge with conformal interval
        N = self.cfg["model"]["hydronet"]["num_nodes"]
        node_feat = self.cfg["model"]["hydronet"]["node_feat_dim"]
        total_feat = 2 * node_feat
        T = 5
        feat = torch.randn(T, N, total_feat)
        edge_index = torch.tensor(
            [[i, i + 1] for i in range(N - 1)] + [[i, i + 1] for i in range(N - 1)], dtype=torch.long
        ).t().contiguous()
        t_span = torch.linspace(0, 1, T)
        with torch.no_grad():
            mu_dis, logvar_dis, _ = self.hydronet(feat, edge_index, t_span)
        mean_dis = mu_dis.detach().cpu().numpy()[..., 0]
        q = getattr(self.hydronet_cal, "q", None)
        band = q if q else float(np.std(mean_dis))
        lo = mean_dis - band
        hi = mean_dis + band

        # 4) flood map probability (placeholder tiles -> deterministic demo)
        img_size = self.cfg["model"]["floodmap"]["img_size"]
        sar_chans = self.cfg["model"]["floodmap"]["sar_chans"]
        opt_chans = self.cfg["model"]["floodmap"]["opt_chans"]
        sar = torch.zeros(1, sar_chans, img_size, img_size)
        opt = torch.zeros(1, opt_chans, img_size, img_size)
        with torch.no_grad():
            flood_prob = self.floodmap.flood_prob(sar, opt).squeeze(0).squeeze(0).cpu().numpy()

        # 5) RiskEngine decisions per monitored region
        decisions = []
        for pol in self.cfg["regions"]:
            engine = RiskEngine(RiskPolicy(**pol))
            decision = engine.evaluate(mean_dis, lo, hi, rainfall_accum_3d, flood_prob)
            decisions.append(engine.to_json(decision))

        # 6) Gemini plain-language brief
        brief = self.gemini.generate(decisions)
        logger.info("Risk brief generated (%d chars)", len(brief))

        out = {
            "cycle": {
                "gfs": gfs.time,
                "imd": imd.time,
                "glofas": glofas.time,
                "rainfall_accum_3d_mm": rainfall_accum_3d,
                "discharge_mean": mean_dis.tolist(),
            },
            "decisions": decisions,
            "brief": brief,
        }
        with open(self.run_dir / "latest_cycle.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        logger.info("Cycle complete. %d regions evaluated.", len(decisions))
        return decisions


if __name__ == "__main__":
    pipeline = ForecastPipeline("config/config.yaml")
    pipeline.run()