"""Download ERA5 rainfall reanalysis for the configured region.

Requires a ~/.cdsapirc (Copernicus Climate Data Store account).
Offline / no credentials -> prints a clear message and exits 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varuna.data.sources import ERA5Downloader

if __name__ == "__main__":
    cfg = yaml.safe_load(open("config/config.yaml", encoding="utf-8"))
    region = cfg["data"]["era5_region"]  # [lat_max, lon_min, lat_min, lon_max]
    years = cfg["data"]["era5_years"]
    dl = ERA5Downloader(region=region, years=years)
    files = dl.download("data/raw/era5")
    print(f"Downloaded {len(files)} files: {[str(f) for f in files]}")