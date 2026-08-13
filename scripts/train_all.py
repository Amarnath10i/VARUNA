"""Training entrypoint for all VARUNA components."""
from __future__ import annotations

import argparse

from varuna.utils.logging import setup_logger

logger = setup_logger("varuna.scripts")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train VARUNA components")
    ap.add_argument("--component", choices=["raincast", "hydronet", "floodmap"], required=True)
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    args = ap.parse_args()

    logger.info("Training %s on %s", args.component, args.device)
    # Real runs wire up the data pipeline (ERA5 / Sentinel) then call:
    #   varuna.engines.train_raincast.train_raincast(...)
    #   varuna.engines.train_hydronet.train_hydronet(...)
    #   varuna.engines.train_floodmap.pretrain_floodmap(...)


if __name__ == "__main__":
    main()