"""Runs the end-to-end forecast cycle and prints the JSON decisions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from varuna.pipeline.forecast import ForecastPipeline

if __name__ == "__main__":
    pipe = ForecastPipeline(Path(__file__).resolve().parents[1] / "config" / "config.yaml", run_dir="runs/forecast")
    decisions = pipe.run()
    print(json.dumps(decisions, indent=2))