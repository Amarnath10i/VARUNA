"""varuna-bench <experiment>: each writes its tables to reports/."""

from __future__ import annotations

import argparse
import logging


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="varuna-bench", description=__doc__)
    ap.add_argument(
        "experiment",
        choices=[
            "conceptual",
            "lstm",
            "lgbm",
            "ungauged",
            "evaluate",
            "floods",
            "conformal",
            "forecast",
        ],
    )
    ap.add_argument("--seeds", type=int, default=3, help="LSTM ensemble size")
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from . import experiments

    getattr(experiments, f"run_{args.experiment}")(args)


if __name__ == "__main__":
    main()
