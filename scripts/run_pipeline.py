#!/usr/bin/env python
"""Entry point: build the extract, run the models, price the policy, report it.

Usage
-----
    python scripts/run_pipeline.py                       # synthetic extract (default)
    python scripts/run_pipeline.py --data-mode real      # IEEE-CIS CSVs in data/raw/
    python scripts/run_pipeline.py --transactions 60000  # smaller, faster run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from frauddetect import config as cfg  # noqa: E402
from frauddetect.report import run_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the payment fraud analytics pipeline.")
    parser.add_argument(
        "--data-mode",
        choices=("synthetic", "real"),
        default="synthetic",
        help="synthetic (reproducible, no download) or real (IEEE-CIS extract required)",
    )
    parser.add_argument(
        "--transactions",
        type=int,
        default=cfg.DEFAULT_N_TRANSACTIONS,
        help="number of synthetic transactions to generate",
    )
    parser.add_argument("--seed", type=int, default=cfg.SEED, help="random seed")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    arguments = parser.parse_args()

    run_pipeline(
        data_mode=arguments.data_mode,
        n_transactions=arguments.transactions,
        seed=arguments.seed,
        verbose=not arguments.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
