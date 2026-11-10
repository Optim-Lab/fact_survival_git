"""Smoke test: train at ONE rolling origin and print the Brier of the facts active there (the full experiment is main.py).

    python scripts/run_experiment.py configs/finwiki.json --data PATH_TO_PICKLE --origin 24250 --epochs 5 [--seed 42] [--device cuda]
"""
#%%
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import neural_predictions, origin_data, prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--data", required=True)
    parser.add_argument("--origin", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = json.load(open(args.config))
    rows, x, tau, labels, info = prepare(config, args.data)
    d = origin_data(config, rows, labels, args.origin)
    survival = neural_predictions(config, x, tau, d, args.seed, [args.epochs], args.device)[args.epochs]
    y = labels[d["test"]]
    for k, h in enumerate(config["horizons"]):
        known = ~np.isnan(y[:, k])
        print(f"horizon {h}: {known.sum():,} scored facts, Brier {np.mean((survival[known, k] - y[known, k]) ** 2):.5f}")
