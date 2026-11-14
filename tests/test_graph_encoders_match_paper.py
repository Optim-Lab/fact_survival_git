"""Reproduction on the real data: refit RE-GCN / LogCL with the NEW code at one ICEWSWiki test origin (seed 42, the selected epoch 5)
and compare with the predictions stored by the paper's run.

    ICEWSWIKI_PKL=...  PAPER_ENCODER_RUNS=.../original_encoder_suite_20261023  [ENCODER_ORIGIN=24153]  python -m pytest tests/test_graph_encoders_match_paper.py -s

One origin takes ~30 min (RE-GCN) / ~1 h (LogCL) on one GPU. The GPU graph code (DGL scatter-add) is not bit-deterministic, so
row-by-row equality is impossible here (bit-equality is checked on a small graph in test_graph_encoders_match_legacy.py).
Measured noise: the PAPER's code re-run at origin 24153 (results/encoder_repro/legacy_rerun.py, 2026-11-12) differs from its own stored
run by mean |diff| 2.4e-3 (RE-GCN) / 5.5e-3 (LogCL) and by up to 0.35 / 0.70 in Brier x1000. The test therefore asks the new code to stay
inside that noise: mean |diff| < 1e-2 and |Brier x1000 difference| < 1.0 at every horizon.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not torch.cuda.is_available() or "ICEWSWIKI_PKL" not in os.environ or "PAPER_ENCODER_RUNS" not in os.environ:
    pytest.skip("needs a GPU, ICEWSWIKI_PKL and PAPER_ENCODER_RUNS", allow_module_level=True)
import main  # noqa: E402

ORIGIN = int(os.environ.get("ENCODER_ORIGIN", 24153))


@pytest.mark.parametrize("name", ["regcn", "logcl"])
def test_one_test_origin_matches_paper_run(name):
    config = json.load(open(ROOT / "configs/icewswiki.json"))
    run = Path(os.environ["PAPER_ENCODER_RUNS"]) / f"icewswiki_{name}_seed42_merged"
    epoch = json.loads((run / "selection.json").read_text())["epoch"]
    paper = pd.read_csv(run / "test_predictions.csv.gz", keep_default_na=False)
    paper = paper[paper.origin == ORIGIN]

    cache = ROOT / f"results/encoder_repro/{name}_seed42_origin{ORIGIN}.csv.gz"          # training takes long: keep the new predictions
    if not cache.exists():
        rows, x, tau, labels, info = main.prepare(config, os.environ["ICEWSWIKI_PKL"])
        survival = main.encoder_predictions(config, name, rows, x, labels, ORIGIN, 42, [epoch], "cuda:0")[epoch]
        at_origin = rows[rows.landmark == ORIGIN]
        pd.concat([at_origin[["subject", "relation", "object"]].assign(horizon=h, p_new=survival[:, k])[~np.isnan(labels[at_origin.index, k])]
                   for k, h in enumerate(config["horizons"])]).to_csv(cache, index=False)
    new = pd.read_csv(cache, keep_default_na=False)

    # A triple can be active twice at one origin (two intervals: no unique key without the raw dates), and the paper's file is in
    # another order. So both sides are sorted by (horizon, triple, probability) and paired by position: exact for a unique triple;
    # the two copies of a repeated triple are paired by the size of their probability. The identity columns must then agree row by row.
    keys = ["horizon", "subject", "relation", "object"]
    new = new.sort_values(keys + ["p_new"], ignore_index=True)
    paper = paper.sort_values(keys + [f"p_{name}"], ignore_index=True)
    assert len(new) == len(paper) and (new[keys].astype(str).to_numpy() == paper[keys].astype(str).to_numpy()).all()
    difference = np.abs(new.p_new - paper[f"p_{name}"])
    brier = lambda p: ((p - paper.survived) ** 2).groupby(paper.horizon).mean() * 1000
    print(f"\n{name} origin {ORIGIN}: {len(new)} rows, max |difference| = {difference.max():.2e}, mean = {difference.mean():.2e}")
    print(pd.DataFrame(dict(new_brier_x1000=brier(new.p_new), paper_brier_x1000=brier(paper[f"p_{name}"]))).round(4).to_string())
    assert difference.mean() < 1e-2
    assert (brier(new.p_new) - brier(paper[f"p_{name}"])).abs().max() < 1.0
