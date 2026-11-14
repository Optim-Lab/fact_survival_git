"""RE-GCN / LogCL + survival head: the simple code must give the SAME predictions as the code that produced the paper
(legacy/modules/regcn_survival.py, legacy/modules/logcl_original_survival.py), on a small synthetic graph.

Needs a GPU (the authors' RE-GCN code runs on CUDA only), DGL, and the authors' code:  REGCN_CODE = RE-GCN repository
(default ../RE-GCN);  LogCL = legacy/vendor/logcl_cpu.
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REGCN_CODE = Path(os.environ.get("REGCN_CODE", ROOT.parent / "RE-GCN"))
if not torch.cuda.is_available() or not REGCN_CODE.exists():
    pytest.skip("needs a GPU and the RE-GCN code (REGCN_CODE)", allow_module_level=True)

sys.path.insert(0, str(ROOT))
from fact_survival.graph_encoders import fit_predict, use_authors_code   # noqa: E402
from fact_survival.labels import known_at, survival_labels                # noqa: E402

use_authors_code(REGCN_CODE, ROOT / "legacy")                             # legacy/ also makes the paper's `datasets.*`, `modules.*` importable


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy_regcn = load_by_path("legacy_regcn", ROOT / "legacy/modules/regcn_survival.py")
legacy_logcl = load_by_path("legacy_logcl", ROOT / "legacy/modules/logcl_original_survival.py")
legacy_logcl.check_upstream_hashes = lambda: None    # it hashes legacy/vendor/LogCL (the untouched upstream copy), not part of legacy/

HORIZONS, ORIGIN, EPOCHS = [1, 2, 4], 11, [1, 3]
SETTINGS = dict(hidden=8, history_length=3, batch_size=64, learning_rate=0.003, weight_decay=0.001)


def synthetic_graph(seed=0):
    """~150 interval facts on 15 entities / 4 relations; one row per (fact, landmark c) with c <= ORIGIN;
    facts that begin at the origin use entities never seen in training (-> id 0)."""
    rng = np.random.default_rng(seed)
    n = 150
    facts = pd.DataFrame(dict(subject=[f"e{i}" for i in rng.integers(0, 15, n)], relation=[f"r{i}" for i in rng.integers(0, 4, n)],
                              object=[f"e{i}" for i in rng.integers(0, 15, n)], begin=rng.integers(0, ORIGIN + 1, n)))
    facts["stop"] = facts.begin + rng.integers(1, 9, n)
    facts.loc[facts.begin == ORIGIN, "subject"] = "new"
    rows = pd.concat([facts[(facts.begin <= c) & (facts.stop > c)].assign(landmark=c) for c in range(ORIGIN + 1)], ignore_index=True)
    labels = survival_labels(rows.stop, rows.stop < 14, rows.landmark, HORIZONS)                # study end 14
    x = rng.normal(size=(len(rows), 12)).astype(np.float32)
    return rows, x, labels


@pytest.mark.parametrize("name", ["regcn", "logcl"])
def test_matches_paper_code(name):
    rows, x, labels = synthetic_graph()
    usable = known_at(ORIGIN, labels, rows.landmark.to_numpy(), HORIZONS)
    new = fit_predict(name, rows, x, usable, HORIZONS, ORIGIN, epochs=EPOCHS, seed=3, device="cuda:0",
                      contrastive_weight=0.1, contrastive_batch=16, **SETTINGS)

    old_rows = rows.rename(columns={"landmark": "origin"}).assign(fact_id=rows.index.astype(str))   # the paper's column names
    settings = {k: SETTINGS[k] for k in ("hidden", "history_length", "batch_size")}               # lr / weight decay were fixed there
    if name == "regcn":
        _, old, _, _ = legacy_regcn.fit_predict_origin(old_rows, x, labels, HORIZONS, ORIGIN, epochs=EPOCHS, seed=3, **settings)
    else:
        _, old, _, _ = legacy_logcl.fit_predict_origin(old_rows, x, labels, HORIZONS, ORIGIN, epochs=EPOCHS, seed=3,
                                                       contrastive_weight=0.1, contrastive_batch=16, **settings)
    for epoch in EPOCHS:
        assert new[epoch].shape == (int((rows.landmark == ORIGIN).sum()), len(HORIZONS))
        np.testing.assert_allclose(new[epoch], old[epoch], atol=1e-6, rtol=0)
