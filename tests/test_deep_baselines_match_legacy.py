"""deep_baselines.py must give the SAME numbers as the paper's code (legacy/experiments/deep_survival_rolling.py).

Same seed + same data -> same weights, same batch order, same predictions (atol 1e-6). Synthetic data; the real data are compared in the run of main.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fact_survival.data import encode_ids                                        # noqa: E402
from fact_survival.deep_baselines import SafeCoxPHLoss, deep_predictions, inputs  # noqa: E402

sys.path.insert(0, str(ROOT / "legacy"))                                          # the paper's code imports datasets.* and modules.*
spec = importlib.util.spec_from_file_location("legacy_deep", ROOT / "legacy/experiments/deep_survival_rolling.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

HORIZONS = [3, 6, 12]
SETTINGS = dict(hidden=16, dropout=0.1, learning_rate=0.003, weight_decay=0.001, batch_size=128)
legacy.HIDDEN, legacy.DROPOUT = SETTINGS["hidden"], SETTINGS["dropout"]            # the paper's code keeps its settings in module constants
legacy.LEARNING_RATE, legacy.WEIGHT_DECAY, legacy.BATCH_SIZE = SETTINGS["learning_rate"], SETTINGS["weight_decay"], SETTINGS["batch_size"]
legacy.select_features(["age"])                                                    # the baselines of the paper's table: age features only (columns 0-4)


def make_data(seed=0, n=700):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12)).astype(np.float32)
    rows = pd.DataFrame({"subject": rng.choice([f"s{i}" for i in range(30)], n), "object": rng.choice([f"o{i}" for i in range(20)], n),
                         "relation": rng.choice([f"r{i}" for i in range(6)], n)})
    train, evaluation = np.arange(550), np.arange(550, n)
    duration = rng.integers(1, 13, size=550)
    event = rng.random(550) < 0.4
    return x, rows, train, evaluation, duration, event


def legacy_inputs(x, rows, train, evaluation):
    entities = pd.factorize(pd.concat((rows.subject, rows.object), ignore_index=True), sort=True)[0]          # the paper's global codes
    categories = np.column_stack((entities[:len(rows)], pd.factorize(rows.relation, sort=True)[0], entities[len(rows):]))
    return legacy.build_inputs(x, categories, train, evaluation)


def test_inputs_match_legacy():
    x, rows, train, evaluation, *_ = make_data()
    ids_train, ids_eval = encode_ids(rows.iloc[train], rows.iloc[evaluation])
    a, b = inputs(x[train][:, :5], ids_train, x[evaluation][:, :5], ids_eval)
    old_a, old_b = legacy_inputs(x, rows, train, evaluation)
    np.testing.assert_allclose(a, old_a, atol=1e-7, rtol=0)
    np.testing.assert_allclose(b, old_b, atol=1e-7, rtol=0)


@pytest.mark.parametrize("name", ["deepsurv", "logistic_hazard"])
def test_predictions_match_legacy_at_two_epochs(name):
    x, rows, train, evaluation, duration, event = make_data()
    ids_train, ids_eval = encode_ids(rows.iloc[train], rows.iloc[evaluation])
    old_train, old_eval = legacy_inputs(x, rows, train, evaluation)
    expected = legacy.fit_origin(name, old_train, old_eval, duration, event, HORIZONS, [1, 3], seed=7)
    got = deep_predictions(name, x[train][:, :5], ids_train, x[evaluation][:, :5], ids_eval, duration, event, HORIZONS, [1, 3], seed=7, **SETTINGS)
    for epoch in (1, 3):
        assert got[epoch].shape == (len(evaluation), len(HORIZONS))
        np.testing.assert_allclose(got[epoch], expected[epoch], atol=1e-6, rtol=0)
        assert (np.diff(got[epoch], axis=1) <= 1e-7).all()                        # survival never increases with the horizon


def test_a_batch_without_an_event_gives_zero_loss():
    log_h = torch.zeros(5, 1, requires_grad=True)
    loss = SafeCoxPHLoss()(log_h, torch.arange(1.0, 6.0), torch.zeros(5))
    loss.backward()
    assert float(loss) == 0.0 and torch.isfinite(log_h.grad).all()
