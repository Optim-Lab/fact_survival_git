"""The simple code must give the SAME predictions as the code that produced the paper (legacy/modules/neural_hazard.py).

Same seed + same data -> same random initial weights, same batch order, same dropout masks -> same numbers.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fact_survival.training import FactSurvival                                      # noqa: E402

spec = importlib.util.spec_from_file_location("legacy_neural_hazard", ROOT / "legacy/modules/neural_hazard.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)                                                       # the paper's code, loaded by path

HORIZONS = [3, 6, 12]
SETTINGS = dict(seed=7, hidden=16, dropout=0.1, batch_size=128, learning_rate=0.003, weight_decay=0.001)


def make_rows(rng, n, unseen_ids=False):
    features = rng.normal(size=(n, 12)).astype(np.float32)
    ids = rng.integers(1, 6, size=(n, 3))
    if unseen_ids:
        ids[:20] = 99                                      # ids that the training rows never contained
    return features, ids, rng.uniform(1.0, 20.0, size=n).astype(np.float32)


def test_matches_legacy_at_two_epochs():
    rng = np.random.default_rng(0)
    features, ids, half_life = make_rows(rng, 600)
    duration = rng.exponential(10.0, size=600)
    labels = (duration[:, None] > np.array(HORIZONS)).astype(np.float32)
    known = rng.random(labels.shape) > 0.25
    known[:, 0] |= rng.random(600) > 0.5                   # the shortest horizon is known most often
    labels[~known] = np.nan                                # NaN = unknown label
    new_rows = make_rows(rng, 150, unseen_ids=True)

    legacy_predictions, _, _ = legacy.fit_predict(
        features, ids, labels, known, half_life, *new_rows, HORIZONS, variant="neural", epochs=[1, 3], **SETTINGS)

    for epochs in (1, 3):                                  # training for E epochs = the snapshot at epoch E of a longer run
        model = FactSurvival(HORIZONS, epochs=epochs, **SETTINGS).fit(features, ids, half_life, labels)
        survival = model.predict(*new_rows)
        assert survival.shape == (150, len(HORIZONS))
        np.testing.assert_allclose(survival, legacy_predictions[epochs], atol=1e-6, rtol=0)
        assert (np.diff(survival, axis=1) <= 1e-7).all()   # survival never increases with the horizon
