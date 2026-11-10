"""baselines.py against the paper's baselines (legacy/modules/model.py) on a real FinWiki origin (needs FINWIKI_PKL)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "legacy"))
from fact_survival.baselines import baseline_predictions                              # noqa: E402


def test_by_hand_global_and_shapes():
    rows = pd.DataFrame({"relation": ["a", "a", "b", "b"] * 5, "age": np.arange(20.0),
                         "object_degree": 1.0, "object_relation_degree": 1.0})
    labels = np.array([[1.0, 1.0], [0.0, np.nan], [1.0, 0.0], [1.0, np.nan]] * 5)
    p = baseline_predictions(rows, labels, rows.iloc[:3], c=1.0)
    assert set(p) == {"global", "relation_age", "object_degree"} and p["global"].shape == (3, 2)
    np.testing.assert_allclose(p["global"][0], [15 / 20, 5 / 10])                    # mean of the known labels per horizon
    assert ((p["relation_age"] > 0) & (p["relation_age"] < 1)).all()


@pytest.mark.skipif("FINWIKI_PKL" not in os.environ, reason="set FINWIKI_PKL to the raw FinWiki file")
@pytest.mark.parametrize("horizon", [3, 12])
def test_real_origin_matches_legacy(horizon):
    from datasets.interval_landmark import load_landmark_rows, month_id, select_rolling_fold          # legacy
    from modules.model import fit_and_predict                                                         # legacy
    from fact_survival.data import landmark_rows, read_facts
    from fact_survival.labels import known_at, survival_labels

    path, origin, c = os.environ["FINWIKI_PKL"], 24228, 0.01
    legacy_rows, meta = load_landmark_rows(Path(path), horizon)                  # rows built for this horizon only
    train, validation = select_rolling_fold(legacy_rows, origin, month_id(meta["valid_end"]))
    expected = fit_and_predict(train, validation, c)

    facts, info = read_facts(path)
    rows = landmark_rows(facts, info["study_end"], info["study_end"] - 3 - 1)    # all horizons share these rows
    y = survival_labels(rows.stop, rows.ended, rows.landmark, [horizon])
    y = known_at(info["study_end"] - 1, y, rows.landmark.to_numpy(), [horizon])  # the study-end rule
    usable = known_at(origin, y, rows.landmark.to_numpy(), [horizon])
    train_idx = np.flatnonzero(~np.isnan(usable[:, 0]))
    test_idx = np.flatnonzero(rows["landmark"].to_numpy() == origin)
    p = baseline_predictions(rows.iloc[train_idx], usable[train_idx], rows.iloc[test_idx], c)

    assert len(train_idx) == len(train) and len(test_idx) == len(validation)
    for name, legacy_name in (("global", "p_global"), ("relation_age", "p_relation_age"), ("object_degree", "p_object_degree")):
        np.testing.assert_allclose(p[name][:, 0], expected[legacy_name].to_numpy(), atol=1e-9)
