"""metrics.py against the paper's metrics (legacy/evaluation/metrics.py) -- on random data, and on a saved real result."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fact_survival.metrics import average_precision, brier, scores                   # noqa: E402

spec = importlib.util.spec_from_file_location("legacy_metrics", ROOT / "legacy/evaluation/metrics.py")
legacy = importlib.util.module_from_spec(spec)
sys.modules["legacy_metrics"] = legacy
spec.loader.exec_module(legacy)


def test_by_hand():
    p = np.array([0.9, 0.2, 0.5])
    y = np.array([1.0, 0.0, np.nan])                               # the third cell is unknown and must not count
    assert brier(p, y) == pytest.approx(((0.9 - 1) ** 2 + (0.2 - 0) ** 2) / 2)
    # one origin: the fact that ended (y = 0) has the higher "will end" score (1 - 0.2 > 1 - 0.9) -> AP = 1
    assert average_precision(p, y, np.array([5, 5, 5])) == pytest.approx(1.0)


def test_random_data_matches_legacy():
    rng = np.random.default_rng(0)
    n = 3000
    origin = rng.integers(0, 12, size=n)
    origin[origin == 11] = 10                                       # origins 0..10 only ...
    y = (rng.random(n) > 0.2).astype(float)
    y[origin == 3] = 1.0                                            # ... and one origin with a single class (AP undefined there)
    p = np.clip(0.8 - 0.4 * (1 - y) + rng.normal(0, 0.2, n), 0.01, 0.99)
    frame = pd.DataFrame({"fact_id": rng.integers(0, 800, n), "origin": origin, "survived": y.astype(int), "p": p})
    expected = legacy.summarize_probability(frame, "p")
    assert brier(p, y) == pytest.approx(expected["row_brier"])
    assert average_precision(p, y, origin) == pytest.approx(expected["event_ap_within_origin"])


@pytest.mark.skipif("RESULT_DIR" not in os.environ, reason="set RESULT_DIR to a finished run folder (test_predictions.csv.gz + test_metrics.csv)")
def test_reproduces_a_saved_result():
    folder = Path(os.environ["RESULT_DIR"])
    predictions = pd.read_csv(folder / "test_predictions.csv.gz", usecols=["origin", "horizon", "survived", "p_neural"])
    saved = pd.read_csv(folder / "test_metrics.csv")
    saved = saved[saved["model"] == "neural"].set_index("horizon")
    for h, part in predictions.groupby("horizon"):
        p, y, o = part["p_neural"].to_numpy(), part["survived"].to_numpy(float), part["origin"].to_numpy()
        assert brier(p, y) == pytest.approx(saved.loc[h, "row_brier"], rel=1e-9)
        assert average_precision(p, y, o) == pytest.approx(saved.loc[h, "event_ap_within_origin"], rel=1e-9)
