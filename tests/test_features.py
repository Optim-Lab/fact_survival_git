"""features.py against real rows printed by the paper's code (legacy load_data, FinWiki), plus a hand-checkable degree example."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fact_survival.features import make_features                      # noqa: E402


def test_real_row_with_all_parts_active():
    # FinWiki, landmark 2019-02: age 13 months; Kazakhstan is the object of 38 active facts (1 with this relation);
    # the subject has 9 active facts (all with this relation); tau = 3 months, taken from the global level.
    x = make_features([13], [38], [1], [9], [9], [3.0], ["global"])
    expected = [2.6391, 0.9286, 1.9459, 1.2528, 0.2412, 3.6636, 0.6931, 2.3026, 2.3026, 1.3863, 0, 0]
    np.testing.assert_allclose(x[0], expected, atol=1e-4)


def test_real_row_at_the_first_landmark():
    # age 0, subject with 5 active facts, tau = 0.5 (no update gap exists yet, global fallback)
    x = make_features([0], [1], [1], [5], [5], [0.5], ["global"])
    expected = [0, 0, 0, 0, 0, 0.6931, 0.6931, 1.7918, 1.7918, 0.4055, 0, 0]
    np.testing.assert_allclose(x[0], expected, atol=1e-4)


def test_half_life_level_flags():
    levels = ["subject_relation", "relation", "global"]
    x = make_features([1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2.0, 2.0, 2.0], levels)
    assert x.shape == (3, 12) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:, 10:], [[1, 0], [0, 1], [0, 0]])
