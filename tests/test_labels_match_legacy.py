"""labels.py must agree with the label contract of the paper's code (legacy/datasets/task.py)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fact_survival.labels import known_at, survival_labels                            # noqa: E402

spec = importlib.util.spec_from_file_location("legacy_task", ROOT / "legacy/datasets/task.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

HORIZONS = np.array([3, 6, 12])


def random_facts(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    begin = rng.integers(0, 40, size=n)
    landmark = begin + rng.integers(0, 10, size=n)
    stop = landmark + rng.integers(1, 30, size=n)                 # stop > landmark: active at the landmark
    return begin, stop, rng.random(n) > 0.4, landmark


def test_worked_example_of_the_docstring():
    stop = np.array([10, 8, 7, 12])
    ended = np.array([True, True, False, False])
    labels = survival_labels(stop, ended, np.full(4, 5), [3])[:, 0]
    np.testing.assert_array_equal(labels, [1, 0, np.nan, 1])


def test_labels_match_legacy():
    begin, stop, ended, landmark = random_facts()
    labels = survival_labels(stop, ended, landmark, HORIZONS)
    for k, h in enumerate(HORIZONS):
        expected = legacy.make_survival_labels(begin, stop, landmark, int(h), event_observed=ended)
        np.testing.assert_array_equal(labels[:, k], expected)


def test_known_at_matches_legacy_mask():
    begin, stop, ended, landmark = random_facts()
    labels = survival_labels(stop, ended, landmark, HORIZONS)
    for origin in (20, 35, 50, 80):
        mask = legacy.mature_label_mask(labels, landmark, HORIZONS, origin)
        np.testing.assert_array_equal(known_at(origin, labels, landmark, HORIZONS), np.where(mask, labels, np.nan))


def test_rejects_facts_that_are_already_over_at_the_landmark():
    with pytest.raises(ValueError):
        survival_labels([9], [True], [9], [3])                    # landmark == stop
