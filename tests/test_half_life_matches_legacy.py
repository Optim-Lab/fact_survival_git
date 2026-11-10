"""half_life.py against a hand-checkable example and against the paper's code (legacy/modules/halo_half_life.py)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fact_survival.half_life import half_lives                                        # noqa: E402

spec = importlib.util.spec_from_file_location("legacy_halo", ROOT / "legacy/modules/halo_half_life.py")
legacy = importlib.util.module_from_spec(spec)
sys.modules["legacy_halo"] = legacy                                                   # dataclasses look the module up by name
spec.loader.exec_module(legacy)


def test_by_hand():
    # starts: (a,r) at 1, 3, 7   (b,r) at 2, 6      -> gaps: a: 2, 4   b: 4        relation r: [2, 4, 4], median 4
    history = dict(history_subject=["a", "a", "a", "b", "b"], history_relation=["r"] * 5, history_begin=[1, 3, 7, 2, 6])
    subject = ["a", "b", "new", "a"]
    relation = ["r", "r", "r", "q"]                                   # "q": a relation without any gap
    tau, level = half_lives(subject, relation, [10, 10, 10, 10], **history)
    np.testing.assert_allclose(tau, [1.5, 2.0, 2.0, 2.0])             # 0.5 x mean(2, 4);  0.5 x median 4;  same;  0.5 x global median 4
    assert list(level) == ["subject_relation", "relation", "relation", "global"]


def test_no_gap_at_all():
    tau, level = half_lives(["a", "z"], ["r", "q"], [5, 5], ["a"], ["r"], [1])      # one start only: there is no gap
    np.testing.assert_allclose(tau, [0.5, 0.5])                                       # 0.5 x 1 time unit
    assert list(level) == ["global", "global"]


def test_only_starts_up_to_the_landmark_are_used():
    history = dict(history_subject=["a", "a", "a"], history_relation=["r"] * 3, history_begin=[1, 3, 7])
    tau, level = half_lives(["a"], ["r"], [5], **history)             # at c = 5 only starts 1 and 3 are known: ONE gap
    assert level[0] == "relation" and tau[0] == 1.0                   # not enough gaps for the pair -> relation median 2 x 0.5
    tau, level = half_lives(["a"], ["r"], [10], **history)            # at c = 10 the pair has two gaps (2, 4)
    assert level[0] == "subject_relation" and tau[0] == 1.5


def test_matches_legacy():
    rng = np.random.default_rng(0)
    n = 500
    facts = pd.DataFrame({"subject": rng.integers(0, 25, n), "relation": rng.integers(0, 5, n),
                          "object": rng.integers(0, 60, n), "begin_time": rng.integers(0, 40, n)})
    facts.loc[0, "relation"] = 9                                      # a relation that occurs once: no gap
    rows = pd.DataFrame({"subject": rng.integers(0, 25, 300), "relation": rng.integers(0, 5, 300),
                         "origin": rng.integers(10, 45, 300)})
    rows.loc[:9, "subject"] = 999                                     # subjects and a relation never seen in the history
    rows.loc[10:14, "relation"] = 888

    expected = legacy.HaloHalfLifeBaseline(begin_col="begin_time", landmark_col="origin").raw_predict(
        rows, facts, 3, return_details=True)
    tau, level = half_lives(rows["subject"], rows["relation"], rows["origin"],
                            facts["subject"], facts["relation"], facts["begin_time"])

    np.testing.assert_allclose(tau, expected["half_life"].to_numpy(), rtol=1e-6)
    assert list(level) == list(expected["half_life_source"])
    assert set(level) == {"subject_relation", "relation", "global"}   # the test really reaches all three levels
