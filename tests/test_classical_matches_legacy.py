"""classical.py (KM, Cox, discrete hazard) against the paper's code (legacy/modules) on a real FinWiki origin (needs FINWIKI_PKL)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "legacy"))
from fact_survival.classical import classical_predictions, follow_up                  # noqa: E402


def small_rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    landmark = rng.integers(0, 20, n)
    stop = landmark + rng.integers(1, 15, n)
    return pd.DataFrame({"relation": rng.choice(["a", "b", "c"], n), "age": rng.integers(0, 10, n),
                         "landmark": landmark, "stop": stop, "ended": rng.random(n) > 0.4})


def test_follow_up_by_hand():
    rows = pd.DataFrame({"landmark": [0, 0, 0, 0], "stop": [2, 10, 4, 4], "ended": [True, True, False, True]})
    duration, event = follow_up(rows, origin=6, horizon=5)
    assert duration.tolist() == [2, 5, 4, 4]                  # capped at min(T-c, origin-c, h)
    assert event.tolist() == [True, False, False, True]       # T=10 ends after the horizon; the censored one is no event


def test_small_data_runs_and_is_a_probability():
    rows = small_rows()
    train = rows[rows.landmark + 5 <= 20]
    test = rows.iloc[:50]
    p = classical_predictions(train, test, origin=20, horizon=5, cox_penalizer=0.1, cox_min_relation_rows=50, discrete_c=1.0)
    for curve in p.values():
        assert curve.shape == (50,) and ((curve >= 0) & (curve <= 1)).all()


@pytest.mark.skipif("FINWIKI_PKL" not in os.environ, reason="set FINWIKI_PKL to the raw FinWiki file")
@pytest.mark.parametrize("horizon", [3, 12])        # origin 24250 is in the test period (an early origin has too few events for Cox)
def test_real_origin_matches_legacy(horizon):
    from modules.proposed_experiment import rolling_outcomes                                    # legacy
    from modules.survival_baselines import DiscreteHazardModel, LandmarkCoxModel, fit_km_probability   # legacy
    from fact_survival.data import landmark_rows, read_facts
    from fact_survival.labels import known_at, survival_labels

    origin = 24250
    facts, info = read_facts(os.environ["FINWIKI_PKL"])
    rows = landmark_rows(facts, info["study_end"], info["study_end"] - 3 - 1)
    c = rows["landmark"].to_numpy()
    y = known_at(info["study_end"] - 1, survival_labels(rows.stop, rows.ended, c, [horizon]), c, [horizon])
    usable = known_at(origin, y, c, [horizon])
    train = rows.iloc[np.flatnonzero(~np.isnan(usable[:, 0]))]
    test = rows.iloc[np.flatnonzero(c == origin)]
    settings = dict(cox_penalizer=0.1, cox_min_relation_rows=50, discrete_c=1.0)
    p = classical_predictions(train, test, origin, horizon, **settings)

    # the paper's code, fed with the same rows
    duration, event = rolling_outcomes(train.stop.to_numpy(), train.ended.to_numpy(), train.landmark.to_numpy(), origin, horizon)
    old_train = pd.DataFrame({"relation": train.relation.to_numpy(), "age_months": train.age.to_numpy(),
                              "residual_duration": duration, "event_observed": event.astype(np.int8)})
    old_test = pd.DataFrame({"relation": test.relation.to_numpy(), "age_months": test.age.to_numpy()})
    np.testing.assert_allclose(p["km"], fit_km_probability(old_train, horizon), atol=1e-12)
    np.testing.assert_allclose(p["cox"], LandmarkCoxModel(0.1, 50).fit(old_train).predict_survival(old_test, horizon), atol=1e-9)
    np.testing.assert_allclose(p["discrete_hazard"], DiscreteHazardModel(1.0).fit(old_train).predict_survival(old_test, horizon), atol=1e-6)
