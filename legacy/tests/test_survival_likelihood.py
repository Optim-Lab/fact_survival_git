"""Discrete event/censor likelihood and rolling-origin causality checks."""
import numpy as np
import torch

from modules.neural_hazard import discrete_survival_nll, fit_predict
from modules.proposed_experiment import rolling_outcomes


def test_event_mass_and_censor_survival_match_hand_calculation():
    log_s = torch.log(torch.tensor([[1., .8, .6, .3], [1., .9, .7, .4]]))
    durations = torch.tensor([2, 2])
    events = torch.tensor([True, False])
    loss = discrete_survival_nll(log_s, durations, events)
    expected = -np.log(.8 - .6) / 2 - np.log(.7) / 2
    np.testing.assert_allclose(loss.item(), expected, rtol=1e-6)


def test_future_endpoint_does_not_enter_rolling_outcomes():
    landmarks = np.array([10, 10, 10, 10])
    stops = np.array([12, 14, 20, 30])
    observed = np.array([True, True, True, False])
    durations, events = rolling_outcomes(stops, observed, landmarks, origin=13, max_horizon=6)
    np.testing.assert_array_equal(durations, [2, 3, 3, 3])
    np.testing.assert_array_equal(events, [True, False, False, False])
    later = stops.copy()
    later[1:] = [100, 200, 300]
    other = rolling_outcomes(later, observed, landmarks, origin=13, max_horizon=6)
    for actual, expected in zip(other, (durations, events)):
        np.testing.assert_array_equal(actual, expected)


def test_likelihood_training_produces_monotone_survival_and_checkpoint():
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 12)).astype(np.float32)
    cat = np.column_stack((np.ones(24, int), np.arange(24) % 2 + 1, np.ones(24, int)))
    durations = np.tile([1, 2, 3, 4], 6)
    events = np.tile([True, False, True, False], 6)
    horizons = [1, 2, 3]
    labels = np.where(durations[:, None] > np.array(horizons), 1.,
                      np.where(events[:, None], 0., np.nan))
    mask = np.isfinite(labels)
    predictions, diagnostics, checkpoint = fit_predict(
        x, cat, labels, mask, np.full(24, 3, np.float32),
        x, cat, np.full(24, 3, np.float32), horizons,
        variant='neural', epochs=[1], batch_size=12,
        objective='survival_likelihood', durations_train=np.minimum(durations, 3),
        events_train=events & (durations <= 3))
    probabilities = predictions[1]
    assert diagnostics['objective'] == checkpoint['objective'] == 'survival_likelihood'
    assert np.isfinite(probabilities).all()
    assert (np.diff(probabilities, axis=1) <= 1e-6).all()


def test_benchmark_keeps_objectives_in_separate_rows(tmp_path):
    import json
    import pandas as pd
    from datasets.task import PROTOCOL_RULES, PROTOCOL_VERSION
    from evaluation.benchmark_table import read_runs

    for objective in ('bce', 'survival_likelihood'):
        folder = tmp_path / objective
        folder.mkdir()
        config = dict(dataset='finwiki', protocol_version=PROTOCOL_VERSION,
                      protocol=PROTOCOL_RULES, input_sha256={'source': 'same'},
                      objective=objective, variants=['neural'], seed=42,
                      partial_development=True)
        (folder / 'run_config.json').write_text(json.dumps(config))
        rows = pd.DataFrame(dict(fact_id=['f1', 'f2'], origin=[10, 10],
                                 horizon=[3, 3], survived=[1, 0],
                                 p_neural=[.9, .1]))
        rows.to_csv(folder / 'development_predictions.csv.gz', index=False,
                    compression='gzip')
    scores = read_runs([tmp_path / 'bce', tmp_path / 'survival_likelihood'],
                       phase='development', allow_partial=True)
    assert set(scores.model) == {'neural', 'neural_survival_likelihood'}


def test_yago_outcomes_are_separate_from_model_features(tmp_path):
    import json
    import pandas as pd
    from datasets.yago import load_yago_labelled_rows

    path = tmp_path / 'yago_survival_rows.csv'
    pd.DataFrame([
        dict(fact_id='event', subject='s1', relation='r', object='o',
             begin_year=2000, landmark_year=2003, age_years=3,
             time_split='development', survival_label_1y=0),
        dict(fact_id='censored', subject='s2', relation='r', object='o',
             begin_year=2000, landmark_year=2003, age_years=3,
             time_split='development', survival_label_1y=1),
    ]).to_csv(path, index=False)
    pd.DataFrame([
        dict(fact_id='event', end_year=2004, boundary_censored=0),
        dict(fact_id='censored', end_year=None, boundary_censored=1),
    ]).to_csv(tmp_path / 'yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv',
              index=False)
    (tmp_path / 'yago_survival_build_summary.json').write_text(
        json.dumps({'observation_end': 2006}))
    rows, labels, metadata = load_yago_labelled_rows(path, [1], return_outcomes=True)
    assert 'end_year' not in rows
    np.testing.assert_array_equal(labels[:, 0], [0, 1])
    np.testing.assert_array_equal(metadata['stop_times'], [2004, 2006])
    np.testing.assert_array_equal(metadata['event_observed'], [True, False])
