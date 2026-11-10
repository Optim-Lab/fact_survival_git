"""DeepSurv / LogisticHazard / DeepHit under the same rolling-origin protocol as the proposed model.

What is identical to the proposed model (modules/proposed_experiment.py):
  * rows, 12 numeric features, labels, masks and origins (load_data, mature_label_mask)
  * for every origin the model is refit on past-only rows (rolling_outcomes clips follow-up)
  * epoch candidates {5, 15}; the epoch is chosen once on the development period and frozen
  * AdamW, learning rate 0.003, weight decay 0.001, batch size 8192, hidden 32 x 2, SiLU, dropout 0.1
  * metrics come from the same metrics_table function

What differs (by necessity): the three models are fitted on (duration, event) pairs, not on
the horizon-wise BCE labels. Durations are measured in the dataset time unit (months; years for YAGO) and capped at the last horizon.
The relation is given as a one-hot vector (the proposed model uses a relation embedding).

Usage:
  python experiments/deep_survival_rolling.py --model deepsurv --phase development --seed 42 --output DIR
  python experiments/deep_survival_rolling.py --model deepsurv --phase test --seed 43 \
         --selection-from DIR_OF_DEVELOPMENT --output DIR_TEST
"""

#%%
import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# pycox is installed only in the lab folder's vendor directory; it is imported, never modified.
sys.path.insert(0, str(PROJECT_ROOT.parent / "fact_survival_lab_20261023/vendor/survival_deps"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchtuples as tt  # noqa: E402
from pycox.models import CoxPH, DeepHitSingle, LogisticHazard  # noqa: E402
from pycox.models.loss import CoxPHLoss  # noqa: E402

from datasets.config import DATASET_CONFIGS  # noqa: E402
from datasets.task import mature_label_mask  # noqa: E402
from modules.proposed_experiment import (  # noqa: E402
    fold_categories, load_data, metrics_table, rolling_outcomes, select_epochs,
)

#%%
HIDDEN, DROPOUT = 32, 0.1
LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE = 0.003, 0.001, 8192
CLIP = 1e-6

# The 12 numeric features come in three groups (same order as modules/proposed_experiment.load_data):
#   age     0-4  log_age, bounded_age, age_hinge_1/3/10                       (conventional survival covariates)
#   degree  5-8  object / object-relation / subject / subject-relation degree  (summaries of the ACTIVE GRAPH)
#   tau     9-11 log_tau, tau_subject_relation, tau_relation                   (half-life prior from the fact history)
# Default = all 12 (the earlier experiments).  "age" alone (+ the relation one-hot) is the graph-free baseline.
FEATURE_GROUPS = {"age": [0, 1, 2, 3, 4], "degree": [5, 6, 7, 8], "tau": [9, 10, 11]}
ACTIVE_COLUMNS = list(range(12))


def select_features(groups):
    """Choose which feature groups the baselines see, e.g. ["age"] or ["age", "degree"]."""
    global ACTIVE_COLUMNS
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s): {unknown}")
    ACTIVE_COLUMNS = sorted(c for g in groups for c in FEATURE_GROUPS[g])


#%%
class SafeCoxPHLoss(CoxPHLoss):
    """pycox's Cox loss divided by the number of events in the mini-batch: a batch WITHOUT any event gives 0/0 = NaN, one NaN
    gradient step destroys all weights and the survival curve becomes invalid (diagnosed 2026-11-02 at ICEWSWiki origin 24069:
    2,474,024 rows = 302 full batches + a last batch of only 40 rows, which has no event with probability ~14%).
    For such a batch the loss is set to 0 with zero gradient (nothing to learn from it).  A batch with at least one event is
    handled exactly as before, so every run that finished earlier is unchanged (a zero-event batch would have produced NaN)."""

    def forward(self, log_h, durations, events):
        if float(events.sum()) == 0.0:
            return log_h.sum() * 0.0
        return super().forward(log_h, durations, events)


def build_inputs(x, categories, train, evaluation):
    """Standardize the 12 numeric features and append a relation one-hot (fold-local)."""
    # float64 statistics: in float32 a column whose training values are all identical gets std ~1e-4 (rounding
    # noise), which slips past the guard below and turns evaluation rows into z-scores of thousands
    # (diagnosed 2026-11-01 at the first FinWiki valid origin; see results/diagnose_baseline_collapse_*).
    x_tr, x_ev = x[train][:, ACTIVE_COLUMNS], x[evaluation][:, ACTIVE_COLUMNS]     # chosen feature groups only
    mean = x_tr.mean(axis=0, dtype=np.float64)
    scale = x_tr.astype(np.float64).std(axis=0)
    scale[scale < 1e-6] = 1.0                         # avoid dividing by ~0 for constant columns
    train_cats, eval_cats, _ = fold_categories(categories, train, evaluation)
    n_relations = int(train_cats[:, 1].max())

    def one_hot(codes):
        out = np.zeros((len(codes), n_relations), dtype=np.float32)
        seen = codes > 0                              # code 0 = relation never seen in training
        out[np.flatnonzero(seen), codes[seen] - 1] = 1.0
        return out

    x_train = np.hstack([(x_tr - mean) / scale, one_hot(train_cats[:, 1])]).astype(np.float32)
    x_eval = np.hstack([(x_ev - mean) / scale, one_hot(eval_cats[:, 1])]).astype(np.float32)
    return x_train, x_eval


def make_model(name, in_features, n_bins, seed, device=None):
    """One small MLP (same size as the proposed model) wrapped by the pycox model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    # DeepHit forces all probability mass inside its time grid (S = 0 at the last grid point), so it
    # gets one extra bin that means "still active beyond the last horizon".
    out_features = 1 if name == "deepsurv" else (n_bins + 1 if name == "deephit" else n_bins)
    net = tt.practical.MLPVanilla(
        in_features, [HIDDEN, HIDDEN], out_features, batch_norm=False, dropout=DROPOUT,
        activation=torch.nn.SiLU, output_bias=(name != "deepsurv"))
    # Same optimizer class as the proposed model (torch AdamW); torchtuples calls it with params=...
    optimizer = partial(torch.optim.AdamW, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    grid = np.arange(1, out_features + 1, dtype="float32")   # time points 1..n_bins (+1 for DeepHit)
    if name == "deepsurv":
        return CoxPH(net, optimizer, device=device, loss=SafeCoxPHLoss())
    if name == "logistic_hazard":
        return LogisticHazard(net, optimizer, duration_index=grid, device=device)
    return DeepHitSingle(net, optimizer, duration_index=grid, device=device)


def initialize_output_layer(net, name, durations, events, n_bins):
    """Start every model from the training data's marginal survival, like the proposed model.

    The proposed model zero-initializes its output weights and sets the baseline log-hazard from
    the observed training survival rates. Without this the pycox models start from random
    outputs and are far off in the first epochs. Here the last Linear layer gets zero weights and
    (for the discrete models) a bias that reproduces the marginal curve of the training rows.
    """
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight)
    if name == "deepsurv":                              # Cox has no intercept; g(x) = 0 at start
        return
    at_risk = np.array([(durations >= k).sum() for k in range(1, n_bins + 1)], dtype=float)
    ended = np.array([((durations == k) & events).sum() for k in range(1, n_bins + 1)], dtype=float)
    hazard = np.divide(ended, at_risk, out=np.zeros(n_bins), where=at_risk > 0)
    for k in range(1, n_bins):                          # bins nobody reached yet reuse the last rate
        if at_risk[k] == 0:
            hazard[k] = hazard[k - 1]
    hazard = np.clip(hazard, 1e-4, 1 - 1e-4)
    if name == "logistic_hazard":                       # one logit per bin
        bias = np.log(hazard / (1 - hazard))
    else:                                               # deephit: softmax over the event-time pmf
        survival = np.cumprod(1 - hazard)
        pmf = np.r_[np.r_[1.0, survival[:-1]] * hazard, survival[-1]]   # last entry = beyond the horizon
        bias = np.log(np.clip(pmf, 1e-8, None))
    with torch.no_grad():
        last.bias.copy_(torch.as_tensor(bias, dtype=torch.float32))


def make_target(name, durations, events, n_bins):
    """DeepSurv uses real durations; the discrete models use the month index (duration - 1)."""
    if name == "deepsurv":
        return durations.astype("float32"), events.astype("float32")
    index, events = (durations - 1).astype("int64"), events.astype("float32")
    if name == "deephit":                                # still active at the cap -> "beyond horizon" bin
        beyond = (durations >= n_bins) & (events == 0)
        index[beyond], events[beyond] = n_bins, 1.0
    return index, events


def survival_at_horizons(model, name, x_eval, horizons):
    """Return S(h | landmark) for every row and horizon, shape (rows, len(horizons))."""
    if name == "deepsurv":
        model.compute_baseline_hazards()                  # needs the training data kept by fit()
    surv = model.predict_surv_df(x_eval, batch_size=BATCH_SIZE)
    times, values = surv.index.to_numpy(dtype=float), surv.to_numpy()
    columns = []
    for h in horizons:
        position = np.searchsorted(times, h, side="right") - 1    # last time point <= h
        columns.append(np.ones(len(x_eval)) if position < 0 else values[position])
    return np.clip(np.column_stack(columns), CLIP, 1 - CLIP)


def fit_origin(name, x_train, x_eval, durations, events, horizons, epochs, seed, device=None):
    """Train once, stopping at every candidate epoch to take a prediction."""
    model = make_model(name, x_train.shape[1], max(horizons), seed, device)
    initialize_output_layer(model.net, name, durations, events, max(horizons))
    target = make_target(name, durations, events, max(horizons))
    predictions, done = {}, 0
    for epoch in sorted(set(epochs)):
        model.fit(x_train, target, batch_size=BATCH_SIZE, epochs=epoch - done, verbose=False)
        done = epoch
        values = survival_at_horizons(model, name, x_eval, horizons)
        if not np.isfinite(values).all():
            raise AssertionError("Invalid survival curve (NaN/Inf) at epoch %d" % epoch)
        rise = float(np.max(np.diff(values, axis=1)))
        if rise > 1e-6:
            # S(h) is non-increasing by construction.  pycox computes the discrete curve in float32 (cumulative sum of log(1-h)),
            # and a rise of ~1e-6 is rounding noise (diagnosed 2026-11-02, FinSurvKG LogisticHazard: rise 1.31e-6 at epoch 100).
            # A rise this small is repaired; anything larger is a real problem and still stops the run.  A curve whose rise is
            # <= 1e-6 is left untouched, so every run that finished earlier is unchanged.
            if rise > 1e-4:
                raise AssertionError(f"Nonmonotone survival curve at epoch {epoch}: max rise {rise:.3g}")
            values = np.minimum.accumulate(values, axis=1)
        predictions[epoch] = values
    return predictions


#%%
def run(args):
    select_features(args.features.split(","))
    config = DATASET_CONFIGS[args.dataset]
    horizons = config["horizons"]
    output = Path(args.output)
    if (output / "run_config.json").exists():
        raise ValueError(f"{output} already holds a run; choose a new --output")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    started = time.time()

    rows, x, categories, labels, _tau, dev_origins, _dev_end, stops, events = load_data(
        args.dataset, config["data_path"], horizons, return_outcomes=True)
    origin_values = rows.origin.to_numpy()
    observed = np.isfinite(labels)

    if args.phase == "development":
        origins = sorted(set(v for values in dev_origins.values() for v in values))
        if args.max_origins:
            origins = origins[: args.max_origins]
        epochs_for = lambda: sorted(set(args.epochs))
    else:
        if args.max_origins:
            raise ValueError("--max-origins is only allowed in the development phase")
        selection = json.loads((Path(args.selection_from) / "selection.json").read_text())
        chosen = selection[args.model]["epoch"]
        epochs_for = lambda: [chosen]
        origins = sorted(rows.loc[rows.split.eq("test"), "origin"].unique())

    frames = []
    for origin in origins:
        mask = mature_label_mask(labels, origin_values, horizons, origin)
        train = np.flatnonzero(mask.any(axis=1))              # rows whose label is already known
        evaluation = np.flatnonzero(origin_values == origin)
        durations, event_at_origin = rolling_outcomes(
            stops[train], events[train], origin_values[train], origin, max(horizons))
        x_train, x_eval = build_inputs(x, categories, train, evaluation)
        predictions = fit_origin(args.model, x_train, x_eval, durations, event_at_origin,
                                 horizons, epochs_for(), args.seed, args.device)
        for k, h in enumerate(horizons):
            if args.phase == "development" and origin not in dev_origins[h]:
                continue
            keep = observed[evaluation, k]
            frame = rows.iloc[evaluation[keep]].copy()
            frame["horizon"] = h
            frame["survived"] = labels[evaluation[keep], k].astype(int)
            for epoch, values in predictions.items():
                name = f"p_{args.model}_e{epoch}" if args.phase == "development" else f"p_{args.model}"
                frame[name] = values[keep, k]
            frames.append(frame)
        print(f"{args.phase} origin={origin} train={len(train):,} elapsed={time.time() - started:.0f}s", flush=True)

    result = pd.concat(frames, ignore_index=True)
    if args.phase == "development":
        result, selection = select_epochs(result, [args.model], output)
        (output / "selection.json").write_text(json.dumps(selection, indent=2))
        print("FROZEN selection:", selection, flush=True)
    result.to_csv(output / f"{args.phase}_predictions.csv.gz", index=False, compression="gzip")
    metrics = metrics_table(result)
    metrics.to_csv(output / f"{args.phase}_metrics.csv", index=False)
    print(metrics[["model", "horizon", "row_brier"]].to_string(index=False), flush=True)
    (output / "run_config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()}, "horizons": horizons,
         "elapsed_seconds": time.time() - started}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", choices=["finwiki", "icewswiki", "yago", "finsurvkg"], default="finwiki")
    parser.add_argument("--model", choices=["deepsurv", "logistic_hazard", "deephit"], required=True)
    parser.add_argument("--phase", choices=["development", "test"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--features", default="age,degree,tau",
                        help="feature groups the model sees (age, degree, tau); 'age' = graph-free baseline")
    parser.add_argument("--selection-from", help="development output directory (test phase only)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default=None, help="e.g. cuda (default: cpu); useful for DeepHit")
    parser.add_argument("--max-origins", type=int, default=0, help="smoke test: first N development origins")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
