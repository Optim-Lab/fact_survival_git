"""DeepSurv and LogisticHazard (pycox): the two neural survival baselines of the paper's tables, refit at every rolling origin.

    survival = deep_predictions("deepsurv", x_train, ids_train, x_eval, ids_eval, duration, event, horizons, epochs=[3],
                                hidden=32, dropout=0.1, learning_rate=0.003, weight_decay=0.001, batch_size=8192, seed=42)
    survival[3]          (n_eval, K)   S-hat_c(h_k | c) after 3 epochs;  one entry per epoch asked for

    x_*        (N, p) float  the numeric inputs. In the paper's tables the baselines see the fact AGE ONLY: p = 5 (columns 0-4 of x_c), no degree, no half-life
    ids_*      (N, 3) int    (subject, relation, object) ids (data.encode_ids). Only the relation is used, as a one-hot vector
    duration   (N,)   int    d_c = min(T - c, o - c, h_K): how long the fact was seen valid at origin o, cut at the last horizon   (classical.follow_up)
    event      (N,)   bool   the fact really ended inside that time (otherwise it is censored)

What is different from the proposed model: it is fitted on (duration, event), not on the horizon labels y_{c,k}.
    deepsurv         Cox model, one output g(x_c);  S(h | c) = S0(h) ^ exp(g)    (baseline S0 estimated from the training rows)
    logistic_hazard  one logit per time step s = 1..h_K: the hazard of ending in step s;  S(h | c) = product of (1 - hazard_s) up to h
The network is a small MLP (2 hidden layers, SiLU, dropout) trained with AdamW. Every setting is chosen by the caller; nothing is decided in here.
Needs pycox 0.3.0 and torchtuples 0.2.2.
"""
#%%
from functools import partial

import numpy as np
import torch
import torchtuples as tt
from pycox.models import CoxPH, LogisticHazard
from pycox.models.loss import CoxPHLoss

CLIP = 1e-6                                          # survival probabilities are kept inside [CLIP, 1 - CLIP]


class SafeCoxPHLoss(CoxPHLoss):
    """pycox's Cox loss divides by the number of events in the mini-batch: a batch WITHOUT an event gives 0/0 = NaN and one NaN step
    destroys all weights. Such a batch has nothing to learn from, so its loss is 0 (batches with an event are unchanged)."""

    def forward(self, log_h, durations, events):
        if float(events.sum()) == 0.0:
            return log_h.sum() * 0.0
        return super().forward(log_h, durations, events)


#%%
def inputs(x_train, ids_train, x_eval, ids_eval):
    """Standardise the numbers with the TRAINING rows (float64 statistics), append the relation one-hot (code 0 = unseen = all zeros)."""
    mean = x_train.mean(axis=0, dtype=np.float64)
    scale = x_train.astype(np.float64).std(axis=0)
    scale[scale < 1e-6] = 1.0                                          # a constant column keeps its scale
    n_relations = int(ids_train[:, 1].max())

    def one_hot(codes):
        out = np.zeros((len(codes), n_relations), dtype=np.float32)
        seen = codes > 0
        out[np.flatnonzero(seen), codes[seen] - 1] = 1.0
        return out

    return tuple(np.hstack([(x - mean) / scale, one_hot(ids[:, 1])]).astype(np.float32)
                 for x, ids in ((x_train, ids_train), (x_eval, ids_eval)))


def make_model(name, in_features, n_steps, *, hidden, dropout, learning_rate, weight_decay, seed, device):
    """MLP (hidden x hidden) wrapped by the pycox model. n_steps = h_K = the number of time steps of the discrete model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_features = 1 if name == "deepsurv" else n_steps

    net = tt.practical.MLPVanilla(in_features, [hidden, hidden], out_features, batch_norm=False, dropout=dropout,
                                  activation=torch.nn.SiLU, output_bias=(name != "deepsurv"))
    optimizer = partial(torch.optim.AdamW, lr=learning_rate, weight_decay=weight_decay)
    if name == "deepsurv":
        return CoxPH(net, optimizer, device=device, loss=SafeCoxPHLoss())
    return LogisticHazard(net, optimizer, duration_index=np.arange(1, n_steps + 1, dtype="float32"), device=device)


def initialize_output_layer(net, name, duration, event, n_steps):
    """Start from the survival of the training rows: output weights 0 and, for logistic_hazard, the bias that gives the observed hazard of every step."""
    last = [m for m in net.modules() if isinstance(m, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight)
    
    if name == "deepsurv":                                             # Cox has no intercept: g(x) = 0 at the start
        return
    
    at_risk = np.array([(duration >= k).sum() for k in range(1, n_steps + 1)], dtype=float)
    ended = np.array([((duration == k) & event).sum() for k in range(1, n_steps + 1)], dtype=float)
    hazard = np.divide(ended, at_risk, out=np.zeros(n_steps), where=at_risk > 0)
    
    for k in range(1, n_steps):                                        # a step nobody reached reuses the previous rate
        if at_risk[k] == 0:
            hazard[k] = hazard[k - 1]
    hazard = np.clip(hazard, 1e-4, 1 - 1e-4)
    
    with torch.no_grad():
        last.bias.copy_(torch.as_tensor(np.log(hazard / (1 - hazard)), dtype=torch.float32))      # logit of the hazard


def target(name, duration, event):
    """deepsurv: the real durations;  logistic_hazard: the index of the time step (duration - 1)."""
    if name == "deepsurv":
        return duration.astype("float32"), event.astype("float32")
    return (duration - 1).astype("int64"), event.astype("float32")


def survival_at_horizons(model, name, x_eval, horizons, batch_size):
    """S-hat(h_k | c) for every row: the curve value at the last time point <= h_k."""
    if name == "deepsurv":
        model.compute_baseline_hazards()                               # Breslow estimate from the training data kept by fit()
    curve = model.predict_surv_df(x_eval, batch_size=batch_size)
    times, values = curve.index.to_numpy(dtype=float), curve.to_numpy()
    columns = []
    for h in horizons:
        position = np.searchsorted(times, h, side="right") - 1
        columns.append(np.ones(len(x_eval)) if position < 0 else values[position])
    return np.clip(np.column_stack(columns), CLIP, 1 - CLIP)


#%%
def deep_predictions(name, x_train, ids_train, x_eval, ids_eval, duration, event, horizons, epochs, *,
                     hidden, dropout, learning_rate, weight_decay, batch_size, seed, device=None):
    """Fit once and take a prediction after every epoch in `epochs` (training goes on between them): {epoch: (n_eval, K)}."""
    
    n_steps = int(max(horizons))

    a, b = inputs(x_train, ids_train, x_eval, ids_eval)
    model = make_model(name, a.shape[1], n_steps, hidden=hidden, dropout=dropout, learning_rate=learning_rate,
                       weight_decay=weight_decay, seed=seed, device=device)
    initialize_output_layer(model.net, name, duration, event, n_steps)
    
    y = target(name, duration, event)
    
    out, done = {}, 0
    
    for epoch in sorted(set(epochs)):
        model.fit(a, y, batch_size=batch_size, epochs=epoch - done, verbose=False)
        done = epoch

        values = survival_at_horizons(model, name, b, horizons, batch_size)

        if not np.isfinite(values).all():

            raise AssertionError(f"invalid survival curve (NaN/Inf) at epoch {epoch}")
        rise = float(np.max(np.diff(values, axis=1)))
        if rise > 1e-6:                                                # S must not rise; a rise of ~1e-6 is float32 rounding in pycox and is repaired
            if rise > 1e-4:
                raise AssertionError(f"non-monotone survival curve at epoch {epoch}: max rise {rise:.3g}")
            values = np.minimum.accumulate(values, axis=1)
        out[epoch] = values
    return out
