"""Landmark-conditional neural survival with positive piecewise hazards.

Inputs contain only information available at the landmark. Category IDs are
fitted on training data; 0 denotes unseen entities/relations. The training loss
is horizon-wise binary cross entropy over mature, observed labels, not an
exact-event-time likelihood. Entity representations encode current graph
degrees by default; ``neural_id`` replaces these with shared learned ID vectors.
"""
#%%
from __future__ import annotations
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def piecewise_log_survival(log_hazard, widths):
    """Integrate positive piecewise rates at the shared horizon boundaries."""
    return -(log_hazard.clamp(-20.0, 12.0).exp() * widths).cumsum(dim=1)


class NeuralLandmarkHazard(nn.Module):
    """Shared representation, relation/bin intercepts, and an optional HALO gate."""

    def __init__(self, numeric_dim, cardinalities, horizons, variant="neural", hidden=32, fixed_neural_weight=None,
                 dropout=0.1):
        super().__init__()
        if variant not in {"neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_prior", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife", "neural_relation", "neural_object_degree", "neural_age", "neural_degree", "neural_gap", "cloglog", "half_life", "halo_residual"}:
            raise ValueError(f"Unknown variant: {variant}")

        self.variant = variant
             # 신경망 생존확률에 부여하는 고정 가중치입니다.
        # None이면 기존처럼 입력별 혼합 비율을 학습합니다.
        if fixed_neural_weight is not None:
            if variant not in {"neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_prior", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife"}:
                raise ValueError("고정 혼합은 neural 계열에서만 사용합니다.")
            if not 0.0 < fixed_neural_weight < 1.0:
                raise ValueError("고정 혼합 가중치는 0과 1 사이여야 합니다.")

        self.fixed_neural_weight = fixed_neural_weight
        self.register_buffer("horizons", torch.as_tensor(horizons, dtype=torch.float32))
        self.register_buffer("widths", torch.diff(self.horizons, prepend=torch.zeros(1)))
        bins = 1 if variant == "half_life" else len(horizons)

        self.baseline = nn.Parameter(torch.zeros(bins))

        self.relation_bias = nn.Embedding(cardinalities[1], bins, padding_idx=0)
        nn.init.zeros_(self.relation_bias.weight)

        self.linear = nn.Linear(numeric_dim, bins, bias=False)
        nn.init.zeros_(self.linear.weight)

        if variant != "cloglog":
            if numeric_dim != 12:
                raise ValueError("Neural encoder requires the documented 12 landmark features")
            self.relation_embedding = nn.Embedding(cardinalities[1], 8, padding_idx=0)
            nn.init.normal_(self.relation_embedding.weight, std=0.05)
            with torch.no_grad():
                self.relation_embedding.weight[0].zero_()

            if variant == "neural_id":
                if cardinalities[0] != cardinalities[2]:
                    raise ValueError("neural_id requires a shared subject/object vocabulary")
                self.entity_embedding = nn.Embedding(cardinalities[0], 8, padding_idx=0)
                nn.init.normal_(self.entity_embedding.weight, std=0.05)
                with torch.no_grad():
                    self.entity_embedding.weight[0].zero_()
            else:
                self.subject_encoder = nn.Linear(2, 8)
                self.object_encoder = nn.Linear(2, 8)

            encoder_layers = [
                nn.Linear(numeric_dim + 32, hidden), nn.SiLU(), nn.Dropout(dropout),
            ]
            if variant != "neural_one_hidden":
                encoder_layers.extend([
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                ])
            self.encoder = nn.Sequential(*encoder_layers)

            self.hazard_head = nn.Linear(hidden, bins, bias=False)
            nn.init.zeros_(self.hazard_head.weight)

            if variant in {"neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_prior", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife"}:
                self.gate = nn.Linear(hidden, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.constant_(self.gate.bias, 3.0)


    def forward(self, x, categories, tau, times=None):
        """Return log survival at native horizons, or supplied nonnegative times."""
        # 기본 구조와 HALO 혼합은 유지하고 지정한 입력만 제거합니다.
        if self.variant in {"neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife"}:
            x = x.clone()
            if self.variant == "neural_no_age":
                x[:, :5] = 0
            elif self.variant == "neural_no_degree":
                x[:, 5:9] = 0
            else:
                # "neural_no_tau_features": half-life input features removed, mixture with the half-life prior kept.
                # "neural_no_halflife": the same, and the mixture is also removed (see the mixture set below).
                x[:, 9:12] = 0

        # Omitted standardized inputs are set to their training means (zero).
        if self.variant in {"neural_relation", "neural_object_degree", "neural_age", "neural_degree"}:
            x = x.clone()
            if self.variant == "neural_relation":
                x.zero_()
            elif self.variant == "neural_object_degree":
                object_degree = x[:, 5].clone()
                x.zero_()
                x[:, 5] = object_degree
                categories = categories.clone()
                categories[:, 1] = 0
            else:
                x[:, 5 if self.variant == "neural_age" else 9:] = 0
        direct_x = x
        if self.variant == "neural_no_direct_degree":
            direct_x = x.clone()
            direct_x[:, 5:9] = 0

        linear_term = self.linear(direct_x)
        if self.variant == "neural_no_linear_term":
            # Check of the x / z overlap: drop the linear effect beta_k^T x; x still reaches the network through z.
            linear_term = torch.zeros_like(linear_term)
        log_hazard = (
            self.baseline
            + self.relation_bias(categories[:, 1])
            + linear_term
        )

        if self.variant != "cloglog":
            if self.variant == "neural_id":
                subject = self.entity_embedding(categories[:, 0])
                obj = self.entity_embedding(categories[:, 2])

            else:
                subject = F.silu(self.subject_encoder(x[:, [7, 8]]))
                obj = F.silu(self.object_encoder(x[:, [5, 6]]))
            # 숫자 degree는 유지하고 파생 표현만 정보 없는 상수로 만듭니다.
            if self.variant == "neural_no_projection":
                subject = torch.zeros_like(subject)
                obj = torch.zeros_like(obj)

            relation = self.relation_embedding(categories[:, 1])

            interaction = subject * obj
            if self.variant == "neural_no_interaction":
                interaction = torch.zeros_like(interaction)

            z = self.encoder(torch.cat(
                (direct_x, subject, relation, obj, interaction), dim=1
            ))

            if self.variant != "neural_linear_only":
                # Check of the x / z overlap: "linear only" drops the neural effect w_k^T z in the log-hazard
                # (z is still computed because the half-life gate is a function of z).
                log_hazard = log_hazard + self.hazard_head(z)

        if self.variant == "halo_residual":
            log_hazard = log_hazard + np.log(np.log(2.0)) - tau[:, None].log()
        hazard = log_hazard.clamp(-20.0, 12.0).exp().expand(-1, len(self.horizons))

        if times is None:
            times = self.horizons
            log_survival = piecewise_log_survival(log_hazard, self.widths)

        else:
            times = torch.as_tensor(times, dtype=x.dtype, device=x.device)
            if times.ndim != 1 or not torch.isfinite(times).all() or (times < 0).any():
                raise ValueError("Prediction times must be finite and nonnegative")
            left = self.horizons - self.widths
            exposure = (times[:, None] - left).clamp(min=0).minimum(self.widths)
            # The final hazard continues beyond the last training horizon.
            exposure[:, -1] += (times - self.horizons[-1]).clamp(min=0)
            log_survival = -(hazard @ exposure.T)

        # "neural_no_prior" (ablation): no half-life mixture, the survival curve is the neural one alone.
        if self.variant in {"neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features"}:
            log_halo = -np.log(2.0) * times[None, :] / tau[:, None]

            if self.fixed_neural_weight is None:
                gate = self.gate(z)
                log_neural_weight = F.logsigmoid(gate)
                log_prior_weight = F.logsigmoid(-gate)

            else:
                pi = self.fixed_neural_weight
                log_neural_weight = np.log(pi)
                log_prior_weight = np.log1p(-pi)

            log_survival = torch.logaddexp(
                log_neural_weight + log_survival,
                log_prior_weight + log_halo,
            )
        return log_survival


def masked_bce(log_survival, labels, mask):
    """Ignore unknown labels before any arithmetic, including unknown NaNs."""
    log_s = log_survival[mask].clamp(max=-1e-7)
    y = labels[mask]
    return -(y * log_s + (1.0 - y) * torch.log(-torch.expm1(log_s))).mean()


def discrete_survival_nll(log_survival, durations, events):
    """Monthly/yearly event mass or right-censor survival, one term per row.

    ``log_survival`` has columns for integer times 0..H. An event at t
    contributes -log(S(t-1)-S(t)); censoring at t contributes -log S(t).
    """
    if log_survival.ndim != 2 or log_survival.shape[0] != len(durations):
        raise ValueError("Survival curve and duration rows must align")

    if durations.ndim != 1 or events.shape != durations.shape:
        raise ValueError("Durations and event flags must be one-dimensional")

    if ((durations < 1) | (durations >= log_survival.shape[1])).any():
        raise ValueError("Durations must be between 1 and the last curve time")

    at_stop = log_survival.gather(1, durations[:, None]).squeeze(1)
    before = log_survival.gather(1, (durations - 1)[:, None]).squeeze(1)
    event_log_mass = before + torch.log(-torch.expm1((at_stop - before).clamp(max=-1e-7)))
    return -torch.where(events, event_log_mass, at_stop).mean()


def _inputs(x, categories, tau, mean, scale, cardinalities):
    x = np.asarray(x, dtype=np.float32)
    raw_categories = np.asarray(categories)
    tau = np.asarray(tau, dtype=np.float32)

    if x.ndim != 2 or x.shape[1] != len(mean) or raw_categories.shape != (len(x), 3):
        raise ValueError("Expected numeric (N, D), categories (N, 3)")

    if not np.issubdtype(raw_categories.dtype, np.integer) or np.any(raw_categories < 0):
        raise ValueError("Categories must be nonnegative integer training-vocabulary IDs")

    if tau.shape != (len(x),) or not np.isfinite(tau).all() or np.any(tau <= 0):
        raise ValueError("Half lives must be finite positive values of shape (N,)")

    if not np.isfinite(x).all():
        raise ValueError("Numeric features must be finite")

    categories = raw_categories.astype(np.int64, copy=True)

    for column, size in enumerate(cardinalities):
        categories[categories[:, column] >= size, column] = 0

    return (
        torch.from_numpy(((x - mean) / scale).astype(np.float32)),
        torch.from_numpy(categories), torch.from_numpy(tau),
    )


def _predict(model, inputs, batch_size):
    model.eval()
    with torch.inference_mode():
        parts = [
            model(*(value[start:start + batch_size] for value in inputs)).exp().numpy()
            for start in range(0, len(inputs[0]), batch_size)
        ]
    return np.concatenate(parts) if parts else np.empty((0, len(model.horizons)))


def predict_checkpoint(checkpoint, x_eval, cat_eval, tau_eval, *, epoch=None, batch_size=8192):
    """Apply a dev-selected checkpoint; never fit on evaluation inputs or labels."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    epoch = max(checkpoint["states"]) if epoch is None else epoch
    model = NeuralLandmarkHazard(**checkpoint["config"])
    model.load_state_dict(checkpoint["states"][epoch])
    inputs = _inputs(x_eval, cat_eval, tau_eval, checkpoint["mean"],
                     checkpoint["scale"], checkpoint["config"]["cardinalities"])
    return _predict(model, inputs, batch_size)


def fit_predict(
    x_train, cat_train, y_train, mask_train, tau_train,
    x_eval, cat_eval, tau_eval, horizons, *, variant="neural", epochs=(5, 15),
    seed=42, hidden=32, batch_size=8192, learning_rate=0.003, weight_decay=0.001, dropout=0.1,
    objective="bce", durations_train=None, events_train=None,
    # objective="bce",
    # durations_train=None,
    # events_train=None,
    fixed_neural_weight=None,
):
# ):
    """Fit once, returning {epoch: probabilities}, diagnostics, and checkpoints.

    Evaluation inputs should be dev rows. Select the epoch on dev, then call
    ``predict_checkpoint`` on test. The checkpoint is serializable with
    ``torch.save`` and includes every requested epoch and training-only scaler.
    Category vocabularies must also be retained by the caller for later data.
    """
    started = time.perf_counter()
    epochs = sorted(set(epochs))
    horizons = np.asarray(horizons, dtype=float)
    x_train, cat_train = np.asarray(x_train), np.asarray(cat_train)
    y_train, mask_train = np.asarray(y_train), np.asarray(mask_train, dtype=bool)
    if not epochs or any(not isinstance(epoch, (int, np.integer)) or epoch < 1 for epoch in epochs):
        raise ValueError("epochs must contain positive integers")
    if horizons.ndim != 1 or not len(horizons) or not np.isfinite(horizons).all() or np.any(np.diff(np.r_[0, horizons]) <= 0):
        raise ValueError("horizons must be finite, positive and strictly increasing")
    if x_train.ndim != 2 or len(x_train) == 0 or cat_train.shape != (len(x_train), 3):
        raise ValueError("Training numeric/categories require aligned nonempty arrays")
    if y_train.shape != (len(x_train), len(horizons)) or mask_train.shape != y_train.shape:
        raise ValueError("Labels and masks must have shape (N, number_of_horizons)")
    if not mask_train.any() or not np.isin(y_train[mask_train], [0, 1]).all():
        raise ValueError("At least one observed binary label is required")
    if objective not in {"bce", "survival_likelihood"}:
        raise ValueError("Unknown objective")
    if objective == "survival_likelihood":
        if not np.array_equal(horizons, horizons.astype(int)):
            raise ValueError("Discrete likelihood requires integer horizons")
        durations_train = np.asarray(durations_train)
        events_train = np.asarray(events_train)
        if (durations_train.shape != (len(x_train),) or
                events_train.shape != (len(x_train),) or
                not np.issubdtype(durations_train.dtype, np.integer) or
                np.any((durations_train < 1) | (durations_train > horizons[-1])) or
                not np.isin(events_train, [False, True]).all()):
            raise ValueError("Likelihood requires valid integer durations and event flags")
        duration_tensor = torch.as_tensor(durations_train, dtype=torch.long)
        event_tensor = torch.as_tensor(events_train, dtype=torch.bool)
        prediction_times = list(range(int(horizons[-1]) + 1))
    if batch_size <= 0 or hidden < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid training hyperparameters")
    torch.manual_seed(seed)
    cardinalities = (cat_train.max(axis=0) + 1).astype(int).tolist()
    if variant == "neural_id":
        cardinalities[0] = cardinalities[2] = max(cardinalities[0], cardinalities[2])
    mean, scale = x_train.mean(axis=0, dtype=np.float64), x_train.std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1.0
    train = _inputs(x_train, cat_train, tau_train, mean, scale, cardinalities)
    evaluation = _inputs(x_eval, cat_eval, tau_eval, mean, scale, cardinalities)
    labels = torch.as_tensor(y_train, dtype=torch.float32)
    mask = torch.as_tensor(mask_train, dtype=torch.bool)
    config = dict(numeric_dim=x_train.shape[1], cardinalities=cardinalities,
                  horizons=horizons.tolist(), variant=variant, hidden=hidden, dropout=dropout)
    if fixed_neural_weight is not None:
        config["fixed_neural_weight"] = float(fixed_neural_weight)

    model = NeuralLandmarkHazard(**config)
    observed_counts = mask_train.sum(axis=0)
    survival = np.divide(np.where(mask_train, y_train, 0).sum(axis=0),
                         observed_counts, out=np.full(len(horizons), 0.5),
                         where=observed_counts > 0)
    # Early origins may have no mature long-horizon labels: initialize those
    # bins from the first observable rate, not an arbitrary 50% survival.
    first_observed = np.flatnonzero(observed_counts)[0]

    base_rate = -np.log(np.clip(survival[first_observed], 1e-4, 1 - 1e-4)) / horizons[first_observed]

    survival[observed_counts == 0] = np.exp(-base_rate * horizons[observed_counts == 0])
    survival = np.minimum.accumulate(np.clip(survival, 1e-4, 1 - 1e-4))
    cumulative = -np.log(survival)
    initial_rates = np.maximum(np.diff(np.r_[0, cumulative]), 1e-4) / np.diff(np.r_[0, horizons])
    if variant == "half_life":
        initial_rates = np.array([cumulative[-1] / horizons[-1]])
    if variant != "halo_residual":
        with torch.no_grad():
            model.baseline.copy_(torch.as_tensor(np.log(initial_rates), dtype=torch.float32))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    predictions, states, history = {}, {}, []
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = torch.randperm(len(x_train))
        loss_sum, count_sum = 0.0, 0
        for indices in order.split(batch_size):
            batch_mask = mask[indices]
            count = int(batch_mask.sum()) if objective == "bce" else len(indices)
            if not count:
                continue

            optimizer.zero_grad(set_to_none=True)
            if objective == "bce":
                log_s = model(*(value[indices] for value in train))
                loss = masked_bce(log_s, labels[indices], batch_mask)

            else:
                log_s = model(*(value[indices] for value in train), times=prediction_times)
                loss = discrete_survival_nll(log_s, duration_tensor[indices], event_tensor[indices])

            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * count
            count_sum += count
        history.append({"epoch": epoch, "loss": loss_sum / count_sum,
                        "elapsed_seconds": time.perf_counter() - started})
        if epoch in epochs:
            predictions[epoch] = _predict(model, evaluation, batch_size)
            states[epoch] = {name: value.detach().clone() for name, value in model.state_dict().items()}

    diagnostics = dict(variant=variant, objective=objective, seed=seed, epochs=max(epochs), history=history,
                       last_loss=history[-1]["loss"], elapsed_seconds=time.perf_counter() - started,
                       parameters=sum(value.numel() for value in model.parameters()),
                       train_rows=len(x_train), observed_labels=int(mask.sum()),
                       device="cpu", threads=torch.get_num_threads(), hidden=hidden, embedding_dim=8,
                       learning_rate=learning_rate, weight_decay=weight_decay, batch_size=batch_size)

    if objective == "survival_likelihood":
        diagnostics.update(event_rows=int(events_train.sum()),
                           censored_rows=int(len(events_train) - events_train.sum()),
                           mean_followup=float(durations_train.mean()))
    checkpoint = dict(config=config, mean=mean, scale=scale, states=states, objective=objective)
    return predictions, diagnostics, checkpoint


if __name__ == "__main__":
    torch.set_num_threads(2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, 12)).astype(np.float32)
    cat = rng.integers(1, 6, size=(96, 3))
    tau = np.full(96, 3, dtype=np.float32)
    horizons = [1, 3, 6]
    duration = rng.exponential(3, size=96)
    y = (duration[:, None] > np.array(horizons)).astype(np.float32)
    mask = rng.random(y.shape) > 0.2
    y[~mask] = np.nan
    for variant in ("neural", "cloglog", "half_life", "halo_residual"):
        prediction, info, checkpoint = fit_predict(
            x, cat, y, mask, tau, x, cat, tau, horizons,
            variant=variant, epochs=[1, 2], batch_size=32,
        )
        assert np.isfinite(prediction[2]).all() and (np.diff(prediction[2], axis=1) <= 1e-7).all()
        assert np.allclose(prediction[2], predict_checkpoint(checkpoint, x, cat, tau, epoch=2))
        unseen = predict_checkpoint(checkpoint, x, np.full_like(cat, 999), tau)
        assert np.isfinite(unseen).all()
        model = NeuralLandmarkHazard(**checkpoint["config"])
        model.load_state_dict(checkpoint["states"][2])
        model.eval()

        inputs = _inputs(x, cat, tau, checkpoint["mean"], checkpoint["scale"], checkpoint["config"]["cardinalities"])
        if variant == "halo_residual":
            untouched = NeuralLandmarkHazard(**checkpoint["config"])
            expected = np.exp(-np.log(2.0) * np.asarray(horizons)[None, :] / tau[:, None])
            assert np.allclose(untouched(*inputs).exp().detach().numpy(), expected, atol=1e-6)
        log_s = model(*inputs, times=[0, 1, 2, 3, 6, 8])
        assert torch.allclose(log_s[:, 0], torch.zeros(96), atol=1e-6)
        assert (torch.diff(log_s, dim=1) <= 1e-6).all()
        loss = masked_bce(model(*inputs), torch.tensor(y), torch.tensor(mask))
        changed = np.where(mask, y, 123.0)
        assert torch.equal(loss, masked_bce(model(*inputs), torch.tensor(changed), torch.tensor(mask)))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert torch.equal(model.relation_bias.weight[0], torch.zeros_like(model.relation_bias.weight[0]))

        if variant != "cloglog":

            assert torch.count_nonzero(model.relation_embedding.weight[0]) == 0
            changed_ids = cat.copy()

            changed_ids[:, [0, 2]] = 0
            assert np.allclose(prediction[2], predict_checkpoint(checkpoint, x, changed_ids, tau))
    print("neural_hazard self-check passed (3 variants, masks, S(0), monotonicity, unseen IDs, gradients)")
