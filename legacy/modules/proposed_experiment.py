"""Run the proposed neural hazard on the presentation's fixed rolling benchmark.

Only the model changes. Stored baseline probabilities are joined by row keys,
never used as training features or labels. Invoke through main.py --model proposed.
"""

#%%
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.config import DATASET_CONFIGS, PROJECT_ROOT
from datasets.interval_landmark import load_landmark_rows, month_id, month_text
from datasets.task import (
    PROTOCOL_RULES, PROTOCOL_VERSION, make_survival_labels, mature_label_mask,
)
from datasets.yago import load_yago_labelled_rows
from evaluation.bootstrap import holm_adjust, multiway_bootstrap
from evaluation.metrics import summarize_probability
from modules.halo_half_life import HaloHalfLifeBaseline, unique_fact_history
from modules.neural_hazard import fit_predict


#%%
def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finwiki_keys(rows):
    return pd.util.hash_pandas_object(
        rows[["subject", "relation", "object", "begin", "end"]], index=False
    ).astype(str)


def load_data(dataset, path, horizons, *, return_outcomes=False):
    """Keep future endpoints in label construction only; features have an allowlist."""
    if dataset == "yago":
        raw, labels, label_meta = load_yago_labelled_rows(path, horizons, return_outcomes=return_outcomes)
        if return_outcomes:
            stops = label_meta["stop_times"]
            event_observed = label_meta["event_observed"]
        rows = raw.rename(columns={"landmark_year": "origin", "begin_year": "begin_time",
                                   "age_years": "age", "time_split": "split"}).copy()
        development_end = 2008
        origins = {h: list(range(2003, 2008 - h + 1)) for h in horizons}
        # Use the full current risk set, BEFORE any horizon observation mask.
        rows["object_active_degree"] = rows.groupby(["origin", "object"])["fact_id"].transform("size")
        rows["object_relation_degree"] = rows.groupby(["origin", "object", "relation"])["fact_id"].transform("size")
        rows["landmark"] = rows.origin.map(lambda v: f"{v:04d}-01")
    else:
        rows, meta = load_landmark_rows(path, min(horizons))
        rows["fact_id"] = finwiki_keys(rows)
        rows["origin"] = rows.landmark.map(month_id)
        rows["begin_time"] = rows.begin.map(month_id)
        rows["age"] = rows.age_months
        # Label meaning comes from the shared task contract. The benchmark's
        # stricter maturity policy (the full horizon must precede study end)
        # remains a separate observation mask.
        from datasets.interval_landmark import end_month_id
        ends = rows.end.map(end_month_id).to_numpy()
        stops = ends
        target = rows.origin.to_numpy()[:, None] + np.asarray(horizons)
        event_observed = ends < month_id(meta["study_end"])
        labels = np.column_stack([
            make_survival_labels(
                rows["begin_time"].to_numpy(),
                ends,
                rows["origin"].to_numpy(),
                horizon,
                event_observed=event_observed,
            )
            for horizon in horizons
        ])
        labels[target >= month_id(meta["study_end"])] = np.nan
        development_end = month_id(meta["valid_end"])
        first = int(rows.origin.min())
        step = DATASET_CONFIGS[dataset]["validation_step_months"]
        origins = {h: list(range(first + h, development_end - h + 1, step)) for h in horizons}

    rows["subject_degree"] = rows.groupby(["origin", "subject"])["fact_id"].transform("size")
    rows["subject_relation_degree"] = rows.groupby(["origin", "subject", "relation"])["fact_id"].transform("size")
    history = unique_fact_history(rows)
    halo = HaloHalfLifeBaseline(begin_col="begin_time", landmark_col="origin")
    details = halo.raw_predict(rows, history, horizons[0], return_details=True)
    tau = details.half_life.to_numpy(np.float32)
    age = rows.age.to_numpy(float)
    log_age = np.log1p(age)

    features = np.column_stack([
        log_age, age / (1 + age),
        *[np.maximum(log_age - np.log1p(k), 0) for k in (1, 3, 10)],
        *[np.log1p(rows[c].to_numpy(float)) for c in
          ("object_active_degree", "object_relation_degree", "subject_degree", "subject_relation_degree")],
        np.log1p(tau),
        (details.half_life_source == "subject_relation").to_numpy(float),
        (details.half_life_source == "relation").to_numpy(float),
    ]).astype(np.float32)
    # Global codes identify strings only; each fold fits its past-only vocabulary.

    entity_codes, _ = pd.factorize(pd.concat((rows.subject, rows.object), ignore_index=True), sort=True)

    categories = np.column_stack((entity_codes[:len(rows)],
                                  pd.factorize(rows.relation, sort=True)[0],
                                  entity_codes[len(rows):]))

    rows = rows[["fact_id", "subject", "relation", "object", "landmark", "origin", "split"]].copy()

    assert not rows.duplicated(["fact_id", "origin"]).any()

    observed = np.isfinite(labels)

    for k in range(len(horizons) - 1):
        both = observed[:, k] & observed[:, k + 1]
        assert (labels[both, k] >= labels[both, k + 1]).all()

    result = rows, features, categories, labels, tau, origins, development_end
    return (*result, stops.astype(np.int64), np.asarray(event_observed, dtype=bool)) if return_outcomes else result



def rolling_outcomes(stops, events, landmarks, origin, max_horizon):
    """Expose only follow-up observed by this origin, capped at the last horizon."""
    stops, events, landmarks = np.asarray(stops), np.asarray(events), np.asarray(landmarks)
    if (stops.shape != landmarks.shape or events.shape != landmarks.shape or
            np.any(stops <= landmarks) or np.any(landmarks >= origin) or max_horizon < 1):

        raise ValueError("Invalid rolling survival outcomes")

    durations = np.minimum.reduce((stops - landmarks, origin - landmarks,
                                   np.full(len(landmarks), max_horizon)))

    observed_events = np.asarray(events, bool) & (stops <= origin) & (stops - landmarks <= max_horizon)

    return durations.astype(np.int64), observed_events


def fold_categories(categories, train, evaluation):
    """Share entity IDs across roles, fitting vocabularies only on training rows."""

    encoded_train, encoded_eval, vocabularies = [], [], []

    entities = np.unique(categories[train][:, [0, 2]])

    for index, column in enumerate(categories.T):
        vocabulary = np.unique(column[train]) if index == 1 else entities
        mapping = np.zeros(max(column.max(), vocabulary.max()) + 1, dtype=np.int64)
        mapping[vocabulary] = np.arange(1, len(vocabulary) + 1)
        encoded_train.append(mapping[column[train]])
        encoded_eval.append(mapping[column[evaluation]])
        vocabularies.append(vocabulary)
    return np.column_stack(encoded_train), np.column_stack(encoded_eval), vocabularies


#%%
def paired_bootstrap(rows, model, reference, repetitions, seed):
    """Paired fact-cluster bootstrap of ROW Brier (same estimand as the table)."""
    difference = (rows[model] - rows.survived) ** 2 - (rows[reference] - rows.survived) ** 2
    groups = pd.DataFrame({"fact": rows.fact_id, "difference": difference}).groupby("fact").difference.agg(["sum", "size"])
    sums, sizes = groups["sum"].to_numpy(), groups["size"].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)

    for index in range(repetitions):
        sampled = rng.integers(len(groups), size=len(groups))
        estimates[index] = sums[sampled].sum() / sizes[sampled].sum()

    low, high = np.quantile(estimates, [.025, .975])

    point = float(difference.mean())

    p = (1 + np.sum(np.abs(estimates - point) >= abs(point))) / (repetitions + 1)
    return dict(row_brier_difference=float(difference.mean()), ci_low=float(low), ci_high=float(high),
                p_value=float(p), facts=len(groups), repetitions=repetitions)


def save_multiway(predictions, output, repetitions, seed, block_months=3):
    """Preserve FinWiki's fact-equal, entity/calendar multiway inference."""
    references = [c for c in ("p_halo_raw", "p_object_degree", "p_cloglog") if c in predictions]
    records = []

    for horizon, group in predictions.groupby("horizon"):
        draw_seed = seed + int(horizon)

        for reference in references:
            records.append(dict(
                horizon=int(horizon), model="neural", reference=reference.removeprefix("p_"),
                estimand="fact_equal_brier", repetitions=repetitions,
                seed=draw_seed, block_months=block_months,
                **multiway_bootstrap(group, "p_neural", reference, repetitions,
                                     block_months, draw_seed),
            ))

    table = pd.DataFrame(records)

    table["holm_adjusted_p"] = holm_adjust(table.p_value.tolist())

    table.to_csv(Path(output) / "test_significance_multiway.csv", index=False)

    return table


def attach_baselines(predictions, dataset, phase, horizons, origins=None):
    """Fail on any missing key, label mismatch, or change to evaluation population."""
    root = PROJECT_ROOT / "results" / PROTOCOL_VERSION
    baseline_root = root / (
        "yago_halo_rolling_test" if dataset == "yago"
        else f"{dataset}_rolling_test"
    )
    config_name = "bootstrap_run_config.json" if dataset == "yago" else "run_config.json"
    baseline_config = json.loads((baseline_root / config_name).read_text())
    if baseline_config.get("protocol_version") != PROTOCOL_VERSION or baseline_config.get(
        "protocol"
    ) != PROTOCOL_RULES:
        raise ValueError(f"{dataset} baseline must be rerun under {PROTOCOL_VERSION}")

    if dataset == "yago":
        name = "development_oof_predictions.csv" if phase == "development" else "test_predictions.csv"
        baseline = pd.read_csv(root / "yago_halo_rolling_test" / name, keep_default_na=False)
        baseline = baseline.rename(columns={"landmark_year": "origin", "label": "baseline_label",
                                             **{c: c.replace("prediction_", "p_") for c in baseline if c.startswith("prediction_")}})
        baseline = baseline[baseline.horizon.isin(horizons)]
        columns = [c for c in baseline if c.startswith("p_") and c != "p_halo_raw"]
        baseline = baseline[["fact_id", "origin", "horizon", "baseline_label", *columns]]
    else:
        frames = []
        for h in horizons:
            name = f"development_oof_h{h}.csv" if phase == "development" else f"test_predictions_h{h}.csv"
            table = pd.read_csv(root / f"{dataset}_rolling_test" / name, keep_default_na=False)
            table["fact_id"] = finwiki_keys(table)
            table["origin"] = table.landmark.map(month_id)
            table["horizon"] = h

            table = table.rename(columns={"survived": "baseline_label"})
            frames.append(table[["fact_id", "origin", "horizon", "baseline_label", "p_global", "p_relation_age", "p_object_degree"]])

        baseline = pd.concat(frames, ignore_index=True)

    if origins is not None:
        baseline = baseline[baseline.origin.isin(origins)]

    keys = ["fact_id", "origin", "horizon"]
    merged = predictions.merge(baseline, on=keys, validate="one_to_one", how="outer", indicator=True)
    if not merged._merge.eq("both").all() or not merged.survived.eq(merged.baseline_label).all():
        raise AssertionError("Proposed-model and presentation-baseline evaluation rows/labels differ")
    return merged.drop(columns=["baseline_label", "_merge"])


def metrics_table(predictions):
    return pd.DataFrame([
        {"horizon": int(h), **summarize_probability(group, column)}
        for h, group in predictions.groupby("horizon")
        for column in group if column.startswith("p_")
    ])


def save_comparisons(predictions, dataset, output, repetitions, seed):
    if "p_neural" not in predictions:
        return

    comparisons = []

    references = [c for c in ("p_halo_raw", "p_object_degree", "p_cloglog") if c in predictions]

    for h, group in predictions.groupby("horizon"):
        for reference in references:
            comparisons.append(dict(horizon=int(h), model="neural", reference=reference.removeprefix("p_"),
                                    **paired_bootstrap(group, "p_neural", reference, repetitions, seed + int(h))))
    table = pd.DataFrame(comparisons)

    table["holm_adjusted_p"] = holm_adjust(table.p_value.tolist())

    table.to_csv(output / "test_significance.csv", index=False)

    if dataset in ("finwiki", "icewswiki"):
        save_multiway(predictions, output, repetitions, seed)


def merge_parts(parts, output, repetitions):
    """Join computational partitions; baseline join rejects missing/extra test rows."""
    if output is None or (output / "run_config.json").exists():
        raise ValueError("Merging requires a fresh --output directory")

    configs = [json.loads((p / "run_config.json").read_text()) for p in parts]

    config = configs[0].copy()

    phase = "development" if config.get("partial_development") else "test"

    for current in configs:
        if (phase == "test" and current["test_evaluation"] != "rolling_origin_partial") or (phase == "development" and not current.get("partial_development")):
            raise ValueError("Each input must be a completed partition of the same phase")
        for key in ("dataset", "data_sha256", "input_sha256", "architecture", "protocol_version", "protocol", "seed", "horizons", "variants", "hidden", "batch_size", "candidate_epochs", "selection", "source_hashes", "objective"):
            if current[key] != config[key]:
                raise ValueError(f"Partition configurations differ: {key}")

    predictions = pd.concat([pd.read_csv(p / f"{phase}_predictions.csv.gz", keep_default_na=False,
                                        dtype={"fact_id": str})
                             for p in parts], ignore_index=True)

    if phase == "development":
        output.mkdir(parents=True, exist_ok=True)
        predictions, config["selection"] = select_epochs(predictions, config["variants"], output)

    learned = {"p_" + variant for variant in config["variants"]} | {"p_halo_raw"}

    predictions = predictions.drop(columns=[c for c in predictions if c.startswith("p_") and c not in learned])

    predictions = attach_baselines(predictions, config["dataset"], phase, config["horizons"])

    output.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output / f"{phase}_predictions.csv.gz", index=False, compression="gzip")

    metrics_table(predictions).to_csv(output / f"{phase}_metrics.csv", index=False)

    if phase == "test":
        save_comparisons(predictions, config["dataset"], output, repetitions, config["seed"])

    config["folds"] = sorted([f for c in configs for f in c["folds"]], key=lambda f: f["origin"])

    config.update(partial_test=False, requested_test_origins=None,
                  partial_development=False, requested_development_origins=None,
                  test_evaluation="rolling_origin" if phase == "test" else "not_run",
                  merged_parts=[str(p.resolve()) for p in parts], merge_code_sha256=file_hash(__file__),
                  elapsed_seconds=sum(c["elapsed_seconds"] for c in configs))

    last_part = max(zip(parts, configs), key=lambda pc: max(f["origin"] for f in pc[1]["folds"]))[0]

    for variant in config["variants"]:
        shutil.copy2(last_part / f"{phase}_{variant}.pt", output / f"{phase}_{variant}.pt")
    (output / "selection.json").write_text(json.dumps(config["selection"], indent=2))

    (output / "run_config.json").write_text(json.dumps(config, indent=2))
    print(f"Merged and verified complete {phase} population: {output}", flush=True)


def select_epochs(predictions, variants, output):
    """Select once on the complete development population, never per partition."""
    candidates = metrics_table(predictions)
    candidates.to_csv(output / "development_candidates.csv", index=False)
    selection = {}
    for variant in variants:
        options = candidates[candidates.model.str.startswith(variant + "_e")]
        scores = options.groupby("model").row_brier.mean()

        chosen = scores.idxmin()

        selection[variant] = dict(epoch=int(chosen.rsplit("_e", 1)[1]),
                                  mean_horizon_brier=float(scores[chosen]))

        predictions[f"p_{variant}"] = predictions[f"p_{chosen}"]

    return predictions.drop(columns=[c for c in predictions if any(
        c.startswith(f"p_{v}_e") for v in variants)]), selection


#%%
def run(args):
    if args.merge_parts:
        merge_parts(args.merge_parts, args.output, args.bootstrap_repetitions or 1000)
        return
    objective = getattr(args, "objective", "bce")

    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if min(args.epochs) < 1 or args.threads < 1 or args.batch_size < 1:

        raise ValueError("epochs, threads and batch size must be positive")

    if args.test_origins and not (args.selection_from and args.evaluate_test):
        raise ValueError("Partial origins require a frozen development selection and --evaluate-test")

    if args.development_origins and (args.evaluate_test or args.selection_from or args.test_origins):
        raise ValueError("Development partitions must be merged before test evaluation")

    config = DATASET_CONFIGS[args.dataset]

    horizons = args.horizons or config["horizons"]

    if horizons != config["horizons"]:
        raise ValueError("Use the presentation's complete ordered horizon set for the joint survival curve")

    source = args.data or config["data_path"]
    data_inputs = [Path(source)]

    if args.dataset == "yago":
        data_inputs.extend([
            Path(source).with_name("yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv"),
            Path(source).with_name("yago_survival_build_summary.json"),
        ])

    output = args.output or PROJECT_ROOT / "results" / PROTOCOL_VERSION / f"proposed_{args.dataset}_seed{args.seed}"

    output.mkdir(parents=True, exist_ok=True)

    if (output / "selection.json").exists():
        raise ValueError(f"Existing run in {output}; choose a new --output to preserve selection provenance")

    torch.set_num_threads(args.threads)
    started = time.time()

    print(f"Loading {args.dataset}; horizons={horizons}; CPU threads={args.threads}", flush=True)
    rows, x, categories, labels, tau, dev_origins, dev_end, stops, events = load_data(
        args.dataset, source, horizons, return_outcomes=True)

    origin_values = rows.origin.to_numpy()
    targets = origin_values[:, None] + np.asarray(horizons)
    observed = np.isfinite(labels)

    fixed_weight = getattr(args, "fixed_neural_weight", None)

    if fixed_weight is not None:
        if not 0.0 < fixed_weight < 1.0:
            raise ValueError("고정 신경망 가중치는 0과 1 사이여야 합니다.")
        if not set(args.variants).issubset({"neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_prior", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife"}):
            raise ValueError(
                "고정 혼합 실험에서는 neural 또는 neural_id만 지정하세요."
            )

    parameters = dict(
        seed=args.seed,
        hidden=args.hidden,
        batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.weight_decay, dropout=args.dropout,
    )
    if fixed_weight is not None:
        parameters["fixed_neural_weight"] = fixed_weight

    manifest = dict(dataset=args.dataset, architecture="structural_neural_hazard_v2", source=str(source), data_sha256=file_hash(source),
                    input_sha256={str(path): file_hash(path) for path in data_inputs},
                    protocol_version=PROTOCOL_VERSION, protocol=PROTOCOL_RULES,
                    requested_test_origins=args.test_origins, partial_test=bool(args.test_origins),
                    requested_development_origins=args.development_origins,
                    partial_development=bool(args.development_origins),
                    horizons=horizons, variants=args.variants, candidate_epochs=args.epochs,
                    category_encoding="shared_entity_ids_training_vocabulary_v1",
                    entity_representation={v: "shared entity ID embedding" if v == "neural_id" else
                                           "degree projection" for v in args.variants if v != "cloglog"},
                    seed=args.seed, hidden=args.hidden, batch_size=args.batch_size, threads=args.threads,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    learning_rate=args.lr, weight_decay=args.weight_decay, dropout=args.dropout, optimizer="AdamW", objective=objective,
                    loss=("unweighted observed-horizon BCE" if objective == "bce" else
                          "discrete event/censor negative log likelihood"),

                    selection="minimum mean horizon development row Brier, independently per variant",

                    test_used_for_selection=False, test_evaluation="not_run", folds=[],

                    feature_names=["log_age", "bounded_age", "age_hinge_1", "age_hinge_3", "age_hinge_10",
                                   "log_object_degree", "log_object_relation_degree", "log_subject_degree",
                                   "log_subject_relation_degree", "log_tau", "tau_subject_relation", "tau_relation"],

                    source_hashes={str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
                                   [PROJECT_ROOT / "main.py", Path(__file__),
                                    PROJECT_ROOT / "modules/neural_hazard.py",
                                    PROJECT_ROOT / "modules/halo_half_life.py",
                                    PROJECT_ROOT / "datasets/task.py",
                                    PROJECT_ROOT / "datasets/interval_landmark.py",
                                    PROJECT_ROOT / "datasets/yago.py"]})
    manifest["fixed_neural_weight"] = fixed_weight
    (output / "run_config.json").write_text(json.dumps(manifest, indent=2))

    selection = {}

    if args.selection_from:
        if not args.evaluate_test:
            raise ValueError("--selection-from requires --evaluate-test")

        selected_config = json.loads((args.selection_from / "run_config.json").read_text())
        if selected_config.get("fixed_neural_weight") != fixed_weight:
            raise ValueError(
                "개발에서 사용한 혼합 가중치와 시험 실행의 가중치가 다릅니다."
            )
        for key in ("dataset", "architecture", "protocol_version", "protocol", "data_sha256", "input_sha256", "horizons", "hidden", "batch_size", "objective"):
            if selected_config[key] != manifest[key]:
                raise ValueError(f"Frozen development configuration differs: {key}")

        if not set(args.variants).issubset(selected_config["variants"]):
            raise ValueError("Requested model was not evaluated in development")

        if selected_config["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("Protocol or model code differs from the frozen development run")

        selection = json.loads((args.selection_from / "selection.json").read_text())
        selection = {variant: selection[variant] for variant in args.variants}

        manifest["selection_from"] = str(args.selection_from.resolve())
        manifest["selection_sha256"] = file_hash(args.selection_from / "selection.json")
        (output / "selection.json").write_text(json.dumps(selection, indent=2))

    phases = ["test"] if args.selection_from else (["development", "test"] if args.evaluate_test else ["development"])

    for phase in phases:
        origins = sorted(set(v for values in dev_origins.values() for v in values)) if phase == "development" else sorted(rows.loc[rows.split.eq("test"), "origin"].unique())
        requested_origins = args.development_origins if phase == "development" else args.test_origins
        if requested_origins:
            if not set(requested_origins).issubset(origins):
                raise ValueError(f"Requested origin is not in the original {phase} period")
            origins = sorted(set(requested_origins))

        frames = []

        for origin in origins:
            mask = mature_label_mask(labels, origin_values, horizons, origin)
            train = np.flatnonzero(mask.any(axis=1))
            evaluation = np.flatnonzero(origin_values == origin)
            train_cats, eval_cats, vocabularies = fold_categories(categories, train, evaluation)
            # Endpoint information is clipped to this rolling origin before fitting.
            # The same training landmark rows are used for both objectives.
            durations, event_at_origin = rolling_outcomes(
                stops[train], events[train], origin_values[train], origin, max(horizons))
            if np.any(targets[train][mask[train]] > origin):
                raise AssertionError("Immature labels entered training")
            record = dict(phase=phase, origin=int(origin), train_rows=len(train),
                          mature_labels=mask.sum(axis=0).tolist(), latest_mature_target=int(targets[mask].max()),
                          evaluation_risk_rows=len(evaluation), variants={})
            predictions = {}
            for variant in args.variants:
                epochs = sorted(set(args.epochs)) if phase == "development" else [selection[variant]["epoch"]]
                values, diagnostics, checkpoint = fit_predict(
                    x[train], train_cats, labels[train], mask[train], tau[train],
                    x[evaluation], eval_cats, tau[evaluation], horizons,
                    variant=variant, epochs=epochs, objective=objective,
                    durations_train=durations if objective == "survival_likelihood" else None,
                    events_train=event_at_origin if objective == "survival_likelihood" else None,
                    **parameters)
                for epoch, value in values.items():
                    name = f"p_{variant}_e{epoch}" if phase == "development" else f"p_{variant}"
                    if not np.isfinite(value).all() or (np.diff(value, axis=1) > 1e-6).any():
                        raise AssertionError("Invalid or nonmonotone survival curve")
                    predictions[name] = value
                record["variants"][variant] = diagnostics
                # Save the last origin model with its fold's vocabulary/scaler, never pickle raw data.
                if origin == origins[-1]:
                    checkpoint["fold_vocabularies"] = vocabularies
                    entity_strings = sorted(set(rows.subject) | set(rows.object))
                    checkpoint["category_strings"] = dict(subject=entity_strings,
                                                          relation=sorted(rows.relation.unique().tolist()),
                                                          object=entity_strings)
                    checkpoint["origin"] = int(origin)
                    torch.save(checkpoint, output / f"{phase}_{variant}.pt")
            for k, h in enumerate(horizons):
                if phase == "development" and origin not in dev_origins[h]:
                    continue
                keep = observed[evaluation, k]
                frame = rows.iloc[evaluation[keep]].copy()
                frame["horizon"] = h
                frame["survived"] = labels[evaluation[keep], k].astype(int)
                frame["p_halo_raw"] = np.clip(np.exp(-np.log(2.) * h / tau[evaluation[keep]]), 1e-6, 1 - 1e-6)
                for name, values in predictions.items():
                    frame[name] = values[keep, k]
                frames.append(frame)
            manifest["folds"].append(record)
            print(f"{phase} origin={origin}: train={len(train):,}, labels={record['mature_labels']}, elapsed={time.time()-started:.1f}s", flush=True)

        predictions = pd.concat(frames, ignore_index=True)
        if phase == "development" and not args.development_origins:
            predictions, selection = select_epochs(predictions, args.variants, output)
            # Persist the choice before looking at any new model test predictions.
            (output / "selection.json").write_text(json.dumps(selection, indent=2))
            print(f"FROZEN selection: {selection}", flush=True)
        if not requested_origins:
            predictions = attach_baselines(predictions, args.dataset, phase, horizons, args.test_origins)
        predictions.to_csv(output / f"{phase}_predictions.csv.gz", index=False, compression="gzip")
        metrics = metrics_table(predictions)
        metrics.to_csv(output / f"{phase}_metrics.csv", index=False)
        print(metrics.pivot(index="model", columns="horizon", values="row_brier").to_string(), flush=True)
        if phase == "test" and "neural" in args.variants and not args.test_origins:
            save_comparisons(predictions, args.dataset, output, args.bootstrap_repetitions or 1000, args.seed)
        manifest["test_evaluation"] = ("rolling_origin_partial" if args.test_origins else "rolling_origin") if phase == "test" else "not_run"
        manifest["elapsed_seconds"] = time.time() - started
        manifest["selection"] = selection
        (output / "run_config.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {output}", flush=True)
