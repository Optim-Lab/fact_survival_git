"""Run the experiment of the paper in four steps. Every setting comes from the config file; nothing is decided in here.

    python main.py develop CONFIG --data PKL --out DIR [--device cuda] [--part 0 --parts 4]   1. train at the development origins, record the Brier at every candidate epoch
    python main.py select  CONFIG --out DIR                                                   2. pick ONE epoch (rule B) -> DIR/selection.json
    python main.py test    CONFIG --data PKL --out DIR [--device cuda] [--part 0 --parts 4]   3. train at the test origins with that epoch; predict; also the simple baselines
    python main.py test    CONFIG --data PKL --out DIR --model deepsurv|logistic_hazard       3'. the same for a deep survival baseline (age features only; epoch from config)
    python main.py develop|select|test CONFIG ... --model regcn|logcl                         1-3 for a graph-encoder row (settings, seeds, code folders: config "encoders")
    python main.py report  CONFIG --out DIR                                                   4. Brier (x10^3) and AP (x100) per model and horizon -> DIR/metrics.csv

Rolling origin o ("now"): the model is refitted from scratch at every origin on the cells whose label is known at o (c + h_k <= o),
then predicts the facts that are active at o (landmark c = o).
    development origins  o <= valid_end   (only used to choose the epoch)
    test origins         o  > valid_end
`--part i --parts n` runs every n-th origin only, so that the long steps can be split over several jobs (the parts are joined by select / report).
"""
#%%
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fact_survival.baselines import baseline_predictions
from fact_survival.classical import follow_up
from fact_survival.data import encode_ids, landmark_rows, read_facts
from fact_survival.deep_baselines import deep_predictions
from fact_survival.features import make_features
from fact_survival.graph_encoders import fit_predict, use_authors_code
from fact_survival.half_life import half_lives
from fact_survival.labels import known_at, survival_labels
from fact_survival.metrics import scores
from fact_survival.training import FactSurvival

MODEL_SETTINGS = ("hidden", "dropout", "learning_rate", "weight_decay", "batch_size")
AGE_COLUMNS = [0, 1, 2, 3, 4]        # columns of x_c that the deep baselines see in the paper's tables: the age features only (no degree, no half-life)
ENCODER_SETTINGS = ("hidden", "history_length", "batch_size", "learning_rate", "weight_decay", "contrastive_weight", "contrastive_batch")
ROOT = Path(__file__).resolve().parent
FIXED_MODELS = ["halo_raw", "global", "relation_age", "object_degree"]                  # the models without a seed
SEEDED = re.compile(r"^(neural|deepsurv|logistic_hazard|regcn|logcl)_s\d+$")                       # a column of one model trained with one seed


#%% 1. Data: one row per (fact, landmark)
def prepare(config, path):
    """Raw file -> rows (one per fact and landmark), features x_c, half-life tau_c, labels y_{c,k} (NaN = unknown), dates info."""
    facts, info = read_facts(path, config["unit"])
    last = config["last_landmark"] or info["study_end"] - min(config["horizons"]) - 1
    
    rows = landmark_rows(facts, info["study_end"], last, config["first_landmark"], config["object_degree"])
    
    # The half-life history is the set of facts that appear in the landmark rows (as in the paper's code). For YAGO the landmarks
    # start in 1992, so a fact that ended before 1992 is not part of the history. FinWiki is not affected (landmarks start at the first begin).
    history = rows[["subject", "relation", "object", "begin"]].drop_duplicates()
    tau, level = half_lives(rows.subject, rows.relation, rows.landmark, history.subject, history.relation, history.begin)
    
    x = make_features(rows.age, rows.object_degree, rows.object_relation_degree,
                      rows.subject_degree, rows.subject_relation_degree, tau, level)
    
    labels = survival_labels(rows.stop, rows.ended, rows.landmark, config["horizons"])
    
    labels = known_at(info["study_end"] - config["study_end_margin"], labels, rows.landmark.to_numpy(), config["horizons"])   # study-end rule:
    # a label is known only if c + h_k <= study_end - margin  (month data: margin 1;  YAGO: margin 0, its last landmark is 2025 - 3 years)
    return rows, x, tau, labels, info


def development_origins(config, rows, info):
    """{h_k: origins used to choose the epoch for that horizon}: from first + h_k to valid_end - h_k, every `development_step`."""
    first = config["development_start"]
    return {h: list(range((first if first is not None else int(rows.landmark.min()) + h),
                          info["valid_end"] - h + 1, config["development_step"])) for h in config["horizons"]}


def test_origins(rows, info):
    return sorted(int(o) for o in rows["landmark"].unique() if o > info["valid_end"])


#%% 2. One origin: what the model may use, what it must predict
def origin_data(config, rows, labels, origin):
    landmark = rows["landmark"].to_numpy()
    usable = known_at(origin, labels, landmark, config["horizons"])          # only what a model standing at `origin` may use
    
    train = np.flatnonzero(~np.isnan(usable).all(axis=1))                    # rows with at least one known label
    
    test = np.flatnonzero(landmark == origin)                                # the facts active at the origin
    train_ids, test_ids = encode_ids(rows.iloc[train], rows.iloc[test])
    
    return dict(usable=usable, train=train, test=test, train_ids=train_ids, test_ids=test_ids)


def neural_predictions(config, x, tau, d, seed, epochs, device):
    """Fit the Full model once (for max(epochs) epochs) and return {epoch: survival (n_test, K)} for every epoch asked."""
    out = {}
    model = FactSurvival(config["horizons"], epochs=max(epochs), seed=seed, device=device,
                         **{k: config[k] for k in MODEL_SETTINGS})
    keep = lambda epoch, m: out.__setitem__(epoch, m.predict(x[d["test"]], d["test_ids"], tau[d["test"]])) if epoch in epochs else None
    model.fit(x[d["train"]], d["train_ids"], tau[d["train"]], d["usable"][d["train"]], after_epoch=keep)
    return out


def encoder_predictions(config, name, rows, x, labels, origin, seed, epochs, device):
    """RE-GCN / LogCL refitted at the origin. Only the rows up to the origin exist for it: the graph never holds a later fact."""
    settings = config["encoders"]
    use_authors_code(ROOT / settings["regcn_code"], ROOT / settings["logcl_code"])
    keep = rows["landmark"].to_numpy() <= origin
    part = rows[keep].reset_index(drop=True)
    usable = known_at(origin, labels[keep], part["landmark"].to_numpy(), config["horizons"])
    return fit_predict(name, part, x[keep], usable, config["horizons"], origin, epochs=epochs, seed=seed, device=device,
                       **{k: settings[k] for k in ENCODER_SETTINGS})          # {epoch: survival} of the rows at the origin, in row order


def seeds_and_epochs(config, model):
    if model == "neural":
        return config["seeds"], config["candidate_epochs"]
    return config["encoders"]["seeds"][model], config["encoders"]["candidate_epochs"]


def predictions_at(config, args, rows, x, tau, labels, d, origin, seed, epochs):
    if args.model == "neural":
        return neural_predictions(config, x, tau, d, seed, epochs, args.device)
    return encoder_predictions(config, args.model, rows, x, labels, origin, seed, epochs, args.device)


def tag(args):                                     # file names: curves_part0.csv (Full model), curves_regcn_part0.csv ...
    return "" if args.model == "neural" else f"{args.model}_"


def selection_file(args):                          # selection.json (Full model), selection_regcn.json ...
    return Path(args.out) / ("selection.json" if args.model == "neural" else f"selection_{args.model}.json")


#%% Step 1: development
def develop(config, args):
    rows, x, tau, labels, info = prepare(config, args.data)
    dev = development_origins(config, rows, info)
    origins = sorted({o for values in dev.values() for o in values})[args.part::args.parts]
    seeds, epochs = seeds_and_epochs(config, args.model)
    records = []
    for origin in origins:
        d = origin_data(config, rows, labels, origin)
        y = labels[d["test"]]
        for seed in seeds:
            by_epoch = predictions_at(config, args, rows, x, tau, labels, d, origin, seed, epochs)
            for epoch, survival in by_epoch.items():
                for k, h in enumerate(config["horizons"]):
                    known = ~np.isnan(y[:, k])
                    if origin in dev[h] and known.any():
                        records.append(dict(seed=seed, origin=origin, horizon=h, epoch=epoch, rows=int(known.sum()),
                                            squared_error=float(((survival[known, k] - y[known, k]) ** 2).sum())))
        print(f"development origin {origin} done", flush=True)
    out = Path(args.out) / "develop"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / f"curves_{tag(args)}part{args.part}.csv", index=False)


#%% Step 2:
def select(config, args):
    """Per seed: pool all development rows, Brier per horizon, mean over horizons.  Mean over seeds.  Lowest mean = the epoch."""
    
    parts = sorted((Path(args.out) / "develop").glob(f"curves_{tag(args)}part*.csv"))
    
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    
    pooled = table.groupby(["seed", "horizon", "epoch"])[["rows", "squared_error"]].sum()
    
    brier = (pooled.squared_error / pooled.rows).groupby(["seed", "epoch"]).mean()          # mean over horizons
    curve = brier.groupby("epoch").mean()                                                   # mean over seeds
    best = int(curve.idxmin())
    selection_file(args).write_text(json.dumps(
        dict(epoch=best, development_brier_x1000=float(curve[best] * 1000), curve_x1000={int(e): float(v * 1000) for e, v in curve.items()},
             seeds=sorted(int(s) for s in table.seed.unique()), parts=[p.name for p in parts]), indent=2))
    print(curve.mul(1000).round(3).to_string(), f"\nselected epoch: {best}")


#%% Step 3: test
def test(config, args):
    epoch = json.loads(selection_file(args).read_text())["epoch"]

    rows, x, tau, labels, info = prepare(config, args.data)
    
    horizons = config["horizons"]
    
    frames = []
    
    for origin in test_origins(rows, info)[args.part::args.parts]:
        d = origin_data(config, rows, labels, origin)
        
        test_rows, y = rows.iloc[d["test"]], labels[d["test"]]
        
        p = {f"{args.model}_s{seed}": predictions_at(config, args, rows, x, tau, labels, d, origin, seed, [epoch])[epoch]
             for seed in seeds_and_epochs(config, args.model)[0]}

        if args.model != "neural":                                           # graph-encoder rows: only the encoder columns
            frames += prediction_frames(origin, horizons, test_rows, y, p)
            print(f"test origin {origin} done", flush=True)
            continue

        p["halo_raw"] = np.clip(2.0 ** (-np.asarray(horizons)[None, :] / tau[d["test"]][:, None]), 1e-6, 1 - 1e-6)   # S^HL alone
        
        p.update(baseline_predictions(rows.iloc[d["train"]], d["usable"][d["train"]], test_rows, config["baseline_c"]))
        
        frames += prediction_frames(origin, horizons, test_rows, y, p)

        print(f"test origin {origin} done", flush=True)

    out = Path(args.out) / "test"

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / f"predictions_{tag(args)}part{args.part}.csv.gz", index=False)


def prediction_frames(origin, horizons, test_rows, y, predictions):
    """One table per horizon: the facts of the origin whose label is known, the label `survived`, and one column per model {name: (n_test, K)}."""
    frames = []
    for k, h in enumerate(horizons):
        known = ~np.isnan(y[:, k])
        frame = pd.DataFrame(dict(origin=origin, horizon=h, subject=test_rows.subject.to_numpy()[known],
                                  relation=test_rows.relation.to_numpy()[known], object=test_rows.object.to_numpy()[known],
                                  begin=test_rows.begin.to_numpy()[known], survived=y[known, k]))
        for name, survival in predictions.items():
            frame[name] = survival[known, k]
        frames.append(frame)
    return frames


#%% Step 3': test of a deep survival baseline
def test_deep(config, args):
    """DeepSurv / LogisticHazard at the test origins: same rows, same origins, but fitted on (duration, event) with the AGE features + relation only."""
    settings = config["deep_baselines"]
    epoch = settings["epochs"][args.model]                                  # chosen once on the validation period (given by the config)

    rows, x, tau, labels, info = prepare(config, args.data)

    horizons = config["horizons"]

    frames = []

    for origin in test_origins(rows, info)[args.part::args.parts]:
        d = origin_data(config, rows, labels, origin)

        test_rows, y = rows.iloc[d["test"]], labels[d["test"]]

        duration, event = follow_up(rows.iloc[d["train"]], origin, max(horizons))        # what was seen of every training fact at the origin

        p = {f"{args.model}_s{seed}": deep_predictions(args.model, x[d["train"]][:, AGE_COLUMNS], d["train_ids"], x[d["test"]][:, AGE_COLUMNS],
                                                      d["test_ids"], duration, event, horizons, [epoch], seed=seed, device=args.device,
                                                      **{k: settings[k] for k in MODEL_SETTINGS})[epoch] for seed in config["seeds"]}

        frames += prediction_frames(origin, horizons, test_rows, y, p)

        print(f"test origin {origin} done", flush=True)

    out = Path(args.out) / "test"

    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / f"predictions_{args.model}_part{args.part}.csv.gz", index=False)


#%% Step 4: the tables
def report(config, args):
    
    files = sorted((Path(args.out) / "test").glob("predictions*_part*.csv.gz"))      # predictions_part*.csv.gz (proposed) and predictions_<model>_part*.csv.gz

    horizons = config["horizons"]

    result = []

    for stem in sorted({f.name.split("_part")[0] for f in files}):                  # one prediction table per kind of model

        table = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files if f.name.startswith(stem + "_part")], ignore_index=True)

        for model in [c for c in table.columns if SEEDED.match(c) or c in FIXED_MODELS]:
            for k, h in enumerate(horizons):

                part = table[table.horizon == h]

                wide = np.full((len(part), len(horizons)), np.nan)

                wide[:, k] = part[model]

                labels = np.full_like(wide, np.nan)

                labels[:, k] = part["survived"]

                s = scores(wide, labels, part["origin"].to_numpy(), horizons)[k]

                result.append(dict(model=model, horizon=h, rows=s["rows"], brier_x1000=s["brier"] * 1000, ap_x100=s["ap"] * 100))

    result = pd.DataFrame(result)

    result["group"] = result.model.str.replace(r"_s\d+$", "", regex=True)           # a model trained with several seeds: mean +- sd over the seeds
    
    summary = result.groupby(["group", "horizon"])[["brier_x1000", "ap_x100"]].agg(["mean", "std"]).round(3)
    
    result.to_csv(Path(args.out) / "metrics_by_seed.csv", index=False)
    
    summary.to_csv(Path(args.out) / "metrics.csv")
    print(summary.to_string())


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("step", choices=["develop", "select", "test", "report"])
    parser.add_argument("config")
    parser.add_argument("--data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--model", choices=["neural", "deepsurv", "logistic_hazard", "regcn", "logcl"], default="neural",
                        help="deepsurv / logistic_hazard: step test only;  regcn / logcl: steps develop, select, test")
    args = parser.parse_args()
    steps = {"develop": develop, "select": select, "test": test_deep if args.model in ("deepsurv", "logistic_hazard") else test, "report": report}
    steps[args.step](json.load(open(args.config)), args)
