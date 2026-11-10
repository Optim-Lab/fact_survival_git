"""Run the experiment of the paper in four steps. Every setting comes from the config file; nothing is decided in here.

    python main.py develop CONFIG --data PKL --out DIR [--device cuda] [--part 0 --parts 4]   1. train at the development origins, record the Brier at every candidate epoch
    python main.py select  CONFIG --out DIR                                                   2. pick ONE epoch (rule B) -> DIR/selection.json
    python main.py test    CONFIG --data PKL --out DIR [--device cuda] [--part 0 --parts 4]   3. train at the test origins with that epoch; predict; also the simple baselines
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
from pathlib import Path

import numpy as np
import pandas as pd

from fact_survival.baselines import baseline_predictions
from fact_survival.data import encode_ids, landmark_rows, read_facts
from fact_survival.features import make_features
from fact_survival.half_life import half_lives
from fact_survival.labels import known_at, survival_labels
from fact_survival.metrics import scores
from fact_survival.training import FactSurvival

MODEL_SETTINGS = ("hidden", "dropout", "learning_rate", "weight_decay", "batch_size")


#%% 1. Data: one row per (fact, landmark)
def prepare(config, path):
    """Raw file -> rows (one per fact and landmark), features x_c, half-life tau_c, labels y_{c,k} (NaN = unknown), dates info."""
    facts, info = read_facts(path, config["unit"])
    last = config["last_landmark"] or info["study_end"] - min(config["horizons"]) - 1
    
    rows = landmark_rows(facts, info["study_end"], last, config["first_landmark"], config["object_degree"])
    
    tau, level = half_lives(rows.subject, rows.relation, rows.landmark, facts.subject, facts.relation, facts.begin)
    
    x = make_features(rows.age, rows.object_degree, rows.object_relation_degree,
                      rows.subject_degree, rows.subject_relation_degree, tau, level)
    
    labels = survival_labels(rows.stop, rows.ended, rows.landmark, config["horizons"])
    
    labels = known_at(info["study_end"] - 1, labels, rows.landmark.to_numpy(), config["horizons"])   # study-end rule
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


#%% Step 1: development
def develop(config, args):
    rows, x, tau, labels, info = prepare(config, args.data)
    dev = development_origins(config, rows, info)
    origins = sorted({o for values in dev.values() for o in values})[args.part::args.parts]
    records = []
    for origin in origins:
        d = origin_data(config, rows, labels, origin)
        y = labels[d["test"]]
        for seed in config["seeds"]:
            by_epoch = neural_predictions(config, x, tau, d, seed, config["candidate_epochs"], args.device)
            for epoch, survival in by_epoch.items():
                for k, h in enumerate(config["horizons"]):
                    known = ~np.isnan(y[:, k])
                    if origin in dev[h] and known.any():
                        records.append(dict(seed=seed, origin=origin, horizon=h, epoch=epoch, rows=int(known.sum()),
                                            squared_error=float(((survival[known, k] - y[known, k]) ** 2).sum())))
        print(f"development origin {origin} done", flush=True)
    out = Path(args.out) / "develop"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / f"curves_part{args.part}.csv", index=False)


#%% Step 2:
def select(config, args):
    """Per seed: pool all development rows, Brier per horizon, mean over horizons.  Mean over seeds.  Lowest mean = the epoch."""
    
    parts = sorted((Path(args.out) / "develop").glob("curves_part*.csv"))
    
    table = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    
    pooled = table.groupby(["seed", "horizon", "epoch"])[["rows", "squared_error"]].sum()
    
    brier = (pooled.squared_error / pooled.rows).groupby(["seed", "epoch"]).mean()          # mean over horizons
    curve = brier.groupby("epoch").mean()                                                   # mean over seeds
    best = int(curve.idxmin())
    (Path(args.out) / "selection.json").write_text(json.dumps(
        dict(epoch=best, development_brier_x1000=float(curve[best] * 1000), curve_x1000={int(e): float(v * 1000) for e, v in curve.items()},
             seeds=config["seeds"], parts=[p.name for p in parts]), indent=2))
    print(curve.mul(1000).round(3).to_string(), f"\nselected epoch: {best}")


#%% Step 3: test
def test(config, args):
    epoch = json.loads((Path(args.out) / "selection.json").read_text())["epoch"]
    
    rows, x, tau, labels, info = prepare(config, args.data)
    
    horizons = config["horizons"]
    
    frames = []
    
    for origin in test_origins(rows, info)[args.part::args.parts]:
        d = origin_data(config, rows, labels, origin)
        
        test_rows, y = rows.iloc[d["test"]], labels[d["test"]]
        
        p = {f"neural_s{seed}": neural_predictions(config, x, tau, d, seed, [epoch], args.device)[epoch] for seed in config["seeds"]}
        
        p["halo_raw"] = np.clip(2.0 ** (-np.asarray(horizons)[None, :] / tau[d["test"]][:, None]), 1e-6, 1 - 1e-6)   # S^HL alone
        
        p.update(baseline_predictions(rows.iloc[d["train"]], d["usable"][d["train"]], test_rows, config["baseline_c"]))
        
        for k, h in enumerate(horizons):
            known = ~np.isnan(y[:, k])
            frame = pd.DataFrame(dict(origin=origin, horizon=h, subject=test_rows.subject.to_numpy()[known],
                                      relation=test_rows.relation.to_numpy()[known], object=test_rows.object.to_numpy()[known],
                                      begin=test_rows.begin.to_numpy()[known], survived=y[known, k]))
        
            for name, survival in p.items():
                frame[name] = survival[known, k]
            frames.append(frame)
        
        print(f"test origin {origin} done", flush=True)
    
    out = Path(args.out) / "test"
    
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / f"predictions_part{args.part}.csv.gz", index=False)


#%% Step 4: the tables
def report(config, args):
    
    parts = sorted((Path(args.out) / "test").glob("predictions_part*.csv.gz"))
    
    table = pd.concat([pd.read_csv(p, keep_default_na=False) for p in parts], ignore_index=True)
    
    models = [c for c in table.columns if c.startswith("neural_s")] + ["halo_raw", "global", "relation_age", "object_degree"]
    
    horizons = config["horizons"]
    
    result = []
    
    for model in models:
        for k, h in enumerate(horizons):
        
            part = table[table.horizon == h]
        
            wide = np.full((len(part), len(horizons)), np.nan)
        
            wide[:, k] = part[model]
        
            labels = np.full_like(wide, np.nan)
        
            labels[:, k] = part["survived"]
        
            s = scores(wide, labels, part["origin"].to_numpy(), horizons)[k]
        
            result.append(dict(model=model, horizon=h, rows=s["rows"], brier_x1000=s["brier"] * 1000, ap_x100=s["ap"] * 100))
    
    result = pd.DataFrame(result)
    
    result["group"] = result.model.where(~result.model.str.startswith("neural_s"), "neural")           # neural: mean +- sd over the seeds
    
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
    args = parser.parse_args()
    {"develop": develop, "select": select, "test": test, "report": report}[args.step](json.load(open(args.config)), args)
