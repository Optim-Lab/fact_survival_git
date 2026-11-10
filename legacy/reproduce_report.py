"""Rebuild benchmark result tables from final row-level predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.task import PROTOCOL_RULES, PROTOCOL_VERSION
from evaluation.bootstrap import holm_adjust
from evaluation.metrics import summarize_probability
from modules.proposed_experiment import paired_bootstrap


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATASET_ORDER = ("yago", "finwiki", "icewswiki")
KEYS = ["fact_id", "origin", "horizon"]
MODEL_LABELS = {
    "halo_raw": "HALO-Surv raw",
    "halo": "HALO-Surv calibrated",
    "km": "KM",
    "global": "Global",
    "relation_age": "Relation-age",
    "object_degree": "Object-degree",
    "discrete_hazard": "Discrete hazard",
    "cloglog": "Linear cloglog",
    "half_life": "Neural constant hazard",
    "neural": "Neural piecewise mixture",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "configs/reproduction_runs.json",
        help="manifest listing the completed development and test runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / PROTOCOL_VERSION / "reproduced_report",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=RESULTS / "reference",
        help="directory containing reference CSVs; use an absent path to skip",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    return parser.parse_args()


def validate_predictions(rows: pd.DataFrame, run_name: str) -> None:
    required = {*KEYS, "survived"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"{run_name}: missing columns {missing}")
    if rows.duplicated(KEYS).any():
        raise ValueError(f"{run_name}: duplicate fact/origin/horizon keys")
    if not rows["survived"].isin([0, 1]).all():
        raise ValueError(f"{run_name}: labels must be zero or one")
    probabilities = rows.filter(regex=r"^p_")
    if probabilities.empty or not np.isfinite(probabilities.to_numpy()).all():
        raise ValueError(f"{run_name}: missing or non-finite probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any().any():
        raise ValueError(f"{run_name}: probability outside [0, 1]")


def assert_same_population(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> None:
    columns = [*KEYS, "survived"]
    if len(reference) != len(candidate):
        raise AssertionError(f"{name}: row count differs")
    if reference[columns].reset_index(drop=True).equals(candidate[columns].reset_index(drop=True)):
        return
    merged = reference[columns].merge(
        candidate[columns], on=KEYS, how="outer", validate="one_to_one",
        suffixes=("_reference", "_candidate"), indicator=True,
    )
    if not merged["_merge"].eq("both").all() or not merged["survived_reference"].eq(
        merged["survived_candidate"]
    ).all():
        raise AssertionError(f"{name}: keys or labels differ")


def assert_same_provenance(reference: dict, candidate: dict, name: str) -> None:
    for key in (
        "dataset", "architecture", "protocol_version", "protocol",
        "input_sha256", "source_hashes", "horizons",
    ):
        if candidate.get(key) != reference.get(key):
            raise AssertionError(f"{name}: provenance differs for {key}")


def assert_neural_monotone(rows: pd.DataFrame, name: str) -> None:
    if "p_neural" not in rows:
        return
    horizons = sorted(rows["horizon"].unique())
    for short, long in zip(horizons, horizons[1:]):
        left = rows.loc[rows.horizon.eq(short), ["fact_id", "origin", "p_neural"]]
        right = rows.loc[rows.horizon.eq(long), ["fact_id", "origin", "p_neural"]]
        paired = left.merge(right, on=["fact_id", "origin"], suffixes=("_short", "_long"))
        if (paired["p_neural_short"] + 1e-12 < paired["p_neural_long"]).any():
            raise AssertionError(f"{name}: neural survival is not monotone")


def read_run(run_name: str) -> tuple[dict, pd.DataFrame]:
    folder = RESULTS / run_name
    config = json.loads((folder / "run_config.json").read_text())
    if config.get("test_evaluation") != "rolling_origin" or config.get("partial_test"):
        raise ValueError(f"{run_name}: not a complete rolling-origin test run")
    if config.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"{run_name}: rerun required under {PROTOCOL_VERSION}")
    if config.get("protocol") != PROTOCOL_RULES:
        raise ValueError(f"{run_name}: protocol rules differ from executable contract")
    rows = pd.read_csv(
        folder / "test_predictions.csv.gz", keep_default_na=False,
        dtype={"fact_id": str},
        usecols=lambda column: column in {*KEYS, "survived"} or column.startswith("p_"),
    )
    validate_predictions(rows, run_name)
    assert_neural_monotone(rows, run_name)
    return config, rows


def metric_rows(rows: pd.DataFrame, dataset: str, seed: int) -> pd.DataFrame:
    records = [
        {"dataset": dataset, "seed": seed, "horizon": int(horizon),
         **summarize_probability(group, column)}
        for horizon, group in rows.groupby("horizon", sort=True)
        for column in group.columns if column.startswith("p_")
    ]
    return pd.DataFrame(records)


def add_prediction_column(target: pd.DataFrame, source: pd.DataFrame, column: str) -> None:
    assert_same_population(target, source, column)
    if target[KEYS].reset_index(drop=True).equals(source[KEYS].reset_index(drop=True)):
        target[column] = source[column].to_numpy()
        return
    values = target[KEYS].merge(source[[*KEYS, column]], on=KEYS, validate="one_to_one")
    target[column] = values[column].to_numpy()


def rebuild(summary: dict, repetitions: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_metrics = []
    significance = []

    for dataset in DATASET_ORDER:
        primary = None
        primary_config = None
        for run_name in summary["runs"][dataset]["test"]:
            config, rows = read_run(run_name)
            seed = int(config["seed"])
            if primary is None:
                primary = rows
                primary_config = config
            else:
                assert_same_provenance(primary_config, config, run_name)
                assert_same_population(primary, rows, run_name)
            all_metrics.append(metric_rows(rows, dataset, seed))

        if dataset == "yago":
            for run_name in summary["yago_comparators"]:
                config, rows = read_run(run_name)
                assert_same_provenance(primary_config, config, run_name)
                assert_same_population(primary, rows, run_name)
                columns = [column for column in rows if column in ("p_cloglog", "p_half_life")]
                if len(columns) != 1:
                    raise AssertionError(f"{run_name}: expected one comparator column")
                column = columns[0]
                add_prediction_column(primary, rows, column)
                metrics = metric_rows(rows, dataset, int(config["seed"]))
                all_metrics.append(metrics.loc[metrics.model.eq(column.removeprefix("p_"))])

        for horizon, group in primary.groupby("horizon", sort=True):
            for reference in ("p_halo_raw", "p_object_degree", "p_cloglog"):
                result = paired_bootstrap(
                    group, "p_neural", reference, repetitions, 42 + int(horizon)
                )
                significance.append({
                    "dataset": dataset, "horizon": int(horizon), "model": "neural",
                    "reference": reference.removeprefix("p_"), **result,
                })

    metrics = pd.concat(all_metrics, ignore_index=True)
    seed_summary = (
        metrics.loc[metrics.model.eq("neural")]
        .groupby(["dataset", "horizon"], sort=True).row_brier
        .agg(["mean", "std", "min", "max", "count"]).reset_index()
    )

    significance = pd.DataFrame(significance)
    for indexes in significance.groupby("dataset", sort=False).groups.values():
        significance.loc[indexes, "holm_adjusted_p"] = holm_adjust(
            significance.loc[indexes, "p_value"].tolist()
        )
    return metrics, seed_summary, significance


def verify_csv(actual: pd.DataFrame, expected_path: Path, sort_by: list[str]) -> None:
    if not expected_path.exists():
        return
    expected = pd.read_csv(expected_path)
    actual = actual[expected.columns]
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    # Stored predictions are float64 after CSV parsing; original metrics may
    # have been computed from float32 tensors before serialization.
    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-10, atol=5e-8)


def latex_table(headers: list[str], rows: list[list[str]], alignment: str) -> str:
    body = "\n".join(" & ".join(row) + r" \\" for row in rows)
    return (
        f"\\begin{{tabular}}{{{alignment}}}\n\\toprule\n"
        + " & ".join(headers) + r" \\" + "\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
    )


def write_tex(metrics: pd.DataFrame, seeds: pd.DataFrame,
              significance: pd.DataFrame, path: Path) -> None:
    pieces = ["% Generated by reproduce_report.py. Do not edit by hand.\n"]
    for dataset in DATASET_ORDER:
        table = metrics.loc[(metrics.dataset.eq(dataset)) & (metrics.seed.eq(42))]
        horizons = sorted(table.horizon.unique())
        pivot = table.pivot(index="model", columns="horizon", values="row_brier")
        order = [model for model in MODEL_LABELS if model in pivot.index]
        rows = [[MODEL_LABELS[model], *[f"{pivot.loc[model, h]:.6f}" for h in horizons]]
                for model in order]
        pieces.append(f"% {dataset}: seed 42 test row Brier\n")
        pieces.append(latex_table(["Model", *map(str, horizons)], rows, "l" + "r" * len(horizons)))

        sig = significance.loc[significance.dataset.eq(dataset)]
        rows = [[str(int(row.horizon)), MODEL_LABELS[row.reference],
                 f"{row.row_brier_difference:+.7f}",
                 f"[{row.ci_low:+.7f}, {row.ci_high:+.7f}]",
                 f"{row.holm_adjusted_p:.4f}"] for row in sig.itertuples()]
        pieces.append(f"% {dataset}: primary paired fact-cluster inference\n")
        pieces.append(latex_table(
            ["Horizon", "Reference", r"$\Delta$", "95\\% CI", "Holm $p$"],
            rows, "rlrrr",
        ))

    rows = [[row.dataset, str(int(row.horizon)),
             f"${row.mean:.6f}\\pm{row.std:.6f}$", str(int(row.count))]
            for row in seeds.itertuples()]
    pieces.append("% Neural seed summary\n")
    pieces.append(latex_table(["Dataset", "Horizon", "Mean $\\pm$ SD", "Seeds"], rows, "lrrr"))
    path.write_text("\n".join(pieces), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    summary = json.loads(args.summary.read_text())
    metrics, seeds, significance = rebuild(summary, args.bootstrap_repetitions)

    args.output.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(args.output / "test_metrics_all_runs.csv", index=False)
    seeds.to_csv(args.output / "neural_seed_summary.csv", index=False)
    significance.to_csv(args.output / "primary_significance.csv", index=False)
    write_tex(metrics, seeds, significance, args.output / "benchmark_tables.tex")

    if args.bootstrap_repetitions == 1000:
        verify_csv(metrics, args.verify / "test_metrics_all_runs.csv",
                   ["dataset", "seed", "horizon", "model"])
        verify_csv(seeds, args.verify / "neural_seed_summary.csv", ["dataset", "horizon"])
        verify_csv(significance, args.verify / "primary_significance.csv",
                   ["dataset", "horizon", "reference"])

    (args.output / "report_manifest.json").write_text(json.dumps({
        "summary": str(args.summary.resolve()),
        "protocol_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL_RULES,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "datasets": list(DATASET_ORDER),
        "verified_against": str(args.verify.resolve()),
        "checks": ["unique keys", "identical seed populations and labels",
                   "probabilities in [0,1]", "monotone neural survival",
                   "shared protocol version, rules, data hashes, and source hashes"],
    }, indent=2), encoding="utf-8")
    print(f"Reproduced and verified benchmark report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
