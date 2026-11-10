"""생존 실험 실행 입구: regcn/logcl/evokg는 원본 인코더에 생존 출력층을 연결합니다.

proposed → modules/proposed_experiment.py (수치 특징 MLP)
regcn → modules/regcn_experiment.py → 저자 RecurrentRGCN
logcl → modules/logcl_experiment.py → 저자 LogCL RecurrentRGCN
evokg → modules/evokg_original_experiment.py → 저자 EmbeddingUpdater
실행 위치는 fact_survival/이며, 모델을 생략하면 baseline입니다.
"""

#%%
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

import pandas as pd

from datasets.config import DATASET_CONFIGS
from datasets.interval_landmark import (
    load_landmark_rows,
    month_id,
    month_text,
    rolling_validation_months,
    select_rolling_fold,
)
from datasets.task import PROTOCOL_RULES, PROTOCOL_VERSION
from datasets.yago import load_yago_landmark_rows
from evaluation.bootstrap import holm_adjust, multiway_bootstrap
from evaluation.metrics import summarize_probability
from modules.model import fit_and_predict
from tqdm.auto import tqdm

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


#%%
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_CONFIGS, default="finwiki")
    parser.add_argument("--model", choices=["baseline", "proposed", "regcn", "logcl", "evokg", "evokg_lifetime", "table"], default="baseline")
    parser.add_argument("--epochs", type=int, nargs="+", default=[5, 15])
    parser.add_argument("--variants", nargs="+", choices=["neural", "neural_one_hidden", "neural_id", "neural_no_interaction", "neural_no_prior", "neural_no_linear_term", "neural_linear_only", "neural_no_direct_degree", "neural_no_projection", "neural_no_age", "neural_no_degree", "neural_no_tau_features", "neural_no_halflife", "neural_relation", "neural_object_degree", "neural_age", "neural_degree", "neural_gap", "cloglog", "half_life", "halo_residual"],
                        default=["neural", "cloglog", "half_life"])
    parser.add_argument("--objective", choices=["bce", "survival_likelihood"], default="bce",
                        help="proposed-model training objective; evaluation remains the same Brier score"
                        )
    parser.add_argument(
        "--fixed-neural-weight",
        type=float,
        default=None,
        help="신경망 혼합 가중치. 생략하면 기존처럼 혼합 비율을 학습합니다.",
    )
    parser.add_argument("--mix-components", type=int, default=3,
                        help="EvoKG lifetime log-normal mixture components")
    parser.add_argument("--regcn-no-messages", action="store_true",
                        help="RE-GCN control: retain time steps and numeric inputs but remove historical graph messages")
    parser.add_argument("--event-context", choices=["none", "vocab", "graph"], default="none",
                        help="RE-GCN past point events: none, matched vocabulary only, or vocabulary plus graph messages")
    parser.add_argument("--history-length", type=int, default=3,
                        help="RE-GCN/LogCL/EvoKG local history steps including the current landmark")
    parser.add_argument("--max-global-edges", type=int, default=None,
                        help="이전 LogCL 전용 옵션; 공식 인코더에서는 지정하지 않습니다")
    parser.add_argument("--contrastive-weight", type=float, default=0.1,
                        help="LogCL local/global InfoNCE weight")
    parser.add_argument("--contrastive-batch", type=int, default=128,
                        help="maximum LogCL in-batch contrastive queries")
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="completed run directories for --model table")
    parser.add_argument("--table-phase", choices=["development", "test"], default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=None,
                        help="EvoKG 기본 100, 나머지 모델 기본 32")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=0.003, help="proposed: AdamW learning rate (default = the old fixed value)")
    parser.add_argument("--weight-decay", type=float, default=0.001, help="proposed: AdamW weight decay (default = the old fixed value)")
    parser.add_argument("--dropout", type=float, default=0.1, help="proposed: hidden-layer dropout (default = the old fixed value)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--selection-from", type=Path,
                        help="reuse a completed development run of the same model for frozen test evaluation")
    parser.add_argument("--test-origins", type=int, nargs="+",
                        help="compute only these native-time origins; output is explicitly a partial test run")
    parser.add_argument("--development-origins", type=int, nargs="+",
                        help="compute a development partition; epoch selection requires merging all partitions")
    parser.add_argument("--merge-parts", type=Path, nargs="+",
                        help="join disjoint completed test-origin runs and verify the full original test population")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the sealed test origins after development evaluation",
    )
    args = parser.parse_args()
    if args.hidden is None:
        args.hidden = 100 if args.model == "evokg" else 32
    return args


#%%
def main() -> None:
    args = get_args()
    if args.regcn_no_messages and args.model != "regcn":
        raise ValueError("--regcn-no-messages requires --model regcn")
    if args.event_context != "none" and args.model != "regcn":
        raise ValueError("--event-context requires --model regcn")
    if args.model != "proposed" and args.objective != "bce":
        raise ValueError("--objective survival_likelihood is supported only by --model proposed")
    if args.model == "table":
        if not args.runs or not args.output:
            raise ValueError("--model table requires --runs and --output")
        from evaluation.benchmark_table import read_runs, write_tables
        summary = write_tables(read_runs(args.runs, phase=args.table_phase),
                               args.output, phase=args.table_phase)
        print(summary.to_string(index=False))
        return
    if args.model == "logcl":
        if args.max_global_edges is not None:
            raise ValueError("원본 LogCL은 과거 이웃 전체를 사용합니다. --max-global-edges를 빼세요.")
        from modules.logcl_experiment import run
        run(args)
        return
    if args.model == "regcn":
        from modules.regcn_experiment import run
        run(args)
        return

    if args.model == "evokg":
        from modules.evokg_original_survival import prepare_original_encoder
        prepare_original_encoder()
        from modules.evokg_original_experiment import run
        run(args)
        return

    if args.model == "evokg_lifetime":
        from modules.evokg_lifetime_experiment import run
        run(args)
        return

    if args.model == "proposed":
        from modules.proposed_experiment import run
        run(args)
        return
    config = DATASET_CONFIGS[args.dataset]
    args.data = args.data or config["data_path"]
    data_inputs = [Path(args.data)]
    if args.dataset == "yago":
        data_inputs.extend([
            Path(args.data).with_name("yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv"),
            Path(args.data).with_name("yago_survival_build_summary.json"),
        ])
    args.output = args.output or config["output_dir"]
    args.horizons = args.horizons or config["horizons"]
    args.bootstrap_repetitions = (
        args.bootstrap_repetitions or config["bootstrap_repetitions"]
    )
    args.output.mkdir(parents=True, exist_ok=True)

    metric_records: list[dict] = []
    comparison_records: list[dict] = []
    test_metric_records: list[dict] = []
    test_comparison_records: list[dict] = []
    metadata: dict[str, dict] = {}

    for horizon in tqdm(args.horizons, desc="horizons", unit="horizon"):
        print(f"\n[H={horizon}] Building landmark rows...")
        if args.dataset == "yago":
            rows, horizon_metadata = load_yago_landmark_rows(args.data, horizon)
        else:
            rows, horizon_metadata = load_landmark_rows(args.data, horizon)

        development_end_text = (
            config["development_end"]
            or horizon_metadata.get("development_end")
            or horizon_metadata["valid_end"]
        )
        development_end = month_id(development_end_text)
        horizon_months = horizon * config["horizon_months"]
        validation_months = rolling_validation_months(
            rows,
            horizon=horizon_months,
            development_end=development_end,
            step_months=config["validation_step_months"],
            development_start=(
                month_id(config["development_start"])
                if config["development_start"]
                else None
            ),
        )
        fold_predictions = []
        fold_manifest = []
        for validation_month in tqdm(validation_months, desc="validation months", unit="month"):
            train, validation = select_rolling_fold(
                rows,
                validation_month=validation_month,
                development_end=development_end,
            )
            if train["survived"].nunique() != 2:
                raise ValueError(
                    f"H={horizon}, validation={month_text(validation_month)}: "
                    "training fold needs both labels"
                )
            predicted = fit_and_predict(train, validation, config["logistic_c"])
            predicted["validation_origin"] = month_text(validation_month)
            fold_predictions.append(predicted)
            print(f"[H={horizon}] development {month_text(validation_month)}: "
                  f"train={len(train):,}, evaluation={len(validation):,}", flush=True)
            fold_manifest.append(
                {
                    "validation_origin": month_text(validation_month),
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "train_latest_target": train["target_month"].max(),
                }
            )

        if not fold_predictions:
            raise ValueError(f"H={horizon}: no valid rolling-origin folds")
        predictions = pd.concat(fold_predictions, ignore_index=True)
        predictions.to_csv(
            args.output / f"development_oof_h{horizon}.csv",
            index=False,
        )

        for column in ("p_global", "p_relation_age", "p_object_degree"):
            metric_records.append(
                {"horizon": horizon, **summarize_probability(predictions, column)}
            )

        for hypothesis, model, reference in (
            ("task_signal", "p_relation_age", "p_global"),
            ("object_increment", "p_object_degree", "p_relation_age"),
        ):
            result = multiway_bootstrap(
                predictions,
                model,
                reference,
                repetitions=args.bootstrap_repetitions,
                block_months=config["calendar_block_months"],
                seed=config["bootstrap_seed"] + horizon,
            )
            comparison_records.append(
                {
                    "horizon": horizon,
                    "hypothesis": hypothesis,
                    "model": model.removeprefix("p_"),
                    "reference": reference.removeprefix("p_"),
                    **result,
                }
            )
        metadata[str(horizon)] = {
            **horizon_metadata,
            "test_evaluation": "not_run",
            "test_rows_used": False,
            "validation_step_months": config["validation_step_months"],
            "folds": fold_manifest,
        }

        if args.evaluate_test:
            test_months = sorted(
                rows.loc[rows["split"] == "test", "landmark"].map(month_id).unique()
            )
            test_predictions = []
            test_manifest = []
            test_end = rows["target_month"].map(month_id).max()
            for test_month in test_months:
                train, test = select_rolling_fold(
                    rows,
                    validation_month=int(test_month),
                    development_end=int(test_end),
                )
                if train["survived"].nunique() != 2:
                    raise ValueError(
                        f"H={horizon}, test={month_text(test_month)}: "
                        "training fold needs both labels"
                    )
                predicted = fit_and_predict(train, test, config["logistic_c"])
                predicted["test_origin"] = month_text(test_month)
                test_predictions.append(predicted)
                print(f"[H={horizon}] test {month_text(test_month)}: "
                      f"train={len(train):,}, evaluation={len(test):,}", flush=True)
                test_manifest.append(
                    {
                        "test_origin": month_text(test_month),
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_latest_target": train["target_month"].max(),
                    }
                )

            if not test_predictions:
                raise ValueError(f"H={horizon}: no test origins")
            test_predictions = pd.concat(test_predictions, ignore_index=True)
            test_predictions.to_csv(
                args.output / f"test_predictions_h{horizon}.csv",
                index=False,
            )
            for column in ("p_global", "p_relation_age", "p_object_degree"):
                test_metric_records.append(
                    {
                        "horizon": horizon,
                        **summarize_probability(test_predictions, column),
                    }
                )
            for hypothesis, model, reference in (
                ("task_signal", "p_relation_age", "p_global"),
                ("object_increment", "p_object_degree", "p_relation_age"),
            ):
                result = multiway_bootstrap(
                    test_predictions,
                    model,
                    reference,
                    repetitions=args.bootstrap_repetitions,
                    block_months=config["calendar_block_months"],
                    seed=config["bootstrap_seed"] + 10_000 + horizon,
                )
                test_comparison_records.append(
                    {
                        "horizon": horizon,
                        "hypothesis": hypothesis,
                        "model": model.removeprefix("p_"),
                        "reference": reference.removeprefix("p_"),
                        **result,
                    }
                )
            metadata[str(horizon)].update(
                {
                    "test_evaluation": "rolling_origin",
                    "test_rows_used": True,
                    "test_folds": test_manifest,
                }
            )

    metrics = pd.DataFrame(metric_records)
    comparisons = pd.DataFrame(comparison_records)
    comparisons["holm_adjusted_p"] = 1.0

    for hypothesis, indexes in comparisons.groupby("hypothesis").groups.items():
        comparisons.loc[indexes, "holm_adjusted_p"] = holm_adjust(
            comparisons.loc[indexes, "p_value"].tolist()
        )
    comparisons["significant_after_holm"] = comparisons["holm_adjusted_p"] < 0.05

    metrics.to_csv(args.output / "metrics.csv", index=False)
    comparisons.to_csv(args.output / "significance.csv", index=False)
    if args.evaluate_test:
        test_metrics = pd.DataFrame(test_metric_records)
        test_comparisons = pd.DataFrame(test_comparison_records)
        test_comparisons["holm_adjusted_p"] = 1.0
        for hypothesis, indexes in test_comparisons.groupby("hypothesis").groups.items():
            test_comparisons.loc[indexes, "holm_adjusted_p"] = holm_adjust(
                test_comparisons.loc[indexes, "p_value"].tolist()
            )
        test_comparisons["significant_after_holm"] = (
            test_comparisons["holm_adjusted_p"] < 0.05
        )
        test_metrics.to_csv(args.output / "test_metrics.csv", index=False)
        test_comparisons.to_csv(args.output / "test_significance.csv", index=False)
    run_config = {
        "data": str(args.data.resolve()),
        "output": str(args.output.resolve()),
        "dataset": args.dataset,
        "data_sha256": file_hash(args.data),
        "input_sha256": {str(path): file_hash(path) for path in data_inputs},
        "protocol_version": PROTOCOL_VERSION,
        "protocol": PROTOCOL_RULES,
        "source_hashes": {
            str(path.relative_to(Path(__file__).resolve().parent)): file_hash(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "datasets/task.py",
                Path(__file__).resolve().parent / "datasets/interval_landmark.py",
                Path(__file__).resolve().parent / "datasets/yago.py",
                Path(__file__).resolve().parent / "modules/model.py",
            )
        },
        "time_unit": config["time_unit"],
        "horizons": args.horizons,
        "logistic_c": config["logistic_c"],
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": config["bootstrap_seed"],
        "evaluation_protocol": (
            "rolling_origin_development_and_test"
            if args.evaluate_test
            else "rolling_origin_development"
        ),
        "test_evaluation": "rolling_origin" if args.evaluate_test else "not_run",
        "test_rows_used": args.evaluate_test,
        "metadata": metadata,
    }
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMetrics")
    print(metrics.to_string(index=False))
    print("\nSignificance")
    print(comparisons.to_string(index=False))
    if args.evaluate_test:
        print("\nTest metrics")
        print(test_metrics.to_string(index=False))
        print("\nTest significance")
        print(test_comparisons.to_string(index=False))
    print(f"\nSaved to: {args.output.resolve()}")


#%%
if __name__ == "__main__":
    main()
