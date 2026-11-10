from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from datasets.yago import load_yago_labelled_rows
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEVELOPMENT_START = 2003
DEVELOPMENT_END = 2008


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-safe YAGO130K survival baseline comparison"
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/YAGO/processed/yago_survival_rows.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/yago_survival_baselines_development"),
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--cox-penalizer",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--logistic-c",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--macro-min-rows",
        type=int,
        default=100,
        help="Macro-relation Brier에 포함할 관계별 최소 validation 행 수",
    )

    return parser.parse_args()


def add_object_degree(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 landmark에서 같은 object로 들어오는 활성 fact의 수를 계산한다.

    endpoint는 사용하지 않으며 현재 landmark의 KG snapshot만 사용한다.
    """
    result = df.copy()

    result["object_degree"] = (
        result.groupby(["landmark_year", "object"])["fact_id"]
        .transform("count")
        .astype(float)
    )

    result["log_object_degree"] = np.log1p(result["object_degree"])
    result["log_age_years"] = np.log1p(result["age_years"].clip(lower=0))

    return result


def valid_development_years(horizon: int) -> list[int]:
    """
    validation label의 전체 관측구간이 2008년을 넘지 않도록 한다.

    validation_year + horizon <= 2008
    """
    final_year = DEVELOPMENT_END - horizon

    if final_year < DEVELOPMENT_START:
        return []

    return list(range(DEVELOPMENT_START, final_year + 1))


def select_fold(
    df: pd.DataFrame,
    horizon: int,
    validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling-origin development fold를 만든다.

    학습:
        landmark < validation_year
        landmark + horizon <= validation_year

    검증:
        landmark == validation_year
        validation_year + horizon <= 2008
    """
    label_column = f"survival_label_{horizon}y"

    if validation_year + horizon > DEVELOPMENT_END:
        raise ValueError(
            f"Invalid fold: validation={validation_year}, "
            f"horizon={horizon}, development_end={DEVELOPMENT_END}"
        )

    allowed = df["time_split"].isin(["train", "development"])

    train_mask = (
        allowed
        & (df["landmark_year"] < validation_year)
        & (df["landmark_year"] + horizon <= validation_year)
        & df[label_column].notna()
    )

    validation_mask = (
        (df["time_split"] == "development")
        & (df["landmark_year"] == validation_year)
        & df[label_column].notna()
    )

    train = df.loc[train_mask].copy()
    validation = df.loc[validation_mask].copy()

    train["label"] = train[label_column].astype(int)
    validation["label"] = validation[label_column].astype(int)

    if train.empty:
        raise ValueError(
            f"Empty training fold: H={horizon}, year={validation_year}"
        )

    if validation.empty:
        raise ValueError(
            f"Empty validation fold: H={horizon}, year={validation_year}"
        )

    # 시간 누출 방지 확인
    assert (train["landmark_year"] + horizon <= validation_year).all()
    assert (validation["landmark_year"] == validation_year).all()
    assert validation_year + horizon <= DEVELOPMENT_END

    return train, validation


def derive_duration_and_event(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    1·2·3년 생존 라벨로부터 연 단위 잔여 생존시간을 만든다.

    예:
        label_1y = 0              -> 1년 이내 종료
        label_1y = 1, label_2y=0 -> 2년 이내 종료
        3년까지 모두 1           -> horizon 시점에서 검열
    """
    n = len(df)

    duration = np.full(n, float(horizon))
    event = np.zeros(n, dtype=int)

    previous_survival = np.ones(n, dtype=bool)

    for step in range(1, horizon + 1):
        column = f"survival_label_{step}y"
        current = df[column].to_numpy()

        observed = ~pd.isna(current)
        terminated = previous_survival & observed & (current == 0)

        duration[terminated] = float(step)
        event[terminated] = 1

        previous_survival = (
            previous_survival
            & observed
            & (current == 1)
        )

    return duration, event


def make_logistic_pipeline(
    include_object_degree: bool,
    c_value: float,
) -> Pipeline:
    categorical = ["relation"]
    numeric = ["log_age_years"]

    if include_object_degree:
        numeric.append("log_object_degree")

    transformer = ColumnTransformer(
        transformers=[
            (
                "relation",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                categorical,
            ),
            (
                "numeric",
                StandardScaler(),
                numeric,
            ),
        ]
    )

    classifier = LogisticRegression(
        C=c_value,
        max_iter=500,
        solver="lbfgs",
        class_weight=None,
    )

    return Pipeline(
        steps=[
            ("features", transformer),
            ("classifier", classifier),
        ]
    )


def fit_predict_km(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    duration, event = derive_duration_and_event(train, horizon)

    model = KaplanMeierFitter()
    model.fit(duration, event_observed=event)

    probability = float(model.predict(horizon))

    return np.full(len(validation), probability, dtype=float)

def prepare_cox_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    min_relation_rows: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cox 모델에서 표본이 너무 적은 관계를 __OTHER__로 묶는다.

    validation 관계 빈도는 사용하지 않고 train 빈도만 사용하므로
    미래정보 누출이 발생하지 않는다.
    """
    relation_counts = train["relation"].value_counts()

    retained_relations = set(
        relation_counts[
            relation_counts >= min_relation_rows
        ].index
    )

    train_relation_values = train["relation"].where(
        train["relation"].isin(retained_relations),
        "__OTHER__",
    )

    validation_relation_values = validation["relation"].where(
        validation["relation"].isin(retained_relations),
        "__OTHER__",
    )

    train_relation = pd.get_dummies(
        train_relation_values,
        prefix="relation",
        drop_first=True,
        dtype=float,
    )

    validation_relation = pd.get_dummies(
        validation_relation_values,
        prefix="relation",
        drop_first=True,
        dtype=float,
    )

    validation_relation = validation_relation.reindex(
        columns=train_relation.columns,
        fill_value=0.0,
    )

    train_numeric = train[
        ["log_age_years", "log_object_degree"]
    ].astype(float).reset_index(drop=True)

    validation_numeric = validation[
        ["log_age_years", "log_object_degree"]
    ].astype(float).reset_index(drop=True)

    train_x = pd.concat(
        [
            train_numeric,
            train_relation.reset_index(drop=True),
        ],
        axis=1,
    )

    validation_x = pd.concat(
        [
            validation_numeric,
            validation_relation.reset_index(drop=True),
        ],
        axis=1,
    )

    # 상수 또는 극단적으로 분산이 낮은 열 제거
    retained_columns = []

    for column in train_x.columns:
        variance = float(train_x[column].var())

        if np.isfinite(variance) and variance > 1e-8:
            retained_columns.append(column)

    train_x = train_x[retained_columns]

    validation_x = validation_x.reindex(
        columns=retained_columns,
        fill_value=0.0,
    )

    return train_x, validation_x


def fit_predict_cox(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    penalizer: float,
) -> np.ndarray:
    train_x, validation_x = prepare_cox_features(
        train,
        validation,
    )

    duration, event = derive_duration_and_event(train, horizon)

    cox_data = train_x.copy()
    cox_data["duration"] = duration
    cox_data["event"] = event

    model = CoxPHFitter(penalizer=penalizer)

    model.fit(
        cox_data,
        duration_col="duration",
        event_col="event",
        show_progress=False,
        batch_mode=True,
    )

    partial_hazard = (
        model.predict_partial_hazard(validation_x)
        .to_numpy()
        .reshape(-1)
    )

    baseline = model.baseline_survival_

    baseline_probability = float(
        np.interp(
            horizon,
            baseline.index.to_numpy(dtype=float),
            baseline.iloc[:, 0].to_numpy(dtype=float),
        )
    )

    prediction = np.power(
        np.clip(baseline_probability, 1e-12, 1.0),
        partial_hazard,
    )

    return np.clip(prediction, 0.0, 1.0)


def build_discrete_rows(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    각 landmark 행을 1년 단위 hazard 학습행으로 변환한다.

    event=1: 해당 연도 구간에서 종료
    event=0: 해당 연도 구간을 생존
    """
    pieces = []
    at_risk = np.ones(len(df), dtype=bool)

    for step in range(1, horizon + 1):
        label_column = f"survival_label_{step}y"
        current_label = df[label_column].to_numpy()

        observed = ~pd.isna(current_label)
        use = at_risk & observed

        if not use.any():
            break

        part = df.loc[
            use,
            [
                "relation",
                "age_years",
                "log_object_degree",
            ],
        ].copy()

        part["step"] = step
        part["future_age"] = (
            part["age_years"] + step - 1
        )
        part["log_future_age"] = np.log1p(
            part["future_age"].clip(lower=0)
        )

        part["event"] = (
            1 - current_label[use].astype(int)
        )

        pieces.append(part)

        at_risk = (
            at_risk
            & observed
            & (current_label == 1)
        )

    if not pieces:
        raise ValueError("No discrete hazard rows were generated.")

    return pd.concat(pieces, ignore_index=True)


def make_discrete_pipeline(c_value: float) -> Pipeline:
    transformer = ColumnTransformer(
        transformers=[
            (
                "relation",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                ["relation"],
            ),
            (
                "numeric",
                StandardScaler(),
                [
                    "log_future_age",
                    "log_object_degree",
                    "step",
                ],
            ),
        ]
    )

    classifier = LogisticRegression(
        C=c_value,
        max_iter=500,
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("features", transformer),
            ("classifier", classifier),
        ]
    )


def fit_predict_discrete_hazard(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    c_value: float,
) -> np.ndarray:
    train_discrete = build_discrete_rows(train, horizon)

    model = make_discrete_pipeline(c_value)

    feature_columns = [
        "relation",
        "log_future_age",
        "log_object_degree",
        "step",
    ]

    model.fit(
        train_discrete[feature_columns],
        train_discrete["event"],
    )

    survival = np.ones(len(validation), dtype=float)

    for step in range(1, horizon + 1):
        step_data = validation[
            [
                "relation",
                "age_years",
                "log_object_degree",
            ]
        ].copy()

        step_data["step"] = step
        step_data["future_age"] = (
            step_data["age_years"] + step - 1
        )
        step_data["log_future_age"] = np.log1p(
            step_data["future_age"].clip(lower=0)
        )

        hazard = model.predict_proba(
            step_data[feature_columns]
        )[:, 1]

        survival *= 1.0 - hazard

    return np.clip(survival, 0.0, 1.0)


def fit_predict_logistic(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    include_object_degree: bool,
    c_value: float,
) -> np.ndarray:
    model = make_logistic_pipeline(
        include_object_degree=include_object_degree,
        c_value=c_value,
    )

    columns = ["relation", "log_age_years"]

    if include_object_degree:
        columns.append("log_object_degree")

    model.fit(train[columns], train["label"])

    return model.predict_proba(validation[columns])[:, 1]


def compute_metrics(
    validation: pd.DataFrame,
    predictions: np.ndarray,
    macro_min_rows: int,
) -> dict:
    y = validation["label"].to_numpy(dtype=float)
    predictions = np.clip(predictions, 0.0, 1.0)

    squared_error = np.square(y - predictions)

    micro_brier = float(np.mean(squared_error))

    fact_error = pd.DataFrame(
        {
            "fact_id": validation["fact_id"].to_numpy(),
            "squared_error": squared_error,
        }
    )

    fact_brier = float(
        fact_error.groupby("fact_id")["squared_error"]
        .mean()
        .mean()
    )

    relation_error = pd.DataFrame(
        {
            "relation": validation["relation"].to_numpy(),
            "squared_error": squared_error,
        }
    )

    relation_summary = (
        relation_error.groupby("relation")["squared_error"]
        .agg(["mean", "count"])
    )

    eligible_relations = relation_summary[
        relation_summary["count"] >= macro_min_rows
    ]

    if eligible_relations.empty:
        macro_relation_brier = np.nan
        macro_relation_count = 0
    else:
        macro_relation_brier = float(
            eligible_relations["mean"].mean()
        )
        macro_relation_count = int(len(eligible_relations))

    return {
        "rows": int(len(validation)),
        "facts": int(validation["fact_id"].nunique()),
        "relations": int(validation["relation"].nunique()),
        "survival_rate": float(y.mean()),
        "micro_brier": micro_brier,
        "fact_brier": fact_brier,
        "macro_relation_brier": macro_relation_brier,
        "macro_relation_count": macro_relation_count,
    }


def evaluate_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    horizon: int,
    validation_year: int,
    args,
) -> list[dict]:
    predictions = {}

    predictions["km"] = fit_predict_km(
        train,
        validation,
        horizon,
    )

    predictions["cox"] = fit_predict_cox(
        train,
        validation,
        horizon,
        penalizer=args.cox_penalizer,
    )

    predictions["discrete_hazard"] = (
        fit_predict_discrete_hazard(
            train,
            validation,
            horizon,
            c_value=args.logistic_c,
        )
    )

    predictions["relation_age"] = fit_predict_logistic(
        train,
        validation,
        include_object_degree=False,
        c_value=args.logistic_c,
    )

    predictions["object_degree"] = fit_predict_logistic(
        train,
        validation,
        include_object_degree=True,
        c_value=args.logistic_c,
    )

    results = []

    for model_name, model_prediction in predictions.items():
        metrics = compute_metrics(
            validation,
            model_prediction,
            macro_min_rows=args.macro_min_rows,
        )

        metrics.update(
            {
                "horizon": horizon,
                "validation_year": validation_year,
                "model": model_name,
                "train_rows": int(len(train)),
                "train_facts": int(train["fact_id"].nunique()),
                "train_latest_landmark": int(
                    train["landmark_year"].max()
                ),
                "label_available_through": int(
                    train["landmark_year"].max() + horizon
                ),
            }
        )

        results.append(metrics)

    return results


def main():
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Survival table not found: {args.data.resolve()}"
        )

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("YAGO130K DEVELOPMENT SURVIVAL BASELINES")
    print("=" * 72)
    print(f"data: {args.data.resolve()}")
    print("test split: SEALED")

    df, _, _ = load_yago_labelled_rows(args.data, args.horizons)

    required_columns = {
        "fact_id",
        "subject",
        "relation",
        "object",
        "begin_year",
        "landmark_year",
        "age_years",
        "time_split",
        "survival_label_1y",
        "survival_label_2y",
        "survival_label_3y",
    }

    missing = sorted(required_columns - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # test는 메모리상 모델 입력에서도 제외한다.
    development_df = df[
        df["time_split"].isin(["train", "development"])
    ].copy()

    del df

    development_df = add_object_degree(development_df)

    all_results = []
    fold_manifest = []

    for horizon in args.horizons:
        years = valid_development_years(horizon)

        if not years:
            print(f"[H={horizon}] No valid development years")
            continue

        print(
            f"\n[H={horizon}] validation years: {years}"
        )

        for validation_year in years:
            train, validation = select_fold(
                development_df,
                horizon=horizon,
                validation_year=validation_year,
            )

            print(
                f"  year={validation_year} | "
                f"train={len(train):,} | "
                f"validation={len(validation):,} | "
                f"train labels available through "
                f"{train['landmark_year'].max() + horizon}"
            )

            fold_manifest.append(
                {
                    "horizon": horizon,
                    "validation_year": validation_year,
                    "train_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "train_latest_landmark": int(
                        train["landmark_year"].max()
                    ),
                    "train_label_available_through": int(
                        train["landmark_year"].max() + horizon
                    ),
                    "validation_label_available_through": int(
                        validation_year + horizon
                    ),
                }
            )

            fold_results = evaluate_fold(
                train,
                validation,
                horizon,
                validation_year,
                args,
            )

            all_results.extend(fold_results)

    if not all_results:
        raise RuntimeError("No folds were evaluated.")

    result_df = pd.DataFrame(all_results)

    summary = (
        result_df.groupby(["horizon", "model"], as_index=False)
        .agg(
            folds=("validation_year", "nunique"),
            rows=("rows", "sum"),
            mean_survival_rate=("survival_rate", "mean"),
            mean_micro_brier=("micro_brier", "mean"),
            mean_fact_brier=("fact_brier", "mean"),
            mean_macro_relation_brier=(
                "macro_relation_brier",
                "mean",
            ),
        )
        .sort_values(
            ["horizon", "mean_fact_brier", "model"]
        )
    )

    result_path = args.output / "fold_metrics.csv"
    summary_path = args.output / "summary_metrics.csv"
    manifest_path = args.output / "fold_manifest.json"

    result_df.to_csv(result_path, index=False)
    summary.to_csv(summary_path, index=False)

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            fold_manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nFold metrics")
    print(result_df.to_string(index=False))

    print("\nSummary")
    print(summary.to_string(index=False))

    print("\nSaved:")
    print(f"- {result_path.resolve()}")
    print(f"- {summary_path.resolve()}")
    print(f"- {manifest_path.resolve()}")


if __name__ == "__main__":
    main()