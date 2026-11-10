"""Paired uncertainty estimation with repeated-landmark dependence."""

#%%
from __future__ import annotations


import numpy as np
import pandas as pd

from datasets.interval_landmark import month_id
from evaluation.metrics import fact_key


#%%
def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


#%%
def multiway_bootstrap(
    rows: pd.DataFrame,
    model_column: str,
    reference_column: str,
    repetitions: int,
    block_months: int,
    seed: int,
) -> dict:
    """Weight fact, shared entity, and calendar block in each bootstrap draw."""
    data = rows.copy()
    data["difference"] = (
        (data["survived"] - data[model_column]) ** 2
        - (data["survived"] - data[reference_column]) ** 2
    )

    fact_index = pd.factorize(fact_key(data), sort=False)[0]
    entities = pd.Index(pd.concat([data["subject"], data["object"]]).unique())
    entity_map = {entity: index for index, entity in enumerate(entities)}
    subject_index = data["subject"].map(entity_map).to_numpy()
    object_index = data["object"].map(entity_map).to_numpy()
    months = data["landmark"].map(month_id).to_numpy()
    blocks, block_index = np.unique(months // block_months, return_inverse=True)

    fact_sizes = np.bincount(fact_index)
    base_weight = 1.0 / fact_sizes[fact_index]
    difference = data["difference"].to_numpy()
    point = float(np.sum(base_weight * difference) / np.sum(base_weight))

    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for repetition in range(repetitions):
        weight = (
            base_weight
            * rng.exponential(size=len(fact_sizes))[fact_index]
            * rng.exponential(size=len(entities))[subject_index]
            * rng.exponential(size=len(entities))[object_index]
            * rng.exponential(size=len(blocks))[block_index]
        )
        estimates[repetition] = np.sum(weight * difference) / np.sum(weight)

    low, high = np.quantile(estimates, [0.025, 0.975])
    left = np.sum(estimates <= 0)
    right = np.sum(estimates >= 0)

    p_value = min(1.0, 2 * (min(left, right) + 1) / (repetitions + 1))
    return {
        "brier_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
        "rows": len(data),
        "facts": len(fact_sizes),
    }

#%%
def multiway_bootstrap(
    rows: pd.DataFrame,
    model_column: str,
    reference_column: str,
    repetitions: int,
    block_months: int,
    seed: int,
    debug: bool = False,
) -> dict:
    """Weight fact, shared entity, and calendar block."""

    # 원본을 변경하지 않도록 복사
    data = rows.copy()

    if debug:
        print("\n=== 1. 함수 입력 ===")
        print("model =", model_column)
        print("reference =", reference_column)
        print("repetitions =", repetitions)
        print("block_months =", block_months)
        print("seed =", seed)
        print("rows =", len(data))

        print("\n입력 데이터 일부:")
        print(
            data[
                [
                    "fact_id",
                    "subject",
                    "object",
                    "landmark",
                    "survived",
                    model_column,
                    reference_column,
                ]
            ].head()
        )

    # 각 행에서 두 모델의 Brier 오차 차이 계산
    data["difference"] = (
        (data["survived"] - data[model_column]) ** 2
        - (
            data["survived"]
            - data[reference_column]
        ) ** 2
    )

    if debug:
        print("\n=== 2. 행별 Brier 차이 ===")
        print(
            data[
                [
                    "survived",
                    model_column,
                    reference_column,
                    "difference",
                ]
            ].head()
        )
        print(
            "difference < 0이면",
            model_column,
            "모델이 더 좋음",
        )

    # 각 행이 어느 fact에 속하는지 정수 번호 부여
    fact_index = pd.factorize(
        fact_key(data),
        sort=False,
    )[0]

    if debug:
        print("\n=== 3. Fact 번호 ===")
        print("fact IDs =", fact_key(data).head().tolist())
        print("fact_index =", fact_index[:10])

    # 전체 subject/object 목록 생성
    entities = pd.Index(
        pd.concat(
            [data["subject"], data["object"]]
        ).unique()
    )

    entity_map = {
        entity: index
        for index, entity in enumerate(entities)
    }

    subject_index = (
        data["subject"]
        .map(entity_map)
        .to_numpy()
    )

    object_index = (
        data["object"]
        .map(entity_map)
        .to_numpy()
    )

    if debug:
        print("\n=== 4. Entity 번호 ===")
        print("entities =", entities[:10].tolist())
        print("entity_map 일부 =", dict(list(entity_map.items())[:10]))
        print("subject_index =", subject_index[:10])
        print("object_index =", object_index[:10])

    # 날짜를 month ID로 바꾼 후 달력 블록 생성
    months = (
        data["landmark"]
        .map(month_id)
        .to_numpy()
    )

    blocks, block_index = np.unique(
        months // block_months,
        return_inverse=True,
    )

    if debug:
        print("\n=== 5. Calendar block ===")
        print("months =", months[:10])
        print("blocks =", blocks[:10])
        print("block_index =", block_index[:10])

    # 각 fact가 몇 행인지 계산
    fact_sizes = np.bincount(fact_index)

    # 같은 fact의 전체 가중치가 1이 되도록 조정
    base_weight = 1.0 / fact_sizes[fact_index]

    difference = data["difference"].to_numpy()

    # 원래 데이터에서의 Brier 차이
    point = float(
        np.sum(base_weight * difference)
        / np.sum(base_weight)
    )

    if debug:
        print("\n=== 6. 기본 가중치와 점추정 ===")
        print("fact_sizes =", fact_sizes[:10])
        print("base_weight =", base_weight[:10])
        print("point estimate =", point)

    # 재현 가능한 난수 생성기
    rng = np.random.default_rng(seed)

    estimates = np.empty(repetitions)

    for repetition in range(repetitions):
        fact_weight = rng.exponential(
            size=len(fact_sizes)
        )
        entity_weight = rng.exponential(
            size=len(entities)
        )
        block_weight = rng.exponential(
            size=len(blocks)
        )

        weight = (
            base_weight
            * fact_weight[fact_index]
            * entity_weight[subject_index]
            * entity_weight[object_index]
            * block_weight[block_index]
        )

        estimates[repetition] = (
            np.sum(weight * difference)
            / np.sum(weight)
        )

        if debug and repetition < 3:
            print(
                f"\n=== 7. Bootstrap 반복 "
                f"{repetition + 1} ==="
            )
            print(
                "fact_weight 일부 =",
                fact_weight[:5],
            )
            print(
                "entity_weight 일부 =",
                entity_weight[:5],
            )
            print(
                "block_weight 일부 =",
                block_weight[:5],
            )
            print(
                "최종 행 가중치 일부 =",
                weight[:5],
            )
            print(
                "이번 Brier 차이 =",
                estimates[repetition],
            )

    # Bootstrap 분포의 2.5%, 97.5%
    low, high = np.quantile(
        estimates,
        [0.025, 0.975],
    )

    # Bootstrap 결과가 0의 양쪽에 얼마나 존재하는지 계산
    left = np.sum(estimates <= 0)
    right = np.sum(estimates >= 0)

    p_value = min(
        1.0,
        2
        * (min(left, right) + 1)
        / (repetitions + 1),
    )

    if debug:
        print("\n=== 8. 최종 결과 ===")
        print("bootstrap estimates =", estimates)
        print("point =", point)
        print("95% CI =", (low, high))
        print("0 이하 개수 =", left)
        print("0 이상 개수 =", right)
        print("p-value =", p_value)

    return {
        "brier_difference": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": float(p_value),
        "rows": len(data),
        "facts": len(fact_sizes),
    }


if __name__ == "__main__":
    example = pd.DataFrame({
        "fact_id": ["A", "A", "B", "B", "C"],
        "subject": ["S1", "S1", "S2", "S2", "S3"],
        "object": ["O1", "O1", "O1", "O2", "O3"],
        "landmark": [
            "2020-01",
            "2020-04",
            "2020-01",
            "2020-04",
            "2020-01",
        ],
        "survived": [1, 0, 1, 1, 0],
        "p_model": [0.8, 0.4, 0.7, 0.9, 0.2],
        "p_reference": [0.6, 0.7, 0.6, 0.6, 0.5],
    })

    result = multiway_bootstrap(
        rows=example,
        model_column="p_model",
        reference_column="p_reference",
        repetitions=5,
        block_months=3,
        seed=42,
        debug=True,
    )

    print("\n반환값:")
    print(result)