"""
YAGO130K의 구간 fact로 landmark 생존 테이블을 생성한다.

핵심 원칙
---------
1. 원본 train/valid/test에 중복된 interval fact를 한 번만 남긴다.
2. 종료연도는 라벨 생성에만 사용한다.
3. 모델 입력 파일에는 종료연도를 저장하지 않는다.
4. landmark 연도를 기준으로 새로운 시간 split을 만든다.
5. 데이터 마지막 연도까지 살아 있는 fact는 우측 검열로 처리한다.

출력
----
1. yago_survival_rows.csv
   모델 학습에 사용하는 안전한 데이터

2. yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv
   라벨 검증 전용 종료일 자료. 모델 feature로 사용 금지.

3. yago_survival_build_summary.json
   중복 제거 및 라벨 분포 요약
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from datasets.task import make_survival_labels as make_fixed_horizon_labels


DEFAULT_TRAIN_WINDOW_YEARS = 10
DEFAULT_HORIZONS = (1, 2, 3)

MISSING_TIME_TOKENS = {
    "",
    "#",
    "##",
    "####",
    "####-##",
    "####-##-##",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "unknown",
    "unk",
}


@dataclass
class IntervalFact:
    """중복을 제거한 하나의 구간 fact."""

    fact_id: str
    subject: str
    relation: str
    object_: str
    begin_year: int
    end_year: int | None
    source_splits: set[str]
    boundary_censored: bool = False


def parse_args() -> argparse.Namespace:
    """명령행 인수를 정의한다."""

    parser = argparse.ArgumentParser(
        description="YAGO130K landmark 생존 테이블을 생성합니다."
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="YAGO train_valid_test.pkl 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="결과를 저장할 디렉터리",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="예측 horizon 연수. 기본값: 1 2 3",
    )
    parser.add_argument(
        "--landmark-start",
        type=int,
        default=None,
        help=(
            "첫 landmark 연도. 생략하면 train cutoff보다 "
            "10년 전부터 시작합니다."
        ),
    )
    parser.add_argument(
        "--landmark-end",
        type=int,
        default=None,
        help=(
            "마지막 landmark 연도. 생략하면 모든 horizon의 "
            "정답을 확인할 수 있는 마지막 연도를 사용합니다."
        ),
    )
    return parser.parse_args()


def is_missing_time(value: Any) -> bool:
    """시간값이 누락됐거나 열린 값인지 확인한다."""

    if value is None:
        return True

    if isinstance(value, float) and math.isnan(value):
        return True

    text = str(value).strip().lower()

    if text in MISSING_TIME_TOKENS:
        return True

    return "#" in text


def parse_year(value: Any) -> int | None:
    """연도 또는 연도형 문자열을 정수 연도로 변환한다."""

    if is_missing_time(value):
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None

    text = str(value).strip()

    year_match = re.fullmatch(r"(-?\d{1,4})", text)
    if year_match:
        return int(year_match.group(1))

    date_match = re.match(r"^(-?\d{4})[-/]", text)
    if date_match:
        return int(date_match.group(1))

    return None


def load_dataset(
    path: Path,
) -> tuple[int, int, list[Any], list[Any], list[Any], Any]:
    """6개 항목으로 구성된 TKG-Benchmark pickle을 읽는다."""

    if not path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")

    # 공식 TKG-Benchmark에서 받은 신뢰할 수 있는 pickle에만 사용한다.
    with path.open("rb") as file:
        container = pickle.load(file)

    if not isinstance(container, (list, tuple)):
        raise TypeError(
            "최상위 pickle 자료형이 list 또는 tuple이 아닙니다."
        )

    if len(container) != 6:
        raise ValueError(
            "pickle 컨테이너는 6개 항목이어야 합니다. "
            f"현재 항목 수: {len(container)}"
        )

    (
        train_split_time,
        valid_split_time,
        train_set,
        valid_set,
        test_set,
        entity_description_dict,
    ) = container

    train_cutoff = parse_year(train_split_time)
    valid_cutoff = parse_year(valid_split_time)

    if train_cutoff is None or valid_cutoff is None:
        raise ValueError(
            "train 또는 valid split 연도를 해석할 수 없습니다."
        )

    if train_cutoff >= valid_cutoff:
        raise ValueError(
            "train cutoff는 valid cutoff보다 앞서야 합니다."
        )

    return (
        train_cutoff,
        valid_cutoff,
        list(train_set),
        list(valid_set),
        list(test_set),
        entity_description_dict,
    )


def record_to_list(record: Any) -> list[Any] | None:
    """fact 레코드를 list로 변환한다."""

    if isinstance(record, (str, bytes)):
        return None

    try:
        return list(record)
    except TypeError:
        return None


def find_observation_end(
    split_records: dict[str, Sequence[Any]],
) -> int:
    """모든 timestamp와 interval에서 마지막 관측연도를 찾는다."""

    observed_years: list[int] = []

    for records in split_records.values():
        for record in records:
            converted = record_to_list(record)

            if converted is None:
                continue

            if len(converted) == 4:
                year = parse_year(converted[3])

                if year is not None:
                    observed_years.append(year)

            elif len(converted) == 5:
                begin_year = parse_year(converted[3])
                end_year = parse_year(converted[4])

                if begin_year is not None:
                    observed_years.append(begin_year)

                if end_year is not None:
                    observed_years.append(end_year)

    if not observed_years:
        raise ValueError("관측 가능한 연도를 찾지 못했습니다.")

    return max(observed_years)


def make_fact_id(
    subject: Any,
    relation: Any,
    object_: Any,
    begin_year: int,
    end_year: int | None,
) -> str:
    """interval 전체를 식별하는 재현 가능한 fact ID를 만든다."""

    raw_key = json.dumps(
        [
            str(subject),
            str(relation),
            str(object_),
            begin_year,
            end_year,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        raw_key.encode("utf-8")
    ).hexdigest()[:20]


def interval_key(
    subject: Any,
    relation: Any,
    object_: Any,
    begin_year: int,
    end_year: int | None,
) -> tuple[str, str, str, int, int | None]:
    """완전히 동일한 interval을 찾기 위한 키를 생성한다."""

    return (
        str(subject),
        str(relation),
        str(object_),
        begin_year,
        end_year,
    )


def collect_unique_intervals(
    split_records: dict[str, Sequence[Any]],
    observation_end: int,
) -> tuple[list[IntervalFact], dict[str, Any]]:
    """
    모든 split의 interval을 합치고 완전 중복을 제거한다.

    동일 interval이 여러 split에 있으면 source_splits에
    원래 소속만 누적한다.
    """

    interval_map: dict[
        tuple[str, str, str, int, int | None],
        IntervalFact,
    ] = {}

    raw_interval_records = 0
    malformed_records = 0
    invalid_intervals = 0

    for split_name, records in split_records.items():
        for record in records:
            converted = record_to_list(record)

            if converted is None or len(converted) != 5:
                continue

            raw_interval_records += 1

            subject, relation, object_, begin_raw, end_raw = converted

            begin_year = parse_year(begin_raw)
            end_year = parse_year(end_raw)

            if begin_year is None:
                malformed_records += 1
                continue

            if end_year is not None and end_year <= begin_year:
                invalid_intervals += 1
                continue

            key = interval_key(
                subject=subject,
                relation=relation,
                object_=object_,
                begin_year=begin_year,
                end_year=end_year,
            )

            if key in interval_map:
                interval_map[key].source_splits.add(split_name)
                continue

            fact_id = make_fact_id(
                subject=subject,
                relation=relation,
                object_=object_,
                begin_year=begin_year,
                end_year=end_year,
            )

            interval_map[key] = IntervalFact(
                fact_id=fact_id,
                subject=str(subject),
                relation=str(relation),
                object_=str(object_),
                begin_year=begin_year,
                end_year=end_year,
                source_splits={split_name},
                # 마지막 관측연도와 같은 종료는 보수적으로 검열 처리
                boundary_censored=(
                    end_year is None
                    or end_year >= observation_end
                ),
            )

    intervals = sorted(
        interval_map.values(),
        key=lambda fact: (
            fact.begin_year,
            fact.relation,
            fact.subject,
            fact.object_,
            fact.fact_id,
        ),
    )

    triple_spans: defaultdict[
        tuple[str, str, str],
        int,
    ] = defaultdict(int)

    for fact in intervals:
        triple_spans[
            (fact.subject, fact.relation, fact.object_)
        ] += 1

    multiple_span_triples = sum(
        count > 1 for count in triple_spans.values()
    )

    summary = {
        "raw_interval_records": raw_interval_records,
        "unique_intervals": len(intervals),
        "duplicate_interval_occurrences_removed": (
            raw_interval_records - len(intervals)
        ),
        "malformed_interval_records": malformed_records,
        "invalid_intervals": invalid_intervals,
        "unique_triples": len(triple_spans),
        "triples_with_multiple_intervals": multiple_span_triples,
    }

    return intervals, summary


def assign_time_split(
    landmark_year: int,
    train_cutoff: int,
    valid_cutoff: int,
) -> str:
    """
    landmark 연도로 모델 split을 배정한다.

    train:
        train cutoff 이하의 landmark

    development:
        train cutoff 이후부터 valid cutoff까지

    test:
        valid cutoff 이후
    """

    if landmark_year <= train_cutoff:
        return "train"

    if landmark_year <= valid_cutoff:
        return "development"

    return "test"


def is_active_at_landmark(
    fact: IntervalFact,
    landmark_year: int,
) -> bool:
    """반개구간 [begin, end)에서 fact가 활성인지 확인한다."""

    if landmark_year < fact.begin_year:
        return False

    if fact.end_year is None:
        return True

    return landmark_year < fact.end_year


def make_survival_label(
    fact: IntervalFact,
    landmark_year: int,
    horizon_years: int,
    observation_end: int,
) -> int | None:
    """
    활성 fact의 horizon 생존 라벨을 생성한다.

    label=0:
        landmark 이후 horizon 안에 종료

    label=1:
        horizon 이후에도 생존

    None:
        관측 종료 때문에 정답을 확정할 수 없음
    """

    event_observed = fact.end_year is not None and not fact.boundary_censored
    stop_year = fact.end_year if event_observed else observation_end
    value = make_fixed_horizon_labels(
        fact.begin_year,
        stop_year,
        landmark_year,
        horizon_years,
        event_observed=event_observed,
    ).item()
    return None if math.isnan(value) else int(value)


def write_endpoint_reference(
    path: Path,
    intervals: Sequence[IntervalFact],
) -> None:
    """
    종료일 검증용 파일을 저장한다.

    이 파일의 end_year는 모델 입력 feature로 사용하면 안 된다.
    """

    fieldnames = [
        "fact_id",
        "subject",
        "relation",
        "object",
        "begin_year",
        "end_year",
        "boundary_censored",
        "source_splits",
    ]

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for fact in intervals:
            writer.writerow(
                {
                    "fact_id": fact.fact_id,
                    "subject": fact.subject,
                    "relation": fact.relation,
                    "object": fact.object_,
                    "begin_year": fact.begin_year,
                    "end_year": (
                        ""
                        if fact.end_year is None
                        else fact.end_year
                    ),
                    "boundary_censored": int(
                        fact.boundary_censored
                    ),
                    "source_splits": "|".join(
                        sorted(fact.source_splits)
                    ),
                }
            )


def write_survival_rows(
    path: Path,
    intervals: Sequence[IntervalFact],
    train_cutoff: int,
    valid_cutoff: int,
    observation_end: int,
    landmark_start: int,
    landmark_end: int,
    horizons: Sequence[int],
) -> tuple[int, dict[str, Any]]:
    """
    모델 입력용 landmark 생존 테이블을 저장한다.

    중요:
    모델 입력 파일에는 end_year를 저장하지 않는다.
    """

    label_columns = [
        f"survival_label_{horizon}y"
        for horizon in horizons
    ]

    fieldnames = [
        "fact_id",
        "subject",
        "relation",
        "object",
        "begin_year",
        "landmark_year",
        "age_years",
        "time_split",
        *label_columns,
    ]

    total_rows = 0

    split_counts: Counter = Counter()
    relation_counts: Counter = Counter()

    label_counts: dict[str, Counter] = {
        f"{horizon}y": Counter()
        for horizon in horizons
    }

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for landmark_year in range(
            landmark_start,
            landmark_end + 1,
        ):
            time_split = assign_time_split(
                landmark_year=landmark_year,
                train_cutoff=train_cutoff,
                valid_cutoff=valid_cutoff,
            )

            for fact in intervals:
                if fact.begin_year > landmark_year:
                    break

                if not is_active_at_landmark(
                    fact=fact,
                    landmark_year=landmark_year,
                ):
                    continue

                row: dict[str, Any] = {
                    "fact_id": fact.fact_id,
                    "subject": fact.subject,
                    "relation": fact.relation,
                    "object": fact.object_,
                    "begin_year": fact.begin_year,
                    "landmark_year": landmark_year,
                    "age_years": landmark_year - fact.begin_year,
                    "time_split": time_split,
                }

                observed_label_exists = False

                for horizon in horizons:
                    label = make_survival_label(
                        fact=fact,
                        landmark_year=landmark_year,
                        horizon_years=horizon,
                        observation_end=observation_end,
                    )

                    column = f"survival_label_{horizon}y"
                    counter_key = f"{horizon}y"

                    if label is None:
                        row[column] = ""
                        label_counts[counter_key]["censored"] += 1
                    else:
                        row[column] = label
                        label_counts[counter_key][f"label_{label}"] += 1
                        observed_label_exists = True

                # 모든 horizon이 검열된 행은 모델 학습에 쓸 수 없으므로 제외
                if not observed_label_exists:
                    continue

                writer.writerow(row)

                total_rows += 1
                split_counts[time_split] += 1
                relation_counts[fact.relation] += 1

    statistics = {
        "total_landmark_rows": total_rows,
        "rows_by_time_split": dict(split_counts),
        "rows_by_relation": dict(
            relation_counts.most_common()
        ),
        "labels_by_horizon": {
            horizon: dict(counts)
            for horizon, counts in label_counts.items()
        },
    }

    return total_rows, statistics


def main() -> None:
    """YAGO landmark 생존 테이블을 생성한다."""

    args = parse_args()

    data_path = args.data.resolve()
    output_directory = args.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    horizons = sorted(set(args.horizons))

    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizon은 1 이상의 정수여야 합니다.")

    (
        train_cutoff,
        valid_cutoff,
        train_set,
        valid_set,
        test_set,
        entity_description_dict,
    ) = load_dataset(data_path)

    split_records = {
        "train": train_set,
        "valid": valid_set,
        "test": test_set,
    }

    observation_end = find_observation_end(split_records)

    intervals, deduplication_summary = collect_unique_intervals(
        split_records=split_records,
        observation_end=observation_end,
    )

    if not intervals:
        raise ValueError("사용 가능한 interval fact가 없습니다.")

    landmark_start = (
        args.landmark_start
        if args.landmark_start is not None
        else train_cutoff - DEFAULT_TRAIN_WINDOW_YEARS
    )

    # 모든 horizon의 정답을 확인할 수 있는 마지막 landmark
    default_landmark_end = observation_end - max(horizons)

    landmark_end = (
        args.landmark_end
        if args.landmark_end is not None
        else default_landmark_end
    )

    if landmark_start > landmark_end:
        raise ValueError(
            "landmark 시작연도가 종료연도보다 늦습니다."
        )

    if landmark_end > default_landmark_end:
        raise ValueError(
            "요청한 landmark 종료연도에서는 가장 긴 horizon의 "
            "정답을 관측할 수 없습니다. "
            f"가능한 마지막 연도: {default_landmark_end}"
        )

    endpoint_path = (
        output_directory
        / "yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv"
    )
    survival_path = (
        output_directory
        / "yago_survival_rows.csv"
    )
    summary_path = (
        output_directory
        / "yago_survival_build_summary.json"
    )

    write_endpoint_reference(
        path=endpoint_path,
        intervals=intervals,
    )

    total_rows, row_statistics = write_survival_rows(
        path=survival_path,
        intervals=intervals,
        train_cutoff=train_cutoff,
        valid_cutoff=valid_cutoff,
        observation_end=observation_end,
        landmark_start=landmark_start,
        landmark_end=landmark_end,
        horizons=horizons,
    )

    summary = {
        "dataset_name": "YAGO130K",
        "source_path": str(data_path),
        "train_cutoff": train_cutoff,
        "valid_cutoff": valid_cutoff,
        "observation_end": observation_end,
        "landmark_start": landmark_start,
        "landmark_end": landmark_end,
        "horizons_years": horizons,
        "entity_description_count": (
            len(entity_description_dict)
            if hasattr(entity_description_dict, "__len__")
            else None
        ),
        "deduplication": deduplication_summary,
        "survival_rows": row_statistics,
        "leakage_rule": (
            "end_year는 라벨 생성 및 감사에만 사용하며 "
            "모델 입력 feature에서 제외한다."
        ),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("YAGO130K LANDMARK SURVIVAL TABLE")
    print("=" * 72)
    print("source:", data_path)
    print("train cutoff:", train_cutoff)
    print("valid cutoff:", valid_cutoff)
    print("observation end:", observation_end)
    print(
        "landmark range:",
        f"{landmark_start}-{landmark_end}",
    )
    print("horizons:", horizons)
    print(
        "raw interval records:",
        f"{deduplication_summary['raw_interval_records']:,}",
    )
    print(
        "unique intervals:",
        f"{deduplication_summary['unique_intervals']:,}",
    )
    print(
        "duplicates removed:",
        f"{deduplication_summary['duplicate_interval_occurrences_removed']:,}",
    )
    print("landmark rows:", f"{total_rows:,}")
    print(
        "rows by split:",
        row_statistics["rows_by_time_split"],
    )
    print(
        "labels by horizon:",
        row_statistics["labels_by_horizon"],
    )
    print("\nSaved:")
    print("-", survival_path)
    print("-", endpoint_path)
    print("-", summary_path)


if __name__ == "__main__":
    main()