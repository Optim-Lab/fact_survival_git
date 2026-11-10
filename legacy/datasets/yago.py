"""Adapt the prepared YAGO survival table to the shared landmark schema."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.task import PROTOCOL_VERSION, make_survival_labels


def load_yago_labelled_rows(
    data_path: Path,
    horizons: list[int],
    *,
    return_outcomes: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load rows and independently rebuild labels from the endpoint audit file.

    Endpoints exist only in the audit file. They are returned as separate
    outcome metadata only when requested, never as model features. Stored
    label columns are accepted only when they exactly match
    ``datasets.task.make_survival_labels``.
    """
    data_path = Path(data_path)
    audit_path = data_path.with_name(
        "yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv"
    )
    summary_path = data_path.with_name("yago_survival_build_summary.json")
    if not audit_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "YAGO label verification requires the endpoint audit CSV and build summary "
            f"next to {data_path}"
        )

    data = pd.read_csv(data_path)
    required = {
        "fact_id", "subject", "relation", "object", "begin_year",
        "landmark_year", "age_years", "time_split",
        *(f"survival_label_{horizon}y" for horizon in horizons),
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"YAGO data is missing columns: {missing}")
    endpoints = pd.read_csv(
        audit_path,
        usecols=["fact_id", "end_year", "boundary_censored"],
    )

    if endpoints["fact_id"].duplicated().any():
        raise ValueError("YAGO endpoint audit contains duplicate fact_id values")

    if not endpoints["boundary_censored"].isin([0, 1, False, True]).all():
        raise ValueError("YAGO boundary_censored must contain only booleans")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observation_end = int(summary["observation_end"])

    endpoint = endpoints.set_index("fact_id").reindex(data["fact_id"])

    if endpoint["boundary_censored"].isna().any():
        raise ValueError("YAGO survival row is missing from the endpoint audit")
    observed_end = endpoint["end_year"].notna() & ~endpoint[
        "boundary_censored"

    ].astype(bool)

    stop = endpoint["end_year"].where(observed_end, observation_end).to_numpy()
    begin = data["begin_year"].to_numpy()
    landmark = data["landmark_year"].to_numpy()
    labels = np.column_stack([
        make_survival_labels(
            begin,
            stop,
            landmark,
            horizon,
            event_observed=observed_end.to_numpy(),
        )
        for horizon in horizons
    ])

    for index, horizon in enumerate(horizons):
        column = f"survival_label_{horizon}y"
        stored = pd.to_numeric(data[column], errors="coerce").to_numpy(float)
        if not np.array_equal(stored, labels[:, index], equal_nan=True):
            raise AssertionError(f"YAGO stored labels differ from shared contract: {column}")
    metadata = {
        "observation_end": observation_end,
        "protocol_version": PROTOCOL_VERSION,
        "label_audit": str(audit_path),
    }
    if return_outcomes:
        metadata["stop_times"] = stop.astype(np.int64)
        metadata["event_observed"] = observed_end.to_numpy(bool)
    return data, labels, metadata


def load_yago_landmark_rows(
    data_path: Path,
    horizon_years: int,
) -> tuple[pd.DataFrame, dict]:

    label = f"survival_label_{horizon_years}y"

    data, rebuilt_labels, audit = load_yago_labelled_rows(data_path, [horizon_years])

    required = {
        "fact_id", "subject", "relation", "object", "begin_year",
        "landmark_year", "age_years", "time_split", label,
    }

    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"YAGO data is missing columns: {missing}")

    # Snapshot features use the complete active risk set. Horizon
    # observability is a label/evaluation mask applied only afterwards.
    data["object_active_degree"] = (
        data.groupby(["landmark_year", "object"])["fact_id"]
        .transform("count")
        .astype(float)
    )

    data["object_relation_degree"] = (
        data.groupby(["landmark_year", "relation", "object"])["fact_id"]
        .transform("count")
        .astype(float)
    )

    observed = np.isfinite(rebuilt_labels[:, 0])
    survived = rebuilt_labels[observed, 0].astype(int)
    data = data.loc[observed].copy()

    rows = pd.DataFrame(
        {
            "fact_id": data["fact_id"],
            "subject": data["subject"],
            "relation": data["relation"],
            "object": data["object"],
            "begin": data["begin_year"].astype(str),
            "landmark": data["landmark_year"].map(lambda year: f"{int(year):04d}-01"),
            "target_month": data["landmark_year"].map(
                lambda year: f"{int(year) + horizon_years:04d}-01"
            ),

            "age_months": data["age_years"].astype(float) * 12,
            "object_active_degree": data["object_active_degree"],
            "object_relation_degree": data["object_relation_degree"],
            "survived": survived,
            "split": data["time_split"],
        }

    ).reset_index(drop=True)

    return rows, {
        "development_start": "2003-01",
        "development_end": "2008-01",
        "study_end": f"{audit['observation_end']:04d}-01",
        "horizon": horizon_years,
        "rows": len(rows),
        "protocol_version": audit["protocol_version"],
        "label_audit": audit["label_audit"],
        "feature_order": "full current risk set -> degree -> observed-label mask",
    }
