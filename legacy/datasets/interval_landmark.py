"""Load a six-item interval TKG pickle and create monthly landmark rows."""

#%%
from __future__ import annotations

import pickle
import pickletools
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.task import PROTOCOL_VERSION, make_survival_labels

#%%
DANGEROUS_PICKLE_OPS = {
    "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST", "NEWOBJ", "NEWOBJ_EX"
}


#%%
def month_id(value: str | int) -> int:
    """Convert YYYY or YYYY-MM[-DD] into a sortable integer month."""
    if isinstance(value, (int, np.integer)):
        return int(value) * 12

    parts = str(value).split("-")
    year, month = map(int, parts[:2]) if len(parts) >= 2 else (int(parts[0]), 1)
    return year * 12 + month - 1

#%%
def end_month_id(value: str | int) -> int:
    """Move a mid-month end date to the following monthly boundary."""
    base = month_id(value)
    parts = str(value).split("-")
    return base + 1 if len(parts) >= 3 and int(parts[2]) > 1 else base


#%%
def month_text(value: int) -> str:
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


#%%
def safe_load_dataset(path: Path) -> list:
    """Load an expected non-executable six-item interval pickle container."""
    if not path.exists():
        raise FileNotFoundError(f"Interval dataset not found: {path}")
    with path.open("rb") as handle:
        bad = [op.name for op, _, _ in pickletools.genops(handle) if op.name in DANGEROUS_PICKLE_OPS]
    if bad:
        raise ValueError(f"unsafe pickle opcode detected: {bad[:3]}")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError("expected a six-item interval dataset container")
    return data


#%%
def unique_spans(train_set: list, valid_set: list, test_set: list) -> pd.DataFrame:
    """Return one row per unique five-field interval fact."""
    facts = set()
    for split in (train_set, valid_set, test_set):
        facts.update(tuple(fact) for fact in split if len(fact) == 5)
    frame = pd.DataFrame(facts, columns=["subject", "relation", "object", "begin", "end"])
    frame["begin_m"] = frame["begin"].map(month_id)
    frame["end_m"] = frame["end"].map(end_month_id)
    if (frame["end_m"] <= frame["begin_m"]).any():
        raise ValueError("found an interval with end <= begin")
    return frame.sort_values(
        ["begin_m", "end_m", "relation", "subject", "object"]
    ).reset_index(drop=True)


#%%
def incident_counts(active_graph: pd.DataFrame) -> dict[str, int]:
    """Count all active edges touching each entity, regardless of direction."""
    entities = pd.concat([active_graph["subject"], active_graph["object"]], ignore_index=True)
    return entities.value_counts().astype(int).to_dict()


#%%
def entity_relation_counts(active_graph: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Count active edges of each relation touching each entity."""
    subject_side = active_graph[["subject", "relation"]].rename(columns={"subject": "entity"})
    object_side = active_graph[["object", "relation"]].rename(columns={"object": "entity"})
    incident = pd.concat([subject_side, object_side], ignore_index=True)
    return incident.groupby(["entity", "relation"]).size().astype(int).to_dict()


#%%
#landmark 만드는
def build_landmark_rows(
    spans: pd.DataFrame,
    horizon: int,
    train_end: int,
    valid_end: int,
    study_end: int,
) -> pd.DataFrame:
    """Create leakage-safe monthly rows with X:(N,F) and y:(N,)."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    frames: list[pd.DataFrame] = []
    first_month = int(spans["begin_m"].min())
    last_landmark = study_end - horizon - 1

    for landmark in range(first_month, last_landmark + 1):
        active = spans[(spans["begin_m"] <= landmark) & (spans["end_m"] > landmark)]
        if active.empty:
            continue

        degree = incident_counts(active)
        relation_degree = entity_relation_counts(active)
        target_month = landmark + horizon

        if landmark <= train_end:
            split = "train" if target_month <= train_end else "embargo"
        elif landmark <= valid_end:
            split = "valid"
        else:
            split = "test"
        # The loop bounds already guarantee target < study_end. Build one
        # columnar frame per month instead of millions of Python row dicts.
        frame = active[["subject", "relation", "object", "begin", "end"]].copy()
        frame["landmark"] = month_text(landmark)
        frame["target_month"] = month_text(target_month)
        frame["age_months"] = landmark - active["begin_m"]
        frame["object_active_degree"] = active["object"].map(degree)
        frame["object_relation_degree"] = [
            relation_degree[(obj, rel)] for obj, rel in zip(active.object, active.relation)
        ]
        frame["survived"] = make_survival_labels(
            active["begin_m"].to_numpy(),
            active["end_m"].to_numpy(),
            landmark,
            horizon,
            event_observed=active["end_m"].to_numpy() < study_end,
        ).astype(int)
        frame["split"] = split
        frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#%%
def rolling_validation_months(
    rows: pd.DataFrame,
    horizon: int,
    development_end: int,
    step_months: int = 3,
    development_start: int | None = None,
) -> list[int]:
    """Return leakage-safe rolling origins before the sealed test period."""
    if horizon <= 0 or step_months <= 0:
        raise ValueError("horizon and step_months must be positive")

    first_landmark = int(rows["landmark"].map(month_id).min())
    first_validation = max(
        first_landmark + horizon,
        development_start if development_start is not None else first_landmark,
    )
    last_validation = development_end - horizon
    if first_validation > last_validation:
        return []
    return list(range(first_validation, last_validation + 1, step_months))


#%%
def select_rolling_fold(
    rows: pd.DataFrame,
    validation_month: int,
    development_end: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one rolling-origin fold without using the sealed test period."""
    landmark = rows["landmark"].map({v: month_id(v) for v in rows["landmark"].unique()})
    target = rows["target_month"].map({v: month_id(v) for v in rows["target_month"].unique()})

    train = rows[(landmark < validation_month) & (target <= validation_month)].copy()
    validation = rows[
        (landmark == validation_month) & (target <= development_end)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError(
            f"empty rolling fold at validation={month_text(validation_month)}"
        )
    if target.loc[train.index].max() > validation_month:#전체 target 중 train에 들어간 행들의 값만 선택
        raise AssertionError("future labels entered rolling training data")
    #검증 행들의 landmark가 정확히 한 종류인지 확인해. nunique()는 서로 다른 값의 개수야.
    if landmark.loc[validation.index].nunique() != 1:
        raise AssertionError("validation fold contains multiple origins")


    return train, validation


#%%
def load_landmark_rows(data_path: Path, horizon: int) -> tuple[pd.DataFrame, dict]:
    """Load an interval pickle and return validated landmark rows plus metadata."""
    train_end_text, valid_end_text, train_set, valid_set, test_set, _ = safe_load_dataset(data_path)
    spans = unique_spans(train_set, valid_set, test_set)
    study_end = int(spans["end_m"].max())
    rows = build_landmark_rows(
        spans,
        horizon,
        month_id(train_end_text),
        month_id(valid_end_text),
        study_end,
    )
    metadata = {
        "train_end": train_end_text,
        "valid_end": valid_end_text,
        "study_end": month_text(study_end),
        "horizon": horizon,
        "rows": len(rows),
        "protocol_version": PROTOCOL_VERSION,
        "feature_order": "full current risk set -> degree -> observed-label mask",
    }
    return rows, metadata
