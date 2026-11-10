"""Row-level and fact-equal Brier scores."""

#%%
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


#%%
FACT_COLUMNS = ["subject", "relation", "object", "begin", "end"]


def fact_key(rows: pd.DataFrame):
    """Return the stable fact identifier available in a dataset."""
    if "fact_id" in rows:
        return rows["fact_id"]
    return pd.MultiIndex.from_frame(rows[FACT_COLUMNS])


#%%
# def summarize_probability(rows: pd.DataFrame, probability_column: str) -> dict:
#     """Summarize one survival-probability vector with shape (N,)."""
#     loss = (rows[probability_column] - rows["survived"]) ** 2
#     fact_brier = loss.groupby(fact_key(rows), sort=False).mean().mean()
#     return {
#         "model": probability_column.removeprefix("p_"),
#         "rows": len(rows),
#         "facts": int(pd.Index(fact_key(rows)).nunique()),
#         "survival_rate": float(rows["survived"].mean()),
#         "row_brier": float(brier_score_loss(rows["survived"], rows[probability_column])),
#         "fact_brier": float(fact_brier),
#     }
def summarize_probability(rows: pd.DataFrame, probability_column: str) -> dict:
    """Fixed-horizon metrics on rows with an observed label."""
    survived = rows["survived"].to_numpy(dtype=int)
    p_survival = rows[probability_column].to_numpy(dtype=float)
    ended = 1 - survived
    p_end = 1 - p_survival

    loss = (rows[probability_column] - rows["survived"]) ** 2
    fact_brier = loss.groupby(fact_key(rows), sort=False).mean().mean()

    has_both_classes = len(set(ended)) == 2
    eligible = []
    if "origin" in rows:
        for _, group in rows.groupby("origin", sort=False):
            events = 1 - group["survived"].to_numpy(dtype=int)
            if len(set(events)) == 2:
                score = 1 - group[probability_column].to_numpy(dtype=float)
                eligible.append((len(group), average_precision_score(events, score)))
    origin_ap = (sum(n * ap for n, ap in eligible) / sum(n for n, _ in eligible)
                 if eligible else float("nan"))
    return {
        "model": probability_column.removeprefix("p_"),
        "rows": len(rows),
        "facts": int(pd.Index(fact_key(rows)).nunique()),
        "survival_rate": float(survived.mean()),
        "event_rate": float(ended.mean()),
        "row_brier": float(brier_score_loss(survived, p_survival)),
        "fact_brier": float(fact_brier),
        "event_ap": float(average_precision_score(ended, p_end))
                    if has_both_classes else float("nan"),
        "event_ap_within_origin": float(origin_ap),
        "ap_eligible_rows": sum(n for n, _ in eligible),
        "event_auroc": float(roc_auc_score(ended, p_end))
                       if has_both_classes else float("nan"),

        "calibration_gap": float(p_end.mean() - ended.mean()),
    }