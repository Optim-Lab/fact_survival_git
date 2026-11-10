"""Verify that a full local run reproduces the frozen primary metrics."""

#%%
from pathlib import Path

import pandas as pd


#%%
RESULT_PATH = Path(__file__).resolve().parent / "results" / "primary_reproduction"
EXPECTED_FACT_BRIER = {
    (3, "global"): 0.0226240113,
    (3, "relation_age"): 0.0218335541,
    (3, "object_degree"): 0.0217191532,
    (6, "global"): 0.0318046661,
    (6, "relation_age"): 0.0295969648,
    (6, "object_degree"): 0.0292837910,
    (12, "global"): 0.0420795899,
    (12, "relation_age"): 0.0371331931,
    (12, "object_degree"): 0.0365345701,
}


#%%
def main() -> None:
    metrics_path = RESULT_PATH / "metrics.csv"
    significance_path = RESULT_PATH / "significance.csv"
    if not metrics_path.exists() or not significance_path.exists():
        raise FileNotFoundError("run the full main.py command before verification")

    metrics = pd.read_csv(metrics_path)
    significance = pd.read_csv(significance_path)
    for key, expected in EXPECTED_FACT_BRIER.items():
        horizon, model = key
        selected = metrics[(metrics["horizon"] == horizon) & (metrics["model"] == model)]
        if len(selected) != 1:
            raise AssertionError(f"missing metric row: {key}")
        actual = float(selected.iloc[0]["fact_brier"])
        if abs(actual - expected) > 1e-8:
            raise AssertionError(f"fact Brier mismatch {key}: {actual} != {expected}")

    if len(significance) != 6:
        raise AssertionError("expected two hypotheses for each of three horizons")
    if not significance["significant_after_holm"].astype(bool).all():
        raise AssertionError("at least one frozen primary comparison is not significant")
    print("REPRODUCTION OK")
    print(f"metrics: {metrics_path}")
    print(f"significance: {significance_path}")


#%%
if __name__ == "__main__":
    main()