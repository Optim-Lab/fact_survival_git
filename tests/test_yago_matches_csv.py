"""data.py on the YAGO pickle must reproduce the YAGO table that the paper's code built (needs YAGO_PKL and YAGO_CSV)."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fact_survival.data import landmark_rows, read_facts                                # noqa: E402
from fact_survival.labels import survival_labels                                        # noqa: E402


@pytest.mark.skipif("YAGO_PKL" not in os.environ or "YAGO_CSV" not in os.environ, reason="set YAGO_PKL and YAGO_CSV")
def test_yago_rows_and_labels_match_the_paper_table():
    facts, info = read_facts(os.environ["YAGO_PKL"], unit="year")
    rows = landmark_rows(facts, info["study_end"], last_landmark=2022, first_landmark=1992, object_degree="incoming")
    labels = survival_labels(rows.stop, rows.ended, rows.landmark, [1, 2, 3])
    key = ["landmark", "subject", "relation", "object", "begin"]
    new = rows[key + ["age"]].assign(y1=labels[:, 0], y2=labels[:, 1], y3=labels[:, 2])
    old = pd.read_csv(os.environ["YAGO_CSV"]).rename(columns={
        "begin_year": "begin", "landmark_year": "landmark", "age_years": "age",
        "survival_label_1y": "y1", "survival_label_2y": "y2", "survival_label_3y": "y3"})[key + ["age", "y1", "y2", "y3"]]
    assert len(new) == len(old)
    unique_new, unique_old = new[~new.duplicated(key, keep=False)], old[~old.duplicated(key, keep=False)]
    merged = unique_new.merge(unique_old, on=key, suffixes=("_new", "_old"))
    assert len(merged) == len(unique_new) == len(unique_old)                     # same rows
    for column in ("age", "y1", "y2", "y3"):                                     # same age and labels (NaN == NaN)
        np.testing.assert_array_equal(merged[column + "_new"].to_numpy(), merged[column + "_old"].to_numpy())
