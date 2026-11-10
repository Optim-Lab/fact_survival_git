"""data.py against the paper's loader (legacy/datasets/interval_landmark.py): a tiny pickle, and the real FinWiki file if given."""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "legacy"))                                                # legacy package `datasets`
from datasets.interval_landmark import load_landmark_rows                            # noqa: E402
from fact_survival.data import landmark_rows, month_id, read_facts                   # noqa: E402

LEGACY_NAMES = {"object_active_degree": "object_degree", "object_relation_degree": "object_relation_degree"}


def compare(path, horizon):
    legacy, meta = load_landmark_rows(Path(path), horizon)
    facts, info = read_facts(path)
    rows = landmark_rows(facts, info["study_end"], info["study_end"] - horizon - 1)
    assert len(rows) == len(legacy)
    for column in ("subject", "relation", "object"):
        assert (rows[column].to_numpy() == legacy[column].to_numpy()).all()
    assert (rows["landmark"].to_numpy() == legacy["landmark"].map(month_id).to_numpy()).all()
    assert (rows["age"].to_numpy() == legacy["age_months"].to_numpy()).all()
    for old, new in LEGACY_NAMES.items():
        assert (rows[new].to_numpy() == legacy[old].to_numpy()).all(), new
    # subject degrees are not in the legacy loader; the paper's code counts them with a groupby on the landmark rows
    key = legacy["landmark"]
    expected = legacy.groupby([key, "subject"])["object"].transform("size")
    assert (rows["subject_degree"].to_numpy() == expected.to_numpy()).all()
    expected = legacy.groupby([key, "subject", "relation"])["object"].transform("size")
    assert (rows["subject_relation_degree"].to_numpy() == expected.to_numpy()).all()
    return rows, info


def test_tiny_pickle(tmp_path):
    facts = [("a", "r", "b", "2020-01", "2020-06"), ("c", "r", "a", "2020-02", "2020-09"),
             ("d", "s", "b", "2020-03-15", "2021-01"), ("a", "r", "e", "2020-04", "2021-02"),
             ("f", "s", "a", "2020-05", "2021-03")]
    path = tmp_path / "tiny.pkl"
    path.write_bytes(pickle.dumps(["2020-06", "2020-12", facts[:3], facts[3:4], facts[4:], None]))
    rows, info = compare(path, 3)
    first = rows[rows["landmark"] == month_id("2020-04")]
    # at 2020-04 the active facts are a-r-b, c-r-a, d-s-b (begins 2020-03-15 -> 2020-04) and a-r-e
    assert len(first) == 4
    assert first.set_index("object").loc["a", "object_degree"] == 3               # 'a' touches c-r-a, a-r-b, a-r-e (either side)
    assert first.set_index("object").loc["a", "object_relation_degree"] == 3      # all three have relation r


@pytest.mark.skipif("FINWIKI_PKL" not in os.environ, reason="set FINWIKI_PKL to the raw FinWiki file to compare on real data")
def test_real_finwiki():
    rows, info = compare(os.environ["FINWIKI_PKL"], 3)
    assert rows["landmark"].nunique() > 30 and len(rows) > 500_000
