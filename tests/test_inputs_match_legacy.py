"""The inputs of the model (rows, features x_c, half-life tau_c, labels y_{c,k}, development origins) must equal the ones that the
paper's code builds: main.prepare() against legacy/modules/proposed_experiment.py::load_data().

Set the data paths you have, the tests of the others are skipped:
    FINWIKI_PKL  ICEWSWIKI_PKL  FINSURVKG_PKL      the raw pickles (new code reads them, and so does the paper's code)
    YAGO_PKL + YAGO_CSV                            raw pickle (new code) and the processed table (the paper's code reads this one)

The two sides list the rows in different orders, so both are sorted by (subject, relation, object, landmark, all features, all labels)
before they are compared; duplicated facts then line up too.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load_by_path("new_main", ROOT / "main.py")
sys.path.insert(0, str(ROOT / "legacy"))                                  # the paper's code imports `datasets.*`
legacy = load_by_path("legacy_proposed_experiment", ROOT / "legacy/modules/proposed_experiment.py")

CASES = {   # name: (config file, environment variable of the file the NEW code reads, file the PAPER's code reads (None = same))
    "finwiki": ("configs/finwiki.json", "FINWIKI_PKL", None),
    "icewswiki": ("configs/icewswiki.json", "ICEWSWIKI_PKL", None),
    "finsurvkg": ("configs/finsurvkg.json", "FINSURVKG_PKL", None),
    "yago": ("configs/yago.json", "YAGO_PKL", "YAGO_CSV"),
}


def order_of(subject, relation, object_, landmark, integers, labels):
    """Row order that depends only on identity columns (ids, landmark, age and the four degrees, labels), never on float noise."""
    table = pd.DataFrame({"subject": subject, "relation": relation, "object": object_, "landmark": landmark})
    for k in range(integers.shape[1]):
        table[f"i{k}"] = integers[:, k]
    for k in range(labels.shape[1]):
        table[f"y{k}"] = np.nan_to_num(labels[:, k], nan=-1.0)
    return table.sort_values(list(table.columns), kind="stable").index.to_numpy()


@pytest.mark.parametrize("name", CASES)
def test_inputs_match_paper_code(name):
    config_file, variable, legacy_variable = CASES[name]
    if variable not in os.environ or (legacy_variable and legacy_variable not in os.environ):
        pytest.skip(f"set {variable}" + (f" and {legacy_variable}" if legacy_variable else ""))
    config = json.load(open(ROOT / config_file))
    rows, x, tau, labels, info = new.prepare(config, os.environ[variable])
    old_rows, old_x, _, old_labels, old_tau, old_origins, old_valid_end = legacy.load_data(
        name, Path(os.environ[legacy_variable or variable]), config["horizons"])

    assert len(rows) == len(old_rows)
    assert info["valid_end"] == old_valid_end
    assert new.development_origins(config, rows, info) == {int(h): list(v) for h, v in old_origins.items()}

    new_integers = rows[["age", "object_degree", "object_relation_degree", "subject_degree", "subject_relation_degree"]].to_numpy()
    # The paper's code returns only identity columns, so age and the four degrees are recovered from its features
    # (columns 0 = log(1 + age), 5-8 = log(1 + degree)): integers, so the rounding is exact.
    old_integers = np.rint(np.expm1(old_x[:, [0, 5, 6, 7, 8]].astype(np.float64))).astype(np.int64)
    a = order_of(rows.subject.to_numpy(), rows.relation.to_numpy(), rows.object.to_numpy(), rows.landmark.to_numpy(), new_integers, labels)
    b = order_of(old_rows.subject.to_numpy(), old_rows.relation.to_numpy(), old_rows.object.to_numpy(), old_rows.origin.to_numpy(), old_integers, old_labels)

    np.testing.assert_array_equal(new_integers[a], old_integers[b])                       # age and the four degrees
    np.testing.assert_array_equal(np.nan_to_num(labels[a], nan=-1), np.nan_to_num(old_labels[b], nan=-1))
    np.testing.assert_allclose(x[a], old_x[b], atol=1e-6, rtol=0)                          # features x_c (float32 rounding only)
    np.testing.assert_allclose(tau[a], old_tau[b], atol=1e-6, rtol=1e-6)                   # half-life tau_c
