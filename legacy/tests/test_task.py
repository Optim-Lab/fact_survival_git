"""Tests for active status, fixed-horizon labels, and right censoring."""

#%%
import json
import unittest
import pickle
import tempfile
from pathlib import Path

import pandas as pd

from datasets.task import is_active, make_survival_label, mature_label_mask
from datasets.config import DATASET_CONFIGS
from datasets.interval_landmark import (
    load_landmark_rows,
    month_id,
    month_text,
    rolling_validation_months,
    select_rolling_fold,
)
from datasets.yago import load_yago_landmark_rows


#%%
class TaskTest(unittest.TestCase):
    @staticmethod
    def _write_yago_fixture(folder, rows, endpoints, observation_end):
        folder = Path(folder)
        path = folder / "yago_survival_rows.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        pd.DataFrame(endpoints).to_csv(
            folder / "yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv",
            index=False,
        )
        (folder / "yago_survival_build_summary.json").write_text(
            json.dumps({"observation_end": observation_end}),
            encoding="utf-8",
        )
        return path

    def test_icewswiki_reuses_monthly_interval_protocol(self) -> None:
        config = DATASET_CONFIGS["icewswiki"]
        for key in ("horizons", "time_unit", "validation_step_months",
                    "calendar_block_months", "logistic_c"):
            self.assertEqual(config[key], DATASET_CONFIGS["finwiki"][key])
        # Daily endpoints follow the same month rounding as FinWiki; point
        # events do not become fabricated duration observations.
        data = ["2011-12-19", "2012-09-02", {
            ("S", "R", "O", "2011-01-15", "2012-02-15"),
            ("S", "event", "O", "2011-01-15"),
            ("T", "R", "P", "2011-01-01", "2014-01-01"),
        }, set(), set(), {}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ICEWSWiki.pkl"
            path.write_bytes(pickle.dumps(data))
            rows, meta = load_landmark_rows(path, 3)
        self.assertEqual(set(rows.relation), {"R"})
        labels = rows[rows.subject.eq("S")].set_index("landmark").survived
        self.assertEqual(labels["2011-11"], 1)
        self.assertEqual(labels["2011-12"], 0)
        self.assertEqual(meta["valid_end"], "2012-09-02")
        self.assertEqual(rows.loc[rows.split.eq("test"), "landmark"].min(), "2012-10")
        self.assertTrue((rows.target_month < meta["study_end"]).all())

    def test_half_open_interval(self) -> None:
        self.assertTrue(is_active(10, 20, 10))
        self.assertTrue(is_active(10, 20, 19))
        self.assertFalse(is_active(10, 20, 20))

    def test_survival_and_termination(self) -> None:
        self.assertEqual(make_survival_label(10, 20, 12, 3, event_observed=True), 1)
        self.assertEqual(make_survival_label(10, 15, 12, 3, event_observed=True), 0)

    def test_right_censoring(self) -> None:
        self.assertIsNone(
            make_survival_label(10, 30, 25, 5, event_observed=False)
        )

    def test_event_and_censoring_boundaries(self) -> None:
        cases = (
            (14, True, 0),
            (15, True, 0),
            (16, True, 1),
            (14, False, None),
            (15, False, None),
            (16, False, 1),
        )
        for stop, event_observed, expected in cases:
            with self.subTest(stop=stop, event_observed=event_observed):
                self.assertEqual(
                    make_survival_label(
                        10, stop, 12, 3, event_observed=event_observed
                    ),
                    expected,
                )

    def test_joint_rolling_mask_uses_only_observed_mature_labels(self) -> None:
        labels = [[1, 1], [0, float("nan")], [1, 1]]
        mask = mature_label_mask(
            labels,
            landmark_times=[10, 11, 12],
            horizons=[1, 3],
            origin=13,
        )
        self.assertEqual(
            mask.tolist(),
            [[True, True], [True, False], [True, False]],
        )

    def test_rolling_fold_uses_only_mature_training_labels(self) -> None:
        rows = pd.DataFrame(
            {
                "landmark": ["2018-01", "2018-04", "2018-07", "2020-10"],
                "target_month": ["2018-04", "2018-07", "2018-10", "2021-01"],
                "survived": [1, 0, 1, 1],
            }
        )
        validation_month = month_id("2018-07")
        train, validation = select_rolling_fold(
            rows,
            validation_month=validation_month,
            development_end=month_id("2020-09"),
        )

        self.assertEqual(train["landmark"].tolist(), ["2018-01", "2018-04"])
        self.assertEqual(validation["landmark"].tolist(), ["2018-07"])
        self.assertNotIn("2020-10", pd.concat([train, validation])["landmark"].tolist())

    def test_rolling_origins_fit_inside_development_period(self) -> None:
        rows = pd.DataFrame(
            {
                "landmark": ["2018-01", "2020-09"],
                "target_month": ["2018-04", "2020-12"],
            }
        )
        origins = rolling_validation_months(
            rows,
            horizon=12,
            development_end=month_id("2020-09"),
            step_months=3,
        )

        self.assertEqual(month_text(origins[0]), "2019-01")
        self.assertEqual(month_text(origins[-1]), "2019-07")

    def test_yago_adapter_uses_shared_schema(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_yago_fixture(
                folder,
                [{
                    "fact_id": "F1", "subject": "S", "relation": "R", "object": "O",
                    "begin_year": 2000, "landmark_year": 2003, "age_years": 3,
                    "time_split": "development", "survival_label_1y": 1,
                }],
                [{"fact_id": "F1", "end_year": 2005, "boundary_censored": 0}],
                observation_end=2006,
            )
            rows, _ = load_yago_landmark_rows(path, 1)

        self.assertEqual(rows.loc[0, "landmark"], "2003-01")
        self.assertEqual(rows.loc[0, "target_month"], "2004-01")
        self.assertEqual(rows.loc[0, "age_months"], 36)

    def test_yago_degree_precedes_horizon_observation_mask(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = self._write_yago_fixture(
                folder,
                [
                    {
                        "fact_id": "F1", "subject": "S1", "relation": "R", "object": "O",
                        "begin_year": 2000, "landmark_year": 2003, "age_years": 3,
                        "time_split": "development", "survival_label_1y": 0,
                    },
                    {
                        "fact_id": "F2", "subject": "S2", "relation": "R", "object": "O",
                        "begin_year": 2001, "landmark_year": 2003, "age_years": 2,
                        "time_split": "development", "survival_label_1y": None,
                    },
                ],
                [
                    {"fact_id": "F1", "end_year": 2004, "boundary_censored": 0},
                    {"fact_id": "F2", "end_year": None, "boundary_censored": 1},
                ],
                observation_end=2004,
            )
            rows, _ = load_yago_landmark_rows(path, 1)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.loc[0, "object_active_degree"], 2)
        self.assertEqual(rows.loc[0, "object_relation_degree"], 2)


#%%
if __name__ == "__main__":
    unittest.main()
