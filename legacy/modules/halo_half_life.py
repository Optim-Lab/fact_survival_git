"""Leakage-safe HALO-style half-life baseline for landmark survival tasks.

This is an adaptation of HALO's update-interval decay, not a reproduction of
HALO's neural attention/GCN classifier.  All update statistics are computed
strictly as of each row's landmark.
"""
#%%
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


GroupKey = Tuple[object, object]


@dataclass
class HalfLifeSnapshot:
    """Half-life estimates available at one landmark/cutoff."""

    group_half_life: Dict[GroupKey, float]
    relation_half_life: Dict[object, float]
    global_half_life: float
    positive_gap_count: int


@dataclass
class HaloHalfLifeBaseline:
    """HALO-style conditional survival probability baseline.

    Parameters use dataset-native integer time units: years for YAGO, months
    for FinWiki, days for a daily dataset.  ``horizon`` must use the same unit.

    Raw prediction:
        p(T > c + H | T > c) = 2 ** (-H / half_life(c))

    The optional Platt calibration is fitted only on supplied training labels.
    """

    subject_col: str = "subject"
    relation_col: str = "relation"
    object_col: str = "object"
    begin_col: str = "begin_time"
    landmark_col: str = "landmark_time"
    min_group_gaps: int = 2
    half_life_multiplier: float = 0.5
    calibration_c: float = 1.0
    probability_epsilon: float = 1e-6

    calibrator_: Optional[LogisticRegression] = field(init=False, default=None)
    calibration_constant_: Optional[float] = field(init=False, default=None)
    calibration_horizon_: Optional[float] = field(init=False, default=None)
    diagnostics_: Dict[str, float] = field(init=False, default_factory=dict)
    _snapshot_cache: Dict[object, HalfLifeSnapshot] = field(
        init=False, default_factory=dict
    )

    def _required_history_columns(self) -> Sequence[str]:
        return (
            self.subject_col,
            self.relation_col,
            self.object_col,
            self.begin_col,
        )

    @staticmethod
    def _check_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing columns: {missing}")

    def clear_cache(self) -> None:
        self._snapshot_cache.clear()

    def prepare_history(self, history: pd.DataFrame) -> pd.DataFrame:
        """Return unique observed starts; endpoint columns are never used."""
        self._check_columns(history, self._required_history_columns(), "history")
        out = history[list(self._required_history_columns())].copy()
        out = out.dropna(subset=list(self._required_history_columns()))
        out = out.drop_duplicates()
        out = out.sort_values(
            [self.subject_col, self.relation_col, self.begin_col, self.object_col],
            kind="stable",
        )
        return out.reset_index(drop=True)

    def fit_half_lives_as_of(
        self, history: pd.DataFrame, cutoff: object
    ) -> HalfLifeSnapshot:
        """Estimate hierarchical half-lives using starts observed by ``cutoff``."""

        if cutoff in self._snapshot_cache:
            return self._snapshot_cache[cutoff]

        hist = self.prepare_history(history)
        hist = hist.loc[hist[self.begin_col] <= cutoff].copy()
        if hist.empty:
            raise ValueError(f"No history is available at cutoff={cutoff!r}")

        # Multiple objects beginning at the same instant represent one update
        # time, not several zero-length updates.
        starts = hist[
            [self.subject_col, self.relation_col, self.begin_col]
        ].drop_duplicates()
        starts = starts.sort_values(
            [self.subject_col, self.relation_col, self.begin_col], kind="stable"
        )

        starts["_gap"] = starts.groupby(
            [self.subject_col, self.relation_col], sort=False
        )[self.begin_col].diff()

        gaps = starts.loc[starts["_gap"] > 0].copy()

        if gaps.empty:
            # One native time unit is the conservative no-update fallback;
            # HALO's multiplier then gives a 0.5-unit half-life.
            global_gap = 1.0
        else:
            global_gap = float(gaps["_gap"].median())

        gap_count = gaps.groupby(
            [self.subject_col, self.relation_col], sort=False
        )["_gap"].size()
        gap_mean = gaps.groupby(
            [self.subject_col, self.relation_col], sort=False
        )["_gap"].mean()
        eligible = gap_mean.loc[gap_count >= self.min_group_gaps]

        group_half_life = {
            tuple(key): max(
                self.probability_epsilon,
                self.half_life_multiplier * float(value),
            )
            for key, value in eligible.items()
        }

        relation_gap = gaps.groupby(self.relation_col, sort=False)["_gap"].median()
        relation_half_life = {
            relation: max(
                self.probability_epsilon,
                self.half_life_multiplier * float(value),
            )
            for relation, value in relation_gap.items()
        }
        global_half_life = max(
            self.probability_epsilon,
            self.half_life_multiplier * global_gap,
        )

        snapshot = HalfLifeSnapshot(
            group_half_life=group_half_life,
            relation_half_life=relation_half_life,
            global_half_life=global_half_life,
            positive_gap_count=len(gaps),
        )
        self._snapshot_cache[cutoff] = snapshot
        return snapshot

    def _half_life_for_rows(
        self, rows: pd.DataFrame, snapshot: HalfLifeSnapshot
    ) -> Tuple[np.ndarray, np.ndarray]:
        values = np.empty(len(rows), dtype=float)
        sources = np.empty(len(rows), dtype=object)

        subjects = rows[self.subject_col].to_numpy()
        relations = rows[self.relation_col].to_numpy()
        for index, (subject, relation) in enumerate(zip(subjects, relations)):
            key = (subject, relation)
            if key in snapshot.group_half_life:
                values[index] = snapshot.group_half_life[key]
                sources[index] = "subject_relation"
            elif relation in snapshot.relation_half_life:
                values[index] = snapshot.relation_half_life[relation]
                sources[index] = "relation"
            else:
                values[index] = snapshot.global_half_life
                sources[index] = "global"
        return values, sources

    def raw_predict(
        self,
        rows: pd.DataFrame,
        history: pd.DataFrame,
        horizon: float,
        *,
        return_details: bool = False,
    ):
        """Predict each row with statistics available at its own landmark."""
        self._check_columns(
            rows,
            [self.subject_col, self.relation_col, self.landmark_col],
            "rows",
        )
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if rows.empty:
            empty = np.array([], dtype=float)
            if return_details:
                return pd.DataFrame(
                    {"half_life": empty, "half_life_source": [], "pred_halo_raw": empty},
                    index=rows.index,
                )
            return empty

        prediction = pd.Series(index=rows.index, dtype=float)
        half_life = pd.Series(index=rows.index, dtype=float)
        source = pd.Series(index=rows.index, dtype=object)

        for landmark, index in rows.groupby(self.landmark_col, sort=True).groups.items():
            subset = rows.loc[index]
            snapshot = self.fit_half_lives_as_of(history, landmark)
            tau, tau_source = self._half_life_for_rows(subset, snapshot)
            raw = np.exp(-np.log(2.0) * float(horizon) / tau)
            raw = np.clip(
                raw, self.probability_epsilon, 1.0 - self.probability_epsilon
            )
            prediction.loc[index] = raw
            half_life.loc[index] = tau
            source.loc[index] = tau_source

        if return_details:
            return pd.DataFrame(
                {
                    "half_life": half_life,
                    "half_life_source": source,
                    "pred_halo_raw": prediction,
                },
                index=rows.index,
            )
        return prediction.to_numpy(dtype=float)

    def fit_calibrator(
        self,
        train_rows: pd.DataFrame,
        history: pd.DataFrame,
        horizon: float,
        label_col: str,
    ) -> "HaloHalfLifeBaseline":
        """Fit one-feature Platt scaling on training rows only."""
        self._check_columns(train_rows, [label_col], "train_rows")
        observed = train_rows.loc[train_rows[label_col].notna()].copy()
        if observed.empty:
            raise ValueError("No observed training labels are available")

        y = observed[label_col].astype(int).to_numpy()
        raw = self.raw_predict(observed, history, horizon)
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)

        self.calibration_horizon_ = float(horizon)
        self.calibrator_ = None
        self.calibration_constant_ = None

        if np.unique(y).size < 2:
            self.calibration_constant_ = float(y.mean())
            slope = 0.0
            intercept = float("nan")
        else:
            model = LogisticRegression(
                C=self.calibration_c,
                solver="lbfgs",
                max_iter=1000,
                random_state=0,
            )
            model.fit(logit, y)
            self.calibrator_ = model
            slope = float(model.coef_[0, 0])
            intercept = float(model.intercept_[0])

        self.diagnostics_ = {
            "calibration_rows": float(len(observed)),
            "calibration_event_rate": float(1.0 - y.mean()),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "raw_brier_train": float(np.mean((y - raw) ** 2)),
        }
        calibrated = self._apply_calibration(raw)
        self.diagnostics_["calibrated_brier_train"] = float(
            np.mean((y - calibrated) ** 2)
        )
        return self

    def _apply_calibration(self, raw: np.ndarray) -> np.ndarray:
        if self.calibration_constant_ is not None:
            return np.full(len(raw), self.calibration_constant_, dtype=float)
        if self.calibrator_ is None:
            raise RuntimeError("Call fit_calibrator before calibrated prediction")
        logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
        return self.calibrator_.predict_proba(logit)[:, 1]

    def predict(
        self,
        rows: pd.DataFrame,
        history: pd.DataFrame,
        horizon: float,
        *,
        calibrated: bool = True,
        return_details: bool = False,
    ):
        if calibrated and self.calibration_horizon_ != float(horizon):
            raise RuntimeError(
                "The calibrator was fitted for a different horizon; "
                "fit one calibrator per horizon."
            )
        details = self.raw_predict(rows, history, horizon, return_details=True)
        if calibrated:
            details["pred_halo"] = self._apply_calibration(
                details["pred_halo_raw"].to_numpy(dtype=float)
            )
        else:
            details["pred_halo"] = details["pred_halo_raw"]

        counts = details["half_life_source"].value_counts(normalize=True)
        self.diagnostics_.update(
            {
                "subject_relation_coverage": float(
                    counts.get("subject_relation", 0.0)
                ),
                "relation_fallback_rate": float(counts.get("relation", 0.0)),
                "global_fallback_rate": float(counts.get("global", 0.0)),
                "half_life_median": float(details["half_life"].median()),
                "half_life_min": float(details["half_life"].min()),
                "half_life_max": float(details["half_life"].max()),
            }
        )
        if return_details:
            return details
        return details["pred_halo"].to_numpy(dtype=float)


def unique_fact_history(
    survival_rows: pd.DataFrame,
    *,
    fact_id_col: str = "fact_id",
    subject_col: str = "subject",
    relation_col: str = "relation",
    object_col: str = "object",
    begin_col: str = "begin_time",
) -> pd.DataFrame:
    """Build start-only history from a landmark table with repeated facts."""
    columns = [fact_id_col, subject_col, relation_col, object_col, begin_col]
    HaloHalfLifeBaseline._check_columns(survival_rows, columns, "survival_rows")
    return survival_rows[columns].drop_duplicates(subset=[fact_id_col]).reset_index(
        drop=True
    )
