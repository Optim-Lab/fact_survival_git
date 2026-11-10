"""The three classical survival baselines of the paper's tables: Kaplan-Meier, Cox, discrete-time hazard.

    p = classical_predictions(train_rows, test_rows, origin, horizon, cox_penalizer=, cox_min_relation_rows=, discrete_c=)
    p["km"], p["cox"], p["discrete_hazard"]       each (n_test,)   S-hat_c(h | c) at ONE horizon h

    train_rows  DataFrame, the rows whose label at horizon h is known at `origin`; columns relation, age, landmark, stop, ended
    test_rows   DataFrame, the rows to predict (landmark == origin); columns relation, age

What the models see: only the relation and the age at the landmark. They are refitted at every origin and every horizon.
Follow-up is cut at what is observable at the origin:   duration = min(T - c, origin - c, h),   event = ended and T <= origin and T - c <= h.

    km               Kaplan-Meier curve of the durations (the same number for every fact)
    cox              Cox model on [age (standardised), relation dummies]; relations with < cox_min_relation_rows rows share "__OTHER__"
    discrete_hazard  logistic regression for "ends in month s" on [relation, age + s - 1] over person-months;
                     S-hat(h) = product over s = 1..h of (1 - hazard_s)
"""
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def follow_up(train_rows, origin, horizon):
    """Duration and event indicator of every training fact, as far as they can be seen at `origin`."""
    c, stop = train_rows["landmark"].to_numpy(), train_rows["stop"].to_numpy()
    duration = np.minimum.reduce((stop - c, origin - c, np.full(len(c), horizon)))
    event = train_rows["ended"].to_numpy(bool) & (stop <= origin) & (stop - c <= horizon)
    return duration.astype(np.int64), event


def cox_survival(train_rows, duration, event, test_rows, horizon, penalizer, min_relation_rows):
    age_mean, age_std = train_rows["age"].mean(), train_rows["age"].std(ddof=0) or 1.0     # age standardised on the training rows
    counts = train_rows["relation"].value_counts()
    kept = set(counts[counts >= min_relation_rows].index.astype(str))                    # rare relations -> "__OTHER__"

    def design(rows, columns=None):
        relation = rows["relation"].astype(str).where(rows["relation"].astype(str).isin(kept), "__OTHER__")
        dummies = pd.get_dummies(relation, prefix="relation", dtype=float)
        if columns is None:
            columns = sorted(dummies.columns)[1:]                                           # drop one reference relation
        return pd.concat([pd.DataFrame({"age_z": (rows["age"].astype(float) - age_mean) / age_std}),
                          dummies.reindex(columns=columns, fill_value=0.0)], axis=1), columns

    x, columns = design(train_rows)
    cox = CoxPHFitter(penalizer=penalizer).fit(x.assign(duration=duration, event=event.astype(int)),
                                               duration_col="duration", event_col="event")
    out = []
    for start in range(0, len(test_rows), 50_000):                                          # chunks keep the memory small
        x_test, _ = design(test_rows.iloc[start:start + 50_000], columns)
        out.append(cox.predict_survival_function(x_test, times=[float(horizon)]).iloc[0].to_numpy(float))
    return np.concatenate(out)


def discrete_hazard_survival(train_rows, duration, event, test_rows, horizon, c):
    # person-months: a fact with duration d gives d rows (month s = 1..d); the event is marked in its last month
    repeats = np.repeat(np.arange(len(duration)), duration)
    month = np.arange(len(repeats)) - np.repeat(np.cumsum(duration) - duration, duration) + 1
    order = np.argsort(month, kind="stable")                  # month by month, like the paper's code (the solver is order-sensitive)
    repeats, month = repeats[order], month[order]
    people = pd.DataFrame({"relation": train_rows["relation"].to_numpy()[repeats],
                           "attained_age": train_rows["age"].to_numpy(float)[repeats] + month - 1})
    ended_now = event[repeats] & (month == duration[repeats])
    model = Pipeline([
        ("features", ColumnTransformer([("relation", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["relation"]),
                                        ("age", StandardScaler(), ["attained_age"])])),
        ("model", LogisticRegression(C=c, max_iter=500, solver="liblinear", random_state=42))]).fit(people, ended_now.astype(int))
    survival = np.ones(len(test_rows))
    for s in range(1, horizon + 1):                                                         # S(h) = product of (1 - hazard_s)
        month_s = pd.DataFrame({"relation": test_rows["relation"].to_numpy(),
                                "attained_age": test_rows["age"].to_numpy(float) + s - 1})
        survival *= 1.0 - model.predict_proba(month_s)[:, 1]
    return survival


def classical_predictions(train_rows, test_rows, origin, horizon, *, cox_penalizer, cox_min_relation_rows, discrete_c):
    duration, event = follow_up(train_rows, origin, horizon)
    km = KaplanMeierFitter().fit(duration, event_observed=event)
    return {"km": np.full(len(test_rows), float(km.predict(horizon))),
            "cox": cox_survival(train_rows, duration, event, test_rows, horizon, cox_penalizer, cox_min_relation_rows),
            "discrete_hazard": discrete_hazard_survival(train_rows, duration, event, test_rows, horizon, discrete_c)}
