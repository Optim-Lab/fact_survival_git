"""Three probability models required for the primary Stage 1 claim."""

#%%
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


#%%
AGE_FEATURES = ["relation", "age_months"]
OBJECT_FEATURES = [
    "relation",
    "age_months",
    "object_active_degree",
    "object_relation_degree",
]


#%%
def build_logistic(numeric_features: list[str], c_value: float) -> Pipeline:
    """Create one-hot relation + standardized numeric logistic regression."""
    features = ColumnTransformer(
        [
            ("relation", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["relation"]),
            (
                "numeric",
                Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                numeric_features,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    max_iter=500,
                    solver="liblinear",
                    random_state=42,
                ),
            ),
        ]
    )


#%%
def fit_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    c_value: float,
) -> pd.DataFrame:
    """Fit on train rows and append three aligned probability vectors (N,)."""
    output = test.copy()
    output["p_global"] = np.full(len(test), float(train["survived"].mean()))

    age_model = build_logistic(["age_months"], c_value)
    age_model.fit(train[AGE_FEATURES], train["survived"])
    output["p_relation_age"] = age_model.predict_proba(test[AGE_FEATURES])[:, 1]

    object_model = build_logistic(
        ["age_months", "object_active_degree", "object_relation_degree"], c_value
    )
    object_model.fit(train[OBJECT_FEATURES], train["survived"])
    output["p_object_degree"] = object_model.predict_proba(test[OBJECT_FEATURES])[:, 1]
    return output
