"""The three simple reference models of the paper's tables, trained like the proposed model (same rows, same labels).

    p = baseline_predictions(train_rows, train_labels, test_rows, c)          # {"global": (n, K), "relation_age": ..., "object_degree": ...}

    train_rows, test_rows  DataFrames with the columns  relation, age, object_degree, object_relation_degree   (from data.py)
    train_labels           (n_train, K)  y_{c,k}: 1 / 0 / NaN (NaN cells are not used)
    c                      inverse regularisation strength of the logistic regression (a setting of the dataset)

    global         the survival rate of the training labels at each horizon (the same number for every fact)
    relation_age   logistic regression on [relation (one-hot), age]
    object_degree  logistic regression on [relation (one-hot), age, object degree, object-relation degree]
One model per horizon h_k, fitted on the cells whose label is known.
"""
#%%
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = {"relation_age": ["age"], "object_degree": ["age", "object_degree", "object_relation_degree"]}


def logistic(numeric, c):
    """One-hot relation (rare relations share one bucket, unseen ones are ignored) + standardised numbers -> logistic regression."""
    prepare = ColumnTransformer([
        ("relation", OneHotEncoder(handle_unknown="ignore", min_frequency=5), ["relation"]),
        ("numbers", Pipeline([("fill", SimpleImputer()), ("scale", StandardScaler())]), numeric)])
    return Pipeline([("prepare", prepare),
                     ("model", LogisticRegression(C=c, max_iter=500, solver="liblinear", random_state=42))])


def baseline_predictions(train_rows, train_labels, test_rows, c):
    train_labels = np.asarray(train_labels)
    out = {name: np.zeros((len(test_rows), train_labels.shape[1])) for name in ("global", *NUMERIC)}
    for k in range(train_labels.shape[1]):                                  # one model per horizon
        known = ~np.isnan(train_labels[:, k])
        y = train_labels[known, k].astype(int)
        
        out["global"][:, k] = y.mean()
        for name, numeric in NUMERIC.items():
            columns = ["relation", *numeric]
            model = logistic(numeric, c).fit(train_rows.iloc[np.flatnonzero(known)][columns], y)
            out[name][:, k] = model.predict_proba(test_rows[columns])[:, 1]
    return out
