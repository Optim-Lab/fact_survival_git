"""The two scores of the paper's tables: Brier (lower is better) and AP (higher is better), for one horizon at a time.

    scores(survival, labels, origin, horizons)  ->  one dict per horizon: rows, event_rate, brier, ap

    survival  (N, K)  S-hat_c(h_k | c), the predicted probability that the fact is still valid at c + h_k
    labels    (N, K)  y_{c,k}: 1 alive, 0 ended, NaN unknown   (unknown cells are never scored)
    origin    (N,)    the origin each row was predicted at

    Brier  = mean over the scored cells of (S-hat - y)^2                                           (tables: x 10^3)
    AP     = average precision for finding the facts that end (event = 1 - y, score = 1 - S-hat); computed inside each
             origin and averaged over the origins, weighted by their number of scored rows           (tables: x 100)
"""
#%%
import numpy as np
from sklearn.metrics import average_precision_score


def brier(p, y):
    known = ~np.isnan(y)                                           # only cells with a known label
    return float(np.mean((p[known] - y[known]) ** 2))


def average_precision(p, y, origin):
    known = ~np.isnan(y)
    score, event, origin = 1 - p[known], 1 - y[known], np.asarray(origin)[known]     # a high score = "this fact will end"
    total, weight = 0.0, 0
   
    for o in np.unique(origin):
   
        at = origin == o
   
        if 0 < event[at].sum() < at.sum():                         # AP needs both ended and surviving facts at this origin
            total, weight = total + at.sum() * average_precision_score(event[at], score[at]), weight + at.sum()
    return total / weight if weight else float("nan")


def scores(survival, labels, origin, horizons):

    return [dict(horizon=h, rows=int((~np.isnan(labels[:, k])).sum()),

                 event_rate=float(1 - np.nanmean(labels[:, k])),

                 brier=brier(survival[:, k], labels[:, k]),

                 ap=average_precision(survival[:, k], labels[:, k], origin))

            for k, h in enumerate(horizons)]
