"""Half-life tau_c of a fact (Appendix "Half-Life Estimation", idea of HALO): how often is its (subject, relation) updated?

    tau, level = half_lives(subject, relation, landmark, history_subject, history_relation, history_begin)

    rows      subject, relation, landmark (N,)     history   subject, relation, start time b of every fact (M,); end times are NEVER used
    tau (N,) float32 = tau_c,  it feeds S^HL(h_k|c) = 2^(-h_k/tau_c)         level (N,) = "subject_relation" | "relation" | "global"

At landmark c only starts b <= c count. A gap = time between two consecutive starts of the same (subject, relation).
    (subject, relation):  >= 2 gaps  ->  tau = 0.5 x mean gap of the pair
    relation:             otherwise  ->  tau = 0.5 x median of all gaps of that relation
    global:               no gap     ->  tau = 0.5 x median of all gaps  (0.5 if there is no gap at all)
"""
#%%
import numpy as np

import pandas as pd

#%%
def update_gaps(starts):
    updates = starts.drop_duplicates().sort_values(["s", "r", "b"])        # same instant, other object = one update
    updates["gap"] = updates.groupby(["s", "r"])["b"].diff()               # time since the previous update of this (s, r)
    return updates[updates["gap"] > 0]


def half_lives(subject, relation, landmark, history_subject, history_relation, history_begin):
    history = pd.DataFrame({"s": history_subject, "r": history_relation, "b": history_begin})
    subject, relation, landmark = np.asarray(subject), np.asarray(relation), np.asarray(landmark)
    tau, level = np.empty(len(landmark)), np.empty(len(landmark), dtype=object)
    for c in np.unique(landmark):    # one pass per landmark c: tau_c uses only the starts known at c
        at = landmark == c                                                 # rows predicted at this landmark
        gaps = update_gaps(history[history["b"] <= c])                     # only what is known at c
       
        pair = gaps.groupby(["s", "r"])["gap"].agg(["mean", "count"])
        pair_gap = pair["mean"][pair["count"] >= 2].reindex(pd.MultiIndex.from_arrays([subject[at], relation[at]])).to_numpy()
       
        relation_gap = gaps.groupby("r")["gap"].median().reindex(relation[at]).to_numpy()
       
        use_pair, use_relation = ~np.isnan(pair_gap), ~np.isnan(relation_gap)
       
        gap = np.where(use_pair, pair_gap, np.where(use_relation, relation_gap, gaps["gap"].median() if len(gaps) else 1.0))
       
        tau[at] = np.maximum(0.5 * gap, 1e-6)                              # tau_c = 0.5 x gap, at least 1e-6
       
        level[at] = np.where(use_pair, "subject_relation", np.where(use_relation, "relation", "global"))
    return tau.astype(np.float32), level
