"""The p = 12 handcrafted features x_c of a fact at its landmark c (Section "Proposed Method", Appendix "Feature Definitions").

    features = make_features(age, object_degree, object_relation_degree, subject_degree, subject_relation_degree, tau, tau_level)
                                                                           # (N, 12) float32; the four degrees come from data.py

    age      (N,)  a_c    age of the fact at c  (c - b), in the time unit of the dataset
    degrees  4 x (N,)     d_c: counts of the facts active at c around the object / (object, relation) / subject / (subject, relation)
    tau      (N,)  tau_c  half-life of the fact (see half_life.py)
    tau_level (N,) str    where tau came from: "subject_relation", "relation" or "global"

    columns   0 log(1 + a)     1 a / (1 + a)     2-4 max(log(1 + a) - log(1 + k), 0) for k = 1, 3, 10      (age, 5)
              5-8 log(1 + d) of the four degrees above                                                       (degree, 4)
              9 log(1 + tau)    10 tau came from (subject, relation)    11 tau came from the relation       (half-life, 3)
Only information available at the landmark is used; the end time T never enters a feature.
"""
#%%
import numpy as np


def make_features(age, object_degree, object_relation_degree, subject_degree, subject_relation_degree, tau, tau_level):
    a = np.asarray(age, dtype=float)
    log_age = np.log1p(a)                                                         # log(1 + a)
    age_part = [log_age, a / (1 + a)] + [np.maximum(log_age - np.log1p(k), 0) for k in (1, 3, 10)]
    degrees = [object_degree, object_relation_degree, subject_degree, subject_relation_degree] # sharing the same entity in active facts at same landmark 

    degree_part = [np.log1p(np.asarray(d, dtype=float)) for d in degrees]         # log(1 + d)
    tau_level = np.asarray(tau_level)
    half_life_part = [np.log1p(np.asarray(tau, dtype=float)),                     # log(1 + tau)
                      tau_level == "subject_relation", tau_level == "relation"]   # 1 if tau came from that level, else 0
    return np.column_stack(age_part + degree_part + half_life_part).astype(np.float32)
