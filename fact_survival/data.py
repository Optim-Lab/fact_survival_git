"""From the raw interval-TKG pickle (FinWiki, ICEWSWiki) to one row per (fact, landmark) -- the input of everything else.

    facts, info = read_facts(path)                                  # one row per interval fact
    rows = landmark_rows(facts, info["study_end"], shortest_horizon)  # one row per (fact, landmark c) with c in [b, T)

    facts  subject, relation, object, begin b, stop T     times are month numbers (year * 12 + month - 1); a fact is valid on [b, T)
    rows   the same columns + landmark c, age a_c = c - b, ended (T < study_end, else censored),
           and the four degrees d_c of the facts ACTIVE at c  (this is the risk set E_c of the paper)

    object_degree           facts touching the object, as subject OR object          object_relation_degree   ... with the same relation
    subject_degree          facts with this subject                                  subject_relation_degree  ... with the same relation
Then the other files take over:   labels.py (stop, ended, landmark) -> y,   half_life.py -> tau,   features.py -> x_c.
"""
#%%
from __future__ import annotations
import pickle

import pickletools

import numpy as np
import pandas as pd

UNSAFE_PICKLE_OPS = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "OBJ", "INST", "NEWOBJ", "NEWOBJ_EX"}   # can run code

#%%
def month_id(value):
    """'2019-02-15' / '2019-02' / '2019' / 2019 -> a sortable month number."""
    if isinstance(value, int):
        return value * 12
    
    parts = str(value).split("-")
    return int(parts[0]) * 12 + (int(parts[1]) - 1 if len(parts) > 1 else 0)

#%%
def end_month_id(value): #how much fact active? 
    """An end date after the 1st of a month belongs to the NEXT month boundary."""
    parts = str(value).split("-")
   
    return month_id(value) + (1 if len(parts) >= 3 and int(parts[2]) > 1 else 0)

#%%
def read_facts(path, unit="month"):
    """unit = "month": dates like '2019-02-15' become month numbers.  unit = "year": the pickle already holds whole years (YAGO)."""
  
    raw = open(path, "rb").read()
  
    if any(op.name in UNSAFE_PICKLE_OPS for op, _, _ in pickletools.genops(raw)):
        raise ValueError("this pickle can execute code; refusing to load it")
   
    train_end, valid_end, train, valid, test, _ = pickle.loads(raw)
   
    unique = {tuple(f) for part in (train, valid, test) for f in part if len(f) == 5}     # (s, r, o, begin, end)
   
    facts = pd.DataFrame(list(unique), columns=["subject", "relation", "object", "begin", "end"])
   
    to_time, to_end = (month_id, end_month_id) if unit == "month" else (int, int)
  
    facts["begin"], facts["stop"] = facts["begin"].map(to_time), facts["end"].map(to_end)
   
    if (facts["stop"] <= facts["begin"]).any():
        raise ValueError("found a fact with end <= begin")
   
    facts = facts.drop(columns="end").sort_values(["begin", "stop", "relation", "subject", "object"], ignore_index=True)
    return facts, dict(train_end=to_time(train_end), valid_end=to_time(valid_end), study_end=int(facts["stop"].max()))

#%%
def landmark_rows(facts, study_end, last_landmark, first_landmark=None, object_degree="touching"):
    """last_landmark: FinWiki-type data use study_end - h_1 - 1;  YAGO uses 2022 (= observation end 2025 - longest horizon 3).
    object_degree: "touching" = facts touching the object as subject OR object (FinWiki, FinSurvKG, ICEWSWiki);
                   "incoming" = facts that have the object as their object (YAGO)."""
  
    frames = []
  
    first_landmark = facts["begin"].min() if first_landmark is None else first_landmark

    for c in range(first_landmark, last_landmark + 1):
  
        active = facts[(facts["begin"] <= c) & (facts["stop"] > c)]        
            # E_c: facts valid at c
        
        if active.empty:
            continue
        
        as_subject = active[["subject", "relation"]].rename(columns={"subject": "entity"})
  
        as_object = active[["object", "relation"]].rename(columns={"object": "entity"})

        edges = pd.concat([as_subject, as_object], ignore_index=True)           # every fact touches two entities

        by_entity = edges["entity"].value_counts()

        by_entity_relation = edges.groupby(["entity", "relation"]).size()

        if object_degree == "touching":
            object_d = active["object"].map(by_entity)
            object_relation_d = by_entity_relation.reindex(pd.MultiIndex.from_arrays([active["object"], active["relation"]])).to_numpy()

        else:                                                                         # "incoming"
            object_d = active.groupby("object")["subject"].transform("size")
            object_relation_d = active.groupby(["object", "relation"])["subject"].transform("size")

        frames.append(active.assign(
            landmark=c, age=c - active["begin"], ended=active["stop"] < study_end,
            object_degree=object_d, object_relation_degree=object_relation_d,
            subject_degree=active.groupby("subject")["object"].transform("size"),
            subject_relation_degree=active.groupby(["subject", "relation"])["object"].transform("size")))

    return pd.concat(frames, ignore_index=True)


#%%
def encode_ids(train_rows, eval_rows):
    """(subject, relation, object) ids, built from the TRAINING rows only: 1, 2, ... in sorted order; unseen = 0.
    Subject and object share one entity vocabulary.   returns two (n, 3) int arrays."""
    entities = np.unique(np.concatenate([train_rows["subject"].to_numpy(), train_rows["object"].to_numpy()]))
    relations = np.unique(train_rows["relation"].to_numpy())

    def encode(rows):
        columns = ((entities, "subject"), (relations, "relation"), (entities, "object"))

        return np.column_stack([pd.Index(vocabulary).get_indexer(rows[name]) + 1 for vocabulary, name in columns])   # -1 + 1 = 0
    return encode(train_rows), encode(eval_rows)
