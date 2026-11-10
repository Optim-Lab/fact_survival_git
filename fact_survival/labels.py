"""Survival labels y_{c,k}: 1 if the fact is still valid at c + h_k, 0 if it ended by then, NaN if unknown.

    labels = survival_labels(stop, ended, landmark, horizons)       # (N, K)   what really happened
    labels = known_at(origin, labels, landmark, horizons)           # (N, K)   hide what is not known yet at `origin`
    !! Never train on survival_labels() directly: its answers use the future. Only known_at() output may go into fit().

    stop      (N,)  T     end time of the fact, or the last time it was observed
    ended     (N,)  bool  True = it really ended at T;  False = still valid when observation stopped (censored)
    landmark  (N,)  c     time of the prediction; the fact is active at c, so c < T
    horizons  (K,)  h_k

Example (c = 5, h = 3, so c + h = 8):  T = 10 -> 1   T = 8, ended -> 0   T = 7, censored -> NaN   T = 12, censored -> 1
(A fact is valid on [b, T): at time T it is already over.)
"""
#%%
import numpy as np

#미래를 본 정답 을만드는 함수
def survival_labels(stop, ended, landmark, horizons):
    stop, landmark, horizons = np.asarray(stop), np.asarray(landmark), np.asarray(horizons)
    if (landmark >= stop).any():
        raise ValueError("every fact must be active at its landmark: landmark < stop")
    alive = stop[:, None] > landmark[:, None] + horizons[None, :]       # T > c + h_k point
    return np.where(alive, 1.0, np.where(np.asarray(ended)[:, None], 0.0, np.nan))   # alive: 1;  over: 0 if seen, else unknown
#alive 1 terminate 0(at observed) , or unknown
#미래누출 방지조건 
def known_at(origin, labels, landmark, horizons):
    closed = np.asarray(landmark)[:, None] + np.asarray(horizons)[None, :] <= origin   # the window up to c + h_k is over
    return np.where(closed, labels, np.nan) #아니면 nan처리

