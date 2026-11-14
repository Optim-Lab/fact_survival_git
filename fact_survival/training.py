"""Fit the Full model on the known labels of one rolling origin, then predict survival probabilities.

    model = FactSurvival(horizons=[3, 6, 12], hidden=32, dropout=0.1, epochs=5,
                         learning_rate=3e-3, weight_decay=1e-3, batch_size=8192, seed=42)
    model.fit(features, ids, half_life, labels)
    survival = model.predict(features, ids, half_life)

    INPUT   features  (N, 12) float  handcrafted features x_c
            ids       (N, 3)  int    (subject, relation, object) ids
            half_life (N,)    float  tau_c
            labels    (N, K)  float  y_{c,k}: 1 = alive at c + h_k, 0 = ended, NaN = unknown (ignored)
    OUTPUT  survival  (N, K)  float  S-hat_c(h_k | c)

fit() reads like a recipe:  1. learn how to prepare the data   2. build the model   3. train   4. done.
Every setting is chosen by the caller; nothing is decided in here.
"""
#%%
import numpy as np

import torch
from torch import nn
import sys
from pathlib import Path
project = Path(__file__).resolve().parents[1]      # 저장소 루트
sys.path.insert(0, str(project))
sys.path.append("..")  # to import fact_survival.hazard_model
from fact_survival.hazard_model import NeuralLandmarkHazard


def masked_bce(log_s, y):
    """Loss L(Theta) of the paper: binary cross-entropy, using only the cells whose label is known."""
    known = ~y.isnan()  #y가 NaN이 아닌 위치를 찾아. ~는 참·거짓을 뒤집는 연산                                  # unknown labels (NaN) never enter the loss
    log_s, y = log_s[known], y[known] #알려진 라벨과 그에 대응하는 예측만 남김.
    log_s = log_s.clamp(max=-1e-7)           #0에 가까워지는걸 제한함.              # S must stay below 1, otherwise log(1 - S) does not exist
    log_one_minus_s = torch.log(-torch.expm1(log_s))      # log(1 - S), accurate even when S is close to 1
    return -(y * log_s + (1 - y) * log_one_minus_s).mean()   # -[ y log S + (1 - y) log(1 - S) ], averaged


def initial_alpha(y, horizons):
    """Where alpha_k (baseline log-hazard of interval k) starts: the hazard that reproduces the survival rate
    seen in the training labels, so that training does not begin from a hazard of 1 per month."""
    K = len(horizons)

    alive = [y[~np.isnan(y[:, k]), k].mean() if (~np.isnan(y[:, k])).any() else None for k in range(K)]
    #각 구간에서 예측된 생존비율을 구함.
    first = next(k for k in range(K) if alive[k] is not None)                  # first horizon that has labels
    
    first_rate = -np.log(np.clip(alive[first], 1e-4, 1 - 1e-4)) / horizons[first]
    
    alive = [a if a is not None else np.exp(-first_rate * horizons[k]) for k, a in enumerate(alive)]
    alive = np.clip(alive, 1e-4, 1 - 1e-4) #survival probabilities are kept inside [1e-4, 1 - 1e-4]
    alive = np.minimum.accumulate(alive)                                       # survival never goes up
    
    cumulative_hazard = -np.log(alive)    #survival                                      # H(h_k)
    hazard = np.diff(np.r_[0, cumulative_hazard]) / np.diff(np.r_[0, horizons])    # hazard inside each interval
    return np.log(np.maximum(hazard, 1e-4))


class FactSurvival:
    def __init__(self, horizons, *, hidden, dropout, epochs, learning_rate, weight_decay, batch_size, seed,
                 device="cpu"):
        self.horizons = np.asarray(horizons, dtype=float)
        self.hidden, self.dropout = hidden, dropout
        self.epochs, self.batch_size, self.seed, self.device = epochs, batch_size, seed, device
        self.learning_rate, self.weight_decay = learning_rate, weight_decay

    def _to_tensors(self, features, ids, half_life):
        """Prepare rows for the model: standardise the features, send unseen ids to 0."""
        ids = np.array(ids, dtype=np.int64)
        for column, size in enumerate(self.vocabulary_sizes):
            ids[ids[:, column] >= size, column] = 0                             # id not in the training rows = unseen
        
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale   # (x - mean) / std
        return [torch.as_tensor(a, device=self.device) for a in
                (x.astype(np.float32), ids, np.asarray(half_life, dtype=np.float32))]

#REAL TRAINING PART 
    def fit(self, features, ids, half_life, labels, after_epoch=None):
        """after_epoch(epoch, self) is called after every epoch (epoch = 1, 2, ...): lets the caller predict at several epochs
        of ONE training run (used to pick the epoch). It does not change the training."""
        torch.manual_seed(self.seed)
        features, ids, y = np.asarray(features), np.asarray(ids), np.asarray(labels, dtype=np.float32)

        # 1. Learn how to prepare the data -- from the training rows only
        self.vocabulary_sizes = (ids.max(axis=0) + 1).tolist()     #열마다 id가 몇개있는지 how many ids exist per column
        
        self.mean = features.mean(axis=0, dtype=np.float64) 
        self.scale = features.std(axis=0, dtype=np.float64)
        self.scale[self.scale < 1e-8] = 1.0                                     # a constant column keeps its scale
        data = self._to_tensors(features, ids, half_life)
        y = torch.as_tensor(y, device=self.device)

        # 2. Build the model and start alpha at the observed survival rates
        #Model is a neural network that predicts the log-hazard of the fact at its landmark c for each horizon h_k. The model takes as input the handcrafted features, the ids of the subject, relation, and object, and the half-life of the fact. The output is a vector of log-hazards for each horizon
        self.model = NeuralLandmarkHazard(self.vocabulary_sizes, self.horizons,
                                          hidden=self.hidden, dropout=self.dropout)
        with torch.no_grad():
            self.model.alpha.copy_(torch.as_tensor(initial_alpha(y.cpu().numpy(), self.horizons), dtype=torch.float32))
        
        self.model.to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # 3. Train: every epoch shuffles the rows, cuts them into mini-batches and nudges the weights on each
        self.loss_per_epoch = []
        for epoch in range(self.epochs):
            self.model.train()                                                  # dropout on
            batch_losses = []
            for rows in torch.randperm(len(y)).split(self.batch_size):
                rows = rows.to(self.device)
                if y[rows].isnan().all():                                       # nothing to learn from in this batch
                    continue
                loss = masked_bce(self.model(*(t[rows] for t in data)), y[rows])   # predict, then compare with the labels
                optimizer.zero_grad()
                loss.backward()                                                 # how should each weight change?
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)          # no single batch may change weights too much
                optimizer.step()                                                # change them a little
                batch_losses.append(loss.item())
            
            self.loss_per_epoch.append(float(np.mean(batch_losses)))            # watch this go down
            if after_epoch is not None:
                after_epoch(epoch + 1, self)

        # 4. Done: self.model, self.mean, self.scale ... are what predict() needs
        return self

    def predict(self, features, ids, half_life):
        self.model.eval()                                                       # dropout off
        with torch.inference_mode():
            return self.model(*self._to_tensors(features, ids, half_life)).exp().cpu().numpy()
