"""The Full model of the paper: neural piecewise-constant hazard, mixed with a half-life curve.

    INPUT   x    (B, 12) float  handcrafted features x_c   (cols 0-4 age, 5-8 degree, 9-11 half-life), standardised
            ids  (B, 3)  int    (subject, relation, object) ids; id 0 = unseen in training
            tau  (B,)    float  half-life tau_c, in the time unit of the dataset
    OUTPUT  log S-hat_c(h_k | c)  (B, K)   log survival probability at the K horizons

Symbols in the comments follow the paper (Section "Proposed Method").
"""
#%%
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

P, Q = 12, 8                                        # p = 12 features;  q = 8 = size of the subject / relation / object vectors
SUBJECT_DEGREES, OBJECT_DEGREES = [7, 8], [5, 6]    # columns of x_c holding d_c^(s) and d_c^(o) (log, standardised)
SUBJECT, RELATION, OBJECT = 0, 1, 2                 # columns of ids


class NeuralLandmarkHazard(nn.Module):
    def __init__(self, vocabulary_sizes, horizons, *, hidden, dropout):
        super().__init__()
        K = len(horizons)
        self.register_buffer("h", torch.as_tensor(horizons, dtype=torch.float32))           # h_1 < ... < h_K
        self.register_buffer("width", torch.diff(self.h, prepend=torch.zeros(1)))           # h_k - h_{k-1}
        # The order of the layers below fixes the random initial weights for a given seed.
        self.alpha = nn.Parameter(torch.zeros(K))                                           # alpha_k   baseline log-hazard
        self.u = nn.Embedding(vocabulary_sizes[RELATION], K, padding_idx=0)                 # u_{r,k}   relation effect
        nn.init.zeros_(self.u.weight)
        self.beta = nn.Linear(P, K, bias=False)                                             # beta_k^T x_c   linear effect
        nn.init.zeros_(self.beta.weight)
        self.A = nn.Embedding(vocabulary_sizes[RELATION], Q, padding_idx=0)                 # A[r] = e^(r)
        nn.init.normal_(self.A.weight, std=0.05)
        with torch.no_grad():
            self.A.weight[0].zero_()                                                        # unseen relation = zero vector
        self.W_s = nn.Linear(2, Q)                                                          # W^(s), b^(s)
        self.W_o = nn.Linear(2, Q)                                                          # W^(o), b^(o)
        self.mlp = nn.Sequential(nn.Linear(P + 4 * Q, hidden), nn.SiLU(), nn.Dropout(dropout),      # MLP_theta : v_c -> z_c
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout))
        self.w = nn.Linear(hidden, K, bias=False)                                           # w_k^T z_c   neural effect
        nn.init.zeros_(self.w.weight)
        self.gate = nn.Linear(hidden, 1)                                                    # w_g, b_g
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 3.0)                                              # pi_c starts at sigmoid(3) = 0.95

    def forward(self, x, ids, tau):
        r = ids[:, RELATION]
        e_s = F.silu(self.W_s(x[:, SUBJECT_DEGREES]))                                       # e_c^(s) = SiLU(W^(s) d_c^(s) + b^(s))
        e_o = F.silu(self.W_o(x[:, OBJECT_DEGREES]))                                        # e_c^(o) = SiLU(W^(o) d_c^(o) + b^(o))
        v = torch.cat((x, e_s, self.A(r), e_o, e_s * e_o), dim=1)                           # v_c = [x_c, e^(s), e^(r), e^(o), e^(s) * e^(o)]
        z = self.mlp(v)                                                                     # z_c = MLP_theta(v_c)
        eta = self.alpha + self.u(r) + self.beta(x) + self.w(z)                             # eta_{c,k} = alpha_k + u_{r,k} + beta_k^T x_c + w_k^T z_c
        lam = eta.clamp(-20.0, 12.0).exp()                                                  # lambda_{c,k} = exp(eta_{c,k}) > 0
        log_s_neural = -(lam * self.width).cumsum(dim=1)                                    # log S^N(h_k|c) = -sum_{j<=k} lambda_{c,j} (h_j - h_{j-1})
        log_s_halflife = -np.log(2.0) * self.h[None, :] / tau[:, None]                      # log S^HL(h_k|c) = -(ln 2) h_k / tau_c
        g = self.gate(z)                                                                    # pi_c = sigmoid(w_g^T z_c + b_g)
        return torch.logaddexp(F.logsigmoid(g) + log_s_neural,                              # log[ pi_c S^N + (1 - pi_c) S^HL ]
                               F.logsigmoid(-g) + log_s_halflife)
