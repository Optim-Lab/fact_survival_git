"""
The encoder reads the graph of the facts active at each landmark (ids from the training rows only, 0 = unseen):
    RE-GCN  (Li et al., SIGIR 2021, github.com/Lee-zix/RE-GCN)        snapshots c-2, c-1, c
    LogCL   (Chen et al., ICDE 2024, github.com/WeiChen3690/LogCL)    snapshots c-3 .. c-1 (local) + all earlier ones (global),
                                                                     plus its local/global contrastive loss (weight given by caller)
Its link-prediction decoder is replaced by   log S(h_k|c) = -sum_{j<=k} exp(w_j^T [e_s, e_r, e_o, x_c] + b_j) (h_j - h_{j-1}).
The original code produced the paper's numbers: legacy/modules/regcn_survival.py, logcl_original_survival.py (tests/test_graph_encoders_match_legacy.py).
"""
#%%
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from fact_survival.training import masked_bce


def use_authors_code(regcn_folder, logcl_folder):
    """RE-GCN: the authors' repository.  LogCL: the folder holding vendor/logcl_cpu (authors' code, device lines patched)."""
    for folder in (regcn_folder, logcl_folder):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))


def initial_bias(labels, horizons):
    """Where b_k starts: the hazard that reproduces the training survival rates (as training.initial_alpha, but -- as in the
    paper's encoder code -- the floor 1e-4 is put on the hazard MASS of interval k, before dividing by its width)."""
    known = ~np.isnan(labels)
    alive = np.divide(np.where(known, labels, 0).sum(axis=0), known.sum(axis=0), out=np.full(labels.shape[1], 0.5), where=known.any(axis=0))
    first = np.flatnonzero(known.any(axis=0))[0]
    rate = -np.log(np.clip(alive[first], 1e-4, 1 - 1e-4)) / horizons[first]
    alive[~known.any(axis=0)] = np.exp(-rate * np.asarray(horizons)[~known.any(axis=0)])   # horizon without labels: constant hazard
    cumulative = -np.log(np.minimum.accumulate(np.clip(alive, 1e-4, 1 - 1e-4)))
    return np.log(np.maximum(np.diff(np.r_[0, cumulative]), 1e-4) / np.diff(np.r_[0, horizons]))


def piecewise_log_survival(log_hazard, width):
    return -(log_hazard.clamp(-20.0, 12.0).exp() * width).cumsum(dim=1)


#%% The graph: one snapshot (set of active facts) per landmark
class Snapshots:
    def __init__(self, rows, train):
        t = rows.iloc[train]
        
        self.entities = {v: i + 1 for i, v in enumerate(sorted(set(t.subject) | set(t.object)))}
        
        self.relations = {v: i + 1 for i, v in enumerate(sorted(set(t.relation)))}
        
        self.ids = torch.tensor(np.column_stack([rows[c].map(m).fillna(0).to_numpy(np.int64) for c, m in
                                                 (("subject", self.entities), ("relation", self.relations), ("object", self.entities))]))
        self.landmark = rows.landmark.to_numpy()
        
        self.at = {}
        
        for c in np.unique(self.landmark):
            edges = self.ids[self.landmark == c]
            self.at[int(c)] = torch.unique(edges[(edges > 0).all(dim=1)], dim=0)      # an edge with an unseen end is left out

    def window(self, last, length):                                                    # snapshots last-length+1 .. last
        return [self.at.get(t, torch.empty((0, 3), dtype=torch.long)) for t in range(last - length + 1, last + 1)]

    def before(self, c):                                                               # every snapshot before c
        return [self.at[t] for t in sorted(self.at) if t < c]


#%% RE-GCN
class REGCN(nn.Module):

    def __init__(self, graph, p, horizons, *, hidden, history_length, **_):

        super().__init__()

        from src.rrgcn import RecurrentRGCN                                            # the authors' encoder (needs a GPU)
        self.graph, self.length = graph, history_length
        self.n_ent, self.n_rel = len(graph.entities) + 1, len(graph.relations) + 1
        self.backbone = RecurrentRGCN(
            decoder_name="convtranse", encoder_name="uvrgcn", num_ents=self.n_ent, num_rels=self.n_rel, num_static_rels=0,
            num_words=0, h_dim=hidden, opn="sub", sequence_len=3, num_bases=100, num_basis=100, num_hidden_layers=2,
            dropout=0.0, self_loop=True, skip_connect=False, layer_norm=True, use_static=False,
            entity_prediction=False, relation_prediction=False, use_cuda=True, gpu=0)
        self.head = nn.Linear(3 * hidden + p, len(horizons))
        nn.init.normal_(self.head.weight, std=0.01)
        h = torch.as_tensor(horizons, dtype=torch.float32)
        self.register_buffer("width", torch.diff(h, prepend=h.new_zeros(1)))

    def encode(self, c):
        from rgcn.utils import build_sub_graph
        device = self.backbone.dynamic_emb.device
        graphs = []
        for edges in self.graph.window(c, self.length):
            g = build_sub_graph(num_nodes=self.n_ent, num_rels=self.n_rel, triples=edges.numpy(), use_cuda=True, gpu=device.index)
            g.r_to_e = g.r_to_e.to(device=device, dtype=torch.long)
            graphs.append(g)
        with torch.cuda.device(device):
            history, _, relations, _, _ = self.backbone(graphs, static_graph=None, use_cuda=True)
        entities = F.normalize(history[-1], dim=1)
        return torch.cat((torch.zeros_like(entities[:1]), entities[1:])), torch.cat((torch.zeros_like(relations[:1]), relations[1:]))

    def decode(self, states, ids, x):
        e, r = states
        return piecewise_log_survival(self.head(torch.cat((e[ids[:, 0]], r[ids[:, 1]], e[ids[:, 2]], x), dim=1)), self.width)

    def forward(self, c, ids, x):                                                      # training: (log S, extra loss)
        return self.decode(self.encode(c), ids, x), 0.0

    def predict(self, c, ids, x, batch_size):                                          # encode once, decode in batches
        states = self.encode(c)
        return torch.cat([self.decode(states, i, v) for i, v in zip(ids.split(batch_size), x.split(batch_size))])


#%% LogCL
def dgl_graph(n_ent, triples, n_rel, frequency=None):
    """The graph input of LogCL's build_sub_graph (local, with inverse edges) / build_graph (global, with frequencies)."""
    import dgl
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    if frequency is None:
        inverse = triples[:, [2, 1, 0]].copy()
        inverse[:, 1] += n_rel
        directed = np.concatenate((triples, inverse))
    else:
        directed = triples
    
    g = dgl.graph((directed[:, 0], directed[:, 2]), num_nodes=n_ent)
    g.ndata["id"] = torch.arange(n_ent).reshape(-1, 1)
    g.ndata["norm"] = g.in_degrees().float().clamp_min(1).reciprocal().reshape(-1, 1)
    
    g.apply_edges(lambda edges: {"norm": edges.dst["norm"] * edges.src["norm"]})
    g.edata["type"] = torch.from_numpy(directed[:, 1].copy())
    
    if frequency is not None:
        g.edata["fre"] = torch.as_tensor(frequency, dtype=torch.long)
        return g
    
    g.uniq_r = np.concatenate((np.unique(triples[:, 1]), np.unique(triples[:, 1]) + n_rel))
    
    spans, subjects = [], []                                                           # r_to_e: subjects of each relation
    
    for r in g.uniq_r:                                                                 # (the inverse relation gets the same subjects, as in LogCL)
        s = np.unique(triples[triples[:, 1] == r % n_rel, 0])
        spans.append((len(subjects), len(subjects) + len(s)))
        subjects.extend(s.tolist())
    g.r_len, g.r_to_e = spans, np.asarray(subjects, dtype=np.int64)
    
    return g


def global_history(snapshots, queries, n_rel):
    """Past edges (with inverse) that start at a query subject or at an object the query (subject, relation) had; + their counts."""
    history = np.concatenate([np.asarray(s, dtype=np.int64).reshape(-1, 3) for s in snapshots] or [np.empty((0, 3), dtype=np.int64)])
    inverse = history[:, [2, 1, 0]].copy()
    inverse[:, 1] += n_rel
    history = np.concatenate((history, inverse))
    if not len(history):
        return history, np.empty(0, dtype=np.int64)
    edges, count = np.unique(history, axis=0, return_counts=True)
    queries = queries.cpu().numpy()
    
    pairs = set(map(tuple, queries[:, :2].tolist()))
    
    keep = np.isin(edges[:, 0], list(set(queries[:, 0].tolist()) | {int(o) for s, r, o in edges if (int(s), int(r)) in pairs}))
    
    return edges[keep], count[keep]


class LogCL(nn.Module):
    def __init__(self, graph, p, horizons, *, hidden, history_length, contrastive_weight, contrastive_batch, **_):
        super().__init__()
        from vendor.logcl_cpu.src.rrgcn import RecurrentRGCN                           # the authors' encoder
        self.graph, self.length, self.cache = graph, history_length, {}
        self.weight, self.sample = contrastive_weight, contrastive_batch
        self.n_ent, self.n_rel = len(graph.entities) + 1, len(graph.relations) + 1
        self.backbone = RecurrentRGCN(
            "convtranse", "uvrgcn", self.n_ent, self.n_rel, 0, 0, hidden, "sub", sequence_len=3, num_bases=1, num_basis=1,
            num_hidden_layers=2, dropout=0.0, self_loop=True, layer_norm=True, use_static=False, pre_type="all", use_cl=True,
            temperature=0.03, use_cuda=False, gpu=torch.device("cpu"))
        
        self.head = nn.Linear(3 * hidden + p, len(horizons))
        
        nn.init.normal_(self.head.weight, std=0.01)
        
        h = torch.as_tensor(horizons, dtype=torch.float32)
        
        self.register_buffer("width", torch.diff(h, prepend=h.new_zeros(1)))

    def query_mask(self, queries):
        """LogCL's query vector per subject: w1[entity at time 0, mean of its query relations]."""
        b = self.backbone
        pairs = torch.unique(queries[(queries[:, :2] > 0).all(dim=1)][:, :2], dim=0)
        relations = b.emb_rel.new_zeros(self.n_ent, b.h_dim)
        if len(pairs):
            relations.index_add_(0, pairs[:, 0], b.emb_rel[pairs[:, 1]])
            relations = relations / torch.bincount(pairs[:, 0], minlength=self.n_ent).clamp_min(1).to(relations.dtype).unsqueeze(1)
        entity = b.w4(torch.cat((b.dynamic_emb, torch.cos(b.weight_t2 * 0 + b.bias_t2).repeat(self.n_ent, 1)), dim=1))
        mask = entity.new_zeros(entity.shape)
        subjects = torch.unique(pairs[:, 0])
        mask[subjects] = b.w1(torch.cat((entity[subjects], relations[subjects]), dim=1))
        return mask

    def forward(self, c, ids, x, contrastive=True):
        device = self.backbone.dynamic_emb.device
    
        self.backbone.gpu = device
    
        queries = self.graph.ids[self.graph.landmark == c].to(device)                  # every fact active at c
    
        if c not in self.cache:
            self.cache[c] = global_history(self.graph.before(c), queries, self.n_rel)
        local = [dgl_graph(self.n_ent, s, self.n_rel) for s in self.graph.window(c - 1, self.length)]
    
        for g in local:
            g.r_to_e = torch.as_tensor(g.r_to_e, dtype=torch.long, device=device)
    
        local = [g.to(device) for g in local]
    
        edges, count = self.cache[c]
    
        global_ = dgl_graph(self.n_ent, edges, self.n_rel, frequency=count).to(device)
    
        e_local, _, r, e_global, r_global, local_states, relation_states = self.backbone.forward(
    
            global_, 0, self.query_mask(queries), local, None, device.type == "cuda")
    
        e_s = 0.9 * e_local.tanh() + (1 - 0.9) * e_global.tanh()                            # subject: 0.9 local + 0.1 global (LogCL decoder)
   
        e_o, e_s, r = e_local.tanh().clone(), e_s.clone(), r.clone()                   # object: local
   
        e_s[0], r[0], e_o[0] = 0, 0, 0                                                 # unseen id = zero vector
   
        log_s = piecewise_log_survival(self.head(torch.cat((e_s[ids[:, 0]], r[ids[:, 1]], e_o[ids[:, 2]], x), dim=1)), self.width)
        
        if not contrastive:
            return log_s
        
        if not self.weight:
            return log_s, 0.0
        
        q = ids[torch.randperm(len(ids), device=device)[:self.sample]]                 # <= 128 rows of the batch
        
        loss = log_s.sum() * 0
        
        if len(q) > 1:
            g = self.backbone.w_cl(torch.cat((self.backbone.his_ent[q[:, 0]], r_global[q[:, 1]]), dim=1))
            for e_t, r_t in zip(local_states, relation_states):
                loss = loss + self.backbone.get_loss_conv(g, self.backbone.w_cl(torch.cat((e_t[q[:, 0]], r_t[q[:, 1]]), dim=1)))
        
        return log_s, self.weight * loss

    def predict(self, c, ids, x, batch_size):                                          # all rows at once (attention sees the whole risk set)
        return self.forward(c, ids, x, contrastive=False)


#%% Training at one origin (same for both encoders)
def fit_predict(name, rows, x, labels, horizons, origin, *, epochs, seed, device, batch_size, learning_rate, weight_decay, **settings):
    x, labels = np.asarray(x, dtype=np.float32), np.asarray(labels, dtype=np.float32)
    landmark = rows.landmark.to_numpy()
    train = np.flatnonzero(~np.isnan(labels).all(axis=1))                              # rows with at least one known label
    test = np.flatnonzero(landmark == origin)
    
    graph = Snapshots(rows, train)
    mean, scale = x[train].mean(axis=0, dtype=np.float64), x[train].std(axis=0, dtype=np.float64)
    
    scale[scale < 1e-8] = 1.0
    
    x = torch.tensor(((x - mean) / scale).astype(np.float32))
    y = torch.tensor(labels, dtype=torch.float32)

    torch.manual_seed(seed)
    device = torch.device(device)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    
    model = {"regcn": REGCN, "logcl": LogCL}[name](graph, x.shape[1], horizons, **settings).to(device)
    
    with torch.no_grad():
        model.head.bias.copy_(torch.as_tensor(initial_bias(labels, horizons), dtype=torch.float32))   # b_k starts at the observed hazard
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    groups = [(int(c), torch.tensor(train[landmark[train] == c])) for c in np.unique(landmark[train])]   # one group per landmark
    
    out = {}
    
    for epoch in range(1, max(epochs) + 1):
        model.train()
        for g in torch.randperm(len(groups)):                                          # landmarks in random order
            c, members = groups[int(g)]
            for batch in members[torch.randperm(len(members))].split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                log_s, extra = model(c, graph.ids[batch].to(device), x[batch].to(device))
                loss = masked_bce(log_s, y[batch].to(device)) + extra                  # survival BCE (+ LogCL contrastive)
                
                loss.backward()
                
                nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                
                optimizer.step()
    
        if epoch in epochs:
            model.eval()
            with torch.inference_mode():
                out[epoch] = model.predict(origin, graph.ids[test].to(device), x[test].to(device), batch_size).exp().cpu().numpy()

        print(f"{name} origin {origin} epoch {epoch} done", flush=True)

    return out
