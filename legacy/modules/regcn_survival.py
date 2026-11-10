"""RE-GCN encoder adaptation for landmark survival (pure PyTorch, CPU pilot).

Architecture reference: Li et al., SIGIR 2021, https://arxiv.org/abs/2104.10353
Reference implementation: https://github.com/Lee-zix/RE-GCN (src/rrgcn.py,
rgcn/layers.py). This independent implementation retains additive relational
messages, relation GRU evolution and entity time gates. It uses active-state
snapshots including the current landmark, replaces ConvTransE/R with the
shared piecewise survival map, and omits optional static-graph supervision.
It is an adaptation, not a reproduction of published link-prediction scores.
"""
#%%
from __future__ import annotations


import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from datasets.task import mature_label_mask
from modules.neural_hazard import masked_bce, piecewise_log_survival


class SnapshotHistory:
    """Full risk-set topology, training-only vocabulary, causal history windows.

    Unknown query categories map to zero. Edges with unknown endpoints or
    relations are excluded, rather than merging unrelated entities into an
    artificial shared graph node. No label or endpoint column is read here.
    Optional point events use a one-step observation lag and a training-only
    vocabulary. include_events=False retains that vocabulary for matched controls.
    """

    def __init__(self, rows, train_indices, cutoff, *, point_events=None, include_events=True):
        train = rows.iloc[train_indices]
        entities = set(train.subject) | set(train.object)
        relations = set(train.relation)
        self.event_vocabulary_cutoff = int(train.origin.max()) if point_events is not None else None
        self.event_vocabulary_rows = 0
        if point_events is not None:
            required = {'subject', 'relation', 'object', 'origin'}
            if not required.issubset(point_events.columns) or not pd.api.types.is_integer_dtype(point_events.origin):
                raise ValueError('Point events require subject/relation/object and integer origin')
            past = point_events.loc[point_events.origin < self.event_vocabulary_cutoff]
            entities |= set(past.subject) | set(past.object)
            relations |= set(past.relation)
            self.event_vocabulary_rows = len(past)
        self.entities = {v: i + 1 for i, v in enumerate(sorted(entities))}
        self.relations = {v: i + 1 for i, v in enumerate(sorted(relations))}
        self.cutoff = int(cutoff)
        self.triples = torch.tensor(np.column_stack([
            rows.subject.map(self.entities).fillna(0).to_numpy(np.int64),
            rows.relation.map(self.relations).fillna(0).to_numpy(np.int64),
            rows.object.map(self.entities).fillna(0).to_numpy(np.int64),
        ]))
        self.snapshots = {}
        self.dropped_edges = {}
        origins = rows.origin.to_numpy()
        for origin in np.unique(origins[origins <= cutoff]):
            edges = self.triples[np.flatnonzero(origins == origin)]
            known = (edges > 0).all(dim=1)
            self.dropped_edges[int(origin)] = int((~known).sum())
            self.snapshots[int(origin)] = torch.unique(edges[known], dim=0)
        self.event_edges = {}
        self.dropped_event_edges = {}
        if point_events is not None:
            # Monthly landmarks are month starts. Even a same-month daily
            # event is withheld; event M first contributes to snapshot M+1.
            visible = point_events.loc[point_events.origin < self.cutoff]
            for event_origin, events in visible.groupby('origin', sort=True):
                origin = int(event_origin) + 1
                edges = torch.tensor(np.column_stack([
                    events.subject.map(self.entities).fillna(0).to_numpy(np.int64),
                    events.relation.map(self.relations).fillna(0).to_numpy(np.int64),
                    events.object.map(self.entities).fillna(0).to_numpy(np.int64),
                ]))
                known = (edges > 0).all(dim=1)
                self.event_edges[origin] = torch.unique(edges[known], dim=0)
                self.dropped_event_edges[origin] = int((~known).sum())
                state = self.snapshots.get(origin, torch.empty((0, 3), dtype=torch.long))
                self.snapshots[origin] = torch.unique(torch.cat((state, self.event_edges[origin])), dim=0) if include_events else state

    def window(self, landmark, length):
        if length < 1 or landmark > self.cutoff:
            raise ValueError("History requires positive length and landmark <= fold cutoff")
        empty = torch.empty((0, 3), dtype=torch.long)
        return [self.snapshots.get(t, empty) for t in range(int(landmark) - length + 1, int(landmark) + 1)]


class REGCNSurvival(nn.Module):
    """Dynamic RE-GCN encoder and a plain piecewise head, without HALO mixture."""

    def __init__(
        self,
        num_entities,
        num_relations,
        numeric_dim,
        horizons,
        hidden=32,
        layers=2,
        message_passing=True,
    ):
        super().__init__()

        import sys
        from pathlib import Path

        regcn_root = Path(__file__).resolve().parents[2] / "RE-GCN"
        if str(regcn_root) not in sys.path:
            sys.path.insert(0, str(regcn_root))

        from src.rrgcn import RecurrentRGCN

        h = torch.as_tensor(horizons, dtype=torch.float32)

        if (
            h.ndim != 1
            or not len(h)
            or not torch.isfinite(h).all()
            or (torch.diff(h, prepend=h.new_zeros(1)) <= 0).any()
        ):
            raise ValueError("예측 기간은 양수이며 증가해야 합니다.")

        if min(num_entities, num_relations, hidden, layers) < 1 or numeric_dim < 0:
            raise ValueError("모형 차원을 확인하세요.")

        if not message_passing:
            raise ValueError("이번 연결에서는 그래프 메시지를 사용하는 조건만 지원합니다.")

        self.num_relations = num_relations
        self.message_passing = message_passing

        # 공식 코드의 인코더를 사용하며, 우선 기존 CPU 학습 경로에 맞춥니다.
        self.backbone = RecurrentRGCN(
            decoder_name="convtranse",
            encoder_name="uvrgcn",
            num_ents=num_entities,
            num_rels=num_relations,
            num_static_rels=0,
            num_words=0,
            h_dim=hidden,
            opn="sub",
            sequence_len=3,
            num_bases=100,
            num_basis=100,
            num_hidden_layers=layers,
            dropout=0.0,
            self_loop=True,
            skip_connect=False,
            layer_norm=True,
            use_static=False,
            entity_prediction=False,
            relation_prediction=False,
            use_cuda=True,
            gpu=0
        )

        # 생존 헤드와 시간 구간은 기존 방식으로 유지합니다.
        self.head = nn.Linear(3 * hidden + numeric_dim, len(h))
        nn.init.normal_(self.head.weight, std=0.01)
        self.register_buffer(
            "widths",
            torch.diff(h, prepend=h.new_zeros(1)),
        )
    def encode(self, snapshots):
        from rgcn.utils import build_sub_graph

        if not snapshots:
            raise ValueError("과거 스냅샷이 하나 이상 필요합니다.")

        device = self.backbone.dynamic_emb.device

        if device.type != "cuda":
            raise ValueError("모델을 먼저 GPU로 이동해야 합니다.")

        graphs = []

        for triples in snapshots:
            if triples.ndim != 2 or triples.shape[1] != 3:
                raise ValueError("스냅샷은 (간선 수, 3) 형태여야 합니다.")

            if triples.dtype != torch.long:
                raise ValueError("스냅샷 ID는 torch.long이어야 합니다.")

            if triples.numel():
                if (triples <= 0).any():
                    raise ValueError("그래프 간선에 미등록 ID 0이 포함됐습니다.")

                if (
                    triples[:, [0, 2]] >= self.backbone.num_ents
                ).any():
                    raise ValueError("개체 ID가 허용 범위를 벗어났습니다.")

                if (triples[:, 1] >= self.num_relations).any():
                    raise ValueError("관계 ID가 허용 범위를 벗어났습니다.")

            graph = build_sub_graph(
                num_nodes=self.backbone.num_ents,
                num_rels=self.num_relations,
                triples=triples.detach().cpu().numpy(),
                use_cuda=True,
                gpu=device.index,
            )

            # 일반 그래프 속성은 별도로 GPU에 옮깁니다.
            graph.r_to_e = graph.r_to_e.to(
                device=device,
                dtype=torch.long,
            )
            graphs.append(graph)
        with torch.cuda.device(device):
            history, _, relations, _, _ = self.backbone(
                graphs,
                static_graph=None,
                use_cuda=True,
            )

        entities = F.normalize(history[-1], dim=1)

        # 생존 헤드에 전달하는 미등록 ID의 벡터는 0으로 고정합니다.
        entities = torch.cat(
            (torch.zeros_like(entities[:1]), entities[1:]),
            dim=0,
        )
        relations = torch.cat(
            (torch.zeros_like(relations[:1]), relations[1:]),
            dim=0,
        )

        return entities, relations

    def decode(self, states, triples, numeric):
        entities, relations = states
        # ② RE-GCN 출력과 수치형 특징 결합
        z = torch.cat((entities[triples[:, 0]], relations[triples[:, 1]], entities[triples[:, 2]], numeric), dim=1)
        # ③ 구간별 생존확률 계산
        return piecewise_log_survival(self.head(z), self.widths)

    def forward(self, snapshots, triples, numeric):
        return self.decode(self.encode(snapshots), triples, numeric)


def fit_predict_origin(rows, x, labels, horizons, origin, *, epochs=2, hidden=32,
                       history_length=3, batch_size=4096, seed=42, message_passing=True,
                       point_events=None, include_events=True):
    """Refit from scratch at one origin; retain every mature training row.

    Historical training queries each receive their own as-of-landmark graph.
    The evaluation risk set is predicted BEFORE the observed-label mask.
    """
    x, labels = np.asarray(x, dtype=np.float32), np.asarray(labels, dtype=np.float32)
    single_epoch = isinstance(epochs, int)
    requested_epochs = [epochs] if single_epoch else sorted(set(epochs))
    if not requested_epochs or min(*requested_epochs, hidden, history_length, batch_size) < 1:
        raise ValueError("Training sizes must be positive")
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError("Expected finite row-aligned numeric features")

    if labels.shape != (len(rows), len(horizons)) or rows.duplicated(["fact_id", "origin"]).any():
        raise ValueError("Expected aligned labels and unique fact/origin keys")

    mask = mature_label_mask(labels, rows.origin.to_numpy(), horizons, origin)

    train = np.flatnonzero(mask.any(axis=1))

    evaluation = np.flatnonzero(rows.origin.to_numpy() == origin)

    if not len(train) or not len(evaluation) or not np.isin(labels[mask], [0, 1]).all():
        raise ValueError("Need mature binary training labels and a nonempty evaluation risk set")

    graph = SnapshotHistory(rows, train, origin, point_events=point_events,
                            include_events=include_events)

    mean, scale = x[train].mean(axis=0, dtype=np.float64), x[train].std(axis=0, dtype=np.float64)

    scale[scale < 1e-8] = 1

    numeric = torch.tensor(((x - mean) / scale).astype(np.float32))

    y, observed = torch.tensor(labels), torch.tensor(mask)

    torch.manual_seed(seed)

    config = dict(num_entities=len(graph.entities) + 1, num_relations=len(graph.relations) + 1,
                  numeric_dim=x.shape[1], horizons=list(horizons), hidden=hidden,
                  message_passing=message_passing)

    if not torch.cuda.is_available():
        raise RuntimeError("GPU를 사용할 수 있는 노드에서 실행하세요.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)

    model = REGCNSurvival(**config).to(device)

    n = mask.sum(axis=0)

    rates = np.divide(np.where(mask, labels, 0).sum(axis=0), n, out=np.full(len(n), .5), where=n > 0)

    first = np.flatnonzero(n)[0]

    base_rate = -np.log(np.clip(rates[first], 1e-4, 1 - 1e-4)) / horizons[first]

    rates[n == 0] = np.exp(-base_rate * np.asarray(horizons)[n == 0])

    cumulative = -np.log(np.minimum.accumulate(np.clip(rates, 1e-4, 1 - 1e-4)))

    with torch.no_grad():
        model.head.bias.copy_(torch.tensor(np.log(np.maximum(np.diff(np.r_[0, cumulative]), 1e-4) / np.diff(np.r_[0, horizons])), dtype=torch.float32))

    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)

    groups = [torch.tensor(train[rows.origin.to_numpy()[train] == t]) for t in np.unique(rows.origin.to_numpy()[train])]

    history, predictions, states_by_epoch = [], {}, {}

    for epoch in range(1, max(requested_epochs) + 1):
        model.train()
        loss_sum = 0.
        count = 0
        for group_id in torch.randperm(len(groups)):
            indices = groups[int(group_id)]

            landmark = int(rows.iloc[int(indices[0])].origin)
            snapshots = graph.window(landmark, history_length)
            # ponytail: recompute history per batch; cache only graph topology,
            # never detached embeddings, to retain encoder gradients in this pilot.

            for batch in indices[torch.randperm(len(indices))].split(batch_size):
                optimizer.zero_grad(set_to_none=True)
                log_s = model(
                    snapshots,
                    graph.triples[batch].to(device),
                    numeric[batch].to(device),
                )
                loss = masked_bce(
                    log_s,
                    y[batch].to(device),
                    observed[batch].to(device),
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite RE-GCN survival loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                size = int(observed[batch].sum())
                loss_sum += float(loss.detach()) * size
                count += size

        history.append(dict(epoch=epoch, loss=loss_sum / count))
        print(f"RE-GCN epoch={epoch}: loss={loss_sum / count:.6f}", flush=True)

        if epoch in requested_epochs:
            model.eval()
            with torch.inference_mode():
                states = model.encode(graph.window(origin, history_length))
                probability = torch.cat([
                    model.decode(
                        states,
                        graph.triples[b].to(device),
                        numeric[b].to(device),
                    ).exp().cpu()
                    for b in torch.tensor(evaluation).split(batch_size)
                ]).numpy()
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any() or (np.diff(probability, axis=1) > 1e-6).any():
                raise AssertionError("Invalid survival probabilities")

            predictions[epoch] = probability

            states_by_epoch[epoch] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }


    diagnostics = dict(train_rows=len(train), evaluation_risk_rows=len(evaluation),
                       mature_labels=mask.sum(axis=0).tolist(), loss_history=history,
                       evaluation_unknown_rows=int((graph.triples[evaluation] == 0).any(dim=1).sum()),
                       evaluation_dropped_graph_edges=graph.dropped_edges[int(origin)],
                       latest_mature_target=int((rows.origin.to_numpy()[:, None] + np.asarray(horizons))[mask].max()))
    if point_events is not None:
        window = range(int(origin) - history_length + 1, int(origin) + 1)
        diagnostics.update(
            event_vocabulary_cutoff=graph.event_vocabulary_cutoff,
            event_vocabulary_rows=graph.event_vocabulary_rows,
            entity_vocabulary_size=len(graph.entities), relation_vocabulary_size=len(graph.relations),
            evaluation_known_event_edges=sum(len(graph.event_edges.get(t, ())) for t in window),
            evaluation_dropped_event_edges=sum(graph.dropped_event_edges.get(t, 0) for t in window),
            event_context_in_messages=bool(include_events and message_passing))

    checkpoint = dict(config=config, state_dict=states_by_epoch[max(requested_epochs)],
                      states=states_by_epoch, mean=mean, scale=scale,
                      entities=graph.entities, relations=graph.relations, origin=int(origin),
                      history_length=history_length)
    checkpoint.update(include_events=include_events, event_vocabulary_cutoff=graph.event_vocabulary_cutoff)

    return evaluation, predictions[requested_epochs[0]] if single_epoch else predictions, diagnostics, checkpoint
