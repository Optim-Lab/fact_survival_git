"""공식 LogCL의 과거 정보 처리 부분에 생존확률 출력을 붙이는 실험입니다.

왜 필요한가: 기존 logcl_survival.py는 논문의 생각을 참고해 다시 만든 코드입니다.
이 파일은 저자 코드를 별도 보관한 뒤, 그 코드의 local/global 인코더를 직접
호출합니다. 최근 그래프의 변화와 오래된 관련 사실을 모두 읽는 연산은 유지하고,
이미 알려진 (주체, 관계, 대상)이 앞으로도 유지될 확률을 새 출력층으로 만듭니다.

원본: vendor/LogCL, commit 0c0f5536c98c67714c2c95fc5b2245a45968d466.
호환 사본: vendor/logcl_cpu. 가져오기 경로 3줄과 GPU 강제 지정 4곳만 바꿨으며,
각 차이와 파일 해시는 vendor/logcl_provenance.json에 기록합니다.

해석의 범위: 논문의 성능 재현은 아닙니다. 입력은 생존 실험의 과거 활성 사실이고,
대상 개체를 고르는 기존 출력층 대신 생존 출력층을 학습합니다. 단어 정적 그래프는
사용하지 않습니다. 학습/평가 모두 공식 학습 경로의 query mask를 일관되게 씁니다.
생존 정답, 종료 시점, 미래 행은 인코더 입력에 넣지 않습니다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

_CPU_DEPENDENCIES = Path(__file__).resolve().parents[1] / 'vendor/cpu_deps'
if _CPU_DEPENDENCIES.exists():
    sys.path.insert(0, str(_CPU_DEPENDENCIES))
import dgl
import numpy as np
import torch
from torch import nn

from datasets.task import mature_label_mask
from modules.neural_hazard import masked_bce, piecewise_log_survival
from modules.regcn_survival import SnapshotHistory
from vendor.logcl_cpu.src.rrgcn import RecurrentRGCN


def check_upstream_hashes():
    """보관한 원본과 CPU 사본이 기록된 내용 그대로인지 확인합니다."""
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'vendor/logcl_provenance.json').read_text())
    for name, info in manifest['files'].items():
        for folder, key in [('LogCL', 'upstream_sha256'), ('logcl_cpu', 'compat_sha256')]:
            digest = hashlib.sha256((root / 'vendor' / folder / name).read_bytes()).hexdigest()
            if digest != info[key]:
                raise AssertionError(f'LogCL source changed: {folder}/{name}')
    return manifest


def _graph(num_entities, triples, num_relations, *, frequencies=None):
    """공식 build_sub_graph/build_graph와 같은 그래프 입력을 만듭니다."""
    triples = np.asarray(triples, dtype=np.int64).reshape(-1, 3)
    if frequencies is None:
        inverse = triples[:, [2, 1, 0]].copy()
        inverse[:, 1] += num_relations
        directed = np.concatenate((triples, inverse))
    else:
        directed = triples
    graph = dgl.graph((directed[:, 0], directed[:, 2]), num_nodes=num_entities)
    degree = graph.in_degrees().float().clamp_min(1)
    graph.ndata['id'] = torch.arange(num_entities).reshape(-1, 1)
    graph.ndata['norm'] = degree.reciprocal().reshape(-1, 1)
    graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    graph.edata['type'] = torch.from_numpy(directed[:, 1].copy())
    if frequencies is not None:
        graph.edata['fre'] = torch.as_tensor(frequencies, dtype=torch.long)
    else:
        # 공식 r2e는 역관계 쪽에도 원래 주체를 모읍니다. 그 동작을 유지합니다.
        relations = np.unique(triples[:, 1])
        graph.uniq_r = np.concatenate((relations, relations + num_relations))
        spans, values = [], []
        for relation in graph.uniq_r:
            subjects = np.unique(triples[triples[:, 1] == relation % num_relations, 0])
            spans.append((len(values), len(values) + len(subjects)))
            values.extend(subjects.tolist())
        graph.r_len = spans
        graph.r_to_e = np.asarray(values, dtype=np.int64)
    return graph


def global_history_triples(snapshots, queries, num_relations):
    """현재 질문과 이어지는 과거 2단계 이웃을 고르고 중복 횟수를 남깁니다."""
    arrays = [np.asarray(value, dtype=np.int64).reshape(-1, 3) for value in snapshots]
    history = np.concatenate(arrays) if arrays else np.empty((0, 3), dtype=np.int64)
    inverse = history[:, [2, 1, 0]].copy()
    inverse[:, 1] += num_relations
    history = np.concatenate((history, inverse))
    if not len(history):
        return history, np.empty(0, dtype=np.int64)
    unique, frequency = np.unique(history, axis=0, return_counts=True)
    if torch.is_tensor(queries):
        queries = queries.detach().cpu().numpy()
    queries = np.asarray(queries, dtype=np.int64).reshape(-1, 3)
    pairs = set(map(tuple, queries[:, :2].tolist()))
    targets = {int(o) for s, r, o in unique if (int(s), int(r)) in pairs}
    subjects = set(queries[:, 0].tolist()) | targets
    keep = np.isin(unique[:, 0], list(subjects))
    return unique[keep], frequency[keep]


class LogCLOriginalSurvival(nn.Module):
    """공식 인코더의 반환값에만 새 생존 출력층을 연결합니다."""

    def __init__(self, num_entities, num_relations, numeric_dim, horizons,
                 hidden=32, layers=2, local_weight=.9, graph_messages=True,
                 head_style='concat'):
        super().__init__()
        h = torch.as_tensor(horizons, dtype=torch.float32)
        if min(num_entities, num_relations, hidden, layers) < 1 or numeric_dim < 0:
            raise ValueError('Positive model dimensions are required')
        if h.ndim != 1 or not len(h) or not torch.isfinite(h).all() or (torch.diff(h, prepend=h.new_zeros(1)) <= 0).any():
            raise ValueError('Horizons must be finite, positive and increasing')
        if not 0 <= local_weight <= 1:
            raise ValueError('local_weight must be between zero and one')
        if head_style not in {'concat', 'residual'}:
            raise ValueError('head_style must be concat or residual')
        self.num_entities, self.num_relations = num_entities, num_relations
        self.local_weight, self.graph_messages = local_weight, graph_messages
        self.head_style = head_style
        self.backbone = RecurrentRGCN(
            'convtranse', 'uvrgcn', num_entities, num_relations, 0, 0, hidden,
            'sub', sequence_len=3, num_bases=1, num_basis=1,
            num_hidden_layers=layers, dropout=0., self_loop=True,
            layer_norm=True, use_static=False, pre_type='all', use_cl=True,
            temperature=.03, use_cuda=False, gpu=torch.device('cpu'))
        self.head = nn.Linear(3 * hidden + numeric_dim if head_style == 'concat' else numeric_dim, len(h))
        nn.init.normal_(self.head.weight, std=.01)
        if head_style == 'residual':
            self.graph_correction = nn.Linear(3 * hidden, len(h), bias=False)
            nn.init.zeros_(self.graph_correction.weight)
        self.register_buffer('widths', torch.diff(h, prepend=h.new_zeros(1)))

    def _query_mask(self, triples):
        # 공식 get_loss의 query construction을 학습/평가 모두에 사용합니다.
        backbone = self.backbone
        relations = backbone.emb_rel.new_zeros(self.num_entities, backbone.h_dim)
        known_queries = triples[(triples[:, :2] > 0).all(dim=1)]
        # 같은 (주체, 관계)는 한 번만 센 뒤 주체별 관계 평균을 한꺼번에 만듭니다.
        pairs = torch.unique(known_queries[:, :2], dim=0)
        subjects = torch.unique(pairs[:, 0])
        if len(pairs):
            relations.index_add_(0, pairs[:, 0], backbone.emb_rel[pairs[:, 1]])
            counts = torch.bincount(pairs[:, 0], minlength=self.num_entities).clamp_min(1)
            relations = relations / counts.to(relations.dtype).unsqueeze(1)
        time_zero = torch.cos(backbone.weight_t2 * 0 + backbone.bias_t2).repeat(self.num_entities, 1)
        entity = backbone.w4(torch.cat((backbone.dynamic_emb, time_zero), dim=1))
        mask = entity.new_zeros(entity.shape)
        mask[subjects] = backbone.w1(torch.cat((entity[subjects], relations[subjects]), dim=1))
        return mask

    def forward(self, snapshots, triples, numeric, historical_snapshots=None,
                *, contrastive=False, contrastive_batch=128, query_triples=None,
                global_inputs=None):
        if not snapshots:
            raise ValueError('At least one past snapshot is required')
        if triples.ndim != 2 or triples.shape[1] != 3 or len(numeric) != len(triples):
            raise ValueError('Expected aligned triples and numeric features')
        device = self.backbone.dynamic_emb.device
        use_cuda = device.type == 'cuda'
        self.backbone.gpu = device
        empty = np.empty((0, 3), dtype=np.int64)
        queries = triples if query_triples is None else query_triples
        local_graphs = [_graph(self.num_entities, value if self.graph_messages else empty,
                               self.num_relations) for value in snapshots]
        for graph in local_graphs:
            graph.r_to_e = torch.as_tensor(graph.r_to_e, dtype=torch.long, device=device)
        local_graphs = [graph.to(device) for graph in local_graphs]
        past = snapshots if historical_snapshots is None else historical_snapshots
        if self.graph_messages:
            global_edges, frequency = (global_history_triples(past, queries, self.num_relations)
                                       if global_inputs is None else global_inputs)
        else:
            global_edges, frequency = empty, np.empty(0, dtype=np.int64)
        global_graph = _graph(self.num_entities, global_edges, self.num_relations, frequencies=frequency).to(device)
        result = self.backbone.forward(global_graph, 0, self._query_mask(queries), local_graphs, None, use_cuda)
        local, _, relation, global_, global_relation, local_states, relation_states = result
        # 공식 decoder처럼 주체는 .9 local+.1 global, 대상은 local 표현을 씁니다.
        fused = self.local_weight * local.tanh() + (1 - self.local_weight) * global_.tanh()
        # 학습에 없던 ID 0은 임의의 학습되지 않은 벡터 대신 정보 없음으로 처리합니다.
        fused = fused.clone()
        relation = relation.clone()
        local_entity = local.tanh().clone()
        fused[0], relation[0] = 0, 0
        local_entity[0] = 0
        representation = torch.cat((fused[triples[:, 0]], relation[triples[:, 1]],
                                    local_entity[triples[:, 2]]), dim=1)
        if self.head_style == 'residual':
            log_hazard = self.head(numeric) + self.graph_correction(representation)
        else:
            log_hazard = self.head(torch.cat((representation, numeric), dim=1))
        log_s = piecewise_log_survival(log_hazard, self.widths)
        if not contrastive:
            return log_s
        sample = torch.randperm(len(triples), device=device)[:contrastive_batch]
        selected = triples[sample]
        loss_cl = log_s.sum() * 0
        if len(selected) > 1:
            query_global = self.backbone.w_cl(torch.cat((self.backbone.his_ent[selected[:, 0]],
                                                       global_relation[selected[:, 1]]), dim=1))
            for entity_state, relation_state in zip(local_states, relation_states):
                query_local = self.backbone.w_cl(torch.cat((entity_state[selected[:, 0]],
                                                           relation_state[selected[:, 1]]), dim=1))
                loss_cl = loss_cl + self.backbone.get_loss_conv(query_global, query_local)
        return log_s, loss_cl


def fit_predict_origin(rows, x, labels, horizons, origin, *, epochs=5, hidden=32,
                       history_length=3, batch_size=4096, seed=42,
                       contrastive_weight=.1, contrastive_batch=128,
                       graph_messages=True, local_weight=.9, head_style='concat'):
    """동일한 과거 정답과 현재 평가 행에서 한 번 학습합니다. 테스트 선택은 안 합니다."""
    check_upstream_hashes()
    x, labels = np.asarray(x, np.float32), np.asarray(labels, np.float32)
    requested = [epochs] if isinstance(epochs, int) else sorted(set(epochs))
    if not requested or min(*requested, hidden, history_length, batch_size, contrastive_batch) < 1 or contrastive_weight < 0:
        raise ValueError('Invalid training configuration')
    if len(x) != len(rows) or labels.shape != (len(rows), len(horizons)):
        raise ValueError('Expected aligned rows, features and labels')
    if (rows.origin.to_numpy() > origin).any():
        raise ValueError('Pass only rows observed by the fold origin')
    mask = mature_label_mask(labels, rows.origin.to_numpy(), horizons, origin)
    train = np.flatnonzero(mask.any(axis=1))
    evaluation = np.flatnonzero(rows.origin.to_numpy() == origin)
    if not len(train) or not len(evaluation) or not np.isin(labels[mask], [0, 1]).all():
        raise ValueError('Need observed training labels and an evaluation risk set')
    graph = SnapshotHistory(rows, train, origin)
    mean, scale = x[train].mean(axis=0, dtype=np.float64), x[train].std(axis=0, dtype=np.float64)
    scale[scale < 1e-8] = 1
    numeric = torch.tensor(((x - mean) / scale).astype(np.float32))
    labels_t, mask_t = torch.tensor(labels), torch.tensor(mask)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    config = dict(num_entities=len(graph.entities) + 1, num_relations=len(graph.relations) + 1,
                  numeric_dim=x.shape[1], horizons=list(horizons), hidden=hidden,
                  graph_messages=graph_messages, local_weight=local_weight, head_style=head_style)
    model = LogCLOriginalSurvival(**config).to(device)
    counts = mask.sum(axis=0)
    survival = np.divide(np.where(mask, labels, 0).sum(axis=0), counts,
                         out=np.full(len(counts), .5), where=counts > 0)
    first = np.flatnonzero(counts)[0]
    base_rate = -np.log(np.clip(survival[first], 1e-4, 1 - 1e-4)) / horizons[first]
    survival[counts == 0] = np.exp(-base_rate * np.asarray(horizons)[counts == 0])
    cumulative = -np.log(np.minimum.accumulate(np.clip(survival, 1e-4, 1 - 1e-4)))
    rates = np.maximum(np.diff(np.r_[0, cumulative]), 1e-4) / np.diff(np.r_[0, horizons])
    with torch.no_grad():
        model.head.bias.copy_(torch.as_tensor(np.log(rates), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    groups = [(int(t), torch.tensor(train[rows.origin.to_numpy()[train] == t]))
              for t in np.unique(rows.origin.to_numpy()[train])]
    snapshots = sorted(graph.snapshots)
    context = {t: [graph.snapshots[value] for value in snapshots if value < t]
               for t in [origin, *[t for t, _ in groups]]}
    queries = {t: graph.triples[rows.origin.to_numpy() == t]
               for t in [origin, *[t for t, _ in groups]]}
    # 같은 시점의 그래프 입력만 재사용합니다. 인코더 표현은 매 배치 다시 계산합니다.
    global_cache = {}
    def global_inputs(t):
        if t not in global_cache:
            global_cache[t] = global_history_triples(context[t], queries[t], config['num_relations'])
        return global_cache[t]
    history, predictions, states = [], {}, {}
    for epoch in range(1, max(requested) + 1):
        model.train()
        total, count = 0., 0
        for group_id in torch.randperm(len(groups)):
            t, indices = groups[int(group_id)]
            window = graph.window(t - 1, history_length)
            for positions in torch.randperm(len(indices)).split(batch_size):
                batch = indices[positions]
                optimizer.zero_grad(set_to_none=True)
                output = model(window, graph.triples[batch].to(device), numeric[batch].to(device), context[t],
                               query_triples=queries[t].to(device), contrastive=bool(contrastive_weight),
                               contrastive_batch=contrastive_batch, global_inputs=global_inputs(t))
                log_s, loss_cl = output if contrastive_weight else (output, 0.)
                loss = masked_bce(log_s, labels_t[batch].to(device), mask_t[batch].to(device)) + contrastive_weight * loss_cl
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite LogCL original survival loss')
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                size = int(mask_t[batch].sum())
                total += float(loss.detach()) * size
                count += size
        history.append(dict(epoch=epoch, loss=total / count))
        print(f'LogCL original survival epoch={epoch}: loss={total / count:.6f}', flush=True)
        if epoch in requested:
            model.eval()
            with torch.inference_mode():
                # 평가 질문 전체를 함께 넣어 배치 분할에 따라 attention이 바뀌지 않게 합니다.
                probability = model(graph.window(origin - 1, history_length), graph.triples[evaluation].to(device),
                                    numeric[evaluation].to(device), context[origin],
                                    global_inputs=global_inputs(origin)).exp().cpu().numpy()
            if not np.isfinite(probability).all() or (np.diff(probability, axis=1) > 1e-6).any():
                raise AssertionError('Invalid LogCL survival curve')
            predictions[epoch] = probability
            states[epoch] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    diagnostics = dict(train_rows=len(train), evaluation_risk_rows=len(evaluation),
                       mature_labels=mask.sum(axis=0).tolist(), loss_history=history,
                       graph_messages=graph_messages, graph_latest_origin=origin - 1, device=str(device),
                       latest_mature_target=int((rows.origin.to_numpy()[:, None] + np.asarray(horizons))[mask].max()),
                       upstream_commit='0c0f5536c98c67714c2c95fc5b2245a45968d466',
                       original_encoder_math=True, original_paper_lp_reproduction=False,
                       input_adaptation='active facts strictly before landmark; no entity-word graph',
                       decoder_adaptation='known triple + numeric to piecewise survival',
                       query_context='complete current risk set, independent of loss minibatch',
                       contrastive_sampling=f'at most {contrastive_batch} training rows per loss batch',
                       head_style=head_style)
    checkpoint = dict(config=config, states=states, mean=mean, scale=scale,
                      entities=graph.entities, relations=graph.relations, origin=int(origin),
                      history_length=history_length, contrastive_weight=contrastive_weight)
    return evaluation, predictions, diagnostics, checkpoint
