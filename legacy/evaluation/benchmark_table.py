"""One Brier table from compatible per-dataset rolling prediction runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from datasets.task import PROTOCOL_RULES, PROTOCOL_VERSION
from evaluation.metrics import summarize_probability

KEYS = ['fact_id', 'origin', 'horizon']

LABELS = {'logcl_bce': 'LogCL-inspired (BCE only)', 'evokg_lifetime': 'EvoKG-inspired + log-normal lifetime', 'evokg': 'EvoKG-inspired + Surv', 'logcl': 'LogCL-inspired + Surv',
          'logcl_original': 'LogCL original encoder + Surv',
          'logcl_original_bce': 'LogCL original encoder + Surv (BCE only)',
          'evokg_original': 'EvoKG original encoder + Surv',
          'regcn': 'RE-GCN + Surv', 'regcn_no_messages': 'RE-GCN without historical messages',
          'regcn_event_vocab': 'RE-GCN intervals (matched event vocabulary)',
          'regcn_events': 'RE-GCN intervals + past events',
          'regcn_events_no_messages': 'RE-GCN without graph messages (matched event vocabulary)',
          'neural_relation': 'Neural relation only', 'neural_object_degree': 'Neural object degree only',
          'neural_age': 'Neural relation + age', 'neural_degree': 'Neural + degree',
          'neural_gap': 'Neural + gap', 'neural': 'Neural piecewise mixture',
          'neural_id': 'Neural piecewise mixture + entity IDs',
          'cloglog': 'Linear cloglog', 'half_life': 'Neural constant hazard',
          'halo_raw': 'HALO-Surv raw', 'halo': 'HALO-Surv calibrated',
          'global': 'Global', 'relation_age': 'Relation-age',
          'object_degree': 'Object-degree', 'km': 'KM',
          'discrete_hazard': 'Discrete hazard'}

MODEL_ORDER = ['global', 'km', 'relation_age', 'object_degree',
               'discrete_hazard', 'halo_raw', 'halo', 'cloglog',
               'half_life', 'neural_relation', 'neural_object_degree', 'neural_age', 'neural_degree', 'neural_gap', 'neural', 'neural_id',
               'regcn_no_messages', 'regcn', 'regcn_events_no_messages', 'regcn_event_vocab', 'regcn_events',
               'logcl_bce', 'logcl', 'evokg', 'evokg_lifetime']


def read_runs(folders, phase='test', *, allow_partial=False):
    """Join probability columns only after exact row/label and data checks."""
    if phase not in ('development', 'test'):
        raise ValueError('phase must be development or test')
    records, populations = [], {}
    for folder in map(Path, folders):
        config = json.loads((folder / 'run_config.json').read_text())
        if config.get('benchmark_eligible') is False and not allow_partial:
            raise ValueError(f'{folder}: pilot result is not a full benchmark')
        if config.get('protocol_version') != PROTOCOL_VERSION or config.get('protocol') != PROTOCOL_RULES:
            raise ValueError(f'{folder}: protocol mismatch')
        if phase == 'test' and config.get('test_evaluation') not in ('rolling_origin', 'rolling_origin_partial'):
            raise ValueError(f'{folder}: no rolling test result')
        if phase == 'test' and config.get('partial_test') and not allow_partial:
            raise ValueError(f'{folder}: partial test result is not a full benchmark')
        if phase == 'development' and config.get('partial_development') and not allow_partial:
            raise ValueError(f'{folder}: partial development result is not a full benchmark')
        rows = pd.read_csv(folder / f'{phase}_predictions.csv.gz', dtype={'fact_id': str}, keep_default_na=False)
        if rows.duplicated(KEYS).any() or not rows.survived.isin([0, 1]).all():
            raise ValueError(f'{folder}: duplicate keys or invalid labels')
        dataset = config['dataset']
        identity = config.get('input_sha256')
        if identity is None:
            raise ValueError(f'{folder}: missing data hash')
        base = rows[[*KEYS, 'survived']]
        if dataset in populations:
            prior, prior_identity = populations[dataset]
            if identity != prior_identity:
                raise ValueError(f'{folder}: dataset hashes differ')
            joined = prior.merge(base, on=KEYS, how='outer', validate='one_to_one',
                                 suffixes=('_left', '_right'), indicator=True)
            if not joined._merge.eq('both').all() or not joined.survived_left.eq(joined.survived_right).all():
                raise ValueError(f'{folder}: evaluation rows or labels differ')
        else:
            populations[dataset] = (base, identity)
        for column in rows.columns:
            if not column.startswith('p_') or re.search(r'_e[0-9]+$', column):
                continue
            if not pd.api.types.is_numeric_dtype(rows[column]) or not rows[column].between(0, 1).all():
                raise ValueError(f'{folder}: invalid probability {column}')
            for horizon, group in rows.groupby('horizon'):
                record = dict(dataset=dataset, seed=int(config.get('seed', 42)),
                              source=str(folder), horizon=int(horizon),
                              **summarize_probability(group, column))
                objective = config.get('objective', 'bce')
                if objective != 'bce' and record['model'] in config.get('variants', []):
                    record['model'] += '_' + objective
                if record['model'] == 'logcl' and config.get('contrastive_weight') == 0:
                    record['model'] = 'logcl_bce'
                architecture = config.get('architecture', '')
                if architecture == 'logcl_original_encoder_survival_v1' and record['model'] in ('logcl', 'logcl_bce'):
                    record['model'] = record['model'].replace('logcl', 'logcl_original', 1)
                if architecture == 'evokg_original_encoder_survival_v1' and record['model'] == 'evokg':
                    record['model'] = 'evokg_original'
                records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError('No probability columns found')
    return table.sort_values(['dataset', 'horizon', 'model', 'seed']).reset_index(drop=True)


def write_tables(metrics, output, *, phase='test', partial=False):
    """Write CSV plus grouped LaTeX row/fact Brier with best/second formatting."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / f'{phase}_metrics_all_runs.csv', index=False)
    # Mean across repeated seeds; each run must use identical evaluation rows.
    summary = metrics.groupby(['dataset', 'horizon', 'model'], as_index=False)[['row_brier', 'fact_brier', 'event_ap', 'event_ap_within_origin', 'event_auroc', 'calibration_gap']].mean()
    summary.to_csv(output / f'{phase}_benchmark_table.csv', index=False)
    datasets = sorted(summary.dataset.unique())
    columns = [(dataset, int(h)) for dataset in datasets
               for h in sorted(summary.loc[summary.dataset.eq(dataset), 'horizon'].unique())]
    models = [name for name in MODEL_ORDER if name in set(summary.model)]
    models += sorted(set(summary.model) - set(models))
    lines = [f'% {phase} Brier; {"PARTIAL ORIGINS" if partial else "complete population"}; lower is better.',
             r'\begin{tabular}{l' + 'r' * len(columns) + '}', r'\toprule',
             'Model & ' + ' & '.join(r'\multicolumn{' + str(sum(d == dataset for d, _ in columns)) +
                                      '}{c}{' + dataset.replace('_', r'\_') + '}' for dataset in datasets) + r' \\',
             ' & ' + ' & '.join(f'{h}{"y" if d == "yago" else "m"}' for d, h in columns) + r' \\', r'\midrule']
    for metric in ('row_brier', 'fact_brier'):
        lines.append(r'\multicolumn{' + str(len(columns) + 1) + '}{l}{' +
                     ('Row Brier $\downarrow$' if metric == 'row_brier' else 'Fact-equal Brier $\downarrow$') + r'} \\')
        for model in models:
            cells = []
            for dataset, horizon in columns:
                group = summary.loc[summary.dataset.eq(dataset) & summary.horizon.eq(horizon)]
                value = group.loc[group.model.eq(model), metric]
                if value.empty:
                    cells.append('--')
                    continue
                score = float(value.iloc[0])
                ranks = sorted(group[metric].unique())
                formatted = f'{score:.6f}'
                if score == ranks[0]:
                    formatted = r'\textbf{' + formatted + '}'
                elif len(ranks) > 1 and score == ranks[1]:
                    formatted = r'\underline{' + formatted + '}'
                cells.append(formatted)
            lines.append(LABELS.get(model, model.replace('_', r'\_')) + ' & ' + ' & '.join(cells) + r' \\')
        if metric == 'row_brier':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    (output / f'{phase}_benchmark_table.tex').write_text('\n'.join(lines) + '\n')
    return summary
