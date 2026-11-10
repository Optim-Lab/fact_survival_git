"""main.py --model logcl에서 저자 인코더를 학습·평가하고 성능표를 저장합니다."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.config import DATASET_CONFIGS, PROJECT_ROOT
from datasets.task import PROTOCOL_RULES, PROTOCOL_VERSION
from evaluation.benchmark_table import read_runs, write_tables
from evaluation.bootstrap import holm_adjust
from modules.proposed_experiment import (
    attach_baselines, file_hash, load_data, metrics_table, paired_bootstrap,
)
from modules.logcl_original_survival import fit_predict_origin, check_upstream_hashes

SOURCE_FILES = [
    'main.py', 'modules/logcl_experiment.py', 'modules/logcl_original_survival.py',
    'vendor/logcl_provenance.json',
    'modules/regcn_survival.py',
    'modules/neural_hazard.py', 'modules/proposed_experiment.py',
    'modules/halo_half_life.py', 'datasets/task.py', 'datasets/config.py',
    'datasets/interval_landmark.py', 'datasets/yago.py',
    'evaluation/metrics.py', 'evaluation/benchmark_table.py',
]


def run(args):
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    if args.merge_parts:
        raise ValueError('LogCL survival adaptation partition merging is not implemented; use a complete run')
    if args.development_origins and (args.evaluate_test or args.selection_from or args.test_origins):
        raise ValueError('Partial development runs cannot select epochs or evaluate test')
    if args.test_origins and not (args.evaluate_test and args.selection_from):
        raise ValueError('Partial test origins require a frozen development selection')
    if args.selection_from and not args.evaluate_test:
        raise ValueError('--selection-from requires --evaluate-test')
    if (min(args.epochs + [args.hidden, args.batch_size, args.threads, args.history_length,
                           args.contrastive_batch]) < 1 or
            not np.isfinite(args.contrastive_weight) or args.contrastive_weight < 0):
        raise ValueError('Epochs, hidden, batch size, threads and history length must be positive')
    config = DATASET_CONFIGS[args.dataset]
    horizons = args.horizons or config['horizons']
    if horizons != config['horizons']:
        raise ValueError('Use the complete configured horizon set')
    source = Path(args.data or config['data_path'])
    output = Path(args.output or PROJECT_ROOT / 'results' / PROTOCOL_VERSION /
                  f'logcl_{args.dataset}_seed{args.seed}')
    if output.exists():
        raise ValueError(f'Choose a fresh output directory: {output}')
    inputs = [source]
    if args.dataset == 'yago':
        inputs += [source.with_name('yago_endpoint_reference_DO_NOT_USE_AS_FEATURE.csv'),
                   source.with_name('yago_survival_build_summary.json')]
    provenance = check_upstream_hashes()
    encoder_hashes = {}
    for name in provenance['files']:
        for folder in ('LogCL', 'logcl_cpu'):
            relative = f'vendor/{folder}/{name}'
            encoder_hashes[relative] = file_hash(PROJECT_ROOT / relative)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    rows, x, _, labels, _, dev_origins, _ = load_data(args.dataset, source, horizons)
    origin_values = rows.origin.to_numpy()
    observed = np.isfinite(labels)
    selected_epochs = sorted(set(args.epochs))
    manifest = dict(
        dataset=args.dataset, architecture='logcl_original_encoder_survival_v1',
        source=str(source), input_sha256={str(p): file_hash(p) for p in inputs},
        source_hashes={name: file_hash(PROJECT_ROOT / name) for name in SOURCE_FILES} | encoder_hashes,
        protocol_version=PROTOCOL_VERSION, protocol=PROTOCOL_RULES,
        horizons=horizons, candidate_epochs=selected_epochs, hidden=args.hidden,
        batch_size=args.batch_size, threads=args.threads, history_length=args.history_length,
        seed=args.seed, torch_version=torch.__version__, numpy_version=np.__version__,
        loss=('shared masked horizon BCE + local/global InfoNCE' if args.contrastive_weight else
              'shared masked horizon BCE'), optimizer='AdamW', learning_rate=.003,
        global_history='causal two-hop neighborhood without sampled-edge cap',
        contrastive_weight=args.contrastive_weight,
        contrastive_batch=args.contrastive_batch,
        weight_decay=.001, head='positive piecewise hazard',
        encoder='official LogCL RecurrentRGCN local/global encoder',
        contrastive='official get_loss_conv local/global query representations',
        original_task='TKG missing-entity prediction',
        adaptation_task='active-fact survival',
        epoch_selection='minimum mean development row Brier across horizons',
        requested_development_origins=args.development_origins,
        partial_development=bool(args.development_origins),
        requested_test_origins=args.test_origins, partial_test=bool(args.test_origins),
        test_evaluation='not_run', test_used_for_selection=False,
        benchmark_eligible=not bool(args.development_origins or args.test_origins), folds=[])
    selection = None
    if args.selection_from:
        prior = json.loads((args.selection_from / 'run_config.json').read_text())
        for key in ('dataset', 'architecture', 'input_sha256', 'source_hashes',
                    'protocol_version', 'protocol', 'horizons', 'hidden',
                    'batch_size', 'history_length', 'seed', 'global_history',
                    'contrastive_weight', 'contrastive_batch'):
            if manifest[key] != prior[key]:
                raise ValueError(f'Frozen development run differs: {key}')
        if prior.get('partial_development') or prior.get('test_used_for_selection'):
            raise ValueError('Selection requires a complete development run')
        selection = json.loads((args.selection_from / 'selection.json').read_text())
        manifest['selection_from'] = str(args.selection_from.resolve())
        manifest['selection_sha256'] = file_hash(args.selection_from / 'selection.json')
    output.mkdir(parents=True)
    phases = ['test'] if selection else (['development', 'test'] if args.evaluate_test else ['development'])
    for phase in phases:
        allowed = (sorted(set(v for origins in dev_origins.values() for v in origins))
                   if phase == 'development' else sorted(rows.loc[rows.split.eq('test'), 'origin'].unique()))
        requested = args.development_origins if phase == 'development' else args.test_origins
        if requested and not set(requested).issubset(allowed):
            raise ValueError(f'Requested {phase} origin is outside the configured period')
        origins = sorted(set(requested)) if requested else allowed
        frames = []
        for origin in origins:
            # Raw data after origin is excluded even from graph construction.
            current = origin_values <= origin
            current_rows = rows.loc[current].reset_index(drop=True)
            current_labels = labels[current]
            epoch_request = [selection['epoch']] if phase == 'test' else selected_epochs
            evaluation, values, diagnostics, checkpoint = fit_predict_origin(
                current_rows, x[current], current_labels, horizons, int(origin),
                epochs=epoch_request, hidden=args.hidden, batch_size=args.batch_size,
                history_length=args.history_length, seed=args.seed,
                contrastive_weight=args.contrastive_weight,
                contrastive_batch=args.contrastive_batch)
            for k, h in enumerate(horizons):
                if phase == 'development' and origin not in dev_origins[h]:
                    continue
                visible = np.isfinite(current_labels[evaluation, k])
                frame = current_rows.iloc[evaluation[visible]].copy()
                frame['horizon'] = h
                frame['survived'] = current_labels[evaluation[visible], k].astype(int)
                if phase == 'development':
                    for epoch, probability in values.items():
                        frame[f'p_logcl_e{epoch}'] = probability[visible, k]
                else:
                    frame['p_logcl'] = values[selection['epoch']][visible, k]
                frames.append(frame)
            manifest['folds'].append(dict(phase=phase, origin=int(origin), **diagnostics))
            checkpoint['fold_origin'] = int(origin)
            torch.save(checkpoint, output / f'{phase}_logcl_last.pt')
            print(f'{phase} origin={origin}: train={diagnostics["train_rows"]:,} '
                  f'risk={diagnostics["evaluation_risk_rows"]:,}', flush=True)
        if not frames:
            raise ValueError(f'No {phase} evaluation rows')
        predictions = pd.concat(frames, ignore_index=True)
        if predictions.duplicated(['fact_id', 'origin', 'horizon']).any():
            raise AssertionError('Duplicate evaluation rows')
        if phase == 'development':
            candidates = metrics_table(predictions)
            candidates.to_csv(output / 'development_candidates.csv', index=False)
            if not args.development_origins:
                scores = candidates.groupby('model').row_brier.mean()
                chosen = scores.idxmin()
                selection = dict(epoch=int(chosen.rsplit('_e', 1)[1]),
                                 mean_horizon_brier=float(scores[chosen]))
                (output / 'selection.json').write_text(json.dumps(selection, indent=2))
                predictions['p_logcl'] = predictions[f'p_logcl_e{selection["epoch"]}']
                predictions = predictions.drop(columns=[c for c in predictions if c.startswith('p_logcl_e')])
            else:
                # A partial origin is a smoke report, never an epoch selection.
                predictions['p_logcl'] = predictions[f'p_logcl_e{max(selected_epochs)}']
                predictions = predictions.drop(columns=[c for c in predictions if c.startswith('p_logcl_e')])
        # Attach stored baseline rows only for a complete population.
        if not requested and selection:
            predictions = attach_baselines(predictions, args.dataset, phase, horizons)
        predictions.to_csv(output / f'{phase}_predictions.csv.gz', index=False, compression='gzip')
        metrics = metrics_table(predictions)
        metrics.to_csv(output / f'{phase}_metrics.csv', index=False)
        if phase == 'test' and not requested:
            comparisons = []
            for h, group in predictions.groupby('horizon'):
                for reference in ('p_global', 'p_relation_age', 'p_object_degree'):
                    if reference in group:
                        comparisons.append(dict(horizon=int(h), model='logcl',
                                                reference=reference.removeprefix('p_'),
                                                **paired_bootstrap(group, 'p_logcl', reference,
                                                                   args.bootstrap_repetitions or 1000,
                                                                   args.seed + int(h))))
            if comparisons:
                significance = pd.DataFrame(comparisons)
                significance['holm_adjusted_p'] = holm_adjust(significance.p_value.tolist())
                significance.to_csv(output / 'test_significance.csv', index=False)
        manifest['test_evaluation'] = ('rolling_origin_partial' if requested else 'rolling_origin') if phase == 'test' else 'not_run'
        manifest['selection'] = selection
        manifest['elapsed_seconds'] = time.perf_counter() - started
        (output / 'run_config.json').write_text(json.dumps(manifest, indent=2))
        if selection or requested:
            combined = read_runs([output], phase=phase, allow_partial=bool(requested))
            write_tables(combined, output, phase=phase, partial=bool(requested))
        print(metrics.pivot(index='model', columns='horizon', values='row_brier').to_string(), flush=True)
    return output
