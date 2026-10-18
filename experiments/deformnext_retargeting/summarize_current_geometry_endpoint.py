"""Audit all six current-controller runs before paired source-cluster analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .historical_geometry_runtime import _seed
from .reproduce_q2c import checked, read, reference, write_new

SEEDS = (101, 202, 303)
ARMS = ('adapter-only', 'adapter-query')
METRICS = ('voxel_iou', 'voxel_f1', 'source_recall64', 'source_recall32', 'stale_leak64', 'stale_leak32',
           'visible_state_acc64', 'visible_state_acc32', 'regional_bce')


def cluster_contrast(baseline, candidate, metric, ids):
    """One mean per source cluster, retaining all model/noise repeats inside."""
    differences, seed_differences = [], {seed: [] for seed in SEEDS}
    for sid in ids:
        values = []
        missing = []
        for seed in SEEDS:
            one_seed = []
            for noise in [0, 1, 2]:
                left, right = baseline[seed][sid, noise][metric], candidate[seed][sid, noise][metric]
                if (left is None) != (right is None):
                    raise ValueError('metric availability changed with method or noise')
                missing.append(left is None)
                if left is not None:
                    if not np.isfinite([left, right]).all():
                        raise ValueError('nonfinite calibration metric')
                    one_seed.append(float(right)-float(left))
            values.extend(one_seed)
            if one_seed:
                seed_differences[seed].append(float(np.mean(one_seed)))
        if any(missing) and not all(missing):
            raise ValueError('region availability changed across model/noise repeats')
        if values:
            differences.append((sid, float(np.mean(values))))
    if not differences:
        return {'mean': None, 'ci95': None, 'source_clusters': 0, 'seed_means': None, 'pair_differences': {}}
    values = np.array([value for _, value in differences], np.float64)
    rng = np.random.default_rng(20260918)
    resampled = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(resampled, [.025, .975]).tolist(),
        'source_clusters': len(values), 'seed_means': {str(seed): float(np.mean(v)) for seed, v in seed_differences.items()},
        'pair_differences': dict(differences)}


def audit_history(history, train_ids, seed):
    if len(history) != 128 or [r['update'] for r in history] != list(range(1, 129)):
        raise ValueError('controller did not complete all128 distinct updates')
    for update, row in enumerate(history):
        epoch, position = divmod(update, 32)
        expected = sorted(train_ids, key=lambda sid: hashlib.sha256(f'{seed}|{epoch}|{sid}'.encode()).hexdigest())[position]
        if (row['epoch'], row['sample_id'], row['noise_seed']) != (
                epoch, expected, _seed('ss-geometry-factorial', seed, expected, epoch)):
            raise ValueError('controller differs from the original fixed pair/noise schedule')
        if not np.isfinite([row['loss'], row['endpoint_loss'], row['gradient_norm']]).all() or row['gradient_norm'] <= 0:
            raise ValueError('nonfinite or disconnected regional training')


def summarize(root, conditions, labels, output):
    cache, authority = read(conditions/'summary.json'), read(labels/'summary.json')
    if cache['status'] != 'PASS' or authority['status'] != 'PASS' or cache['population'] != authority['population']:
        raise ValueError('current condition/GT population is incomplete or different')
    train_ids = [r['sample_id'] for r in cache['records'] if r['role'] == 'train']
    calibration = {r['sample_id']: r for r in cache['records'] if r['role'] == 'calibration'}
    ids = sorted(calibration)
    if len(train_ids) != 32 or len(ids) != 32 or len({r['source_cluster_id'] for r in calibration.values()}) != 32:
        raise ValueError('expected current TRAIN32/cal32 with unique source clusters')
    for row in cache['records']:
        with np.load(checked(row['geometry_query']), allow_pickle=False) as data:
            protocol = json.loads(str(data['protocol']))
            if (str(data['sample_id']) != row['sample_id'] or protocol['consumer'] != cache['consumer']
                    or protocol['input_GT_geometry_or_latent_opened']
                    or protocol['O_states'] != ['M0']*4 or protocol['D_states'] != ['M1']):
                raise ValueError('query is not the fixed RGB-only original/desired predicted geometry')
    evaluations, histories, artifacts, audits = {}, {}, [], []
    for arm in ARMS:
        for seed in SEEDS:
            folder = root/arm/f'seed{seed}'
            result = read(folder/'summary.json')
            if (result['status'] != 'PASS' or (result['arm'], result['seed'], result['updates']) != (arm, seed, 128)
                    or result['calibration_clusters'] != 32 or result['calibration_noise_seeds'] != 3
                    or result['production_promotion'] or result['production_global96_recipe_changed']):
                raise ValueError('incomplete or different current frozen-controller run')
            freeze = read(folder/'checkpoint-freeze.json')
            if freeze['checkpoint'] != result['checkpoint'] or freeze['calibration_GT_values_opened'] or freeze['completed_updates'] != 128:
                raise ValueError('calibration was not frozen behind fixed-final checkpoint')
            saved = torch.load(checked(result['checkpoint']), map_location='cpu', weights_only=True)
            if (saved['consumer'] != cache['consumer'] or saved['conditions'] != reference(conditions/'summary.json')
                    or saved['regions'] != reference(labels/'summary.json') or saved['updates'] != 128
                    or saved['arm'] != arm or saved['seed'] != seed or saved['train_pairs'] != sorted(train_ids)):
                raise ValueError('checkpoint consumer, supervised split or inputs differ')
            weights = saved['adapter']
            width = 3075 if arm == 'adapter-only' else 3107
            if (tuple(weights['residual.0.weight'].shape) != (width,)
                    or any(k.startswith('content_encoder.') for k in weights) != (arm == 'adapter-query')
                    or not saved['optimizer']['state']
                    or any(int(state['step']) != 128 for state in saved['optimizer']['state'].values())):
                raise ValueError('physical channel removal or actual optimizer budget differs')
            history = [json.loads(line) for line in (folder/'train-log.jsonl').read_text().splitlines()]
            audit_history(history, train_ids, seed)
            histories[arm, seed] = history
            for case in ['native', 'normal', 'projected-source-off'] + (['shuffled-query'] if arm == 'adapter-query' else []):
                path = folder/'evaluation'/(case+'.json')
                evaluation = read(path)
                if (evaluation['status'] != 'PASS' or evaluation['case'] != case
                        or evaluation['sampling_steps'] != 12 or evaluation['guidance'] != 3.
                        or evaluation['rescale_t'] != 1. or evaluation['generation_batch'] != 32):
                    raise ValueError('current fixed-final evaluator protocol differs')
                rows = {(r['sample_id'], r['noise_index']): r for r in evaluation['per_asset']}
                if len(rows) != 96 or len(evaluation['per_asset']) != 96 or set(rows) != {(sid, n) for sid in ids for n in [0, 1, 2]}:
                    raise ValueError('calibration32/three-noise population differs')
                if any(r['source_cluster_id'] != calibration[r['sample_id']]['source_cluster_id'] for r in rows.values()):
                    raise ValueError('calibration source cluster changed')
                evaluations[arm, seed, case] = rows
                artifacts.append(reference(path))
            artifacts += [reference(folder/name) for name in ['summary.json', 'checkpoint-final.pt', 'checkpoint-freeze.json', 'train-log.jsonl']]
            audits.append({'arm': arm, 'seed': seed, 'updates': 128, 'physical_residual_width': width,
                'no_GT_query_input': True, 'checkpoint': result['checkpoint']})
    for seed in SEEDS:
        left, right = histories['adapter-only', seed], histories['adapter-query', seed]
        if any(any(a[k] != b[k] for k in ['update', 'epoch', 'sample_id', 'noise_seed']) for a, b in zip(left, right)) or left[0]['loss'] != right[0]['loss']:
            raise ValueError('paired data/noise/initial native regional loss differs')
    reference_rows = evaluations['adapter-only', SEEDS[0], 'native']
    for (arm, seed, case), rows in evaluations.items():
        for key, row in rows.items():
            if row['target_voxels_sha256'] != reference_rows[key]['target_voxels_sha256']:
                raise ValueError('GT voxels changed with method/model/noise')
            if case == 'native' and row['predicted_voxels_sha256'] != reference_rows[key]['predicted_voxels_sha256']:
                raise ValueError('frozen native predictions differ across seeds or devices; audit before attributing effects')
    # Independently reconstruct the evaluation GT from the current actual inputs.
    label_lookup = {r['sample_id']: r for r in authority['records']}
    for sid in ids:
        with np.load(checked(label_lookup[sid]['artifact']), allow_pickle=False) as data:
            target = np.zeros((64, 64, 64), bool)
            target[tuple(data['target64'].astype(np.int64).T)] = True
        digest = hashlib.sha256(np.packbits(target.reshape(-1)).tobytes()).hexdigest()
        if any(reference_rows[sid, noise]['target_voxels_sha256'] != digest for noise in [0, 1, 2]):
            raise ValueError('evaluator GT differs from the actual SS-input label authority')
    view = lambda arm, case: {seed: evaluations[arm, seed, case] for seed in SEEDS}
    definitions = {
        'query_minus_adapter': (view('adapter-only', 'normal'), view('adapter-query', 'normal')),
        'adapter_minus_native': (view('adapter-only', 'native'), view('adapter-only', 'normal')),
        'query_minus_native': (view('adapter-query', 'native'), view('adapter-query', 'normal')),
        'adapter_source_ON_minus_projected_OFF': (view('adapter-only', 'projected-source-off'), view('adapter-only', 'normal')),
        'query_source_ON_minus_projected_OFF': (view('adapter-query', 'projected-source-off'), view('adapter-query', 'normal')),
        'query_intact_minus_joint_shuffle': (view('adapter-query', 'shuffled-query'), view('adapter-query', 'normal'))}
    contrasts = {name: {metric: cluster_contrast(left, right, metric, ids) for metric in METRICS}
                 for name, (left, right) in definitions.items()}
    result = {'status': 'PASS_COMPLETE_CURRENT_REGIONAL_PILOT', 'training_runs': 6, 'model_seeds': list(SEEDS),
        'updates_per_run': 128, 'calibration_source_clusters': 32, 'calibration_noise_seeds': 3,
        'paired_training_schedule_and_initial_loss_exact': True, 'all_frozen_native_prediction_hashes_exact': True,
        'all_GT_hashes_exact_and_verified_against_actual_SS_input': True,
        'contrast_sign': 'candidate minus baseline; negative is improvement for stale_leak and regional_bce',
        'aggregation': 'model/noise repeats averaged within unique source cluster, then source-cluster bootstrap10000 seed20260918',
        'uncertainty_scope': 'source-cluster CI conditional on these3 training seeds; seed means also reported',
        'contrasts': contrasts, 'run_audits': audits, 'inputs': artifacts,
        'condition_authority': reference(conditions/'summary.json'), 'GT_authority': reference(labels/'summary.json'),
        'query_uses_original_M0_camera08_in_addition_to_native_four_RGB': True,
        'projected_source_OFF_keeps_native_four_RGB_global_and_negative': True,
        'same_input_information_geometry_control': 'intact versus joint-tuple shuffle',
        'primary14_micro_IoU_comparison_replaced': False, 'production_promotion': False,
        'PBR_or_Full_selection_complete': False}
    write_new(output, result)
    print(json.dumps({'status': result['status'], 'training_runs': 6,
        'source_recall32': {key: {k: v for k, v in value['source_recall32'].items() if k != 'pair_differences'}
                            for key, value in contrasts.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'conditions', 'labels', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summarize(args.root, args.conditions, args.labels, args.output)


if __name__ == '__main__':
    main()
