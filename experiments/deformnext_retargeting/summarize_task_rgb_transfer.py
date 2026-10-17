"""Audit the matched six-run current-data RGB pilot and pair-cluster contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .reproduce_q2c import checked, read, reference, write_new

SEEDS = (42017, 42018, 42019)
METRICS = ('task_rgb_l1', 'raw_soft_rgb_l1', 'learned_top1_rgb_l1', 'raw_top1_rgb_l1',
           'DINO_teacher_cosine_error', 'RAW_DINO_teacher_cosine_error',
           'exact_patch_top1_accuracy', 'RAW_exact_patch_top1_accuracy', 'proposal_recall')
BASELINES = ('raw_soft_rgb_l1', 'raw_top1_rgb_l1', 'RAW_DINO_teacher_cosine_error',
             'RAW_exact_patch_top1_accuracy', 'proposal_recall')


def paired_ci(values):
    """Average seeds/cues/queries within each unique source cluster first."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('expected finite pair-cluster means')
    rng = np.random.default_rng(20260916)
    bootstrap = values[rng.integers(len(values), size=(10000, len(values)))].mean(1)
    return {'mean': float(values.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist(),
            'independent_source_clusters': len(values)}


def audit_sampling(first, second):
    if len(first) != 1200 or len(second) != 1200:
        raise ValueError('matched pilot must have all1200 updates')
    for update, (left, right) in enumerate(zip(first, second), 1):
        if any(left[key] != right[key] for key in ('update', 'sample_id', 'sampled_query_sha256')):
            raise ValueError('paired training sample/query sequence differs')
        if left['update'] != update or not np.isfinite([left['loss'], right['loss']]).all():
            raise ValueError('invalid optimizer update history')
    if first[0]['loss'] != second[0]['loss']:
        raise ValueError('initial matched task losses differ')


def summarize(inputs, root, output):
    population = read(inputs / 'summary.json')
    if population['status'] != 'PASS' or not population['full_population'] or len(population['records']) != 320:
        raise ValueError('expected the frozen256/64 population cache')
    calibration = {r['sample_id']: r for r in population['records'] if r['role'] == 'calibration'}
    train = {r['sample_id']: r for r in population['records'] if r['role'] == 'train' and r['eligible_queries']}
    pair_ids = sorted(sid for sid, row in calibration.items() if row['eligible_queries'])
    if (len(calibration) != 64 or len(train) != 251 or len(pair_ids) != 61
            or len({calibration[sid]['source_cluster_id'] for sid in pair_ids}) != 61
            or {r['source_cluster_id'] for r in train.values()} & {r['source_cluster_id'] for r in calibration.values()}):
        raise ValueError('frozen role/source-cluster membership differs')
    runs, logs, artifacts = {}, {}, [reference(inputs / 'summary.json')]
    for arm, name in [('D0-PAIRWISE', 'd0'), ('D1-Q2C', 'd1')]:
        for seed in SEEDS:
            folder = root / name / f'seed{seed}'
            result = read(folder / 'summary.json')
            if (result['status'] != 'PASS' or (result['arm'], result['seed']) != (arm, seed)
                    or result['completed_updates'] != 1200 or result['training_assets_with_queries'] != 251
                    or result['evaluated_calibration_assets'] != 61
                    or not result['initial_D0_D1_RAW_scores_exact']
                    or not result['calibration_opened_after_final_checkpoint_freeze']
                    or result['production_promotion'] or result['SS_gain_claim']):
                raise ValueError('incomplete or different current-task pilot')
            freeze = read(folder / 'checkpoint-freeze.json')
            if freeze['checkpoint'] != result['checkpoint'] or freeze['learned_calibration_values_opened']:
                raise ValueError('evaluation preceded the fixed checkpoint freeze')
            state = torch.load(checked(result['checkpoint']), map_location='cpu', weights_only=True)
            if (state['input_summary'] != reference(inputs / 'summary.json')
                    or state['completed_updates'] != 1200 or (state['arm'], state['seed']) != (arm, seed)
                    or 'matcher_contract' in state or 'schema' in state):
                raise ValueError('pilot checkpoint identity or production isolation differs')
            log = [json.loads(line) for line in (folder / 'train-log.jsonl').read_text().splitlines()]
            if any(row['sample_id'] not in train for row in log):
                raise ValueError('unexpected training asset')
            rows = {r['sample_id']: r for r in result['calibration_rows']}
            if len(rows) != 64 or len(result['calibration_rows']) != 64 or set(rows) != set(calibration):
                raise ValueError('calibration identity/exclusion population differs')
            for sid, row in rows.items():
                if (row['source_cluster_id'] != calibration[sid]['source_cluster_id']
                        or row['queries'] != calibration[sid]['eligible_queries']):
                    raise ValueError('calibration geometry/query population differs')
                if row['queries'] and (len(row['cue_errors']) != 16 or
                        not np.isfinite([row[m] for m in METRICS] + row['cue_errors']).all()):
                    raise ValueError('nonfinite or incomplete evaluation cue bank')
            runs[name, seed], logs[name, seed] = rows, log
            artifacts += [reference(folder / file) for file in
                          ['summary.json', 'checkpoint-freeze.json', 'checkpoint-final.pt', 'train-log.jsonl']]
    reference_rows = runs['d0', SEEDS[0]]
    for seed in SEEDS:
        audit_sampling(logs['d0', seed], logs['d1', seed])
        for name in ['d0', 'd1']:
            for sid in pair_ids:
                if any(runs[name, seed][sid][m] != reference_rows[sid][m] for m in BASELINES):
                    raise ValueError('fixed RAW proposals/values/control metrics differ')
    means = {name: {metric: np.mean([[runs[name, seed][sid][metric] for sid in pair_ids]
                        for seed in SEEDS], 0) for metric in METRICS} for name in ['d0', 'd1']}
    contrasts = {metric: paired_ci(means['d0'][metric] - means['d1'][metric]) for metric in
                 ['task_rgb_l1', 'learned_top1_rgb_l1', 'DINO_teacher_cosine_error']}
    contrasts['exact_patch_top1_accuracy'] = paired_ci(
        means['d1']['exact_patch_top1_accuracy'] - means['d0']['exact_patch_top1_accuracy'])
    seed_gains = [float(np.mean([runs['d0', seed][sid]['task_rgb_l1'] -
                  runs['d1', seed][sid]['task_rgb_l1'] for sid in pair_ids])) for seed in SEEDS]
    cue_gains = [float(np.mean([runs['d0', seed][sid]['cue_errors'][cue] -
                 runs['d1', seed][sid]['cue_errors'][cue] for seed in SEEDS for sid in pair_ids])) for cue in range(16)]
    report = {'status': 'PASS_MATCHED_CURRENT_TASK', 'scope': 'new current-DeformNeXt controlled patch-RGB variant',
        'training_runs': 6, 'model_seeds': list(SEEDS), 'optimizer_updates_per_run': 1200,
        'training_assets_with_queries': 251, 'calibration_assets': 64, 'calibration_source_clusters_with_queries': 61,
        'aggregation': 'query/channel/cue mean -> model-seed mean inside each unique source cluster -> cluster mean',
        'bootstrap_resamples': 10000, 'bootstrap_seed': 20260916,
        'all3600_paired_training_sample_and_query_hashes_exact': True, 'all_three_initial_losses_exact': True,
        'fixed_RAW_control_metrics_exact_across_all_six_runs': True,
        'arms': {name: {m: float(values.mean()) for m, values in metrics.items()} for name, metrics in means.items()},
        'D1_improvement_over_D0': contrasts, 'task_rgb_seed_gains': seed_gains,
        'task_rgb_cue_layout_gains': cue_gains, 'worst_quarter_cue_gain': float(np.mean(sorted(cue_gains)[:4])),
        'pair_task_rgb_gains': {sid: float(value) for sid, value in
            zip(pair_ids, means['d0']['task_rgb_l1'] - means['d1']['task_rgb_l1'])},
        'excluded_no_query_calibration_assets': [sid for sid in calibration if sid not in pair_ids],
        'inputs': artifacts, 'validation508_task_values_opened': False,
        'production_promotion': False, 'SS_generation_gain_claim': False,
        'uncertainty_scope': 'paired source-cluster variation on calibration61; not an unseen-test or population-of-training-seeds CI'}
    write_new(output, report)
    print(json.dumps({k: report[k] for k in ['status', 'arms', 'D1_improvement_over_D0', 'task_rgb_seed_gains']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['inputs', 'root', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summarize(args.inputs, args.root, args.output)


if __name__ == '__main__':
    main()
