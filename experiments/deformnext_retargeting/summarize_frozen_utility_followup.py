"""Fixed-final quality decision for the frozen-prior utility follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .compare_ss_noise_panels import compare
from .frozen_matcher_utility import validate_profile
from .semantic_topk_search import write_new
from .semantic_topk_selection import supported
from .summarize_semantic_topk_search import aggregate, task_gradient_audit, validate_evaluation
from .summarize_ss_module_study import audit_final_replay, audit_training


NAME = 'frozen-utility-c1'


def read(path):
    return json.loads(path.read_text())


def validate_saved_audits(final, qa, epochs):
    """Require actual frozen-head, trained-utility and epoch logging evidence."""
    expected = {
        'status': 'PASS_ACTUAL_FINAL_TENSORS_AND_OPTIMIZER',
        'frozen_base_tensors_exact': 701, 'LoRA_tensors_updated': 360,
        'matching_tensors_present': {'decoder': 18, 'score': 6},
        'matching_tensors_changed': {'decoder': [], 'score': []},
        'physical_deformation_query_channels': 0, 'retargeter_input_channels': 2048,
        'optimizer_states': 371, 'actual_optimizer_step': 480,
    }
    if any(final.get(key) != value for key, value in expected.items()):
        raise ValueError('final frozen-matcher/utility optimizer audit differs')
    for key in ['all1061_denoiser_tensors_finite', 'all_conditioner_tensors_finite',
                'all_optimizer_moments_finite']:
        if final.get(key) is not True:
            raise ValueError('final nonfinite/unaudited tensor state')
    if 'content_output.weight' not in final['task_attention_tensors_changed']:
        raise ValueError('zero-start utility output did not learn')
    if (qa.get('status') != 'PASS_TRAINED_SS_CACHE_AMP_CFG_CONTRACT'
            or qa.get('optimizer_updates') != 0
            or qa.get('saved_frozen_head_exact_to_selected_checkpoint') is not True
            or len(qa.get('frozen_head_state_keys', [])) != 24
            or qa.get('actual_trainable_matcher_saved_state_changed_keys') != []):
        raise ValueError('actual saved utility GPU conditioning proof missing')
    if len(epochs) != 5:
        raise ValueError('five actual epoch audits required')
    for number, epoch in enumerate(epochs, 1):
        expected_epoch = {
            'status': 'PASS_ACTUAL_EPOCH_OPTIMIZER_LOGGING_AND_BATCH',
            'epoch': number, 'step': 96 * number, 'training_seed': 101,
            'global_batch': 96, 'world_size': 2, 'microbatch_per_rank': 48,
            'accumulation': 1, 'actual_optimizer_states': 371,
            'all_optimizer_steps': 96 * number,
            'actual_remote_four_numeric_fields_verified': True,
            'actual_remote_PNG_byte_exact': True,
        }
        if any(epoch.get(key) != value for key, value in expected_epoch.items()):
            raise ValueError('epoch budget/optimizer/actual W&B evidence differs')


def summarize(root, campaign):
    if root.resolve() != (campaign / 'frozen-utility-followup-v1').resolve():
        raise ValueError('follow-up must use its declared campaign namespace')
    cfg = read(root / 'configs' / (NAME + '.json'))
    validate_profile(cfg, root)
    if cfg['training_seed'] != 101 or cfg['ckpt'] != 'none':
        raise ValueError('fresh original-SS4210 seed101 recipe required')
    original = read(campaign / 'seven-arm-final-report.json')
    decision = read(campaign / 'candidate-selection-v1.json')
    if decision['status'] != 'NO_PROMISING_COMBINATION_YET' or decision['selected_candidate'] is not None:
        raise ValueError('preserved negative first-round decision differs')
    training = audit_training(root, NAME, original['training_audits']['static-c1']['initial_iou'])
    records = {NAME: [read(root / 'evaluations' / NAME / f'noise-{n}/evaluation.json') for n in range(3)]}
    previous = campaign.parent / 'ss-seed-confirmation-v1'
    sources = {
        'v1': (previous, 'v1'), 'previous-c1': (previous, 'c1'),
        'static-top1-c1': (campaign, 'static-top1-c1'),
        'top1-reference': (campaign, 'top1-reference'),
        'task-full': (campaign, 'task-full'),
        'task-native-spatial': (campaign, 'task-native-spatial'),
    }
    inputs = [root / 'configs' / (NAME + '.json'), campaign / 'candidate-selection-v1.json',
              campaign / 'seven-arm-final-report.json']
    for name, (base, arm) in sources.items():
        paths = [base / 'evaluations' / arm / f'noise-{n}/evaluation.json' for n in range(3)]
        records[name] = [read(path) for path in paths]
        inputs.extend(paths)
    inputs.extend(root / 'evaluations' / NAME / f'noise-{n}/evaluation.json' for n in range(3))
    for values in records.values():
        for value in values:
            validate_evaluation(value)
    training['final128_replays_epoch32'] = audit_final_replay(training, records[NAME][0])
    training['task_branch_gradients'] = task_gradient_audit(root, NAME)
    final_folder = root / 'final-weight-audits-v1' / ('train-' + NAME)
    epoch_folders = [root / 'epoch-audits-v1' / ('train-' + NAME) / f'epoch{n:02d}' for n in range(1, 6)]
    qa_path = root / 'functional-QA-v1' / ('s101-' + NAME) / 'summary.json'
    validate_saved_audits(read(final_folder / 'audit.json'), read(qa_path),
                          [read(folder / 'audit.json') for folder in epoch_folders])
    inputs.extend([qa_path, final_folder / 'audit.json'])
    for folder in [*epoch_folders, final_folder]:
        path = folder / 'file4-closure.json'
        closure = read(path)
        if closure.get('status') != 'PASS' or closure.get('local_equals_actual_file4_SHA256') is not True:
            raise ValueError('actual file4 closed evidence required')
        audit_path = folder / 'audit.json'
        wanted = str(audit_path.relative_to(campaign.parent))
        entries = [entry for entry in closure['files'] if entry['relative_path'] == wanted]
        if (len(entries) != 1 or entries[0]['sha256'] != hashlib.sha256(audit_path.read_bytes()).hexdigest()
                or entries[0]['bytes'] != audit_path.stat().st_size):
            raise ValueError('closed audit bytes differ from the actual file4 receipt')
        inputs.extend([path, folder / 'audit.json'])
    contrasts = {name: compare([(str(n), records[name][n], records[NAME][n]) for n in range(3)])
                 for name in sources}
    eligible = all(supported(contrasts[name]) for name in ['v1', 'static-top1-c1', 'top1-reference'])
    report = {
        'status': 'PROMISING_FROZEN_UTILITY_REQUIRES_CONFIRMATION' if eligible else 'NO_FROZEN_UTILITY_IMPROVEMENT_CONFIRMED',
        'candidate': NAME, 'eligible_for_training_seed_confirmation': eligible,
        'models': {name: aggregate(values) for name, values in records.items()},
        'contrasts': contrasts, 'training_audit': training,
        'inputs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
                   for p in dict.fromkeys(inputs)],
        'selection_rule': 'Positive source-cluster 95% CI versus V1, same-retargeter Top1, and no-retargeter Top1; positive direction at all3noises. Fixed raw480 only.',
        'scope': 'Exploratory fixed128/93source/3noise, training seed101. Repeated generation noises are not independent training seeds. No multiplicity adjustment.',
        'first_round_negative_preserved': True, 'model_adopted': False, 'full_goal_complete': False,
        'next': 'Fresh seed202 utility and matched Top1+C1, then reserved121 source-disjoint validation.' if eligible
                else 'Retain negative evidence and continue diagnosis/model search; no adoption.',
    }
    output = root / 'fixed-final-decision-v1.json'
    if output.exists() or output.with_suffix('.md').exists():
        raise FileExistsError('preserve actual follow-up result')
    lines = ['# Frozen matcher utility: fixed-final decision', '', report['status'], '',
             '| Model | IoU | F1 |', '| --- | ---: | ---: |']
    for name, value in report['models'].items():
        lines.append(f"| {name} | {value['micro_voxel_iou']:.6f} | {value['micro_voxel_f1']:.6f} |")
    lines += ['', '| Utility versus | IoU gain, pp | 95% source-cluster CI, pp |', '| --- | ---: | --- |']
    for name, value in contrasts.items():
        lo, hi = value['paired_cluster_bootstrap_ci95']
        lines.append(f"| {name} | {100*value['gain']:+.3f} | [{100*lo:+.3f}, {100*hi:+.3f}] |")
    lines += ['', report['scope'], '', report['selection_rule'], '', report['next']]
    write_new(output, report)
    with output.with_suffix('.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    print(json.dumps({'status': report['status'], 'eligible': eligible, 'model_adopted': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    args = parser.parse_args()
    summarize(args.root, args.campaign_root)


if __name__ == '__main__':
    main()
