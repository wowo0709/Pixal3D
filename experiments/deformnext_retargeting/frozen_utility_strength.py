"""Isolate utility logit strength while retaining the frozen semantic matching prior."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from .frozen_matcher_utility import validate_profile as validate_reference
from .matching_v2_ss import train
from .semantic_topk_confirmation import seed_config
from .semantic_topk_confirmation import make_jobs
from .semantic_topk_functional_qa import job as functional_job
from .matching_v2_queue import gpu_ids


NAME = 'utility-scale16-c1'
NAMESPACE = 'utility-strength-followup-v1'
INITIAL_SCALE = 16.


def validate_profile(cfg, root):
    reference = copy.deepcopy(cfg)
    task = reference['trainer']['args']['deform_conditioner']['task_attention']
    scale = task.pop('initial_content_scale', None)
    if isinstance(scale, bool) or scale != INITIAL_SCALE:
        raise ValueError('strength study requires explicit initial_content_scale=16')
    if cfg['experiment']['arm'] != NAME or cfg['training_seed'] != 101:
        raise ValueError('declared fresh seed101 utility-strength arm required')
    expected_recipe = (root / 'configs' / (NAME + '.json')).resolve()
    expected_run = (root / 'runs_ss64' / NAME).resolve()
    if (Path(cfg['config']).resolve() != expected_recipe
            or Path(cfg['experiment']['recipe_path']).resolve() != expected_recipe
            or Path(cfg['output_dir']).resolve() != expected_run):
        raise ValueError('strength study must not reuse the reference recipe or output directory')
    # The existing strict profile still checks every frozen-matcher/task switch,
    # temperature, native C1 channels, optimizer, budget and per-rank batch.
    validate_reference(reference, root)


def recipe(source, root):
    """Change one model initialization value; keep source optimization/data intact."""
    cfg = seed_config(source, root, NAME, 101)
    cfg['trainer']['args']['deform_conditioner']['task_attention']['initial_content_scale'] = INITIAL_SCALE
    cfg.update(master_port='30444', wandb_id='ss-frozen-utility-scale16-c1-s101',
               wandb_name='SS frozen matching utility scale16 C1 seed101')
    cfg['experiment'].update(release_status='pending-native-and-memory-proof')
    validate_profile(cfg, root)
    return cfg


def summarize(root, campaign):
    from .summarize_frozen_utility_followup import summarize as summarize_saved_profile
    summarize_saved_profile(root, campaign, name=NAME, namespace=NAMESPACE,
                            profile_validator=validate_profile,
                            extra_sources={'frozen-utility-c1': (
                                campaign / 'frozen-utility-followup-v1', 'frozen-utility-c1')})


def jobs(root, campaign, previous_work, devices, *, allowed_gpus):
    assigned = gpu_ids(','.join(allowed_gpus))
    if len(devices) != 2 or len(set(devices)) != 2 or any(device not in assigned for device in devices):
        raise ValueError('two distinct assigned physical GPUs required')
    result = make_jobs(root, previous_work, NAME, devices, [], 101, prefix='')
    command = result[0]['command']
    command[command.index('experiments.deformnext_retargeting.matching_v2_ss')] = (
        'experiments.deformnext_retargeting.frozen_utility_strength')
    qa = functional_job(root, root, NAME, 101, previous_work / 'panels/panel-128.json')
    result.append(qa)
    result.append({'name': 'summarize-utility-strength-fixed-final', 'resource': 'cpu', 'priority': 1,
        'depends_on': [f'final-{NAME}-noise{noise}' for noise in range(3)] + [qa['name']],
        'requires': [str(root / 'epoch-audits-v1' / ('train-' + NAME) / f'epoch{e:02d}/file4-closure.json')
                     for e in range(1, 6)]
                    + [str(root / 'final-weight-audits-v1' / ('train-' + NAME) / 'file4-closure.json')],
        'outputs': [str(root / 'fixed-final-decision-v1.json')],
        'command': ['taskset', '-c', '84,85', 'env', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=1',
                    sys.executable, '-u', '-m', 'experiments.deformnext_retargeting.frozen_utility_strength',
                    '--mode', 'summarize', '--root', str(root), '--campaign-root', str(campaign)]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'summarize'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--campaign-root', type=Path)
    args = parser.parse_args()
    if args.mode == 'train':
        if args.config is None:
            parser.error('--config is required for training')
        validate_profile(json.loads(args.config.read_text()), args.root)
        train(args)
    else:
        if args.campaign_root is None:
            parser.error('--campaign-root is required for summary')
        summarize(args.root, args.campaign_root)


if __name__ == '__main__':
    main()
