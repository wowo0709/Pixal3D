"""Soften the frozen score prior while retaining zero-start task utility."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

from .frozen_matcher_utility import validate_profile as validate_frozen
from .frozen_utility_strength import validate_profile as validate_strength
from .matching_v2_ss import train
from .loader_runtime_contract import validate_loader_runtime
from .semantic_topk_confirmation import seed_config
from .semantic_topk_search import write_new


NAME = 'utility-softprior-c1'
NAMESPACE = 'utility-softprior-followup-v1'
SOURCES = [
    'experiments/deformnext_retargeting/utility_softprior_followup.py',
    'experiments/deformnext_retargeting/loader_runtime_contract.py',
    'experiments/deformnext_retargeting/matching_v2_ss.py',
    'pixal3d/modules/deform_conditioner.py', 'pixal3d/modules/task_matching_attention.py',
    'pixal3d/datasets/deformnext.py',
    'pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py',
    'experiments/multiview_finetune/runtime.py',
]


def validate_profile(cfg, root):
    validate_loader_runtime(cfg)
    if (root.name != NAMESPACE or cfg['experiment']['arm'] != NAME
            or cfg['trainer']['name'] != 'ImageConditionedProjFlowMatchingCFGTrainer'
            or cfg['training_seed'] != 101 or cfg['experiment'].get('training_seed') != 101):
        raise ValueError('fresh seed101 patch-query soft-prior namespace and trainer required')
    expected = root / 'configs' / (NAME + '.json')
    run = root / 'runs_ss64' / NAME
    if (Path(cfg['config']).resolve() != expected.resolve()
            or Path(cfg['experiment']['recipe_path']).resolve() != expected.resolve()
            or Path(cfg['output_dir']).resolve() != run.resolve()):
        raise ValueError('soft-prior aliases must belong to the new arm')
    reference = copy.deepcopy(cfg)
    spec = reference['trainer']['args']['deform_conditioner']
    if (spec.get('score_temperature') != .05
            or spec['task_attention'].pop('initial_content_scale', None) != 16.
            or 'voxel_chunk_size' in spec):
        raise ValueError('retain patch queries and utility scale16; change frozen score T to.05 only')
    spec['score_temperature'] = .005
    validate_frozen(reference, root)


def recipe(source, root):
    validate_strength(source, Path(source['config']).parent.parent)
    cfg = seed_config(source, root, NAME, 101)
    cfg['trainer']['args']['deform_conditioner']['score_temperature'] = .05
    cfg.update(master_port='30447', wandb_id='ss-frozen-utility-softprior-c1-s101',
               wandb_name='SS frozen utility scale16 soft score prior T0.05 C1 seed101')
    cfg['experiment'].update(release_status='pending-native-and-DDP48-proof',
        matching_temperature=.05,
        comparison_scope='Single-change development search;freshSS4210,raw480,global96,world2,micro48,accumulation1',
        query_model_change='Frozen score temperature.005 to.05 only;retain patch query and zero-start utility scale16')
    validate_profile(cfg, root)
    return cfg


def source_record():
    repo = Path(__file__).resolve().parents[2]
    return {'commit_at_entry': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'actual_component_SHA256': {path: hashlib.sha256((repo / path).read_bytes()).hexdigest() for path in SOURCES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'train'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.mode == 'prepare':
        prepared = recipe(cfg, args.root)
        prepared['experiment']['source_commit'] = source_record()['commit_at_entry']
        write_new(args.root / 'configs' / (NAME + '.json'), prepared)
        print('PREPARED_NOT_GPU_VERIFIED_OR_TRAINED', flush=True)
        return
    validate_profile(cfg, args.root)
    before = source_record()
    folder = args.root / 'verification' / NAME
    write_new(folder / 'entry-source-contract.json', before)
    train(args)
    after = source_record()
    if before['actual_component_SHA256'] != after['actual_component_SHA256']:
        raise RuntimeError('soft-prior source changed during actual execution')
    write_new(folder / 'source-binding-train.json', {
        'status': 'PASS_SOFT_PRIOR_SOURCE_BINDING', **before,
        'commit_at_exit': after['commit_at_entry'], 'source_stable_through_execution': True,
        'actual_native_proof_SHA256': hashlib.sha256((folder / 'native-proof.json').read_bytes()).hexdigest(),
        'model_quality_claimed': False})


if __name__ == '__main__':
    main()
