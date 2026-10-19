"""Prepare the native voxel-query candidate without changing a running model."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .frozen_matcher_utility import validate_profile as validate_frozen
from .frozen_utility_strength import validate_profile as validate_strength
from .matching_v2_ss import native_proof, train
from .matching_v2_queue import gpu_ids
from .semantic_topk_confirmation import seed_config
from .semantic_topk_search import write_new


NAME = 'voxel-query-c1'
NAMESPACE = 'voxel-query-followup-v1'
TRAINER = 'VoxelQueryImageConditionedProjFlowMatchingCFGTrainer'
SOURCES = [
    'experiments/deformnext_retargeting/matching_v2_ss.py',
    'experiments/deformnext_retargeting/voxel_query_followup.py',
    'pixal3d/modules/deform_conditioner.py', 'pixal3d/modules/task_matching_attention.py',
    'pixal3d/modules/voxel_query_matching.py', 'pixal3d/datasets/deformnext.py',
    'pixal3d/trainers/__init__.py', 'pixal3d/trainers/flow_matching/voxel_query_matching.py',
    'pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py',
    'experiments/multiview_finetune/runtime.py',
]


def validate_profile(cfg, root):
    if (root.name != NAMESPACE or cfg['experiment']['arm'] != NAME
            or cfg['trainer']['name'] != TRAINER or cfg['training_seed'] != 101
            or cfg['experiment'].get('training_seed') != 101):
        raise ValueError('explicit fresh seed101 native voxel-query namespace and trainer required')
    expected = root / 'configs' / (NAME + '.json')
    run = root / 'runs_ss64' / NAME
    if (any(Path(cfg[key]).resolve() != path.resolve() for key, path in [('config', expected), ('output_dir', run)])
            or Path(cfg['experiment']['recipe_path']).resolve() != expected.resolve()):
        raise ValueError('voxel-query recipe/output aliases must belong to its new namespace')
    reference = copy.deepcopy(cfg)
    spec = reference['trainer']['args']['deform_conditioner']
    if spec.pop('voxel_chunk_size', None) != 128 or spec['task_attention'].pop('initial_content_scale', None) != 16.:
        raise ValueError('declared voxel chunk128 and unchanged reference utility scale16 required')
    validate_frozen(reference, root)


def recipe(source, root):
    validate_strength(source, Path(source['config']).parent.parent)
    cfg = seed_config(source, root, NAME, 101)
    cfg['trainer']['name'] = TRAINER
    cfg['trainer']['args']['deform_conditioner']['voxel_chunk_size'] = 128
    cfg.update(master_port='30446', wandb_id='ss-voxel-query-c1-s101',
               wandb_name='SS native voxel-query frozen semantic Top64 C1 seed101')
    cfg['experiment'].update(release_status='pending-native-and-DDP48-proof',
        query_source='native-four-view-projected-DINO',
        query_model_change='Utility query per native voxel;original 2D FRAME pool,D1,score and C1 target-query channels retained')
    validate_profile(cfg, root)
    return cfg


def source_record():
    repo = Path(__file__).resolve().parents[2]
    return {'commit_at_entry': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'actual_component_SHA256': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in SOURCES}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'proof', 'train'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Scale16 reference for prepare; new voxel-query recipe otherwise')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.mode == 'prepare':
        output = args.root / 'configs' / (NAME + '.json')
        prepared = recipe(cfg, args.root)
        prepared['experiment']['source_commit'] = source_record()['commit_at_entry']
        write_new(output, prepared)
        print(json.dumps({'status': 'PREPARED_NOT_GPU_VERIFIED_OR_TRAINED', 'config': str(output)}), flush=True)
        return
    validate_profile(cfg, args.root)
    visible = gpu_ids(os.environ.get('CUDA_VISIBLE_DEVICES', ''))
    assigned = gpu_ids(os.environ.get('MATCHING_V2_ALLOWED_GPUS', ''))
    if any(device not in assigned for device in visible):
        raise ValueError('native voxel-query GPU must be explicitly assigned')
    before = source_record()
    folder = args.root / 'verification' / NAME
    if args.mode == 'proof':
        native_proof(cfg, folder)
    else:
        # The shared launch retains actual native SS-loss/backward checks, a
        # two-update micro48/DDP2 benchmark, and fresh SS4210 LoRA continuation.
        # No smaller microbatch is permitted by this candidate's profile.
        write_new(folder / 'entry-source-contract.json', before)
        train(args)
    after = source_record()
    if before['actual_component_SHA256'] != after['actual_component_SHA256']:
        raise RuntimeError('native voxel-query source changed during actual execution')
    proof = folder / 'native-proof.json'
    write_new(folder / ('query-source-binding-' + args.mode + '.json'), {
        'status': 'PASS_NATIVE_VOXEL_QUERY_SOURCE_BINDING', **before,
        'commit_at_exit': after['commit_at_entry'],
        'actual_native_proof_SHA256': hashlib.sha256(proof.read_bytes()).hexdigest(),
        'source_stable_through_execution': True, 'mode': args.mode,
        'model_quality_claimed': False})


if __name__ == '__main__':
    main()
