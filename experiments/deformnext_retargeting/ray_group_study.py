"""Shared camera-ray query baseline for semantic, score/attention and decoder arms."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .loader_runtime_contract import validate_loader_runtime
from .matching_v2_queue import gpu_ids
from .matching_v2_ss import native_proof, train
from .semantic_topk_confirmation import seed_config
from .semantic_topk_search import write_new


NAMESPACE = 'ray-group-study-v1'
TRAINER = 'RayGroupImageConditionedProjFlowMatchingCFGTrainer'
ARMS = {
    'ray-top1': ('semantic-top1', True),
    'ray-full': ('semantic-attention', True),
    'ray-native-spatial': ('native-spatial-attention', True),
    'ray-no-decoder': ('semantic-attention', False),
    'ray-weighted': ('semantic-weighted', True),
    'ray-no-score-prior': ('semantic-attention', True),
}
STATIC = {'ray-top1', 'ray-weighted'}
SOURCES = [
    'experiments/deformnext_retargeting/ray_group_study.py',
    'experiments/deformnext_retargeting/matching_v2_ss.py',
    'experiments/deformnext_retargeting/loader_runtime_contract.py',
    'pixal3d/modules/ray_group_query.py', 'pixal3d/modules/ray_group_matching.py',
    'pixal3d/modules/deform_conditioner.py', 'pixal3d/modules/r9_matching.py',
    'pixal3d/modules/task_matching_attention.py', 'pixal3d/datasets/deformnext.py',
    'pixal3d/trainers/__init__.py', 'pixal3d/trainers/flow_matching/ray_group_matching.py',
    'pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py',
    'experiments/multiview_finetune/runtime.py',
]


def recipe(source, root, name):
    if name not in ARMS:
        raise ValueError('declared ray study arm required')
    cfg = seed_config(source, root, name, 101)
    cfg['trainer']['name'] = TRAINER
    spec = cfg['trainer']['args']['deform_conditioner']
    spec.update(projection_mode='native-bilinear', ray_mode=ARMS[name][0], ray_chunk_size=64,
        correspondence_mode='native-spatial' if name == 'ray-native-spatial' else 'semantic',
        candidate_aggregation='top1' if name == 'ray-top1' else 'weighted', score_temperature=.05)
    spec['task_attention'] = (None if name in STATIC else dict(decoder=ARMS[name][1],
        score_prior=name != 'ray-no-score-prior', train_decoder=False, train_score=False, content_mode='residual-zero',
        initial_content_scale=16., chunk_size=64))
    for args in [cfg['dataset']['args'], cfg['validation']['dataset_args']]:
        args.update(match_root=None, task_matching_features=False)
    cfg.update(master_port=str(30450 + list(ARMS).index(name)),
        wandb_id='ss-ray-group-v1-s101-' + name, wandb_name='SS shared camera-ray query ' + name + ' seed101')
    cfg['experiment'].update(conditioning_schema='deformnext-native-ray-group-v1',
        comparison_scope='Shared camera-ray query/native projector/C1 retargeter;freshSS4210/raw480/global96/DDP2/micro48/acc1',
        release_status='pending-ray-native-and-DDP-memory-proof', ray_group_mode=ARMS[name][0],
        query_unit='one query per source-camera native32patch ray slot',
        query_source='mean fourRGB native projected DINO over observed target-frustum voxel members',
        matching_query_source='mean edited-target FRAME over same ray group;fresh frozen RAW64 retrieval',
        retarget_query_source='mean edited-target native DINO over same ray group',
        matching_temperature=.05, matching_decoder='frozen D1-Q2C' if ARMS[name][1] else 'physically removed D1 blocks',
        matching_score_prior=name != 'ray-no-score-prior', matching_head_frozen=True, physical_deformation_query_channels=0,
        correspondence='original native spatial3' if name == 'ray-native-spatial' else 'fresh ray-level frozen Omega RAW64',
        utility_mode='none;exact matched Top1 feature' if name == 'ray-top1' else 'zero-output residual utility,scale16',
        optimizer_expected_states=366 if name in STATIC else 371,
        retargeter=True, inference_gt_inputs=False,
        old_2D_match_cache_used=False, new_ray_matching_cache_used=False,
        ray_no_decoder_scope='D1 removal with its selected frozen scalar initialization;not fully retrained optimalD0 head')
    cfg['experiment']['template_source_commit'] = source['experiment'].get('source_commit')
    cfg['experiment']['source_commit'] = source_record()['commit_at_entry']
    if name == 'ray-weighted':
        cfg['experiment']['utility_mode'] = 'frozen score-weighted Top64;no utility module'
    elif name == 'ray-no-score-prior':
        cfg['experiment']['utility_mode'] = 'learned utility without scalar score prior;same RAW64/D1/R'
    validate_profile(cfg, root)
    return cfg


def validate_profile(cfg, root):
    name = cfg['experiment']['arm']
    if root.name != NAMESPACE or name not in ARMS or cfg['trainer']['name'] != TRAINER:
        raise ValueError('declared common ray-group namespace/arm/trainer required')
    validate_loader_runtime(cfg)
    recipe_path, run = root / 'configs' / (name + '.json'), root / 'runs_ss64' / name
    if (any(Path(cfg[key]).resolve() != value.resolve() for key, value in
            [('config', recipe_path), ('output_dir', run), ('default_output_dir', run)])
            or Path(cfg['experiment']['recipe_path']).resolve() != recipe_path.resolve()):
        raise ValueError('recipe and run aliases must belong to the ray study')
    args, spec = cfg['trainer']['args'], cfg['trainer']['args']['deform_conditioner']
    if (cfg['training_seed'] != 101 or cfg['experiment']['training_seed'] != 101
            or cfg.get('ckpt') != 'none' or cfg.get('load_dir') != '' or cfg['num_gpus'] != 2
            or args['max_steps'] != 480 or args['batch_size_per_gpu'] != 48 or args['batch_split'] != 1
            or cfg['experiment']['required_microbatch_per_gpu'] != 48):
        raise ValueError('freshSS4210/seed101/raw480/global96/DDP2/micro48/acc1 required')
    expected = dict(arm='c1', implementation='r9-native-v5', key_dim=1024, value_dim=1024,
        query_dim=1024, topk=64, hidden_dim=256, g_max=.25, freeze_q2c=True,
        projection_mode='native-bilinear', ray_mode=ARMS[name][0], ray_chunk_size=64,
        correspondence_mode='native-spatial' if name == 'ray-native-spatial' else 'semantic',
        candidate_aggregation='top1' if name == 'ray-top1' else 'weighted', score_temperature=.05)
    if any(spec.get(key) != value for key, value in expected.items()) or spec.get('deformation_dim') is not None:
        raise ValueError('common ray/native C1 architecture,matching and physical no-DQ contract differs')
    task = (None if name in STATIC else dict(decoder=ARMS[name][1], score_prior=name != 'ray-no-score-prior',
        train_decoder=False, train_score=False, content_mode='residual-zero', initial_content_scale=16., chunk_size=64))
    if spec.get('task_attention') != task:
        raise ValueError('ray attention,Top1 or physical decoder ablation differs')
    for data in [cfg['dataset']['args'], cfg['validation']['dataset_args']]:
        if (data.get('match_root') is not None or data.get('task_matching_features') is not False
                or data['transport_implementation'] != 'r9-native-v5' or data['transport_arm'] != 'c1'):
            raise ValueError('reuse RGB/FRAME/DINO inputs without old2D matches or GT spatial addresses')
    if args.get('task_attention_optimizer_eps') is not None:
        raise ValueError('ray unit study does not change the task optimizer')


def source_record():
    repo = Path(__file__).resolve().parents[2]
    return {'commit_at_entry': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'component_SHA256': {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in SOURCES},
        'runtime_backend': {key: os.environ.get(key) for key in ['ATTN_BACKEND', 'PYTORCH_CUDA_ALLOC_CONF']}}


def validate_runtime_backend():
    if (os.environ.get('ATTN_BACKEND') != 'sdpa'
            or os.environ.get('PYTORCH_CUDA_ALLOC_CONF') != 'expandable_segments:True'):
        raise ValueError('use the actual reference SS runtime:ATTN_BACKEND=sdpa,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare', 'proof', 'train'], required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--arm', choices=list(ARMS))
    parser.add_argument('--proof-output', type=Path)
    parser.add_argument('--proof-memory-fraction', type=float)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.mode == 'prepare':
        if args.arm is None:
            parser.error('prepare requires a declared arm')
        cfg = recipe(cfg, args.root, args.arm)
        write_new(args.root / 'configs' / (args.arm + '.json'), cfg)
        print(json.dumps({'status': 'PREPARED_NOT_GPU_VERIFIED', 'arm': args.arm}), flush=True)
        return
    validate_profile(cfg, args.root)
    validate_runtime_backend()
    if args.mode != 'proof' and (args.proof_output is not None or args.proof_memory_fraction is not None):
        parser.error('standalone proof output/memory limit are not training settings')
    visible, allowed = gpu_ids(os.environ.get('CUDA_VISIBLE_DEVICES', '')), gpu_ids(os.environ.get('MATCHING_V2_ALLOWED_GPUS', ''))
    if not visible or any(device not in allowed for device in visible):
        raise ValueError('explicit assigned physical GPUs required')
    folder = args.proof_output or args.root / 'verification' / cfg['experiment']['arm']
    owned = args.root / 'verification' / cfg['experiment']['arm']
    if not folder.resolve().is_relative_to(owned.resolve()):
        raise ValueError('proof output must stay inside its declared ray arm verification directory')
    before = source_record()
    if args.mode == 'proof':
        if (folder / 'native-proof.json').exists():
            raise FileExistsError('preserve existing proof;use a new owned output directory')
        import torch
        if args.proof_memory_fraction is not None:
            if len(visible) != 1 or not 0 < args.proof_memory_fraction < 1:
                raise ValueError('standalone one-GPU fractional allocator limit required')
            torch.cuda.set_per_process_memory_fraction(args.proof_memory_fraction, 0)
        write_new(folder / 'entry-source-contract.json', before)
        native_proof(cfg, folder)
        peak = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        write_new(folder / 'standalone-memory-v1.json', {'peak_allocated_GiB': peak,
            'allocator_fraction_limit': args.proof_memory_fraction, 'optimizer_updates': 0})
    else:
        write_new(folder / 'entry-source-contract.json', before)
        train(args)
    after = source_record()
    if before['component_SHA256'] != after['component_SHA256']:
        raise RuntimeError('ray study numerical source changed during actual execution')
    write_new(folder / ('ray-source-binding-' + args.mode + '.json'), {
        'status': 'PASS_ACTUAL_RAY_MODEL_SOURCE_BINDING', **before,
        'commit_at_exit': after['commit_at_entry'], 'source_stable_through_execution': True,
        'mode': args.mode, 'model_quality_claimed': False})


if __name__ == '__main__':
    main()
