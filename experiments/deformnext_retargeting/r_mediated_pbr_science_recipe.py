"""Fresh original-MV PBR task recipes, separate from engineering/proxy weights."""
import copy

from .probe_r_mediated_pbr_native import MODES, validate_config as validate_proof


NAME = 'RMediatedPBRScienceTrainer'
SCHEMA = 'r-mediated-native-PBR-task-cold-v1'


def make_science_config(proof, original, validation, *, world, seed, run_id, output_dir, port):
    cfg = copy.deepcopy(proof)
    mode = proof['experiment']['PBR_source_mode']
    cfg.update(output_dir=str(output_dir), training_seed=seed, use_wandb=True,
        wandb_entity='youngwoojeon-unist', wandb_project='pixal3d-deformnext-retargeting',
        wandb_id=run_id, wandb_name=run_id, num_gpus=world, num_nodes=1, node_rank=0,
        master_addr='127.0.0.1', master_port=str(port), load_dir='', load_ckpt=None,
        ckpt='none', auto_retry=0, tryrun=False, profile=False, multiview_runtime=True,
        trainer=dict(name=NAME, args=expected_trainer_args(proof, world)), validation=copy.deepcopy(validation),
        experiment=dict(PBR_source_mode=mode, conditioning_schema=SCHEMA,
            source_multiview_step=16850, native_task_objective='PBR_sparse_flow_mse',
            additional_source_teacher_supervision=False, source_teacher_updates=0,
            validate_at_start=True, training_assets=9139, epochs=5, updates_per_epoch=380,
            world_size=world, global_batch=24, microbatch_per_gpu=24//world, accumulation=1,
            target_modality='rgb', auto_continue_after_pilot=False, engineering_only=False))
    validate_science(cfg, proof, original)
    return cfg


def expected_trainer_args(proof, world):
    if world not in {2, 4}:
        raise ValueError('explicit measured DDP2 or4 required')
    value = copy.deepcopy(proof['trainer']['args'])
    value.update(batch_size_per_gpu=24//world, batch_split=1, max_steps=1900,
        num_workers=2, i_save=380, i_log=10, i_print=10, max_checkpoints=None)
    return value


def validate_science(cfg, proof, original):
    mode = cfg['experiment'].get('PBR_source_mode')
    validate_proof(proof, original, mode)
    e, v, t = cfg['experiment'], cfg['validation'], cfg['trainer']
    world = cfg.get('num_gpus')
    if (mode not in MODES or world not in {2, 4}
            or cfg['models'] != original['models'] or cfg['models'] != proof['models']
            or cfg['data_dir'] != proof['data_dir'] or cfg['dataset'] != proof['dataset']
            or t['name'] != NAME or t['args'] != expected_trainer_args(proof, world)
            or cfg.get('training_seed') not in {101, 202}
            or cfg.get('use_wandb') is not True or cfg.get('auto_retry') != 0
            or cfg.get('load_dir') != '' or cfg.get('load_ckpt') is not None or cfg.get('ckpt') != 'none'
            or cfg.get('num_nodes') != 1 or cfg.get('node_rank') != 0
            or cfg.get('tryrun') is not False or cfg.get('profile') is not False
            or cfg.get('multiview_runtime') is not True
            or cfg.get('wandb_entity') != 'youngwoojeon-unist'
            or cfg.get('wandb_project') != 'pixal3d-deformnext-retargeting'
            or not cfg.get('wandb_id') or cfg.get('wandb_name') != cfg['wandb_id']
            or e.get('conditioning_schema') != SCHEMA or e.get('source_multiview_step') != 16850
            or e.get('native_task_objective') != 'PBR_sparse_flow_mse'
            or e.get('additional_source_teacher_supervision') is not False
            or e.get('source_teacher_updates') != 0 or e.get('validate_at_start') is not True
            or (e.get('training_assets'), e.get('epochs'), e.get('updates_per_epoch')) != (9139, 5, 380)
            or (e.get('world_size'), e.get('global_batch'), e.get('microbatch_per_gpu'), e.get('accumulation'))
                != (world, 24, 24//world, 1)
            or e.get('target_modality') != 'rgb' or e.get('auto_continue_after_pilot') is not False
            or e.get('engineering_only') is not False):
        raise ValueError('fresh original16850/native PBR global24/1900/dropout.1/LoRA/source schema required')
    if (v['dataset_name'] != cfg['dataset']['name']
            or v['dataset_args'] != dict(cfg['dataset']['args'], expected_split='validation')
            or v.get('every_steps') != 380 or v.get('max_assets') != 4
            or v.get('assets_per_source') is not None or not v.get('roots_file')
            or v['quick_metrics'].get('enabled') is not True
            or v['quick_metrics'].get('panel_size') != 4
            or v['quick_metrics'].get('generation_batch_size') != 1
            or v['quick_metrics'].get('sampling_steps') != 12
            or v['quick_metrics'].get('render_resolution') != 256
            or not v['quick_metrics'].get('panel_file') or not v['quick_metrics'].get('seed_map_file')):
        raise ValueError('initial and every380 fixed4 native12/CFG3 anchor rendering protocol required')
    return mode


def validate_initial4(actual, reference):
    """Require original cold32 lineage and actual matching prediction hashes."""
    from .evaluate_pbr_streaming import aggregate_metrics
    from .evaluate_r_mediated_pbr_initial import IDENTITY_FIELDS
    if (reference['status'] != 'PASS_ACTUAL_NATIVE_PBR_COLD_GENERATED32'
            or reference['source_multiview_step'] != 16850 or len(reference['per_asset']) != 32
            or reference['optimizer_updates'] != 0 or not reference['original_and_source_states_unchanged']
            or actual['step'] != 0 or len(actual['quick_per_asset']) != 4):
        raise ValueError('actual step0 fixed4 and complete cold original-PBR32 reference required')
    lookup = {row['sample_id']: row for row in reference['per_asset']}
    rows = actual['quick_per_asset']
    ids = [r['sample_id'] for r in rows]
    if len(lookup) != 32 or len(set(ids)) != 4 or actual['quick_panel_ids'] != ids:
        raise ValueError('unique fixed4 subset of original32 required')
    for row in rows:
        prior = lookup.get(row['sample_id'])
        if (prior is None or any(row[k] != prior[k] for k in IDENTITY_FIELDS)
                or row['metrics'] != prior['metrics']):
            raise ValueError('initial PBR actual inputs/noise/prediction/rendering differ before first update')
    if actual['quick_metrics'] != aggregate_metrics([r['metrics'] for r in rows]):
        raise ValueError('initial4 aggregate must reproduce actual per-asset pixel-error pooling')
