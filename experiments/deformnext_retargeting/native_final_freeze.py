"""Freeze native ablation compositions from development evidence, before test.

All four planned models remain in the table, including negative results. No
candidate is invented to satisfy the older three-role campaign schema.
"""
import copy
from pathlib import Path
import re

from .evaluate_bound_population import validate_population
from .native_fullcascade_contract import PAIRS
from .native_population_contract import SCHEMA


METRIC_ROLES = dict(source_preservation=['LPIPS', 'DreamSim', 'CLIP'],
                    global_naturalness=['FID', 'KID'],
                    target_alignment=['PSNR', 'SSIM', 'M_IoU', 'CD', 'FScore'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_spec(spec):
    require(isinstance(spec, dict) and set(spec) == {'path', 'SHA256'}
            and isinstance(spec['path'], str) and Path(spec['path']).is_absolute()
            and isinstance(spec['SHA256'], str)
            and re.fullmatch('[0-9a-f]{64}', spec['SHA256']) is not None,
            'one explicit absolute file and actual SHA256 required')


def pbr_binding(config, completion, hashes):
    """Describe the actual saved native1900 NN/source/config, never a pilot."""
    root = Path(config['output_dir'])
    mode = config['experiment']['PBR_source_mode']
    require(mode in PAIRS.values(), 'declared native PBR mode required')
    expected = 0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19
    require(completion.get('status') == 'PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE'
            and completion.get('mode') == mode and completion.get('actual_optimizer_updates') == 1900
            and completion.get('training_seed') == config['training_seed'] == 101
            and completion.get('source_multiview_step') == 16850
            and completion.get('source_tensors') == expected
            and completion.get('stage_training_complete') is True
            and completion.get('frozen_base_matcher_and_anchors_exact') is True
            and completion.get('post_training_DDP_parameters_exact') is True
            and tuple(completion.get(k) for k in ('world_size', 'global_batch', 'microbatch', 'accumulation'))
                == (4, 24, 6, 1), 'complete own native1900 training and integrity evidence required')
    spec = config['trainer']['args'].get('deform_conditioner')
    require((spec is None if mode == 'vanilla' else spec is not None and spec['appearance_mode'] == mode),
            'physical source module and checkpoint mode must agree')
    require(config['experiment']['native_task_objective'] == 'PBR_sparse_flow_mse'
            and config['experiment']['source_teacher_updates'] == 0
            and config['experiment']['engineering_only'] is False,
            'actual native objective without proxy teacher initialization required')
    def file(path):
        result = dict(path=str(path), SHA256=hashes[str(path)])
        validate_spec(result)
        return result
    return dict(schema='native-cold-PBR1900-frozen-binding-v1', mode=mode,
                training_seed=101, step=1900, source_tensors=expected,
                native_config=file(root/'config.json'),
                checkpoint=file(root/'ckpts/denoiser_step0001900.pt'),
                conditioner_checkpoint=None if mode == 'vanilla' else
                    file(root/'ckpts/deform_conditioner_step0001900.pt'))


def make_freeze(*, ss_models, pbr_models, original_stages, rows, train_rows,
                validation_rows, seeds, parities, development, development_evidence,
                ss_evaluations, cascade_statistics, heldout_quality_read_before_freeze):
    require(heldout_quality_read_before_freeze is False,
            'candidate test quality must not be consulted before freezing')
    require(set(ss_models) == set(PAIRS) == set(parities)
            and set(ss_evaluations) == set(PAIRS)
            and set(pbr_models) == set(PAIRS.values()), 'all four planned native compositions required')
    panel = [r['sample_id'] for r in rows]
    validate_population(rows, scope='final-test508', panel=panel, seeds=seeds,
                        train_rows=train_rows, validation_rows=validation_rows)
    clusters = {r['sample_id']: r['source_cluster_id'] for r in rows}
    require(len(set(clusters.values())) == 134, 'complete test508/134-source population required')
    require(set(development) == {'native_SS128', 'native_cascade32', 'native_two_seeds'},
            'native SS, full-cascade and independent-training-seed development evidence required')
    ss, cascade, replication = (development[k] for k in ('native_SS128', 'native_cascade32', 'native_two_seeds'))
    require(ss.get('status') == 'PASS_PAIRED_NATIVE_FOUR_MODEL_DEV128_THREE_NOISE_ANALYSIS'
            and ss.get('assets') == 128 and ss.get('source_clusters') == 93
            and ss.get('held_out_test508_evaluated') is False,
            'actual native four-model development128 results required')
    require(cascade.get('status') == 'PASS_ACTUAL_NATIVE_SS480_PBR1900_EIGHT_MODEL32_EIGHT_METRICS'
            and cascade.get('assets') == 32 and cascade.get('views') == 768
            and cascade.get('fixed_original_Shape_models') is True
            and cascade.get('source_LPIPS_DreamSim_CLIP_and_target_PSNR_SSIM_MIoU_CD_FScore_measured') is True,
            'actual native eight-metric full-cascade development required')
    require(replication.get('status') == 'PASS_ACTUAL_NATIVE_TWO_TRAINING_SEEDS_PAIRED128_THREE_GENERATION_NOISES'
            and replication.get('training_seeds') == [101, 202]
            and replication.get('actual_complete480_input_noise_timestep_global_traces_equal_within_seed') is True
            and replication.get('actual_training_streams_differ_between_101_and_202') is True
            and replication.get('actual_original_frozen_NN_base_tensors_exact_in_all_four_final_checkpoints') is True,
            'actual independent native training confirmation required, regardless of sign')
    require(len(development_evidence) >= 3, 'pinned development evidence files required')
    for spec in development_evidence:
        validate_spec(spec)
    require(set(original_stages) == {'shape512', 'shape1024', 'pbr1024'},
            'fixed original Shape512/1024 and original PBR isolation prior required')
    for name, step in [('shape512', 4210), ('shape1024', 16850), ('pbr1024', 16850)]:
        stage = original_stages[name]
        require(stage['step'] == step, 'user-specified original multi-view stage steps required')
        for key in ('config', 'checkpoint'):
            validate_spec(stage[key])
        require(stage['checkpoint']['path'].endswith(f'denoiser_step{step:07d}.pt'),
                'original stage checkpoint name/step mismatch')
        require(cascade_statistics['original_stage_checkpoint_SHA256'][name] == stage['checkpoint']['SHA256'],
                'fixed stage differs from the actual development cascade')
    models = {}
    for variant, mode in PAIRS.items():
        ss_binding, pbr = ss_models[variant], pbr_models[mode]
        require(ss_binding.get('schema') == SCHEMA and ss_binding.get('variant') == variant
                and ss_binding.get('step') == 480 and ss_binding.get('training_seed') == 101
                and pbr.get('schema') == 'native-cold-PBR1900-frozen-binding-v1'
                and pbr.get('mode') == mode and pbr.get('step') == 1900,
                'own native480/native1900 matched model composition required')
        for binding, vanilla in ((ss_binding, variant == 'vanilla'), (pbr, mode == 'vanilla')):
            for key in ('native_config', 'checkpoint'):
                validate_spec(binding[key])
            require((binding['conditioner_checkpoint'] is None) == vanilla,
                    'source weights must be physically present exactly for source models')
            if not vanilla:
                validate_spec(binding['conditioner_checkpoint'])
        measured = ss_evaluations[variant]
        require(measured.get('status') == 'PASS' and measured.get('native_variant') == variant
                and measured.get('evaluation_mode') == 'final128'
                and measured.get('step') == 480 and len(measured['per_asset']) == 128,
                'own actual native480 development128 evaluation required')
        for key, measured_key in [('checkpoint', 'checkpoint'), ('native_config', 'arm_config'),
                                  ('conditioner_checkpoint', 'conditioner_checkpoint')]:
            spec = ss_binding[key]
            require(measured[measured_key] == (None if spec is None else spec['path']),
                    'saved SS path differs from development model')
            if spec is not None:
                require(measured['verified_inputs'][spec['path']] == spec['SHA256'],
                        'saved SS SHA differs from actual evaluated bytes')
        require(cascade_statistics['trained_PBR_checkpoint_SHA256'][mode] == pbr['checkpoint']['SHA256'],
                'saved PBR differs from the actual development cascade')
        parity = parities[variant]
        require(parity.get('status') == 'PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY'
                and parity.get('model_binding') == ss_binding
                and parity.get('independent_model_loads') == 2
                and parity.get('reference_prefix32_records_exact') is True
                and parity.get('all36_records_and_archived_occupancy_exact') is True
                and parity.get('tail_generation_batch') == 4 and parity.get('optimizer_updates') == 0,
                'actual own validation36/tail4 two-load parity required')
        models[variant] = dict(ss64=ss_binding, shape512=original_stages['shape512'],
                               shape1024=original_stages['shape1024'], pbr1024=pbr)
    require(len({m['ss64']['checkpoint']['SHA256'] for m in models.values()}) == 4
            and len({m['pbr1024']['checkpoint']['SHA256'] for m in models.values()}) == 4,
            'four distinct actually trained models, no duplicated candidate role')
    return copy.deepcopy(dict(status='FROZEN_FROM_DEVELOPMENT_ONLY',
        heldout_quality_read_before_freeze=False, development_evidence=development_evidence,
        models=models, population=dict(sample_ids=panel, source_clusters_by_sample=clusters,
            rendered_cameras=[8, 9, 10], generation_seeds=seeds),
        primary_metric_families=METRIC_ROLES,
        input_cameras=[8, 9, 10, 11], input_states=['M1', 'M0', 'M0', 'M0'], target_modality='rgb',
        original_zero_controls=['source4-M0-M0-M0-M0', 'edited4-M1-M0-M0-M0'],
        original_PBR_for_SS_isolation=original_stages['pbr1024'],
        original_zero_control_execution_not_proven_by_freeze=True,
        all_planned_models_retained_regardless_of_development_sign=True,
        automatic_model_adoption=False, forced_attention_success=False,
        final508_SS_generated=False, full_cascade_generated=False, all_ten_metrics_measured=False,
        full_goal_complete=False))
