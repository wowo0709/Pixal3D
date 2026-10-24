"""Bind physical shared-evidence controls to the unchanged DEV cascade.

These controls retain their own names, schemas and source tensor counts.
Admission never selects a model or permits held-out inference. Numerical
Shape/PBR generation and rendering use the existing native entries.
"""
from pathlib import Path
import re

from .dev128_pipeline_contract import validate_generation, validate_population
from .evaluate_shared_evidence_ablation_checkpoint import (
    BUNDLE_SCHEMA, control_metadata, validate_bundle,
)
from .shared_evidence_ablation_recipe import SCHEMA as CONDITIONING_SCHEMA
from .shared_evidence_ablation_recipe import STUDY_SCHEMA, VARIANTS, SOURCE_COUNTS


SCHEMA = 'shared-evidence-ablation-SS480-DEV-frozen-binding-v1'
CASE_STATUS = 'PASS_ACTUAL_SHARED_ABLATION_DEV128_NATIVE_PIPELINE_CASE'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def model_binding(bundle, cfg, hashes):
    """Construct identity only after validating the actual own science recipe."""
    arm = validate_bundle(bundle, cfg)
    require(type(bundle['step']) is int and bundle['step'] == 480,
            'completed own480 checkpoint required for final-context DEV cascade')
    result = dict(schema=SCHEMA, module_arm=arm, variant=arm,
        runtime_family='shared_evidence_ablation', checkpoint_bundle_schema=BUNDLE_SCHEMA,
        step=480, source_tensors=SOURCE_COUNTS[arm],
        evidence_mode=VARIANTS[arm][1], shared_carrier_use_evidence=True,
        learned_projection_tensors=0, conditioning_mode='edited-rgb',
        source_teacher_updates=0, native_task_objective='SS_flow_mse',
        **control_metadata(cfg, arm))
    for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
        path = bundle[key]
        result[key] = dict(path=path, SHA256=hashes.get(path))
    validate_ss_binding(result, arm, seed=cfg['training_seed'])
    return result


def validate_ss_binding(binding, arm, *, seed=101):
    require(arm in VARIANTS and type(seed) is int and seed in (101, 202),
            'explicit physical shared-evidence control and integer seed required')
    parent, mode, control, count = VARIANTS[arm]
    expected = dict(schema=SCHEMA, module_arm=arm, variant=arm,
        runtime_family='shared_evidence_ablation', checkpoint_bundle_schema=BUNDLE_SCHEMA,
        step=480, source_tensors=count, training_seed=seed,
        conditioning_schema=CONDITIONING_SCHEMA, ablation_study_schema=STUDY_SCHEMA,
        ablation_parent_variant=parent, ablation_control=control,
        retargeter_present=control != 'without-retargeting', evidence_mode=mode,
        shared_carrier_use_evidence=True, learned_projection_tensors=0,
        learned_attention_head_tensors=7 if mode == 'learned-anchored' else 0,
        conditioning_mode='edited-rgb', source_teacher_updates=0,
        native_task_objective='SS_flow_mse')
    require(all(binding.get(k) == v and type(binding[k]) is type(v)
                for k,v in expected.items()),
            'own physical control, same480 step, source/head count and recipe required')
    source_hashes = binding.get('ablation_source_SHA256')
    require(isinstance(source_hashes, dict) and bool(source_hashes)
        and all(Path(p).is_absolute() and isinstance(d, str)
                and re.fullmatch('[0-9a-f]{64}', d) for p,d in source_hashes.items()),
        'actual own native-source SHA256 identities required')
    for key in ('native_config', 'checkpoint', 'conditioner_checkpoint'):
        spec = binding.get(key)
        require(isinstance(spec, dict) and set(spec) == {'path', 'SHA256'}
            and isinstance(spec['path'], str) and Path(spec['path']).is_absolute()
            and isinstance(spec['SHA256'], str)
            and re.fullmatch('[0-9a-f]{64}', spec['SHA256']) is not None,
            'each own physical control requires actual config and both checkpoint identities')
    root = Path(binding['native_config']['path']).parent
    require(binding['native_config']['path'] == str(root/'config.json')
        and binding['checkpoint']['path'] == str(root/'ckpts/denoiser_step0000480.pt')
        and binding['conditioner_checkpoint']['path'] == str(root/'ckpts/deform_conditioner_step0000480.pt'),
        'config, denoiser and source checkpoint must belong to one actual480 root')


def validate_ss_evaluation(record, binding, arm, *, seed=101):
    validate_ss_binding(binding, arm, seed=seed)
    expected = dict(status='PASS', stage='ss64', step=480, weights='raw',
        native_variant=arm, actual_source_tensors=SOURCE_COUNTS[arm], actual_LoRA_tensors=360,
        optimizer_updates_during_evaluation=0, target_modality='rgb',
        input_states=['M1','M0','M0','M0'], input_camera_ids=[8,9,10,11],
        generation_batch_size=8, sampling_steps=12, guidance=3.,
        deterministic_algorithms=True, evaluation_mode='final128',
        conditioning_schema=CONDITIONING_SCHEMA, ablation_study_schema=STUDY_SCHEMA,
        ablation_parent_variant=binding['ablation_parent_variant'],
        ablation_control=binding['ablation_control'], retargeter_present=binding['retargeter_present'],
        training_seed=seed, learned_projection_tensors=0,
        learned_attention_head_tensors=binding['learned_attention_head_tensors'],
        shared_carrier_use_evidence=True, ablation_source_SHA256=binding['ablation_source_SHA256'])
    require(all(record.get(k) == v and type(record[k]) is type(v)
                for k,v in expected.items()),
            'actual own control forward with unchanged RGB roles, source structure and sampler required')
    for key, field in [('native_config','arm_config'), ('checkpoint','checkpoint'),
                       ('conditioner_checkpoint','conditioner_checkpoint')]:
        spec = binding[key]
        require(record.get(field) == spec['path']
            and record.get('verified_inputs', {}).get(spec['path']) == spec['SHA256'],
            'actual saved evaluation must use the bound own checkpoint bytes')


def validate_archive128(result, arm, reference, *, binding):
    """Admit all128 own predictions; reference GT is paired for scoring only."""
    validate_ss_evaluation(result, binding, arm, seed=binding['training_seed'])
    archive = result.get('native_SS_artifacts', {})
    rows, refs = result.get('per_asset', []), reference.get('per_asset', [])
    artifacts = archive.get('artifacts', [])
    require(len(rows) == len(refs) == len(artifacts) == archive.get('archived_asset_count') == 128
        and archive.get('status') == 'PASS_ACTUAL_ORIGINAL_SHARED_ABLATION_SS128_CAPTURE'
        and archive.get('same_forward_prediction_and_target_capture') is True
        and archive.get('predicted_SS_is_available_for_full_cascade') is True
        and archive.get('GT_SS_is_for_scoring_only') is True,
        'complete own128 prediction capture required; GT may only be used for scoring')
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == 128 and len({r['source_cluster_id'] for r in rows}) == 93,
            'exact existing unique128/93-source DEV panel required')
    for row, ref, artifact in zip(rows, refs, artifacts):
        require(all(row[k] == ref[k] for k in ('sample_id','source_cluster_id','seed','target_voxels_sha256'))
            and artifact['sample_id'] == row['sample_id'] and artifact['seed'] == row['seed']
            and artifact['target_voxels_SHA256'] == row['target_voxels_sha256']
            and all(artifact['occupancy_counts'][k] == row[k] for k in ('tp','fp','fn')),
            'every own128 asset, GT, noise and archived occupancy count must match')
    return ids
