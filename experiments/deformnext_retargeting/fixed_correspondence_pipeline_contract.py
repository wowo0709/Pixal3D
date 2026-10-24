"""Own physical Fixed correspondence identities for the original DEV cascade."""
from pathlib import Path
import re

from .dev128_pipeline_contract import validate_generation, validate_population
from .evaluate_fixed_correspondence_checkpoint import BUNDLE_SCHEMA, validate_bundle, control_metadata
from .fixed_correspondence_native_recipe import SCHEMA as CONDITIONING_SCHEMA
from .fixed_correspondence_native_recipe import STUDY_SCHEMA, VARIANTS


SCHEMA = 'shared-fixed-correspondence-SS480-DEV-binding-v1'
CASE_STATUS = 'PASS_ACTUAL_FIXED_CORRESPONDENCE_DEV128_NATIVE_PIPELINE_CASE'


def require(ok,message):
    if not ok:
        raise ValueError(message)


def model_binding(bundle,cfg,hashes):
    arm=validate_bundle(bundle,cfg)
    require(bundle['step']==480,'own completed480 pair required for final-context DEV cascade')
    binding=dict(schema=SCHEMA,module_arm=arm,variant=arm,runtime_family='fixed_correspondence',
        checkpoint_bundle_schema=BUNDLE_SCHEMA,step=480,source_tensors=10,
        conditioning_mode='edited-rgb',source_teacher_updates=0,native_task_objective='SS_flow_mse',
        **control_metadata(cfg,arm))
    for key in ('native_config','checkpoint','conditioner_checkpoint'):
        binding[key]=dict(path=bundle[key],SHA256=hashes.get(bundle[key]))
    validate_ss_binding(binding,arm,seed=cfg['training_seed'])
    return binding


def validate_ss_binding(binding,arm,*,seed=101):
    require(arm in VARIANTS and type(seed) is int and seed in (101,202),
        'explicit own Fixed correspondence identity and integer seed required')
    expected=dict(schema=SCHEMA,module_arm=arm,variant=arm,runtime_family='fixed_correspondence',
        checkpoint_bundle_schema=BUNDLE_SCHEMA,step=480,source_tensors=10,training_seed=seed,
        conditioning_schema=CONDITIONING_SCHEMA,correspondence_study_schema=STUDY_SCHEMA,
        correspondence_parent_variant='E_FIXED_RAW',correspondence_mode=VARIANTS[arm],
        actual_candidate_budget=3 if arm=='M_FIXED_SPATIAL' else 64,
        matching_decoder_physically_absent=arm!='M_FIXED_D1_COS',matching_scalar_score_physically_absent=True,
        evidence_mode='fixed-prior',retargeter_present=True,shared_carrier_use_retarget_evidence=True,
        learned_projection_tensors=0,learned_attention_head_tensors=0,conditioning_mode='edited-rgb',
        source_teacher_updates=0,native_task_objective='SS_flow_mse')
    require(all(binding.get(k)==v and type(binding[k]) is type(v) for k,v in expected.items()),
        'own raw/spatial/decoder physical matcher, actual candidate budget and same delivery required')
    pins=binding.get('correspondence_source_SHA256')
    require(isinstance(pins,dict) and bool(pins) and all(Path(p).is_absolute()
        and isinstance(d,str) and re.fullmatch('[0-9a-f]{64}',d) for p,d in pins.items()),
        'actual own numerical-source pins required')
    for key in ('native_config','checkpoint','conditioner_checkpoint'):
        item=binding.get(key)
        require(isinstance(item,dict) and set(item)=={'path','SHA256'}
            and isinstance(item['path'],str) and Path(item['path']).is_absolute()
            and isinstance(item['SHA256'],str) and re.fullmatch('[0-9a-f]{64}',item['SHA256']),
            'actual own configuration and both checkpoint identities required')
    root=Path(binding['native_config']['path']).parent
    require(binding['native_config']['path']==str(root/'config.json')
        and binding['checkpoint']['path']==str(root/'ckpts/denoiser_step0000480.pt')
        and binding['conditioner_checkpoint']['path']==str(root/'ckpts/deform_conditioner_step0000480.pt'),
        'own same480 root required; another model cannot be relabelled as a correspondence control')


def validate_ss_evaluation(record,binding,arm,*,seed=101):
    validate_ss_binding(binding,arm,seed=seed)
    expected=dict(status='PASS',stage='ss64',step=480,weights='raw',native_variant=arm,
        actual_source_tensors=10,actual_LoRA_tensors=360,optimizer_updates_during_evaluation=0,
        target_modality='rgb',input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
        generation_batch_size=8,sampling_steps=12,guidance=3.,deterministic_algorithms=True,
        evaluation_mode='final128')
    expected.update({k:binding[k] for k in ['training_seed','conditioning_schema','correspondence_study_schema',
        'correspondence_parent_variant','correspondence_mode','actual_candidate_budget',
        'matching_decoder_physically_absent','matching_scalar_score_physically_absent',
        'evidence_mode','retargeter_present','shared_carrier_use_retarget_evidence',
        'learned_projection_tensors','learned_attention_head_tensors','correspondence_source_SHA256']})
    require(all(record.get(k)==v and type(record[k]) is type(v) for k,v in expected.items()),
        'actual own physical forward, native sampler and RGB roles required')
    for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),
                       ('conditioner_checkpoint','conditioner_checkpoint')]:
        item=binding[key]
        require(record.get(field)==item['path']
            and record.get('verified_inputs',{}).get(item['path'])==item['SHA256'],
            'actual saved evaluation must bind the own480 model bytes')


def validate_archive128(result,arm,reference,*,binding):
    validate_ss_evaluation(result,binding,arm,seed=binding['training_seed'])
    archive=result.get('native_SS_artifacts',{});rows=result.get('per_asset',[])
    refs=reference.get('per_asset',[]);artifacts=archive.get('artifacts',[])
    require(len(rows)==len(refs)==len(artifacts)==archive.get('archived_asset_count')==128
        and archive.get('status')=='PASS_ACTUAL_ORIGINAL_FIXED_CORRESPONDENCE_SS128_CAPTURE'
        and archive.get('same_forward_prediction_and_target_capture') is True
        and archive.get('predicted_SS_is_available_for_full_cascade') is True
        and archive.get('GT_SS_is_for_scoring_only') is True,
        'complete own128 actual predictions and scoring-only labels required')
    ids=[r['sample_id'] for r in rows]
    require(len(set(ids))==128 and len({r['source_cluster_id'] for r in rows})==93,
        'unique existing128/93-source DEV panel required')
    for row,ref,artifact in zip(rows,refs,artifacts):
        require(all(row[k]==ref[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256'])
            and artifact['sample_id']==row['sample_id'] and artifact['seed']==row['seed']
            and artifact['target_voxels_SHA256']==row['target_voxels_sha256']
            and all(artifact['occupancy_counts'][k]==row[k] for k in ['tp','fp','fn']),
            'all own128 candidate predictions must retain sample/GT/noise/count pairing')
    return ids
