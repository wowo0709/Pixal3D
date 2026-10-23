"""Validate actual physical SS480/PBR1900 bindings, including seed202 evidence."""
from pathlib import Path

from .common_geometry_pbr_contract import MODES
from .consistent_evidence_native_recipe import VARIANTS as FOLLOWUPS
from .native_final_freeze import validate_spec
from .pipeline_population_contract_v2 import ALIASES, SOURCE_COUNTS, family, schema_for


def require(ok,message):
    if not ok:raise ValueError(message)


def _own_files(binding,*,absent,step):
    for key in ['native_config','checkpoint']:validate_spec(binding[key])
    root=Path(binding['native_config']['path']).parent
    require(Path(binding['native_config']['path'])==root/'config.json'
        and Path(binding['checkpoint']['path'])==root/'ckpts'/f'denoiser_step{step:07d}.pt'
        and (binding['conditioner_checkpoint'] is None)==absent,
        'actual own config and NN/source checkpoint paths required')
    if not absent:
        validate_spec(binding['conditioner_checkpoint'])
        require(Path(binding['conditioner_checkpoint']['path'])==root/'ckpts'/f'deform_conditioner_step{step:07d}.pt',
            'actual source checkpoint must belong to the same model root and step')


def validate_ss_binding(binding,arm,*,seed=101):
    require(type(seed) is int and seed in (101,202),'actual independent integer training seed required')
    require(binding.get('schema')==schema_for(arm) and binding.get('module_arm')==arm
        and binding.get('variant')==ALIASES.get(arm,arm) and binding.get('runtime_family')==family(arm)
        and type(binding.get('step')) is int and binding['step']==480
        and type(binding.get('training_seed')) is int and binding['training_seed']==seed
        and type(binding.get('source_tensors')) is int and binding['source_tensors']==SOURCE_COUNTS[arm]
        and binding.get('conditioning_mode')=='edited-rgb'
        and binding.get('source_teacher_updates')==0 and binding.get('native_task_objective')=='SS_flow_mse',
        'own actual native480 physical SS family and seed required')
    if arm in FOLLOWUPS:
        require(binding.get('checkpoint_bundle_schema')=='consistent-raw-ray-evidence-SS-checkpoint-pair-v1'
            and binding.get('evidence_mode')==FOLLOWUPS[arm]
            and binding.get('shared_carrier_use_retarget_evidence') is True
            and binding.get('learned_projection_tensors')==0
            and binding.get('learned_attention_head_tensors')==(7 if arm=='E_LEARNED_RAW' else 0),
            'actual raw shared-evidence checkpoint and physical identity V required')
    _own_files(binding,absent=arm=='V1',step=480)


def validate_ss_evaluation(record,binding,arm,*,seed=101):
    validate_ss_binding(binding,arm,seed=seed)
    require(record.get('status')=='PASS' and record.get('stage')=='ss64'
        and record.get('step')==480 and record.get('weights')=='raw'
        and record.get('native_variant')==ALIASES.get(arm,arm)
        and type(record.get('actual_source_tensors')) is int and record['actual_source_tensors']==SOURCE_COUNTS[arm]
        and record.get('actual_LoRA_tensors')==360 and record.get('optimizer_updates_during_evaluation')==0
        and record.get('target_modality')=='rgb' and record.get('input_states')==['M1','M0','M0','M0']
        and record.get('input_camera_ids')==[8,9,10,11]
        and record.get('generation_batch_size')==8 and record.get('sampling_steps')==12
        and record.get('guidance')==3. and record.get('deterministic_algorithms') is True,
        'actual native raw480 SS evaluation with the same RGB roles/cameras/sampler required')
    for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),
                      ('conditioner_checkpoint','conditioner_checkpoint')]:
        spec=binding[key]
        require(record.get(field)==(None if spec is None else spec['path']),
            'actual SS evaluation belongs to another model')
        if spec is not None:
            require(record.get('verified_inputs',{}).get(spec['path'])==spec['SHA256'],
                'actual SS evaluation used different model bytes')
    if arm in FOLLOWUPS:
        require(record.get('conditioning_schema')=='consistent-raw-ray-evidence-native-v1'
            and record.get('followup_study_schema')=='consistent-raw-ray-evidence-followup-v1'
            and record.get('shared_carrier_use_retarget_evidence') is True
            and record.get('learned_projection_tensors')==0
            and record.get('learned_attention_head_tensors')==(7 if arm=='E_LEARNED_RAW' else 0)
            and type(record.get('training_seed')) is int and record['training_seed']==seed,
            'actual shared-evidence forward and independent training seed required')


def pbr_source_count(mode):
    require(mode in MODES,'actual native PBR mode required')
    return 0 if mode=='vanilla' else 10 if mode=='top1-only' else 19


def validate_pbr_binding(binding,mode,*,seed=101):
    require(type(seed) is int and seed in (101,202)
        and binding.get('schema')=='native-cold-PBR1900-frozen-binding-v1'
        and binding.get('mode')==mode and type(binding.get('step')) is int and binding['step']==1900
        and type(binding.get('training_seed')) is int and binding['training_seed']==seed
        and type(binding.get('source_tensors')) is int and binding['source_tensors']==pbr_source_count(mode),
        'own actual native1900 PBR binding and integer training seed required')
    _own_files(binding,absent=mode=='vanilla',step=1900)


def pbr_binding(config,completions,hashes):
    """Bind all four actual completed ranks without changing a seed202 to101."""
    e,t=config['experiment'],config['trainer']['args']
    mode,seed=e['PBR_source_mode'],config['training_seed'];count=pbr_source_count(mode)
    require(type(seed) is int and seed in (101,202) and set(completions)=={'0','1','2','3'}
        and Path(config['output_dir']).is_absolute() and e.get('source_multiview_step')==16850
        and e.get('training_assets')==9139 and e.get('epochs')==5
        and e.get('native_task_objective')=='PBR_sparse_flow_mse'
        and e.get('source_teacher_updates')==0 and e.get('engineering_only') is False
        and tuple(e.get(k) for k in ['world_size','global_batch','microbatch_per_gpu','accumulation'])==(4,24,6,1)
        and config.get('num_gpus')==4 and t.get('batch_size_per_gpu')==6 and t.get('batch_split')==1
        and t.get('max_steps')==1900 and t.get('lora')==dict(rank=16,alpha=16,dropout=0.),
        'actual fresh five-epoch native PBR recipe and complete four ranks required')
    spec=t.get('deform_conditioner')
    require((spec is None if mode=='vanilla' else spec is not None and spec.get('appearance_mode')==mode),
        'actual physical PBR source module must agree with mode')
    for rank,row in completions.items():
        require(row.get('status')=='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE'
            and type(row.get('rank')) is int and row['rank']==int(rank)
            and type(row.get('training_seed')) is int and row['training_seed']==seed
            and row.get('mode')==mode and row.get('actual_optimizer_updates')==1900
            and row.get('source_multiview_step')==16850 and row.get('source_tensors')==count
            and row.get('actual_optimizer_states')==360+count
            and tuple(row.get(k) for k in ['world_size','global_batch','microbatch','accumulation'])==(4,24,6,1)
            and all(row.get(k) is True for k in ['stage_training_complete','initial4_identity_before_first_update',
                'frozen_base_matcher_and_anchors_exact','post_training_DDP_parameters_exact']),
            'each actual PBR rank must finish1900, preserve its prior and synchronize parameters')
    root=Path(config['output_dir'])
    def file(path):
        spec=dict(path=str(path),SHA256=hashes.get(str(path)));validate_spec(spec);return spec
    binding=dict(schema='native-cold-PBR1900-frozen-binding-v1',mode=mode,training_seed=seed,
        step=1900,source_tensors=count,native_config=file(root/'config.json'),
        checkpoint=file(root/'ckpts/denoiser_step0001900.pt'),conditioner_checkpoint=None if mode=='vanilla'
        else file(root/'ckpts/deform_conditioner_step0001900.pt'))
    validate_pbr_binding(binding,mode,seed=seed)
    return binding
