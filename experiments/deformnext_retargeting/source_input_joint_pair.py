"""Matched warm-source scientific recipes and actual engineering prerequisites.

These helpers make no admission decision, create no GPU job, and never turn
CPU fixtures or a source-only prototype into native training evidence.
"""
import copy
import math
from types import SimpleNamespace

from .member_feature_recipe import validate_feature_recipe
from .source_input_joint_runtime import science_recipe_binding, TEACHER_INITIAL_FUNCTION

VARIANTS={'original':'original-NoDQ','contrast':'source-contrast-NoDQ'}
ADDRESSES=('config','output_dir','default_output_dir','name','wandb_name','wandb_id')


def science_config(saved,*,variant,recipe,output,run_id,source_commit):
    if variant not in VARIANTS.values():raise ValueError('one declared nineteen-tensor design required')
    cfg=copy.deepcopy(saved)
    cfg.update(config=str(recipe),output_dir=str(output),default_output_dir=str(output),
        name=run_id,wandb_id=run_id,wandb_name=run_id,use_wandb=True,
        load_dir='',load_ckpt=None,ckpt='none',auto_retry=0)
    exp=cfg['experiment']
    for key in ['native_warm_engineering_only','full_training_not_queued','scientific_training_queued']:
        exp.pop(key,None)
    exp.update(source_input_design=variant,additional_source_teacher_supervision=True,
        initial_function=TEACHER_INITIAL_FUNCTION,capacity_probe=False,
        source_commit=source_commit,recipe_path=str(recipe),
        release_status='REQUIRES_ACTUAL_SOURCE72_WARM_CAPACITY_AND_DISK_ADMISSION',
        source_teacher_assets=1536,source_teacher_updates=112,
        warm_actual_initial_function_is_native_V1=False,
        primary_comparison='Both teacher112-warmed Original19/Contrast19; same native SS4210 and TRAIN9149/480/global96/micro96/acc1')
    return cfg


def validate_pair(configs,teacher_specs):
    if set(configs)!=set(VARIANTS)or set(teacher_specs)!=set(VARIANTS):
        raise ValueError('exactly the original and contrast paired scientific recipes required')
    normalized=[]
    for key,variant in VARIANTS.items():
        cfg=configs[key];spec=teacher_specs[key]
        if (type(cfg.get('training_seed'))is not int or cfg['training_seed']!=101
                or spec.get('variant')!=variant
                or cfg['experiment'].get('source_teacher_assets')!=1536
                or cfg['experiment'].get('source_teacher_updates')!=112
                or cfg['experiment'].get('warm_actual_initial_function_is_native_V1')is not False):
            raise ValueError('primary seed101 and matched teacher112 warm initialization required')
        holder=SimpleNamespace(validate_feature_recipe=validate_feature_recipe)
        with science_recipe_binding(holder,variant=variant,teacher_enabled=True):
            holder.validate_feature_recipe(cfg)
        value=copy.deepcopy(cfg)
        for field in ADDRESSES:value.pop(field,None)
        value['experiment'].pop('source_input_design')
        value['experiment'].pop('recipe_path',None)
        normalized.append(value)
    if normalized[0]!=normalized[1]:
        raise ValueError('matched native model/data/optimizer/validation protocol differs between paired designs')
    if (configs['original']['output_dir']==configs['contrast']['output_dir']
            or configs['original']['wandb_id']==configs['contrast']['wandb_id']):
        raise ValueError('distinct owned scientific outputs and WandB run IDs required')
    return dict(models=list(VARIANTS),training_seed=101,global_batch=96,microbatch=96,
        accumulation=1,world_size_each=1,updates_each=480,TRAIN_assets=9149,
        teacher_initialization_updates_each=112,source_tensors_each=19,
        initial_native_identity_claim=False,quality_improvement_proven=False)


def positive(value):return type(value)in(float,int)and math.isfinite(value)and value>0


def validate_engineering(models,teacher_specs):
    if set(models)!=set(VARIANTS)or set(teacher_specs)!=set(VARIANTS):
        raise ValueError('actual parity/capacity evidence required for both designs')
    paired=[]
    for key,variant in VARIANTS.items():
        p=models[key]['parity'];c=models[key]['capacity'];r=models[key]['runtime'];spec=teacher_specs[key]
        if (p.get('status')!='PASS_ACTUAL_NATIVE_SS_WARM_TEACHER_FIELDS_VELOCITY_AND_GRADIENT'
                or p.get('variant')!=variant or p.get('split')!='train'
                or (p.get('source_tensors'),p.get('NN_LoRA_tensors'),p.get('optimizer_states'),
                    p.get('optimizer_updates'),p.get('native_SS_probe_batch'))!=(19,360,0,0,1)
                or not p.get('actual_same_noise_velocity_exact')
                or not p.get('source_and_NN_LoRA_weights_exact_after_backward')
                or not p.get('frozen_matcher_unchanged')or not p.get('native_global_and_negative_unchanged')
                or not p.get('actual_native_global_projection_negative_exact_to_standalone')
                or not all(p['actual_native_global_projection_negative_exact_to_standalone'].values())
                or not positive(p.get('warm_source_projection_correction_RMS'))
                or not positive(p.get('actual_NN_LoRA_gradient_L1'))
                or len(p.get('actual_source_gradient_L1',{}))!=19
                or not all(positive(v)for v in p['actual_source_gradient_L1'].values())):
            raise ValueError('actual native warm fields/velocity/gradient proof required: '+key)
        if (c.get('status')!='PASS_ACTUAL_MEMBER_FEATURE_SS_FULLBATCH96_CAPACITY'
                or tuple(c.get(k)for k in ['world_size','global_batch','microbatch','accumulation',
                    'trainable_NN_LoRA_tensors','trainable_source_tensors','actual_optimizer_states'])!=(1,96,96,1,360,19,379)
                or c.get('actual_optimizer_steps')!=[3]
                or not c.get('frozen_parameters_and_buffers_exact')
                or not c.get('all_source_tensors_positive_finite_gradient_by_update3')
                or c.get('scientific_training_updates')!=0 or c.get('scientific_checkpoint_exported')is not False
                or c.get('actual_maximum_FFN_expansion_batch',97)>24
                or not positive(c.get('peak_allocated_GiB'))):
            raise ValueError('actual disposable native micro96 capacity proof required: '+key)
        init=p.get('initialization',{});other=r.get('initialization',[])
        if (r.get('status')!='PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE'or r.get('phase')!='capacity'
                or r.get('source_input_design')!=variant or len(other)!=1):
            raise ValueError('actual completed capacity source initializer required: '+key)
        for record in [init,other[0]]:
            if any(record.get(k)!=v for k,v in dict(variant=variant,checkpoint_SHA256=spec['checkpoint_SHA256'],
                    audit_SHA256=spec['audit_SHA256'],source_tensors=19,source_teacher_updates=112,
                    additional_TRAIN_M1_source9_10_supervision=True,frozen_matcher_exact=True,
                    whole_conditioner_loaded_after_utility=True,cold_native_identity_expected=False).items()):
                raise ValueError('same complete teacher112 state required for proof/capacity/science: '+key)
        paired.append(tuple(p[k]for k in ['sample_id','conditioning_input_sha256','actual_RGB_tensor_SHA256','actual_GT_SS_latent_SHA256']))
    if paired[0]!=paired[1]:raise ValueError('same actual TRAIN RGB/latent case required for both warm proofs')
    return dict(actual_both_warm_native_parity_and_micro96_capacity=True,
        fresh_scientific_optimizer_required=True,scientific_updates_already_completed=0)
