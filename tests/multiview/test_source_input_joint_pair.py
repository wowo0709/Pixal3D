"""CPU comparison/admission fixtures; never actual native GPU proof records."""
import copy

import pytest

from experiments.deformnext_retargeting.source_input_joint_pair import (
    VARIANTS,science_config,validate_pair,validate_engineering,
)
from test_member_feature_runtime import recipe


def pair():
    saved=recipe();saved.update(models={'denoiser':{'unit_only':True}},data_dir='unit-TRAIN9149.jsonl')
    saved['trainer']['args']['optimizer']['args'].update(eps=1e-8,weight_decay=.01)
    saved['experiment'].update(native_warm_engineering_only=True,scientific_training_queued=False)
    configs={key:science_config(saved,variant=variant,recipe='/unit/'+key+'.json',
        output='/unit/'+key,run_id='unit-'+key,source_commit='unit-only-not-real')for key,variant in VARIANTS.items()}
    specs={key:dict(variant=variant,checkpoint_SHA256=key+'-unit-weight',audit_SHA256=key+'-unit-audit')for key,variant in VARIANTS.items()}
    return configs,specs


def test_both_science_recipes_remain_fresh_warm_and_do_not_mutate_reference():
    saved=recipe();before=copy.deepcopy(saved)
    c=science_config(saved,variant='original-NoDQ',recipe='/unit/cfg',output='/unit/out',run_id='unit',source_commit='unit')
    assert saved==before and c['experiment']['additional_source_teacher_supervision']
    assert c['load_dir']==''and c['load_ckpt']is None
    configs,specs=pair();assert validate_pair(configs,specs)['updates_each']==480
    assert not validate_pair(configs,specs)['quality_improvement_proven']


@pytest.mark.parametrize('damage',['missing','seed','cold','optimizer','cached-projection','model',
    'same-output','same-wandb','wrong-teacher','wrong-common-LR','hidden-deformation'])
def test_cannot_release_an_unmatched_or_changed_native_protocol(damage):
    configs,specs=pair();c=configs['contrast']
    if damage=='missing':configs.pop('original');specs.pop('original')
    elif damage=='seed':c['training_seed']=202
    elif damage=='cold':c['experiment']['additional_source_teacher_supervision']=False
    elif damage=='optimizer':c['trainer']['args']['optimizer']['args']['eps']=1e-6
    elif damage=='cached-projection':c['dataset']['args']['projection_root']='/different'
    elif damage=='model':c['models']['denoiser']['different']=True
    elif damage=='same-output':c['output_dir']=configs['original']['output_dir']
    elif damage=='same-wandb':c['wandb_id']=configs['original']['wandb_id']
    elif damage=='wrong-teacher':specs['contrast']['variant']='original-NoDQ'
    elif damage=='wrong-common-LR':
        for v in configs.values():v['trainer']['args']['optimizer']['args']['lr']=1e-4
    else:c['trainer']['args']['deform_conditioner']['deformation_dim']=2048
    with pytest.raises(ValueError):validate_pair(configs,specs)


def proofs(specs):
    result={}
    for key,variant in VARIANTS.items():
        init=dict(variant=variant,checkpoint_SHA256=specs[key]['checkpoint_SHA256'],
            audit_SHA256=specs[key]['audit_SHA256'],source_tensors=19,source_teacher_updates=112,
            additional_TRAIN_M1_source9_10_supervision=True,frozen_matcher_exact=True,
            whole_conditioner_loaded_after_utility=True,cold_native_identity_expected=False)
        p=dict(status='PASS_ACTUAL_NATIVE_SS_WARM_TEACHER_FIELDS_VELOCITY_AND_GRADIENT',variant=variant,
            split='train',source_tensors=19,NN_LoRA_tensors=360,optimizer_states=0,optimizer_updates=0,native_SS_probe_batch=1,
            actual_same_noise_velocity_exact=True,source_and_NN_LoRA_weights_exact_after_backward=True,
            frozen_matcher_unchanged=True,native_global_and_negative_unchanged=True,
            actual_native_global_projection_negative_exact_to_standalone={'global':True,'proj':True,'negative':True},
            warm_source_projection_correction_RMS=.01,actual_NN_LoRA_gradient_L1=1.,
            actual_source_gradient_L1={str(i):.01 for i in range(19)},initialization=copy.deepcopy(init),
            sample_id='unit-TRAIN-case',conditioning_input_sha256='unit-RGB-input',
            actual_RGB_tensor_SHA256='unit-RGB-tensor',actual_GT_SS_latent_SHA256='unit-GT-latent')
        c=dict(status='PASS_ACTUAL_MEMBER_FEATURE_SS_FULLBATCH96_CAPACITY',world_size=1,global_batch=96,
            microbatch=96,accumulation=1,trainable_NN_LoRA_tensors=360,trainable_source_tensors=19,
            actual_optimizer_states=379,actual_optimizer_steps=[3],frozen_parameters_and_buffers_exact=True,
            all_source_tensors_positive_finite_gradient_by_update3=True,scientific_training_updates=0,
            scientific_checkpoint_exported=False,actual_maximum_FFN_expansion_batch=24,peak_allocated_GiB=1.)
        r=dict(status='PASS_COMPLETED_NATIVE_SOURCE_INPUT_PHASE',phase='capacity',source_input_design=variant,initialization=[init])
        result[key]=dict(parity=p,capacity=c,runtime=r)
    return result


def test_requires_both_actual_proof_types_and_same_teacher_without_claiming_training():
    _,specs=pair();models=proofs(specs)
    result=validate_engineering(models,specs)
    assert result['actual_both_warm_native_parity_and_micro96_capacity']
    assert result['scientific_updates_already_completed']==0


@pytest.mark.parametrize('damage',['no-runtime','wrong-teacher','zero-gradient','bool-gradient','NN-gradient',
    'optimizer','micro48','capacity-export','paired-RGB','frozen-matcher','cold-identity'])
def test_incomplete_partial_or_different_native_proof_cannot_release_science(damage):
    _,specs=pair();models=proofs(specs);m=models['contrast']
    if damage=='no-runtime':m['runtime']['phase']='science'
    elif damage=='wrong-teacher':m['runtime']['initialization'][0]['checkpoint_SHA256']='other'
    elif damage=='zero-gradient':m['parity']['actual_source_gradient_L1']['0']=0.
    elif damage=='bool-gradient':m['parity']['actual_source_gradient_L1']['0']=True
    elif damage=='NN-gradient':m['parity']['actual_NN_LoRA_gradient_L1']=0.
    elif damage=='optimizer':m['parity']['optimizer_states']=1
    elif damage=='micro48':m['capacity']['microbatch']=48
    elif damage=='capacity-export':m['capacity']['scientific_checkpoint_exported']=True
    elif damage=='paired-RGB':m['parity']['actual_RGB_tensor_SHA256']='different-RGB'
    elif damage=='frozen-matcher':m['parity']['frozen_matcher_unchanged']=False
    else:m['parity']['initialization']['cold_native_identity_expected']=True
    with pytest.raises(ValueError):validate_engineering(models,specs)
