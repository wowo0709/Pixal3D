"""Own recipe/schema admission and native numerical invariants, without CPU NN forwards."""
import ast
import copy
import inspect
import json
from pathlib import Path
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.deformnext_retargeting import consistent_evidence_native_runtime as parent_runtime
from experiments.deformnext_retargeting import fixed_correspondence_native_runtime as runtime
from experiments.deformnext_retargeting import evaluate_fixed_correspondence_checkpoint as evaluator
from experiments.deformnext_retargeting.fixed_correspondence_native_recipe import (
    VARIANTS, make_config, validate_config, study_metadata,
)
from test_fixed_correspondence_evidence import make
from test_shared_evidence_ablation import matcher
from test_shared_evidence_ablation_runtime import fixture_config as old_fixture


def fixture(tmp_path,arm='M_FIXED_RAW',phase='science'):
    _,parent=old_fixture(tmp_path)
    path=tmp_path/'parent.json'
    cfg=make_config(parent,variant=arm,phase=phase,parent_config=path,
        output_dir=str(tmp_path/arm/phase/'native'),recipe_path=str(tmp_path/'M-recipe.json'),
        run_id=arm+'-'+phase,source_commit='3'*40,assigned_GPUs=['6'],CPU_affinity_per_GPU={'6':[76,77]},
        proof_json=str(tmp_path/'M-proof.json'),capacity_json=str(tmp_path/'M-capacity.json'))
    return cfg,parent


@pytest.mark.parametrize('arm',VARIANTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_same_fixed_parent_only_declared_matcher_and_run_metadata_change(tmp_path,arm,phase):
    cfg,parent=fixture(tmp_path,arm,phase)
    assert validate_config(cfg,phase)==arm
    for key in ['models','dataset','validation','data_dir']:
        assert cfg[key]==parent[key]
    assert cfg['trainer']['args']==parent['trainer']['args']
    assert cfg['experiment']['shared_carrier_use_retarget_evidence'] is True
    assert cfg['experiment']['actual_candidate_budget']==(3 if arm=='M_FIXED_SPATIAL' else 64)


@pytest.mark.parametrize('damage',['data','initializer','batch','accum','LR','DQ','candidate','schema','source','parent','matcher','alias'])
def test_rejects_changes_outside_the_correspondence_substitution(tmp_path,damage):
    cfg,_=fixture(tmp_path);e=cfg['experiment'];t=cfg['trainer']['args']
    if damage=='data':cfg['dataset']['args']['expected_split']='test'
    elif damage=='initializer':t['continuation_ckpt']['denoiser']='/wrong.pt'
    elif damage=='batch':t['batch_size_per_gpu']=48
    elif damage=='accum':t['batch_split']=2
    elif damage=='LR':t['optimizer']['args']['lr']=1e-4
    elif damage=='DQ':e['physical_deformation_query_channels']=1024
    elif damage=='candidate':e['actual_candidate_budget']=3
    elif damage=='schema':e['conditioning_schema']='consistent-raw-ray-evidence-native-v1'
    elif damage=='source':e['correspondence_source_SHA256'][next(iter(e['correspondence_source_SHA256']))]='0'*64
    elif damage=='parent':Path(e['correspondence_parent_science_config']).write_text('{}')
    elif damage=='matcher':Path(t['deform_conditioner']['q2c_checkpoint']).write_bytes(b'changed')
    else:e['native_variant']='E_FIXED_RAW'
    with pytest.raises(ValueError):validate_config(cfg,'science')


def bundle(cfg):
    root=Path(cfg['output_dir']);arm=cfg['experiment']['native_variant']
    return dict(schema=evaluator.BUNDLE_SCHEMA,variant=arm,step=480,native_config=str(root/'config.json'),
        checkpoint=str(root/'ckpts/denoiser_step0000480.pt'),
        conditioner_checkpoint=str(root/'ckpts/deform_conditioner_step0000480.pt'),source_tensors=10,
        source_teacher_updates=0,native_task_objective='SS_flow_mse',native_validation=dict(step=480),
        **evaluator.control_metadata(cfg,arm))


@pytest.mark.parametrize('arm',VARIANTS)
def test_own_bundle_never_aliases_the_parent_or_other_matching_modes(tmp_path,arm):
    cfg,_=fixture(tmp_path,arm);b=bundle(cfg)
    assert evaluator.validate_bundle(b,cfg)==arm
    b['variant']='E_FIXED_RAW'
    with pytest.raises(ValueError):evaluator.validate_bundle(b,cfg)


@pytest.mark.parametrize('damage',['schema','source-count','foreign-checkpoint','budget','decoder','score','parent-source'])
def test_wrong_physical_bundle_cannot_pass_metadata_binding(tmp_path,damage):
    cfg,_=fixture(tmp_path);b=bundle(cfg)
    if damage=='schema':b['schema']='consistent-raw-ray-evidence-SS-checkpoint-pair-v1'
    elif damage=='source-count':b['source_tensors']=19
    elif damage=='foreign-checkpoint':b['checkpoint']='/another/denoiser_step0000480.pt'
    elif damage=='budget':b['actual_candidate_budget']=3
    elif damage=='decoder':b['matching_decoder_physically_absent']=False
    elif damage=='score':b['matching_scalar_score_physically_absent']=False
    else:b['correspondence_source_SHA256']={}
    with pytest.raises(ValueError):evaluator.validate_bundle(b,cfg)


@pytest.mark.parametrize('arm',VARIANTS)
def test_parameter_groups_allow_no_matcher_but_never_an_extra_head(matcher,arm):
    denoiser=nn.Module()
    for i in range(360):denoiser.register_parameter('lora_'+str(i),nn.Parameter(torch.ones(1)))
    source=make(arm);trainer=SimpleNamespace(models=dict(denoiser=denoiser),deform_conditioner=source)
    a,b=runtime.parameter_groups(trainer,arm);assert (len(a),len(b))==(360,10)
    source.ray_readout=nn.Linear(1,1)
    with pytest.raises(ValueError):runtime.parameter_groups(trainer,arm)


def admission(cfg,arm,phase):
    initial=dict(initial_LoRA_SHA256='a'*64,initial_source_SHA256='b'*64)
    r=dict(status='PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD' if phase=='proof' else
        'PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY',variant=arm,source_tensors=10,LoRA_tensors=360,
        actual_initial_core10_equal_Fixed=True,learned_projection_tensors=0,learned_attention_head_tensors=0,
        evidence_mode='fixed-prior',retargeter_present=True,
        shared_carrier_use_retarget_evidence=True,**initial,**study_metadata(cfg,arm))
    r.update(optimizer_updates=0,scientific_training_updates=0,fields_equal_to_native={'global':True,'proj':True},
        actual_selected_evidence_equals_R_appearance=True,temporary_optimizer_updates=3,actual_optimizer_states=370,
        actual_optimizer_steps=[3],world_size=1,global_batch=96,microbatch=96,accumulation=1,
        all_source_gradients_positive_finite_at_update3=True,frozen_parameters_buffers_exact=True,
        actual_initial_native32_recorded_identity=True)
    return r,initial


@pytest.mark.parametrize('phase',['proof','capacity'])
@pytest.mark.parametrize('arm',VARIANTS)
def test_only_own_same_physical_schema_gpu_admission_can_be_used(tmp_path,phase,arm):
    cfg,_=fixture(tmp_path,arm);record,initial=admission(cfg,arm,phase)
    runtime.validate_own_admission(record,cfg,arm,phase=phase,initial=initial)
    record['variant']='E_FIXED_RAW'
    with pytest.raises(ValueError):runtime.validate_own_admission(record,cfg,arm,phase=phase,initial=initial)


@pytest.mark.parametrize('field',['conditioning_schema','initial_source_SHA256','correspondence_source_SHA256',
    'matching_decoder_physically_absent','matching_scalar_score_physically_absent',
    'frozen_parameters_buffers_exact','all_source_gradients_positive_finite_at_update3','microbatch'])
def test_capacity_cannot_substitute_another_schema_or_incomplete_update(tmp_path,field):
    cfg,_=fixture(tmp_path);r,initial=admission(cfg,'M_FIXED_RAW','capacity')
    r[field]=False if isinstance(r[field],bool) else 48 if field=='microbatch' else 'corrupt'
    with pytest.raises(ValueError):runtime.validate_own_admission(r,cfg,'M_FIXED_RAW',phase='capacity',initial=initial)


@pytest.mark.parametrize('name',['tensor_sha','state_sha','rng_identity','dataset_identity','native_condition_only'])
def test_native_noise_data_and_condition_helpers_are_unchanged(name):
    def tree(f):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))),include_attributes=False)
    assert tree(getattr(runtime,name))==tree(getattr(parent_runtime,name))


def test_flow_function_and_backward_proof_changes_only_report_metadata():
    def tree(f):
        t=ast.parse(textwrap.dedent(inspect.getsource(f)))
        for n in ast.walk(t):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='source_metadata':
                n.args=n.args[:1]
        return ast.dump(t,include_attributes=False)
    assert tree(runtime.proof)==tree(parent_runtime.proof)
