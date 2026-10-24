"""Own-family admission and immutable native numerical path; no CPU NN forwards."""
import ast
import copy
import hashlib
import inspect
import json
import textwrap
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from experiments.deformnext_retargeting import consistent_evidence_native_runtime as parent_runtime
from experiments.deformnext_retargeting import shared_evidence_ablation_native_runtime as runtime
from experiments.deformnext_retargeting import evaluate_shared_evidence_ablation_checkpoint as evaluator
from experiments.deformnext_retargeting.shared_evidence_ablation_recipe import (
    NAME, SCHEMA, STUDY_SCHEMA, VARIANTS, SOURCE_COUNTS, make_config, validate_config,
)
from test_consistent_ray_evidence import config as parent_config
from test_shared_evidence_ablation import matcher, make


def fixture_config(tmp_path, variant='A_FIXED_UNIFORM', phase='science'):
    parent, reference = parent_config(VARIANTS[variant][0])
    frozen = tmp_path/'matcher-fixture.bin'; frozen.write_bytes(b'non-NN config identity fixture')
    digest = hashlib.sha256(frozen.read_bytes()).hexdigest()
    for cfg in [parent, reference]:
        cfg['trainer']['args']['deform_conditioner']['q2c_checkpoint'] = str(frozen)
        cfg['dataset']['args']['expected_q2c_sha256'] = digest
        cfg['validation']['dataset_args']['expected_q2c_sha256'] = digest
    ref = tmp_path/'reference.json'; ref.write_text(json.dumps(reference))
    parent['experiment'].update(reference_science_config=str(ref),
        reference_science_config_SHA256=hashlib.sha256(ref.read_bytes()).hexdigest(),
        source_commit='1'*40, recipe_path=str(tmp_path/'parent-recipe.json'),
        cold_proof_json=str(tmp_path/'parent-proof.json'),
        micro96_capacity_json=str(tmp_path/'parent-capacity.json'))
    parent.update(output_dir=str(tmp_path/'parent-native'), wandb_id='parent', wandb_name='parent')
    path = tmp_path/'parent.json'; path.write_text(json.dumps(parent))
    cfg = make_config(parent, variant=variant, phase=phase, parent_config=path,
        output_dir=str(tmp_path/variant/phase/'native'), recipe_path=str(tmp_path/'own-recipe.json'),
        run_id=variant+'-'+phase, source_commit='2'*40, assigned_GPUs=['2'], CPU_affinity_per_GPU={'2':[88,89]},
        proof_json=str(tmp_path/'own-proof.json'), capacity_json=str(tmp_path/'own-capacity.json'))
    return cfg, parent


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('phase', ['proof','capacity','science'])
def test_own_phase_keeps_actual_parent_data_model_and_optimization(tmp_path, variant, phase):
    cfg, parent = fixture_config(tmp_path,variant,phase)
    assert validate_config(cfg,phase) == variant
    assert cfg['trainer']['name'] == NAME and cfg['trainer']['args'] == parent['trainer']['args']
    for key in ['dataset','validation','models','data_dir','training_seed','load_dir','ckpt','load_ckpt']:
        assert cfg[key] == parent[key]
    assert cfg['experiment']['parent_source_commit'] == '1'*40
    assert cfg['experiment']['source_commit'] == '2'*40
    assert cfg['experiment']['actual_source_tensors'] == SOURCE_COUNTS[variant]
    assert cfg['experiment']['retargeter_present'] is (VARIANTS[variant][2] != 'without-retargeting')
    assert 'followup_study_schema' not in cfg['experiment']
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('damage', ['batch','accum','budget','rank','lr','data','target','matching','resume',
    'teacher','count','prior','fake-DQ','base','phase','source-hash','parent-alias','provenance'])
def test_rejects_changed_contract_or_mislabelled_parent(tmp_path, damage):
    cfg,_ = fixture_config(tmp_path); e,t = cfg['experiment'],cfg['trainer']['args']
    if damage == 'batch':t['batch_size_per_gpu'] = 48
    elif damage == 'accum':t['batch_split'] = 2
    elif damage == 'budget':t['max_steps'] = 960
    elif damage == 'rank':t['lora']['rank'] = 8
    elif damage == 'lr':t['optimizer']['args']['lr'] = 1e-4
    elif damage == 'data':cfg['data_dir'] = 'other split'
    elif damage == 'target':e['target_modality'] = 'depth'
    elif damage == 'matching':t['deform_conditioner']['score_temperature'] = .5
    elif damage == 'resume':cfg['load_dir'] = '/trained/parent'
    elif damage == 'teacher':e['source_teacher_updates'] = 3
    elif damage == 'count':e['actual_source_tensors'] = 17
    elif damage == 'prior':e['ablation_control'] = 'without-logit-prior'
    elif damage == 'fake-DQ':t['deform_conditioner']['deformation_dim'] = 2048
    elif damage == 'base':t['continuation_ckpt']['denoiser'] = 'new initializer'
    elif damage == 'phase':cfg['use_wandb'] = False
    elif damage == 'source-hash':e['ablation_source_SHA256'][next(iter(e['ablation_source_SHA256']))] = '0'*64
    elif damage == 'parent-alias':e['native_variant'] = 'E_FIXED_RAW'
    else:e['source_commit'] = 'bad'
    with pytest.raises(ValueError):validate_config(cfg,'science')


@pytest.mark.parametrize('which', ['parent','matcher'])
def test_changed_parent_or_frozen_matcher_cannot_redefine_the_control(tmp_path, which):
    cfg,_ = fixture_config(tmp_path)
    path = (cfg['experiment']['ablation_parent_science_config'] if which == 'parent' else
            cfg['trainer']['args']['deform_conditioner']['q2c_checkpoint'])
    with open(path,'ab') as f:f.write(b' ')
    with pytest.raises(ValueError):validate_config(cfg,'science')


def bundle(cfg):
    from pathlib import Path
    root=Path(cfg['output_dir']);variant=cfg['experiment']['native_variant']
    return dict(schema=evaluator.BUNDLE_SCHEMA,variant=variant,step=480,
        checkpoint=str(root/'ckpts/denoiser_step0000480.pt'),
        conditioner_checkpoint=str(root/'ckpts/deform_conditioner_step0000480.pt'),
        native_config=str(root/'config.json'),source_tensors=SOURCE_COUNTS[variant],
        source_teacher_updates=0,native_task_objective='SS_flow_mse',native_validation={'step':480},
        **evaluator.control_metadata(cfg,variant))


@pytest.mark.parametrize('variant', VARIANTS)
def test_own_checkpoint_pair_is_required_even_without_R(tmp_path, variant):
    cfg,_ = fixture_config(tmp_path,variant); b=bundle(cfg)
    assert evaluator.validate_bundle(b,cfg)==variant
    assert b['conditioner_checkpoint'] is not None


@pytest.mark.parametrize('damage', ['legacy-schema','parent-name','parent-source','count','control','R','seed','source-code'])
def test_parent_checkpoint_or_another_control_cannot_alias_own_pair(tmp_path,damage):
    cfg,_=fixture_config(tmp_path,'A_LEARNED_NO_R');b=bundle(cfg)
    if damage=='legacy-schema':b['schema']='consistent-raw-ray-evidence-SS-checkpoint-pair-v1'
    elif damage=='parent-name':b['variant']='E_LEARNED_RAW'
    elif damage=='parent-source':b['conditioner_checkpoint']='/parent/native/ckpts/deform_conditioner_step0000480.pt'
    elif damage=='count':b['source_tensors']=17
    elif damage=='control':b['ablation_control']='without-logit-prior'
    elif damage=='R':b['retargeter_present']=True
    elif damage=='seed':b['training_seed']=202
    else:b['ablation_source_SHA256']={}
    with pytest.raises(ValueError):evaluator.validate_bundle(b,cfg)


def admission(cfg, phase):
    variant=cfg['experiment']['native_variant'];initial=dict(initial_LoRA_SHA256='a'*64,initial_source_SHA256='b'*64)
    r=dict(variant=variant,source_tensors=SOURCE_COUNTS[variant],LoRA_tensors=360,
        actual_initial_retained_parameters_equal_parent=True,
        retargeter_present=VARIANTS[variant][2]!='without-retargeting',
        learned_attention_head_tensors=7 if VARIANTS[variant][1]=='learned-anchored' else 0,
        **initial,**runtime.study_metadata(cfg,variant))
    if phase=='proof':
        r.update(status='PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD',
            optimizer_updates=0,scientific_training_updates=0,fields_equal_to_native={'global':True,'proj':True,'negative':True},
            actual_initial_head_probability_matches_declared_rule=None if VARIANTS[variant][1]=='fixed-prior' else True,
            actual_UniformK_candidate_weights_exact=True if VARIANTS[variant][2]=='uniform-k' else None)
    else:
        r.update(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY',
            temporary_optimizer_updates=3,scientific_training_updates=0,actual_optimizer_states=360+SOURCE_COUNTS[variant],
            actual_optimizer_steps=[3],world_size=1,global_batch=96,microbatch=96,accumulation=1,
            all_source_gradients_positive_finite_at_update3=True,frozen_parameters_buffers_exact=True,
            actual_initial_native32_recorded_identity=True)
    return r,initial


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('phase', ['proof','capacity'])
def test_own_admission_is_distinct_from_parent_and_wrong_weights(tmp_path,variant,phase):
    cfg,_=fixture_config(tmp_path,variant);r,initial=admission(cfg,phase)
    runtime.validate_own_admission(r,cfg,variant,phase=phase,initial=initial)
    for key,value in [('variant',VARIANTS[variant][0]),('initial_source_SHA256','c'*64),
                      ('ablation_source_SHA256',{}),('retargeter_present',not r['retargeter_present'])]:
        broken=dict(r);broken[key]=value
        with pytest.raises(ValueError):runtime.validate_own_admission(broken,cfg,variant,phase=phase,initial=initial)


@pytest.mark.parametrize('variant', VARIANTS)
def test_actual_parameter_groups_keep360_LoRA_and_only_declared_source(matcher,variant):
    denoiser=nn.Module()
    for i in range(360):denoiser.register_parameter('lora_'+str(i),nn.Parameter(torch.ones(())))
    trainer=SimpleNamespace(models={'denoiser':denoiser},deform_conditioner=make(variant))
    lora,source=runtime.parameter_groups(trainer,variant)
    assert len(lora)==360 and len(source)==SOURCE_COUNTS[variant]
    trainer.deform_conditioner.ablation='reference'
    with pytest.raises(ValueError):runtime.parameter_groups(trainer,variant)


@pytest.mark.parametrize('name', ['tensor_sha','state_sha','rng_identity','dataset_identity','native_condition_only'])
def test_native_input_noise_and_identity_helpers_are_unchanged(name):
    def tree(obj):return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(obj))))
    assert tree(getattr(runtime,name))==tree(getattr(parent_runtime,name))


def test_native_training_step_changes_only_optional_R_gradient_logging():
    def step(module):
        tree=ast.parse(textwrap.dedent(inspect.getsource(module.native_training)))
        return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='run_step')
    class RemoveLoggingOrdering(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='layers' for t in node.targets):return None
            return self.generic_visit(node)
        def visit_If(self,node):
            if ast.unparse(node.test)=='module.retargeter is not None':return None
            return self.generic_visit(node)
    assert ast.dump(RemoveLoggingOrdering().visit(step(runtime)))==ast.dump(RemoveLoggingOrdering().visit(step(parent_runtime)))
