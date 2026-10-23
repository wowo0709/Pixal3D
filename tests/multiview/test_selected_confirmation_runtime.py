"""Metadata and observer lifecycle tests; no neural construction or forward."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.selected_confirmation_runtime import (
    RUNTIMES, observed_trainer, source_digest, validate_admission,
    validate_engineering, validate_recipe, validate_observer_values,
)
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS, ALIASES, SOURCE_COUNTS, family
from test_module_confirmation_recipe import inputs as legacy_configs
from test_pipeline_followup_completion_v2 import configs as followup_configs


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_all_physical_models_keep_original_numerics_and_only_record_devices(tmp_path,arm,phase):
    fixture=followup_configs if arm.startswith('E_') else legacy_configs
    original,confirmed=fixture(tmp_path,arm,phase)
    before=deepcopy(original)
    confirmed['experiment'].update(assigned_GPUs=['6','7'],CPU_affinity_per_GPU={'6':[76,77],'7':[78,79]})
    assert validate_recipe(original,confirmed,arm=arm,phase=phase)==ALIASES.get(arm,arm)
    assert family(arm) in RUNTIMES and original==before


@pytest.mark.parametrize('damage',['LR','microbatch','seed','float-seed','new-scientific-field'])
def test_operational_normalization_cannot_hide_scientific_changes(tmp_path,damage):
    a,b=followup_configs(tmp_path)
    b['experiment'].update(assigned_GPUs=['6'],CPU_affinity_per_GPU={'6':[76,77]})
    if damage=='LR':b['trainer']['args']['optimizer']['args']['lr']=1e-3
    elif damage=='microbatch':b['trainer']['args']['batch_size_per_gpu']=48
    elif damage=='seed':b['training_seed']=101
    elif damage=='float-seed':b['training_seed']=202.
    else:b['experiment']['allow_warm_start']=True
    with pytest.raises(ValueError):validate_recipe(a,b,arm='E_LEARNED_RAW',phase='science')


@pytest.mark.parametrize('gpus,cpus',[
    ('6',{'6':[76,77]}),(['6','6'],{'6':[76,77]}),([6],{'6':[76,77]}),
    (['6'],{'7':[78,79]}),(['6'],{'6':[True]}),(['6'],{'6':[]}),(['6'],{'6':[76,76]}),
])
def test_operational_exemption_still_requires_explicit_valid_device_assignment(tmp_path,gpus,cpus):
    a,b=followup_configs(tmp_path)
    b['experiment'].update(assigned_GPUs=gpus,CPU_affinity_per_GPU=cpus)
    with pytest.raises(ValueError):validate_recipe(a,b,arm='E_LEARNED_RAW',phase='science')


def reports(arm='E_LEARNED_RAW'):
    count=SOURCE_COUNTS[arm]
    initial=dict(initial_LoRA_SHA256='a'*64,initial_source_SHA256='b'*64 if count else None,
        post_constructor_RNG=dict(CPU='cpu202',CUDA='cuda202',numpy='numpy202',python='python202'),
        TRAIN_order_SHA256='train9149')
    startup=dict(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP',variant=ALIASES.get(arm,arm),
        source_tensors=count,LoRA_tensors=360,global_batch=96,microbatch=96,accumulation=1,
        actual_optimizer_states=0,**initial)
    proof=dict(status='PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD',variant=startup['variant'],
        source_tensors=count,LoRA_tensors=360,optimizer_updates=0,
        actual_CPU_CUDA_numpy_python_RNG_preserved=True,
        fields_equal_to_native={k:True for k in ['global','proj','negative_global','negative_proj']},
        actual_native_loss=.15,LoRA_gradient_L1=1.,same_noise_velocity_SHA256='velocity',
        same_noise_native_flow_loss=.15,**initial)
    if arm.startswith('E_'):
        proof.update(learned_projection_tensors=0,actual_initial_core10_equal_Top1=True,
            shared_carrier_use_retarget_evidence=True,actual_selected_evidence_equals_R_appearance=True,
            actual_initial_head_probability_equals_frozen_prior=True if arm=='E_LEARNED_RAW' else None)
    capacity=dict(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY',variant=startup['variant'],
        source_tensors=count,LoRA_tensors=360,global_batch=96,microbatch=96,accumulation=1,world_size=1,
        temporary_optimizer_updates=3,actual_optimizer_states=360+count,actual_optimizer_steps=[3],
        scientific_training_updates=0,scientific_checkpoint_exported=False,
        actual_initial_native32_recorded_identity=True,frozen_parameters_buffers_exact=True,
        all_source_gradients_positive_finite_at_update3=True if count else None,**initial)
    reference=deepcopy(startup);reference.update(variant='vanilla',source_tensors=0,initial_source_SHA256=None)
    rp,rc=deepcopy(proof),deepcopy(capacity)
    for r in [rp,rc]:r.update(variant='vanilla',source_tensors=0,initial_source_SHA256=None)
    rc['actual_optimizer_states']=360
    trace=[dict(step=i+1,asset_ids=[['train']*96],x_t_SHA256=str(i),t_SHA256='t',global_SHA256='RGB4') for i in range(5)]
    values=dict(own_proof=proof,own_capacity=capacity,reference_startup=reference,reference_proof=rp,
        reference_capacity=rc,own_capacity_trace=trace[:3],reference_capacity_trace=deepcopy(trace[:3]),
        reference_trace=trace,own_initial_identity=dict(status='PASS_ACTUAL_INITIAL_NATIVE32_RECORDED_METRIC_IDENTITY',
        assets=32,optimizer_updates=0))
    return startup,values


@pytest.mark.parametrize('arm',ARMS)
def test_own_engineering_and_same_seed_v1_are_compared_without_source_aliases(arm):
    startup,v=reports(arm)
    validate_admission(startup,v['own_proof'],v['own_capacity'],v['reference_startup'],arm=arm)
    validate_engineering(v,arm=arm)


@pytest.mark.parametrize('key',['initial_source_SHA256','initial_source10_SHA256','initial_source12_SHA256','initial_source19_SHA256'])
def test_existing_runtime_digest_names_are_preserved(key):
    assert source_digest({key:'actual-source-bytes'})=='actual-source-bytes'
    with pytest.raises(ValueError):source_digest({key:'actual-source-bytes','unrelated':True,**(
        {'initial_source10_SHA256':'other'} if key!='initial_source10_SHA256' else {'initial_source_SHA256':'other'})})
    with pytest.raises(ValueError):source_digest({})


@pytest.mark.parametrize('damage',[None,'other-V1-root','other-proof','other-capacity','seed101-reference','short-prefix'])
def test_shared_evidence_native_checks_and_observer_read_the_same_actual_artifacts(tmp_path,damage):
    _,cfg=followup_configs(tmp_path)
    folder=tmp_path/'V1';folder.mkdir();_,ref=legacy_configs(folder,'V1')
    _,values=reports();values['reference_config']=ref
    e=cfg['experiment']
    plan=dict(schema='actual-selected-SS202-paired-prefix-admission-v1',module_arm='E_LEARNED_RAW',training_seed=202,
        reference_startup={'path':e['paired_reference_native_root']+'/startup.json'},
        reference_trace={'path':e['paired_reference_native_root']+'/training-noise-trace.jsonl'},
        own_proof={'path':e['cold_proof_json']},own_capacity={'path':e['micro96_capacity_json']})
    if damage=='other-V1-root':plan['reference_trace']['path']='/other-V1/training-noise-trace.jsonl'
    elif damage=='other-proof':plan['own_proof']['path']='/seed101/proof.json'
    elif damage=='other-capacity':plan['own_capacity']['path']='/seed101/capacity.json'
    elif damage=='seed101-reference':ref['training_seed']=101
    elif damage=='short-prefix':values['reference_trace'].pop()
    if damage:
        with pytest.raises(ValueError):validate_observer_values(plan,values,cfg,'E_LEARNED_RAW')
    else:validate_observer_values(plan,values,cfg,'E_LEARNED_RAW')


@pytest.mark.parametrize('damage',['warm-LoRA','warm-source','RNG','TRAIN-order','wrong-family','missing-negative',
    'false-field','micro48','optimizer-not-fresh','short-capacity','frozen-base','no-source-gradient',
    'reused-source','native-output','different-capacity-noise','short-trace','initial32','no-LoRA-gradient',
    'reference-failed','reference-capacity','different-E','learned-V','initial-prior'])
def test_missing_or_unpaired_gpu_evidence_cannot_admit_science(damage):
    s,v=reports();p,c,r=v['own_proof'],v['own_capacity'],v['reference_startup']
    if damage=='warm-LoRA':s['initial_LoRA_SHA256']='trained'
    elif damage=='warm-source':s['initial_source_SHA256']='trained'
    elif damage=='RNG':s['post_constructor_RNG']['CUDA']='shifted'
    elif damage=='TRAIN-order':s['TRAIN_order_SHA256']='other'
    elif damage=='wrong-family':p['variant']='FULL'
    elif damage=='missing-negative':p['fields_equal_to_native'].pop('negative_global')
    elif damage=='false-field':p['fields_equal_to_native']['proj']=False
    elif damage=='micro48':c['microbatch']=48
    elif damage=='optimizer-not-fresh':s['actual_optimizer_states']=377
    elif damage=='short-capacity':c['actual_optimizer_steps']=[2]
    elif damage=='frozen-base':c['frozen_parameters_buffers_exact']=False
    elif damage=='no-source-gradient':c['all_source_gradients_positive_finite_at_update3']=False
    elif damage=='reused-source':c['initial_source_SHA256']='other'
    elif damage=='native-output':p['same_noise_velocity_SHA256']='other'
    elif damage=='different-capacity-noise':v['own_capacity_trace'][0]['x_t_SHA256']='other'
    elif damage=='short-trace':v['own_capacity_trace'].pop()
    elif damage=='initial32':v['own_initial_identity']['optimizer_updates']=3
    elif damage=='no-LoRA-gradient':p['LoRA_gradient_L1']=0.
    elif damage=='reference-failed':v['reference_proof']['status']='FAILED'
    elif damage=='reference-capacity':v['reference_capacity']['scientific_checkpoint_exported']=True
    elif damage=='different-E':p['actual_selected_evidence_equals_R_appearance']=False
    elif damage=='learned-V':p['learned_projection_tensors']=2
    else:p['actual_initial_head_probability_equals_frozen_prior']=False
    with pytest.raises(ValueError):
        validate_admission(s,p,c,r,arm='E_LEARNED_RAW')
        validate_engineering(v,arm='E_LEARNED_RAW')


@pytest.mark.parametrize('damage',[None,'prefix','incomplete','exception','frozen-base','optimizer-count','optimizer-step'])
def test_observer_preserves_native_return_and_restores_factory_even_on_failure(tmp_path,monkeypatch,damage):
    from experiments.deformnext_retargeting import probe_member_feature_fullbatch as frozen
    startup,values=reports()
    (tmp_path/'startup.json').write_text(json.dumps(startup))
    calls=[]
    digest=['frozen']
    monkeypatch.setattr(frozen,'_frozen_digest',lambda model:digest[0])
    class Parent:
        output_dir=str(tmp_path)
        models={'denoiser':object()}
        optimizer=SimpleNamespace(state={i:{'step':480} for i in range(377)})
        def run(self):
            for self.step in range(5):self.run_step([{'_sha256':['train']*96}])
            if damage=='exception':raise RuntimeError('actual-native-failure')
            if damage=='frozen-base':digest[0]='changed'
            if damage=='optimizer-count':self.optimizer.state.pop(0)
            if damage=='optimizer-step':self.optimizer.state[0]['step']=479
            self.step=479 if damage=='incomplete' else 480
            return 'original-native-return'
        def run_step(self,packs):
            calls.append(self.step)
            r=values['reference_trace'][self.step]
            self._paired_trace=[{k:r[k] for k in ['x_t_SHA256','t_SHA256','global_SHA256']}]
            if damage=='prefix':self._paired_trace[0]['t_SHA256']='different'
            return 'original-step-return'
    original=lambda variant:Parent
    runtime=SimpleNamespace(selected_trainer=original)
    check=SimpleNamespace(inputs={})
    with (pytest.raises((ValueError,RuntimeError)) if damage else __import__('contextlib').nullcontext()):
        with observed_trainer(runtime,arm='E_LEARNED_RAW',values=values,check=check):
            assert runtime.selected_trainer('E_LEARNED_RAW')().run()=='original-native-return'
    assert runtime.selected_trainer is original
    assert calls==([0] if damage=='prefix' else [0,1,2,3,4])
    output=tmp_path/'confirmation-paired-prefix-and-frozen.json'
    assert output.exists() is (damage is None)
    if output.exists():
        report=json.loads(output.read_text())
        assert report['updates']==480 and report['training_seed']==202 and not report['full_goal_complete']
