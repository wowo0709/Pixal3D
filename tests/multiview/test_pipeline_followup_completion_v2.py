"""No neural execution: physical identity, held-out release and replication IO."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.deformnext_retargeting import module_confirmation_recipe as old_recipe
from experiments.deformnext_retargeting import unified_population_contract as old_population
from experiments.deformnext_retargeting.consistent_evidence_native_recipe import VARIANTS
from experiments.deformnext_retargeting.evaluate_consistent_evidence_checkpoint import BUNDLE_SCHEMA
from experiments.deformnext_retargeting.pipeline_confirmation_recipe_v2 import make_confirmation, validate_confirmation
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import (
    ARMS, ENGINEERING, FINAL, SOURCE_COUNTS, SCHEMA, bundle_validator,
    evaluator_binding, family, model_binding, required_sources, schema_for, validate_plan,
)
from experiments.deformnext_retargeting.pipeline_training_seed_summary_v2 import summarize
from test_consistent_ray_evidence import config
from test_native_population_contract import inputs as population_inputs
from test_unified_population_contract import inputs as old_inputs
from test_module_training_seed_summary import inputs as old_replication_inputs


def configs(tmp_path, arm='E_LEARNED_RAW', phase='science', population=False):
    original, reference = config(arm)
    original.update(output_dir='/original/'+arm,wandb_id='original-'+arm)
    original['experiment'].update(recipe_path='/original/'+arm+'/recipe.json',
        paired_reference_native_root='/original/V1/native',cold_proof_json='/original/'+arm+'/proof.json',
        micro96_capacity_json='/original/'+arm+'/capacity.json')
    if population:
        original['validation'].update(roots_file='/validation.jsonl')
        original['validation']['dataset_args'].update(match_root=None,projection_root='/val-proj',expected_q2c_sha256='D1')
    reference['validation']=deepcopy(original['validation'])
    def write(path,value):
        path.write_text(json.dumps(value));return path,hashlib.sha256(path.read_bytes()).hexdigest()
    ref_path, digest=write(tmp_path/'original-reference.json',reference)
    original['experiment'].update(reference_science_config=str(ref_path),reference_science_config_SHA256=digest)
    ref202=old_recipe.make_reference(reference,output_dir='/reference-only/new',run_id='ref202',
        recipe_path=str(tmp_path/'reference202.json'))
    path202,hash202=write(tmp_path/'reference202.json',ref202)
    confirmed=make_confirmation(original,arm=arm,phase=phase,output_dir='/confirmation/'+arm,
        run_id=arm+'202',recipe_path=str(tmp_path/'confirmation.json'),
        reference_path=path202,reference_SHA256=hash202,paired_reference_native_root='/confirmation/V1/native',
        cold_proof_json='/confirmation/'+arm+'/proof.json',micro96_capacity_json='/confirmation/'+arm+'/capacity.json')
    return original,confirmed


@pytest.mark.parametrize('arm',VARIANTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_own_seed202_changes_only_seed_run_and_recorded_artifact_paths(tmp_path,arm,phase):
    a,b=configs(tmp_path,arm,phase)
    report=validate_confirmation(a,b,arm=arm,phase=phase)
    assert b['trainer']==a['trainer'] and b['validation']==a['validation'] and b['dataset']==a['dataset']
    assert b['training_seed']==202 and b['use_wandb'] is (phase=='science')
    assert report['actual_GPU_parity_capacity_still_required'] and not report['new_training_queued']
    assert not report['candidate_selected'] and b['experiment']['native_variant']==arm


@pytest.mark.parametrize('damage',['same-seed','float-seed','FFN','data','LoRA','R-width','cap',
    'same-proof','same-capacity','same-V1','relative-path','extra-metadata','changed-utility'])
def test_a_new_seed_cannot_hide_changed_design_or_old_proof(tmp_path,damage):
    a,b=configs(tmp_path);e,t=b['experiment'],b['trainer']['args']
    if damage=='same-seed':b['training_seed']=101
    elif damage=='float-seed':b['training_seed']=202.
    elif damage=='FFN':t['ffn_chunk_assets']=12
    elif damage=='data':b['data_dir']='different'
    elif damage=='LoRA':t['lora']['rank']=8
    elif damage=='R-width':t['deform_conditioner']['hidden_dim']=128
    elif damage=='cap':e['residual_logit_bound']=1.
    elif damage.startswith('same-'):
        key={'same-proof':'cold_proof_json','same-capacity':'micro96_capacity_json','same-V1':'paired_reference_native_root'}[damage]
        e[key]=a['experiment'][key]
    elif damage=='relative-path':e['paired_reference_native_root']='relative/V1'
    elif damage=='extra-metadata':e['ignore_this_field']=True
    else:t['ray_feature_utility']={'checkpoint':'teacher'}
    with pytest.raises(ValueError):validate_confirmation(a,b,arm='E_LEARNED_RAW',phase='science')


def bound_inputs(tmp_path,arm='E_LEARNED_RAW',scope=FINAL):
    cfg,_=configs(tmp_path,arm,population=True)
    root=cfg['output_dir']
    bundle=dict(schema=BUNDLE_SCHEMA,variant=arm,step=480,native_config=root+'/config.json',
        checkpoint=root+'/ckpts/denoiser_step0000480.pt',conditioner_checkpoint=root+'/ckpts/deform_conditioner_step0000480.pt',
        source_teacher_updates=0,native_task_objective='SS_flow_mse',native_validation={'step':480})
    hashes={bundle[k]:'a'*64 for k in ['native_config','checkpoint','conditioner_checkpoint']}
    own=model_binding(bundle,cfg,hashes,arm=arm)
    p,_,_,rows,ids,seeds,kw=population_inputs(scope)
    p.update(module_arm=arm,model_binding=own)
    kw['binding']=own;kw['freeze']['models']['Top1']['ss64']=own;kw['parity']['model_binding']=own
    return p,bundle,cfg,rows,ids,seeds,kw


@pytest.mark.parametrize('arm',VARIANTS)
@pytest.mark.parametrize('scope',[ENGINEERING,FINAL])
def test_followups_keep_own_schema_and_all_original_population_guards(tmp_path,arm,scope):
    p,b,c,rows,ids,seeds,kw=bound_inputs(tmp_path,arm,scope)
    before=deepcopy(c)
    rebound=validate_plan(p,b,c,rows,ids,seeds,**kw)
    assert c==before and kw['binding']['schema']==SCHEMA
    assert kw['binding']['module_arm']==arm and kw['binding']['source_tensors']==SOURCE_COUNTS[arm]
    assert kw['binding']['runtime_family']=='consistent_evidence'
    if scope==FINAL:
        assert rebound['validation']['dataset_args']['expected_split']=='test'
        assert rebound['validation']['dataset_args']['match_root'] is None
        rebound['validation']=before['validation']
    assert rebound==before


@pytest.mark.parametrize('damage',['old-schema','other-source','fake-count','alias-Full','cap',
    'one-load','tail','no-freeze','test-read','overlap','QA','q2c','float-batch'])
def test_wrong_physical_model_or_missing_evidence_never_releases_test(tmp_path,damage):
    p,b,c,rows,ids,seeds,kw=bound_inputs(tmp_path)
    if damage=='old-schema':kw['binding']['schema']=old_population.SCHEMA
    elif damage=='other-source':b['conditioner_checkpoint']='/other/Full/source.pt'
    elif damage=='fake-count':kw['binding']['source_tensors']=19
    elif damage=='alias-Full':p['module_arm']='FULL'
    elif damage=='cap':kw['binding']['residual_logit_bound']=1.
    elif damage=='one-load':kw['parity']['independent_model_loads']=1
    elif damage=='tail':kw['parity']['tail_generation_batch']=0
    elif damage=='no-freeze':kw['freeze']=None
    elif damage=='test-read':kw['freeze']['heldout_quality_read_before_freeze']=True
    elif damage=='overlap':rows[0]['source_cluster_id']=kw['train_rows'][0]['source_cluster_id']
    elif damage=='QA':kw['cache_QA']['all508_paired_coverage']=False
    elif damage=='q2c':kw['cache_QA']['frozen_TRAIN_q2c_SHA256']='other'
    else:p['generation_batch_size']=8.
    with pytest.raises(ValueError):validate_plan(p,b,c,rows,ids,seeds,**kw)


@pytest.mark.parametrize('arm',ARMS)
def test_required_sources_and_actual_evaluator_keep_physical_family(arm):
    with evaluator_binding(arm) as core:
        assert callable(core.main) and callable(core.validate_panel)
        if arm in VARIANTS:assert core.__name__.endswith('evaluate_consistent_evidence_checkpoint')
    paths=required_sources(arm)
    assert paths and all(Path(p).is_file() for p in paths)
    assert any(p.endswith('/evaluate_pipeline_bound_population_v2.py') for p in paths)
    if arm in VARIANTS:
        assert schema_for(arm)==SCHEMA
    else:
        assert schema_for(arm)==old_population.SCHEMA and family(arm)==old_population.family(arm)


@pytest.mark.parametrize('arm',tuple(a for a in ARMS if a not in VARIANTS))
def test_original_twelve_bindings_are_byte_for_byte_same_structures(tmp_path,arm):
    p,b,c,rows,ids,seeds,kw,hashes=old_inputs(tmp_path,arm)
    assert model_binding(b,c,hashes,arm=arm)==old_population.model_binding(b,c,hashes,arm=arm)
    assert validate_plan(p,b,c,rows,ids,seeds,**kw)==old_population.validate_plan(p,b,c,rows,ids,seeds,**kw)


def replication_inputs(tmp_path,arm='E_LEARNED_RAW'):
    data=old_replication_inputs(tmp_path)
    folder=tmp_path/'followup';folder.mkdir()
    original,confirmed=configs(folder,arm)
    results,cfgs,traces,starts=data
    for seed,cfg in [(101,original),(202,confirmed)]:
        old=cfgs[seed].pop('FULL');cfgs[seed][arm]=cfg
        oldroot,newroot=old['output_dir'],cfg['output_dir']
        records=results[seed].pop('FULL')
        for r in records:
            r.update(native_variant=arm,actual_source_tensors=SOURCE_COUNTS[arm],
                conditioning_schema='consistent-raw-ray-evidence-native-v1',
                followup_study_schema='consistent-raw-ray-evidence-followup-v1',
                shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
                learned_attention_head_tensors=7 if arm=='E_LEARNED_RAW' else 0,training_seed=seed)
            for k in ['checkpoint','conditioner_checkpoint','arm_config']:r[k]=r[k].replace(oldroot,newroot)
            r['verified_inputs']={k.replace(oldroot,newroot):v for k,v in r['verified_inputs'].items()}
        results[seed][arm]=records
        traces[seed][arm]=traces[seed].pop('FULL')
        starts[seed][arm]=starts[seed].pop('FULL')
        starts[seed][arm].update(variant=arm,source_tensors=SOURCE_COUNTS[arm])
    return data


@pytest.mark.parametrize('arm',VARIANTS)
def test_real_model_identity_can_be_compared_across_two_independent_seeds(tmp_path,arm):
    report=summarize(*replication_inputs(tmp_path,arm),selected=arm,draws=100)
    assert report['measured_models']==['V1','C_DECODER',arm]
    assert report['training_seeds']==[101,202] and report['assets']==128
    assert report['physical_family_explicit'] and not report['automatic_model_adoption']


@pytest.mark.parametrize('damage',['same-stream','architecture','noise','bundle-hash','seed-paths','old-head','wrong-runtime'])
def test_false_independence_or_other_model_cannot_be_called_replication(tmp_path,damage):
    r,c,t,s=replication_inputs(tmp_path)
    arm='E_LEARNED_RAW'
    if damage=='same-stream':t[202]=deepcopy(t[101])
    elif damage=='architecture':c[202][arm]['trainer']['args']['ffn_chunk_assets']=12
    elif damage=='noise':r[202][arm][0]['per_asset'][0]['seed']+=1
    elif damage=='bundle-hash':r[202][arm][0]['verified_inputs'].clear()
    elif damage=='seed-paths':c[202][arm]['experiment']['cold_proof_json']=c[101][arm]['experiment']['cold_proof_json']
    elif damage=='old-head':r[202][arm][0]['learned_projection_tensors']=2
    else:r[202][arm][0]['conditioning_schema']='unified-module-projected-native-v1'
    with pytest.raises(ValueError):summarize(r,c,t,s,selected=arm,draws=100)
