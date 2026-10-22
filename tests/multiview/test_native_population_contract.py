from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.evaluate_native_bound_population import dataset_binding
from experiments.deformnext_retargeting.native_population_contract import (
    ENGINEERING, FINAL, MODE, model_binding, validate_capture, validate_plan, validate_repeatability,
)
from test_r_top1_native_recipe import control


def inputs(scope=FINAL):
    cfg=control()[0]
    cfg.update(output_dir='/native/Top1',training_seed=101)
    cfg['validation']['roots_file']='/validation.jsonl'
    cfg['validation']['dataset_args'].update(match_root=None,projection_root='/val-proj',expected_q2c_sha256='D1')
    bundle=dict(variant='R_top1_only',step=480,native_config='/native/Top1/config.json',
        checkpoint='/native/Top1/ckpts/denoiser_step0000480.pt',
        conditioner_checkpoint='/native/Top1/ckpts/deform_conditioner_step0000480.pt')
    hashes={p:'a'*64 for p in [bundle['native_config'],bundle['checkpoint'],bundle['conditioner_checkpoint']]}
    binding=model_binding(bundle,cfg,hashes)
    def rows(prefix,count):
        return [dict(sample_id=prefix+str(i),source_uid=prefix+'-uid'+str(i),
            source_cluster_id=prefix+'-cluster'+str(i%134),split=prefix)for i in range(count)]
    train=rows('train',9149);val=rows('validation',508);test=rows('test',508)
    data=val if scope==ENGINEERING else test
    ids=[r['sample_id']for r in data][:36 if scope==ENGINEERING else 508]
    seeds={sid:100+i for i,sid in enumerate(ids)}
    roots=dict(match_root=None,projection_root='/val-proj')if scope==ENGINEERING else dict(match_root='/test-matches',projection_root='/test-proj')
    manifest='/validation.jsonl'if scope==ENGINEERING else'/test.jsonl'
    plan=dict(population_scope=scope,mode=MODE,expected_assets=len(ids),generation_batch_size=8,
        deterministic_algorithms=True,model_binding=binding,panel='/panel.json',seeds='/seeds.json',model_alias='Top1',
        population_inputs=dict(manifest=dict(path=manifest,SHA256='manifest'),panel=dict(path='/panel.json'),seeds=dict(path='/seeds.json')),
        dataset_roots=roots)
    freeze=dict(status='FROZEN_FROM_DEVELOPMENT_ONLY',heldout_quality_read_before_freeze=False,
        development_evidence=[dict(path='/DEV/audit.json',SHA256='DEV')],models={'Top1':dict(ss64=binding)},
        population=dict(sample_ids=ids,source_clusters_by_sample={r['sample_id']:r['source_cluster_id']for r in data},rendered_cameras=[8,9,10]))
    qa=dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',published_manifest=manifest,
        published_manifest_SHA256='manifest',SS_PBR_cache_ready=True,all508_paired_coverage=True,no_test_asset_dropped=True,
        no_candidate_model_forward_or_test_quality_metric=True,matching_head_training_or_calibration_updates=0,
        Omega_completed_updates=23597,frozen_TRAIN_q2c_SHA256='D1')
    parity=dict(status='PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY',model_binding=binding,independent_model_loads=2,
        reference_prefix32_records_exact=True,all36_records_and_archived_occupancy_exact=True,tail_generation_batch=4,optimizer_updates=0)
    kwargs=dict(train_rows=train,validation_rows=val,binding=binding,freeze=freeze,cache_QA=qa,cache_QA_plan=roots.copy(),parity=parity)
    return plan,bundle,cfg,data,ids,seeds,kwargs


def test_final_changes_only_dataset_selection_and_preserves_online_matching():
    args=list(inputs());original=deepcopy(args[2]);rebound=validate_plan(*args[:6],**args[6])
    assert args[2]==original
    assert rebound['validation']['roots_file']=='/test.jsonl'
    assert rebound['validation']['dataset_args']['expected_split']=='test'
    assert rebound['validation']['dataset_args']['projection_root']=='/test-proj'
    assert rebound['validation']['dataset_args']['match_root']is None
    reverse=deepcopy(rebound);reverse['validation']=original['validation']
    assert reverse==original


def test_engineering36_includes_tail4_and_is_identical_to_original_recipe():
    args=inputs(ENGINEERING)
    assert validate_plan(*args[:6],**args[6])==args[2]
    assert len(args[4])==36 and len(args[4])%8==4


@pytest.mark.parametrize('damage',[
    'seed-bool','batch','model-binding','scope','panel-path','test-overlap','VAL-as-test','QA','QA-root',
    'q2c','Omega','no-freeze','test-read-before-freeze','freeze-model','freeze-population',
    'one-proof-load','proof-weights','proof-prefix','proof-tail',
])
def test_final_rejects_population_frozen_weight_and_cache_confound(damage):
    p,b,c,r,ids,seeds,kw=inputs()
    if damage=='seed-bool':seeds[ids[0]]=True
    elif damage=='batch':p['generation_batch_size']=4
    elif damage=='model-binding':p['model_binding']=dict(p['model_binding'],step=96)
    elif damage=='scope':p['population_scope']='validation-used-as-test'
    elif damage=='panel-path':p['panel']='/other-panel'
    elif damage=='test-overlap':r[0]['source_uid']=kw['train_rows'][0]['source_uid']
    elif damage=='VAL-as-test':r[0]['split']='validation'
    elif damage=='QA':kw['cache_QA']['SS_PBR_cache_ready']=False
    elif damage=='QA-root':kw['cache_QA_plan']['projection_root']='/other'
    elif damage=='q2c':kw['cache_QA']['frozen_TRAIN_q2c_SHA256']='other'
    elif damage=='Omega':kw['cache_QA']['Omega_completed_updates']=0
    elif damage=='no-freeze':kw['freeze']=None
    elif damage=='test-read-before-freeze':kw['freeze']['heldout_quality_read_before_freeze']=True
    elif damage=='freeze-model':kw['freeze']['models']['Top1']['ss64']=dict(kw['binding'],step=96)
    elif damage=='freeze-population':kw['freeze']['population']['rendered_cameras']=[8]
    elif damage=='one-proof-load':kw['parity']['independent_model_loads']=1
    elif damage=='proof-weights':kw['parity']['model_binding']=dict(kw['binding'],step=96)
    elif damage=='proof-prefix':kw['parity']['reference_prefix32_records_exact']=False
    elif damage=='proof-tail':kw['parity']['tail_generation_batch']=0
    with pytest.raises(ValueError):validate_plan(p,b,c,r,ids,seeds,**kw)


@pytest.mark.parametrize('damage',['manifest','projection'])
def test_engineering_cannot_silently_use_different_validation_data(damage):
    p,b,c,r,ids,seeds,kw=inputs(ENGINEERING)
    if damage=='manifest':p['population_inputs']['manifest']['path']='/test.jsonl'
    else:p['dataset_roots']['projection_root']='/other'
    with pytest.raises(ValueError):validate_plan(p,b,c,r,ids,seeds,**kw)


def test_binding_restores_native_function_and_dictionary_after_exception():
    p,b,c,r,ids,seeds,kw=inputs(ENGINEERING)
    original=deepcopy(c);function=lambda *_:None;core=SimpleNamespace(validate_panel=function)
    rebound=deepcopy(c);rebound['validation']['roots_file']='/temporary'
    with pytest.raises(RuntimeError,match='owned'):
        with dataset_binding(core,lambda *_:(ids,seeds,rebound)):
            assert core.validate_panel(p,b,c)==(ids,seeds)
            assert c['validation']['roots_file']=='/temporary'
            raise RuntimeError('owned failure')
    assert c==original and core.validate_panel is function
    with pytest.raises(ValueError,match='never applied'):
        with dataset_binding(core,lambda *_:None):pass
    assert core.validate_panel is function


def test_model_binding_rejects_other_checkpoint_or_incomplete_hashes():
    _,b,c,*_=inputs();hashes={p:'a'*64 for p in[b['native_config'],b['checkpoint'],b['conditioner_checkpoint']]}
    with pytest.raises(ValueError):model_binding(dict(b,checkpoint='/other/ckpts/denoiser_step0000480.pt'),c,hashes)
    with pytest.raises(ValueError):model_binding(dict(b,step=True),c,hashes)
    hashes[b['conditioner_checkpoint']]='short'
    with pytest.raises(ValueError):model_binding(b,c,hashes)


def test_capture_requires_actual_sample_noise_GT_and_occupancy_counts():
    row=dict(sample_id='one',seed=4,target_voxels_sha256='GT',tp=3,fp=2,fn=1)
    result=dict(status='PASS',optimizer_updates_during_evaluation=0,quick_panel_ids=['one'],per_asset=[row])
    captured=[dict(sample_id='one',seed=4,target_voxels_SHA256='GT',occupancy_counts=dict(tp=3,fp=2,fn=1))]
    validate_capture(result,captured,1)
    captured[0]['occupancy_counts']['fp']=8
    with pytest.raises(ValueError):validate_capture(result,captured,1)


def repeat_inputs():
    row=dict(sample_id='one',seed=4,target_voxels_sha256='GT',tp=3,fp=2,fn=1)
    rows=[dict(row,sample_id=str(i))for i in range(36)]
    entries=[dict(sample_id=str(i),seed=4,target_voxels_SHA256='GT',prediction_voxels_SHA256='prediction',
        prediction_latent_SHA256='latent',target_latent_SHA256='GT-latent',occupancy_counts=dict(tp=3,fp=2,fn=1))for i in range(36)]
    r=dict(status='PASS',optimizer_updates_during_evaluation=0,quick_panel_ids=[str(i)for i in range(36)],
        per_asset=rows,quick_metrics={'voxel_iou':.5},manifest_sha256='manifest',actual_loaded_trainable_SHA256='weights',
        frozen_native_model_binding={'checkpoint':'fixed'},native_SS_artifacts=dict(archived_asset_count=36,artifacts=entries))
    a=dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION',population_scope=ENGINEERING,assets=36,
        actual_generation_batch=8,tail_generation_batch=4,sampling_steps=12,guidance=3.,optimizer_updates=0,
        actual_reference_prefix32_records_exact=True,actual_saved_SS_occupancy_counts_verified=True,
        model_binding=r['frozen_native_model_binding'])
    return [r,deepcopy(r)],[a,deepcopy(a)]


def test_two_native36_runs_must_repeat_actual_latent_and_occupancy_arrays():
    results,audits=repeat_inputs()
    assert validate_repeatability(results,audits)['independent_model_loads']==2


@pytest.mark.parametrize('damage',['one-load','metrics','GT','latent','tail','not-prefix'])
def test_incomplete_or_nonrepeating_engineering_never_releases_test(damage):
    r,a=repeat_inputs()
    if damage=='one-load':r.pop();a.pop()
    elif damage=='metrics':r[1]['quick_metrics']['voxel_iou']=.4
    elif damage=='GT':r[1]['native_SS_artifacts']['artifacts'][0]['target_voxels_SHA256']='other'
    elif damage=='latent':r[1]['native_SS_artifacts']['artifacts'][-1]['prediction_latent_SHA256']='other'
    elif damage=='tail':a[1]['tail_generation_batch']=0
    elif damage=='not-prefix':a[1]['actual_reference_prefix32_records_exact']=False
    with pytest.raises(ValueError):validate_repeatability(r,a)
