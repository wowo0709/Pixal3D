"""Typed metadata and actual stored array checks; no neural execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.native_population_contract import model_binding as old_binding
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS, ENGINEERING, model_binding
from experiments.deformnext_retargeting import selected_population_validation as module
from experiments.deformnext_retargeting.selected_population_validation import validate_records, verify_archive
from test_prepare_selected_confirmation import Files
from test_native_population_contract import repeat_inputs
from test_selected_population_plan_factory import inputs as model_inputs


def inputs(tmp_path, arm='E_LEARNED_RAW', legacy=False):
    request, values = model_inputs(tmp_path, arm); cfg, bundle = values['config'], values['bundle']
    binding = request['model_binding']; hashes = {binding[k]['path']:binding[k]['SHA256'] for k in
        ['native_config','checkpoint','conditioner_checkpoint'] if binding[k] is not None}
    reference = deepcopy(values['reference']); reference.update(conditioning_mode='edited-rgb', manifest_sha256='manifest')
    base_results, base_audits = repeat_inputs(); row_template = base_results[0]['per_asset'][0]
    reference['per_asset'] = [dict(row_template, **r, source_cluster_id='cluster'+str(i))
        for i,r in enumerate(reference['per_asset'])]
    saved_binding = old_binding(bundle,cfg,hashes) if legacy else binding
    results, audits = [], []
    for _ in range(2):
        result = deepcopy(reference); result.update(evaluation_mode='bound-population', population_scope=ENGINEERING,
            diagnostic_only=True, frozen_native_model_binding=deepcopy(saved_binding),
            per_asset=deepcopy(reference['per_asset'][:36]), quick_panel_ids=reference['quick_panel_ids'][:36],
            actual_loaded_trainable_SHA256='weights', quick_metrics={'voxel_iou':.5})
        artifacts=[]
        for i,row in enumerate(result['per_asset']):
            a=deepcopy(base_results[0]['native_SS_artifacts']['artifacts'][i]);a.update(sample_id=row['sample_id'],seed=row['seed'])
            artifacts.append(a)
        result['native_SS_artifacts']=dict(status='PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE',
            archived_asset_count=36,artifacts=artifacts)
        audit=deepcopy(base_audits[0]);audit['model_binding']=deepcopy(saved_binding)
        results.append(result);audits.append(audit)
    return results,audits,reference,dict(arm=arm,bundle=bundle,config=cfg,hashes=hashes)


@pytest.mark.parametrize('arm', ARMS)
def test_every_current_family_keeps_its_actual_physical_binding_and_predictions(tmp_path,arm):
    r,a,ref,kw=inputs(tmp_path,arm);before=deepcopy((r,a,ref,kw));out=validate_records(r,a,ref,**kw)
    assert (r,a,ref,kw)==before and out['model_binding']['module_arm']==arm
    assert out['independent_model_loads']==2 and out['tail_generation_batch']==4
    assert all(not x['metadata_schema_only'] for x in out['original_evaluator_binding_lineage'])


@pytest.mark.parametrize('arm',['V1','C_DECODER','FULL'])
def test_same_actual_native_weights_reuse_legacy_36_loading_evidence_without_new_forward(tmp_path,arm):
    r,a,ref,kw=inputs(tmp_path,arm,legacy=True);out=validate_records(r,a,ref,**kw)
    assert out['model_binding']['module_arm']==arm and out['no_new_neural_forward_or_training']
    assert all(x['metadata_schema_only'] and not x['actual_measurements_or_architecture_changed']
        for x in out['original_evaluator_binding_lineage'])


@pytest.mark.parametrize('damage',['one-load','legacy-other-family','other-binding','source-count','other-NN',
    'source-bytes','latent','occupancy','actual-prefix','prefix-flag','tail-seed','tail-noise-constant',
    'GT','native-bundle','manifest','scope','diagnostic','not128reference','audit-binding','one-tail','float-binding'])
def test_schema_compatibility_never_masks_different_models_inputs_or_arrays(tmp_path,damage):
    r,a,ref,kw=inputs(tmp_path)
    if damage=='one-load':r.pop();a.pop()
    elif damage=='legacy-other-family':
        for value,audit in zip(r,a):
            value['frozen_native_model_binding']['schema']='native-cold-SS480-frozen-binding-v1'
            audit['model_binding']=deepcopy(value['frozen_native_model_binding'])
    elif damage=='other-binding':r[0]['frozen_native_model_binding']['checkpoint']['SHA256']='b'*64
    elif damage=='source-count':r[0]['actual_source_tensors']=19
    elif damage=='other-NN':r[0]['checkpoint']='/other.pt'
    elif damage=='source-bytes':r[0]['verified_inputs'][r[0]['conditioner_checkpoint']]='b'*64
    elif damage=='latent':r[1]['native_SS_artifacts']['artifacts'][-1]['prediction_latent_SHA256']='other'
    elif damage=='occupancy':r[1]['native_SS_artifacts']['artifacts'][-1]['prediction_voxels_SHA256']='other'
    elif damage=='actual-prefix':r[0]['per_asset'][0]['tp']+=1
    elif damage=='prefix-flag':a[0]['actual_reference_prefix32_records_exact']=False
    elif damage=='tail-seed':r[1]['per_asset'][-1]['seed']+=1
    elif damage=='tail-noise-constant':
        for result in r:
            result['per_asset'][-1]['seed']+=1;result['native_SS_artifacts']['artifacts'][-1]['seed']+=1
    elif damage=='GT':r[0]['per_asset'][0]['target_voxels_sha256']='other'
    elif damage=='native-bundle':r[0]['native_task_checkpoint_bundle']='/other-bundle.json'
    elif damage=='manifest':r[0]['manifest_sha256']='other'
    elif damage=='scope':r[0]['population_scope']='final-test508'
    elif damage=='diagnostic':r[0]['diagnostic_only']=False
    elif damage=='not128reference':ref['evaluation_mode']='archive32'
    elif damage=='audit-binding':a[0]['model_binding']['source_tensors']=19
    elif damage=='one-tail':a[0]['tail_generation_batch']=0
    else:
        r[0]['frozen_native_model_binding']['source_tensors']=17.;a[0]['model_binding']=deepcopy(r[0]['frozen_native_model_binding'])
    with pytest.raises(ValueError):validate_records(r,a,ref,**kw)


@pytest.mark.parametrize('damage',[None,'latent','occupancy','counts','sample','grid','file-bytes'])
def test_real_saved_arrays_are_hashed_and_counts_recomputed(tmp_path,damage):
    pred=np.zeros(64**3,dtype=bool);pred[:3]=True;gt=np.zeros_like(pred);gt[:1]=True
    arrays=dict(sample_id=np.asarray('one'),seed=np.int64(4),occupancy_shape=np.array([64,64,64]),
        prediction_latent=np.zeros((1,2,2,2),dtype=np.float32),target_latent=np.ones((1,2,2,2),dtype=np.float32),
        prediction_occupancy_packed=np.packbits(pred),target_occupancy_packed=np.packbits(gt))
    if damage=='grid':arrays['occupancy_shape']=np.array([32,32,32])
    path=tmp_path/'sample.npz';np.savez_compressed(path,**arrays)
    artifact=dict(sample_id='one',seed=4,artifact_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        occupancy_counts=dict(tp=1,fp=2,fn=0))
    for key in ['prediction_latent','target_latent']:
        v=arrays[key];h=hashlib.sha256((str(v.dtype)+repr(v.shape)).encode());h.update(v.tobytes());artifact[key+'_SHA256']=h.hexdigest()
    for prefix in ['prediction','target']:
        artifact[prefix+'_voxels_SHA256']=hashlib.sha256(arrays[prefix+'_occupancy_packed'].tobytes()).hexdigest()
    receipt=tmp_path/'file4-closure.json';receipt.write_text(json.dumps(dict(status='PASS',local_equals_actual_file4_SHA256=True,
        files=[dict(relative_path='sample.npz',sha256=artifact['artifact_SHA256'],bytes=path.stat().st_size)])))
    if damage=='latent':artifact['prediction_latent_SHA256']='b'*64
    elif damage=='occupancy':artifact['prediction_voxels_SHA256']='b'*64
    elif damage=='counts':artifact['occupancy_counts']['tp']=2
    elif damage=='sample':artifact['sample_id']='other'
    elif damage=='file-bytes':path.write_bytes(path.read_bytes()+b'changed')
    if damage:
        with pytest.raises(ValueError):verify_archive(path,artifact,ClosedInputs(tmp_path),receipt)
    else:assert verify_archive(path,artifact,ClosedInputs(tmp_path),receipt)['actual_archived_latent_and_occupancy_bytes_checked']


def closed_inputs(tmp_path,arm='E_LEARNED_RAW',legacy=False):
    """Synthetic measured records and weights-as-bytes, not actual NN proof."""
    results,audits,reference,kw=inputs(tmp_path,arm,legacy)
    files=Files(tmp_path);cfg,bundle=deepcopy(kw['config']),deepcopy(kw['bundle'])
    root=tmp_path/'model/native';cfg['output_dir']=str(root)
    bundle['native_config']=str(root/'config.json');bundle['checkpoint']=str(root/'ckpts/denoiser_step0000480.pt')
    bundle['conditioner_checkpoint']=None if arm=='V1' else str(root/'ckpts/deform_conditioner_step0000480.pt')
    config_spec=files.add(root/'config.json',cfg);hashes={config_spec['path']:config_spec['SHA256']}
    for key in ['checkpoint','conditioner_checkpoint']:
        if bundle[key] is None:continue
        path=Path(bundle[key]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'SYNTHETIC metadata-only checkpoint')
        digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path)]=digest
        files.entries.append(dict(relative_path=str(path.relative_to(tmp_path)),sha256=digest,bytes=path.stat().st_size))
    bundle_spec=files.add(tmp_path/'bundle.json',bundle)
    own=model_binding(bundle,cfg,hashes,arm=arm);saved=old_binding(bundle,cfg,hashes) if legacy else own
    for record in [reference,*results]:
        for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),('conditioner_checkpoint','conditioner_checkpoint')]:
            record[field]=bundle[key]
        record.update(verified_inputs=deepcopy(hashes),native_task_checkpoint_bundle=bundle_spec['path'])
    pred=np.zeros(64**3,dtype=bool);pred[:3]=True;gt=np.zeros_like(pred);gt[:1]=True
    pred_hash=hashlib.sha256(np.packbits(pred).tobytes()).hexdigest();gt_hash=hashlib.sha256(np.packbits(gt).tobytes()).hexdigest()
    for row in reference['per_asset']:
        row.update(target_voxels_sha256=gt_hash,tp=1,fp=2,fn=0,voxel_iou=1/3,voxel_precision=1/3,voxel_recall=1.,voxel_f1=.5)
    repetitions=[]
    for index,(result,audit) in enumerate(zip(results,audits)):
        folder=tmp_path/f'repeat{index}/native';folder.mkdir(parents=True)
        result.update(per_asset=deepcopy(reference['per_asset'][:36]),frozen_native_model_binding=deepcopy(saved),quick_metrics={'voxel_iou':1/3})
        audit['model_binding']=deepcopy(saved)
        artifacts=[]
        for i,row in enumerate(result['per_asset']):
            arrays=dict(sample_id=np.asarray(row['sample_id']),seed=np.int64(row['seed']),occupancy_shape=np.array([64,64,64]),
                prediction_latent=np.full((1,2,2,2),i,dtype=np.float32),target_latent=np.full((1,2,2,2),i+1,dtype=np.float32),
                prediction_occupancy_packed=np.packbits(pred),target_occupancy_packed=np.packbits(gt))
            stored=folder/f'{i}.npz';np.savez_compressed(stored,**arrays);digest=hashlib.sha256(stored.read_bytes()).hexdigest()
            files.entries.append(dict(relative_path=str(stored.relative_to(tmp_path)),sha256=digest,bytes=stored.stat().st_size))
            artifact=dict(sample_id=row['sample_id'],seed=row['seed'],artifact=stored.name,artifact_SHA256=digest,
                prediction_voxels_SHA256=pred_hash,target_voxels_SHA256=gt_hash,occupancy_counts=dict(tp=1,fp=2,fn=0))
            for key in ['prediction_latent','target_latent']:
                v=arrays[key];d=hashlib.sha256((str(v.dtype)+repr(v.shape)).encode());d.update(v.tobytes());artifact[key+'_SHA256']=d.hexdigest()
            artifacts.append(artifact)
        result['native_SS_artifacts']['artifacts']=artifacts
        pop=dict(population_scope=ENGINEERING,expected_assets=36,generation_batch_size=8,deterministic_algorithms=True,
            mode='bound-population',bundle=bundle_spec['path'],model_binding=saved,output=str(folder),
            wrapper_output=str(folder.parent/'wrapper'),source_SHA256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in module.population_sources(arm)})
        pop_spec=files.add(folder.parent/'population-plan.json',pop)
        result['verified_population_inputs']={pop_spec['path']:pop_spec['SHA256']}
        repetitions.append(dict(population_plan=pop_spec,evaluation=files.add(folder/'evaluation.json',result),
            audit=files.add(folder/'population-audit.json',audit),terminal=files.add(folder.parent/'wrapper/terminal.json',
                dict(PID=2147483000+index,birth=1.+index,returncode=0))))
    plan=dict(schema=module.SCHEMA,module_arm=arm,canonical_local_root=str(tmp_path),bundle=bundle_spec,
        reference=files.add(tmp_path/'reference.json',reference),repetitions=repetitions,output=str(tmp_path/'proof'),
        source_SHA256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in module.required_sources(arm)})
    planpath=tmp_path/'reduction-plan.json';files.add(planpath,plan);files.close()
    return planpath,plan,files


@pytest.mark.parametrize('arm,legacy',[('V1',True),('C_DECODER',True),('FULL',True),('E_LEARNED_RAW',False)])
def test_closed_IO_checks_all72_actual_stored_synthetic_arrays_and_distinct_load_identities(tmp_path,monkeypatch,arm,legacy):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');path,plan,_=closed_inputs(tmp_path,arm,legacy)
    proof=module.execute(path)
    assert proof['actual_saved_archive_files_verified']==72 and proof['actual_saved_latent_and_occupancy_array_bytes_verified']
    assert proof['model_binding']['module_arm']==arm and proof['independent_model_loads']==2
    assert (Path(plan['output'])/'audit.json').is_file()
    with pytest.raises(FileExistsError):module.execute(path)


@pytest.mark.parametrize('damage',['duplicate-process','other-output','missing-source','unreported-plan','array-bytes'])
def test_closed_IO_rejects_forged_process_provenance_or_missing_actual_arrays(tmp_path,monkeypatch,damage):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');path,plan,files=closed_inputs(tmp_path)
    if damage=='duplicate-process':
        first=json.loads(Path(plan['repetitions'][0]['terminal']['path']).read_text())
        target=Path(plan['repetitions'][1]['terminal']['path'])
        plan['repetitions'][1]['terminal']=files.add(target,first)
    elif damage in ['other-output','missing-source']:
        item=plan['repetitions'][1]['population_plan'];value=json.loads(Path(item['path']).read_text())
        if damage=='other-output':value['output']='/different/output'
        else:value['source_SHA256']={}
        changed=files.add(tmp_path/'changed-population.json',value)
        plan['repetitions'][1]['population_plan']=changed
        result_path=Path(plan['repetitions'][1]['evaluation']['path']);result=json.loads(result_path.read_text())
        result['verified_population_inputs']={changed['path']:changed['SHA256']}
        plan['repetitions'][1]['evaluation']=files.add(result_path.parent/'changed-evaluation.json',result)
    elif damage=='unreported-plan':plan['repetitions'][1]['population_plan']['SHA256']='b'*64
    else:(tmp_path/'repeat1/native/35.npz').write_bytes(b'changed')
    # Publish a distinct root receipt, preserving each input's exact entry.
    if damage!='array-bytes':
        target=tmp_path/'damaged-reduction-plan.json';files.add(target,plan)
        unique={e['relative_path']:e for e in files.entries};files.entries=list(unique.values());files.close();path=target
    with pytest.raises(ValueError):module.execute(path)
    assert not Path(plan['output']).exists()
