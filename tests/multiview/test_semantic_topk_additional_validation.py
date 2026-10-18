import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.semantic_topk_additional_validation import jobs_and_contract,validate,summarize


def reservation():
    return {'status':'RESERVED_NOT_EVALUATED','additional_assets':121,'additional_source_clusters':39,
        'panel_SHA256':'fixed-before-selection','seed_maps':{str(n):{'path':f'map{n}.json'} for n in range(3)}}


def test_additional_panel_waits_for_selection_and_keeps_both_training_seeds_and_all_controls(tmp_path):
    (tmp_path/'configs').mkdir();(tmp_path/'configs/zero-shot-reference.json').write_text(json.dumps({'trainer':{'args':{'continuation_ckpt':{'denoiser':'SS4210'}}}}))
    jobs,contract=jobs_and_contract(tmp_path,tmp_path/'previous','task-no-prior',reservation(),'selection')
    assert len(jobs)==43 and len(contract['models'])==14
    assert contract['models']['s202-task-no-prior']['training_seed']==202
    assert contract['models']['s101-task-no-prior']['training_seed']==101
    assert {'previous-v1','previous-c1','s101-static-top1-c1','s101-task-native-spatial','zero-source-only','zero-edited-rgb'}<=contract['models'].keys()
    assert all('selection' in j['depends_on'] for j in jobs[:-1])
    assert all(j['resource']=='evaluation' and 'gpu' not in j and 'gpus' not in j for j in jobs[:-1])
    assert all('--panel-file' in j['command'] and '--generation-batch' in j['command'] for j in jobs[:-1])
    assert 'summarize-semantic-topk-seed202' in jobs[-1]['depends_on']


def test_native_spatial_control_cannot_be_used_as_selected_semantic_candidate(tmp_path):
    with pytest.raises(ValueError,match='selected semanticTopK'):
        jobs_and_contract(tmp_path,tmp_path,'task-native-spatial',reservation(),'selection')


def test_additional_validation_rejects_changed_source_seed_depth_or_checkpoint():
    spec={'step':480,'conditioning_mode':'edited-rgb','checkpoint':'raw480','conditioner_checkpoint':'cond480','config':'fixed.json'}
    e={'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
        'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'conditioning_mode':'edited-rgb',
        'target_modality':'rgb','diagnostic_only':False,'checkpoint':'raw480','conditioner_checkpoint':'cond480','arm_config':'fixed.json',
        'input_camera_ids':[8,9,10,11],'input_states':['M1','M0','M0','M0'],'quick_panel_ids':['a'],
        'per_asset':[{'sample_id':'a','source_cluster_id':'held-source','seed':17,'target_voxels_sha256':'a'*64,'target_voxel_shape':[64,64,64]}]}
    rows={'a':{'source_cluster_id':'held-source'}};seeds={'a':17};validate(e,spec,['a'],rows,seeds)
    for key,value in [('target_modality','depth'),('checkpoint','different'),('optimizer_updates_during_evaluation',1)]:
        bad=copy.deepcopy(e);bad[key]=value
        with pytest.raises(ValueError):validate(bad,spec,['a'],rows,seeds)
    bad=copy.deepcopy(e);bad['per_asset'][0]['source_cluster_id']='selection-source'
    with pytest.raises(ValueError,match='source/noise'):validate(bad,spec,['a'],rows,seeds)


@pytest.mark.parametrize('candidate',['task-full','previous-c1'])
def test_complete_synthetic_additional_panel_pairs_all_models_without_claiming_goal_complete(tmp_path,candidate):
    root=tmp_path/'run';folder=root/'additional-validation-v1';folder.mkdir(parents=True);(root/'configs').mkdir()
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    ids=[f'extra-{i}' for i in range(121)]
    valrows=[{'sample_id':sid,'source_cluster_id':'extra-source-'+str(i%39)} for i,sid in enumerate(ids)]
    devrows=[{'sample_id':f'dev-{i}','source_cluster_id':'dev-source-'+str(i%93)} for i in range(128)]
    valrows+=devrows+[{'sample_id':f'unused-{i}','source_cluster_id':'dev-source-'+str(i%93)} for i in range(259)]
    val=tmp_path/'validation.jsonl';val.write_text(''.join(json.dumps(r)+'\n' for r in valrows))
    train=tmp_path/'train.jsonl';train.write_text(json.dumps({'sample_id':'training','source_cluster_id':'train-source'})+'\n')
    save(root/'configs/top1-reference.json',{'data_dir':str(train),'validation':{'roots_file':str(val)}})
    save(root/'configs/zero-shot-reference.json',{'trainer':{'args':{'continuation_ckpt':{'denoiser':'SS4210'}}}})
    panel=folder/'panel-source-disjoint-121.json';save(panel,{'sample_ids':ids})
    r=reservation();r.update(panel_SHA256=sha(panel),manifest_SHA256={'train':sha(train),'validation':sha(val)})
    maps={}
    for n in range(3):
        path=tmp_path/f'noise{n}.json';maps[n]={sid:i+1000*n for i,sid in enumerate(ids)};save(path,maps[n])
        r['seed_maps'][str(n)]={'path':str(path),'sha256':sha(path)}
    save(folder/'panel-contract.json',r)
    _,contract=jobs_and_contract(root,tmp_path/'previous',candidate,r,'selection')
    contract['reservation_SHA256']=sha(folder/'panel-contract.json');save(folder/'evaluation-contract.json',contract)
    save(root/'candidate-selection-v1.json',{'selected_candidate':candidate,'status':'PROMISING_COMBINATION_FOR_SEED_CONFIRMATION'})
    save(root/'evaluations/top1-reference/noise-0/evaluation.json',{'per_asset':devrows})
    for label,spec in contract['models'].items():
        first='previous-c1' if candidate=='previous-c1' else 's101-'+candidate
        tp=60 if label in [first,'s202-'+candidate] else 50
        for n in range(3):
            e={'status':'PASS','stage':'ss64','step':spec['step'],'weights':'raw','sampling_steps':12,'guidance':3.,
                'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'conditioning_mode':spec['conditioning_mode'],
                'target_modality':'rgb','diagnostic_only':False,'checkpoint':spec['checkpoint'],
                'conditioner_checkpoint':spec['conditioner_checkpoint'],'arm_config':spec['config'],
                'input_camera_ids':[8,9,10,11],'input_states':['M0']*4 if spec['conditioning_mode']=='source-only' else ['M1','M0','M0','M0'],
                'quick_panel_ids':ids,'per_asset':[{'sample_id':sid,'source_cluster_id':'extra-source-'+str(i%39),
                    'seed':maps[n][sid],'tp':tp,'fp':10,'fn':100-tp,'target_voxels_sha256':hashlib.sha256(sid.encode()).hexdigest(),
                    'target_voxel_shape':[64,64,64]} for i,sid in enumerate(ids)]}
            save(folder/'evaluations'/label/f'noise-{n}/evaluation.json',e)
    summarize(root);report=json.loads((folder/'final-report.json').read_text())
    assert len(report['models'])==14 and not report['full_goal_complete']
    assert report['positive_CI_and_all_noises_on_both_additional_training_seeds']
    assert report['per_training_seed_matched_Top1']['202']['paired_target_hashes_verified']==363
    assert report['per_training_seed_matched_Top1']['101']['source_clusters']==39
    if candidate=='previous-c1':
        assert 's101-previous-c1' not in contract['models']
        assert contract['models']['previous-c1']['config'].endswith('ss-seed-confirmation-v1/configs/c1.json')
        assert report['per_training_seed_matched_Top1']['101']==report['contrasts']['previous-c1-vs-s101-top1-reference']
