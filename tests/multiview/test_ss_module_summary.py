import json

import pytest

from experiments.deformnext_retargeting.summarize_ss_module_study import audit_training,result_contrasts,audit_validation_panel,audit_final_replay
from experiments.deformnext_retargeting import summarize_ss_module_study
from experiments.deformnext_retargeting.queue_baseline_repeats import jobs,protocol_replay_jobs
from pathlib import Path
import shutil
import sys


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def complete_run(tmp_path):
    arm='model';run=tmp_path/'runs_ss64'/arm
    ids=[f'object{i}' for i in range(32)];panel=tmp_path/'panel.json';write(panel,{'sample_ids':ids})
    write(run/'completed.json',{'status':'PASS','updates':480,'epochs':5,'global_batch':96,'world_size':2})
    write(run/'progress.json',{'stage':'completed','current':480})
    write(tmp_path/'configs/model.json',{'trainer':{'args':{'deform_conditioner':{'arm':'c1'}}},
        'validation':{'quick_metrics':{'panel_file':str(panel)}},
        'experiment':{'head_sha256':'head','omega_checkpoint':'frozen'}})
    write(tmp_path/'verification/model/native-proof.json',{'status':'PASS','condition_max_abs_errors':{'global':0,'proj':0},'fusion_gate_gradient':.001,'adapter_output_gradient_l1':.01})
    per_asset=[{'sample_id':sid,'source_cluster_id':'source'+str(i),'seed':i,'tp':21,'fn':4,'fp':25} for i,sid in enumerate(ids)]
    validation=[{'step':step,'quick_metrics':{'voxel_iou':.42},'quick_panel_ids':ids,'quick_per_asset':per_asset} for step in [0,96,192,288,384,480]]
    (run/'validation.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in validation))
    for rank in [0,1]:
        rows=[{'step':step,'effective_batch_size':96,'world_size':2,'accumulation_packs':1,
            'microbatch_assets':[48],'compute_seconds':17.2,'peak_allocated_gib':47.75,'loss':{'loss':.1}} for step in range(1,481)]
        (run/('runtime_batch.jsonl' if rank==0 else 'runtime_batch.rank1.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in rows))
    for prefix in ['denoiser','deform_conditioner','misc']:
        p=run/'ckpts'/f'{prefix}_step0000480.pt';p.parent.mkdir(exist_ok=True);p.write_bytes(b'saved')
    return tmp_path,run


def test_summary_checks_every_update_rank_epoch_and_checkpoint(complete_run):
    study,run=complete_run
    audit=audit_training(study,'model',.42)
    assert [x['updates'] for x in audit['rank_runtime']]==[480,480]
    records=(run/'runtime_batch.rank1.jsonl').read_text().splitlines()
    (run/'runtime_batch.rank1.jsonl').write_text('\n'.join(records[:-1])+'\n')
    with pytest.raises(ValueError,match='optimizer updates'):
        audit_training(study,'model',.42)


def test_completed_marker_cannot_hide_missing_validation_or_query_proof(complete_run):
    study,run=complete_run
    p=study/'configs/model.json';cfg=json.loads(p.read_text());cfg['trainer']['args']['deform_conditioner']['arm']='c2';write(p,cfg)
    with pytest.raises(ValueError,match='query proof missing'):
        audit_training(study,'model',.42)
    cfg['trainer']['args']['deform_conditioner']['arm']='c1';write(p,cfg)
    lines=(run/'validation.jsonl').read_text().splitlines();(run/'validation.jsonl').write_text('\n'.join(lines[1:])+'\n')
    with pytest.raises(ValueError,match='validations incomplete'):
        audit_training(study,'model',.42)


def test_primary_summary_contains_every_planned_module_contrast():
    names=['v1','original-c1','p1-native-grid','p2-matcher-grid','b2-top1','b2-top1-r','b3-attention-r','c1-retarget-r','c2-query-r']
    results={}
    for i,name in enumerate(names):
        results[name]={'records':[{'sampling_steps':12,'guidance':3.,'per_asset':[
            {'sample_id':'object','source_cluster_id':'source','seed':noise,'tp':5+i%3,'fn':5-i%3,'fp':3}]} for noise in range(3)]}
    comparisons=result_contrasts(results)
    for name in ['b2-top1-vs-v1','b2-top1-r-vs-b2-top1','b3-attention-r-vs-b2-top1-r',
        'c1-retarget-r-vs-b3-attention-r','c2-query-r-vs-c1-retarget-r','p2-matcher-grid-vs-p1-native-grid']:
        assert name in comparisons
        assert comparisons[name]['sampling_noise_seeds']==3
        assert comparisons[name]['training_seed_replication'] is False


def test_baseline_repeats_preserve_zero_shot_modality_and_saved_weights():
    scheduled=jobs(Path('/data'),Path('/work'),Path('/config'),Path('/weights/initializer.pt'))
    assert len(scheduled)==10
    assert len({j['name'] for j in scheduled})==10
    for job in scheduled:
        command=job['command']
        assert job['resource']=='evaluation'
        assert command[command.index('--generation-batch')+1]=='8'
        assert '--panel-file' in command and '--seed-map-file' in command
        assert 'train.py' not in command
        if 'zero-' in job['name']:
            mode=command[command.index('--conditioning-mode')+1]
            assert mode in {'source-only','edited-rgb'}
            assert command[command.index('--checkpoint')+1]=='/weights/initializer.pt'
            assert '--arm-config' not in command
        else:
            assert '--conditioner-checkpoint' in command
            assert command[command.index('--step')+1]=='480'


def test_epoch_panel_guard_detects_changed_noise_even_when_aggregate_is_unchanged(complete_run):
    study,run=complete_run;records=[json.loads(s) for s in (run/'validation.jsonl').read_text().splitlines()]
    ids=records[0]['quick_panel_ids'];audit_validation_panel(records,ids)
    records[1]['quick_per_asset'][0]['seed']+=1
    with pytest.raises(ValueError,match='noise, cluster or target changed'):
        audit_validation_panel(records,ids)


def test_protocol_replays_preserve_legacy_outputs_and_use_saved_weights():
    scheduled=protocol_replay_jobs(Path('/data'),Path('/work'),Path('/config'))
    assert len(scheduled)==2
    for arm,job in zip(['v1','c1'],scheduled):
        command=job['command']
        assert job['resource']=='evaluation' and 'train.py' not in command
        assert command[command.index('--checkpoint')+1]==f'/work/evaluation-checkpoints/{arm}/denoiser_step0000480.pt'
        assert command[command.index('--seed-map-file')+1]=='/work/panels/seed-map-extra-noise-1.json'
        assert command[command.index('--generation-batch')+1]=='8'
        assert command[command.index('--output')+1]==f'/data/baseline-noise-confirmation/protocol-replay/noise-1/{arm}'
        assert ('--conditioner-checkpoint' in command)==(arm=='c1')
        assert 'gpu' not in job and 'gpus' not in job


def test_replay_guard_requires_explicit_protocol_and_exact_per_asset_counts(tmp_path):
    for arm in ['v1','c1']:
        rows=[dict(sample_id=str(i),source_cluster_id=str(i),seed=i,tp=5,fp=3,fn=2) for i in range(128)]
        old={'status':'PASS','optimizer_updates_during_evaluation':0,'per_asset':rows}
        write(tmp_path/f'noise-confirmation/noise-1/{arm}/evaluation.json',old)
        write(summarize_ss_module_study.baseline_paths(tmp_path,arm)[1],dict(old,sampling_steps=12,guidance=3.))
    assert summarize_ss_module_study.audit_baseline_replays(tmp_path)['v1']['exact_seed_cluster_tp_fp_fn_replay']
    p=summarize_ss_module_study.baseline_paths(tmp_path,'v1')[1];fresh=json.loads(p.read_text())
    del fresh['guidance'];write(p,fresh)
    with pytest.raises(ValueError,match='replay protocol'):
        summarize_ss_module_study.audit_baseline_replays(tmp_path)
    fresh['guidance']=3.;fresh['per_asset'][0].update(tp=6,fn=1);write(p,fresh)
    with pytest.raises(ValueError,match='does not replay historical'):
        summarize_ss_module_study.audit_baseline_replays(tmp_path)


def test_complete_primary_report_covers_all_fourteen_models(complete_run,monkeypatch):
    template,run=complete_run;root=template/'campaign';study=root/'module-study-native-v4';reference=root/'retarget-native-v3'
    write(root/'selected-head.json',{'head_sha256':'head','arm':'h3'})
    for directory,names in [(study,[x[0] for x in summarize_ss_module_study.ARMS]),(reference,['p1-native-grid','p2-matcher-grid'])]:
        for name in names:
            shutil.copytree(run,directory/'runs_ss64'/name)
            cfg=json.loads((template/'configs/model.json').read_text())
            spec=next((a for n,a,_ in summarize_ss_module_study.ARMS if n==name),'c1')
            cfg['trainer']['args']['deform_conditioner']['arm']=spec
            write(directory/'configs'/(name+'.json'),cfg)
            proof=json.loads((template/'verification/model/native-proof.json').read_text())
            if spec=='c2':proof['query_channels']={'extra_channels':2048,'query_weight_gradient_l1':.01}
            write(directory/'verification'/name/'native-proof.json',proof)
    names=['zero-source-only','zero-edited-rgb','v1','original-c1','b1','b2','b3','p1-native-grid','p2-matcher-grid',*[x[0] for x in summarize_ss_module_study.ARMS]]
    for index,name in enumerate(names):
        if name.startswith('zero-'):
            paths=[root/f'evaluations/initializer/{name.removeprefix("zero-")}/evaluation.json',*[root/f'baseline-noise-confirmation/noise-{n}/{name}/evaluation.json' for n in [1,2]]]
        elif name in ['v1','original-c1']:paths=summarize_ss_module_study.baseline_paths(root,'v1' if name=='v1' else 'c1')
        elif name in ['b1','b2','b3']:paths=[root/f'evaluations/{name}/normal/evaluation.json',*[root/f'baseline-noise-confirmation/noise-{n}/{name}/evaluation.json' for n in [1,2]]]
        else:paths=summarize_ss_module_study.evaluation_paths(root,reference if name.startswith('p') else study,name)
        for noise,path in enumerate(paths):
            rows=[{'sample_id':f'object{i}','source_cluster_id':'source'+str(i),'seed':i+noise*1000,'tp':10+index,'fn':15-index,'fp':5} for i in range(128)]
            if noise==0:
                for row in rows[:32]:row.update(tp=21,fn=4,fp=25)
            write(path,{'status':'PASS','sampling_steps':12,'guidance':3.,'optimizer_updates_during_evaluation':0,'per_asset':rows})
            if noise==1 and name in ['v1','original-c1']:
                arm='v1' if name=='v1' else 'c1'
                write(root/f'noise-confirmation/noise-1/{arm}/evaluation.json',{'status':'PASS','optimizer_updates_during_evaluation':0,'per_asset':rows})
    monkeypatch.setattr(sys,'argv',['summary','--root',str(root)])
    summarize_ss_module_study.main()
    result=json.loads((study/'final-primary-report.json').read_text())
    assert result['status']=='primary_matrix_complete'
    assert len(result['models'])==14
    assert len(result['training_audit'])==7
    assert result['training_seed_replication'] is False
    assert all(r['final128_replays_epoch32'] for r in result['training_audit'].values())
    assert all(r['exact_seed_cluster_tp_fp_fn_replay'] for r in result['baseline_protocol_replays'].values())
    assert 'sampling_steps' not in json.loads((root/'noise-confirmation/noise-1/v1/evaluation.json').read_text())
    report=(study/'final-primary-report.md').read_text()
    assert 'zero-source-only' in report and 'c2-query-r' in report
    assert not list(study.glob('*.tmp'))
    # Even unchanged voxel counts cannot turn an inference diagnostic into
    # a normal ablation record.
    diagnostic=study/'noise-confirmation/noise-2/b2-top1/evaluation.json'
    value=json.loads(diagnostic.read_text());value.update(transport_control='source-off',diagnostic_only=True)
    write(diagnostic,value)
    with pytest.raises(ValueError,match='nonmatching final generation'):
        summarize_ss_module_study.main()


def test_final_replay_guard_detects_different_saved_weight_results(complete_run):
    _,run=complete_run;validation=[json.loads(s) for s in (run/'validation.jsonl').read_text().splitlines()]
    final={'per_asset':[dict(row) for row in validation[-1]['quick_per_asset']]}
    audit={'validation':validation}
    assert audit_final_replay(audit,final)
    final['per_asset'][0]['tp']-=1;final['per_asset'][0]['fn']+=1
    with pytest.raises(ValueError,match='same-checkpoint'):
        audit_final_replay(audit,final)
