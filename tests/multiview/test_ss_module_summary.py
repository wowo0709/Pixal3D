import json

import pytest

from experiments.deformnext_retargeting.summarize_ss_module_study import audit_training,result_contrasts
from experiments.deformnext_retargeting.queue_baseline_repeats import jobs
from pathlib import Path


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def complete_run(tmp_path):
    arm='model';run=tmp_path/'runs_ss64'/arm
    write(run/'completed.json',{'status':'PASS','updates':480,'epochs':5,'global_batch':96,'world_size':2})
    write(run/'progress.json',{'stage':'completed','current':480})
    write(tmp_path/'configs/model.json',{'trainer':{'args':{'deform_conditioner':{'arm':'c1'}}},
        'experiment':{'head_sha256':'head','omega_checkpoint':'frozen'}})
    write(tmp_path/'verification/model/native-proof.json',{'status':'PASS','condition_max_abs_errors':{'global':0,'proj':0},'fusion_gate_gradient':.001,'adapter_output_gradient_l1':.01})
    validation=[{'step':step,'quick_metrics':{'voxel_iou':.42}} for step in [0,96,192,288,384,480]]
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
