import copy
import json
from types import SimpleNamespace

import pytest

from experiments.deformnext_retargeting.semantic_topk_selection import CANDIDATES,SELECT_JOB,selection,supported,queue_candidate


def contrast(gain=.01,low=.001,noises=None):
    return {'gain':gain,'paired_cluster_bootstrap_ci95':[low,gain+.01],
        'per_noise':[{'gain':x} for x in (noises if noises is not None else [gain]*3)]}


def reports():
    names=['top1-reference','static-top1-c1','previous-c1','previous-v1','task-native-spatial',*CANDIDATES]
    models={name:{'micro_voxel_iou':.43,'micro_voxel_f1':.6} for name in names}
    models['task-full']['micro_voxel_iou']=.45
    seven={'status':'PASS_SEVEN_ARM_FIXED_FINAL_DEVELOPMENT_SEARCH','models':copy.deepcopy(models),'contrasts':{}}
    matched={'status':'PASS_MATCHED_RETARGETER_TOP1_FIXED_FINAL','models':copy.deepcopy(models),'contrasts':{}}
    for name in CANDIDATES:
        for baseline in ['previous-v1','top1-reference']:
            seven['contrasts'][name+'-vs-'+baseline]=contrast()
        matched['contrasts'][name+'-vs-static-top1-c1']=contrast()
    return seven,matched


def test_selects_observed_best_only_after_both_complete_reports():
    a,b=reports();r=selection(a,b)
    assert r['selected_candidate']=='task-full' and r['full_goal_complete'] is False
    a['status']='RUNNING'
    with pytest.raises(ValueError,match='both complete'):selection(a,b)


@pytest.mark.parametrize('winner',['task-no-decoder','task-no-prior'])
def test_does_not_force_decoder_or_scalar_prior_when_ablation_wins(winner):
    a,b=reports()
    for report in [a,b]:report['models'][winner]['micro_voxel_iou']=.46
    assert selection(a,b)['selected_candidate']==winner


def test_negative_or_noise_unstable_search_cannot_be_reported_as_positive():
    a,b=reports()
    for name in CANDIDATES:a['contrasts'][name+'-vs-top1-reference']=contrast(low=-.001)
    result=selection(a,b)
    assert result['status']=='NO_PROMISING_COMBINATION_YET' and result['selected_candidate'] is None
    assert not supported(contrast(noises=[.02,.02,-.001]))
    assert not supported(contrast(gain=-.01,low=-.02))


def test_best_control_cannot_be_hidden_by_semantic_candidate_selection():
    a,b=reports()
    for report in [a,b]:report['models']['task-native-spatial']['micro_voxel_iou']=.5
    result=selection(a,b)
    assert result['selected_candidate']=='task-full'
    assert result['best_overall_development_model']=='task-native-spatial'


def test_inconsistent_fixed_final_reports_are_rejected():
    a,b=reports();b['models']['static-c1']['micro_voxel_iou']+=.01
    with pytest.raises(ValueError,match='disagree'):selection(a,b)


def test_selected_candidate_is_fresh_and_uses_different_port_from_concurrent_baseline(tmp_path,monkeypatch):
    root=tmp_path/'result';work=tmp_path/'work';root.mkdir();work.mkdir();(root/'configs').mkdir()
    a,b=reports()
    for filename,value in [('seven-arm-final-report.json',a),('matched-retargeter-top1-report.json',b)]:
        (root/filename).write_text(json.dumps(value))
    original={'trainer':{'args':{'max_steps':480,'batch_size_per_gpu':48,'batch_split':1,
        'continuation_ckpt':{'denoiser':'SS4210'}}},'training_seed':101,'num_gpus':2,'experiment':{'arm':'task-full'}}
    (root/'configs/task-full.json').write_text(json.dumps(original))
    (root/'configs/zero-shot-reference.json').write_text(json.dumps(original))
    extra=root/'additional-validation-v1';extra.mkdir()
    (extra/'panel-contract.json').write_text(json.dumps({'status':'RESERVED_NOT_EVALUATED',
        'additional_assets':121,'additional_source_clusters':39,'panel_SHA256':'reserved-panel',
        'seed_maps':{str(n):{'path':str(tmp_path/f'noise{n}.json')} for n in range(3)}}))
    baseline=root/'seed-confirmation-v1/configs/top1-reference.json';baseline.parent.mkdir(parents=True)
    baseline.write_text(json.dumps({'master_port':'30440'}));baseline_bytes=baseline.read_bytes()
    jobs=[{'name':SELECT_JOB},*[{'name':f'confirm-seed202-final-top1-reference-noise{n}'} for n in range(3)]]
    (work/'queue-jobs.json').write_text(json.dumps(jobs))
    monkeypatch.setattr('experiments.deformnext_retargeting.semantic_topk_selection.subprocess.check_output',lambda *args,**kw:'test-source\n')
    queue_candidate(SimpleNamespace(root=root,work_root=work,previous_work=tmp_path/'panels',gpus='0,1'))
    cfg=json.loads((root/'seed-confirmation-v1/configs/task-full.json').read_text())
    assert cfg['training_seed']==202 and cfg['trainer']==original['trainer'] and cfg['num_gpus']==2
    assert cfg['master_port']=='30441' and baseline.read_bytes()==baseline_bytes
    actual=json.loads((work/'queue-jobs.json').read_text());new=actual[len(jobs):]
    assert len(new)==50 and new[0]['gpus']==['0','1'] and new[0]['depends_on']==[SELECT_JOB]
    assert any(j['name']=='summarize-selected-model-correspondence' for j in new)
    assert json.loads((root/'selected-correspondence-v1/contract.json').read_text())['candidate']=='task-full'
    extra_jobs=[j for j in new if j['name'].startswith('additional121-')]
    assert len(extra_jobs)==42 and all('gpu' not in j and 'gpus' not in j for j in extra_jobs)
    assert not any('resume' in j['command'] for j in new)
    assert json.loads((root/'candidate-selection-v1.json').read_text())['selected_candidate']=='task-full'
