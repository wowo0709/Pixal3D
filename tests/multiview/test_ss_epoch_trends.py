import json
import pytest

from experiments.deformnext_retargeting.summarize_ss_epoch_trends import completed_lines,record_epoch_contrasts


def test_inflight_jsonl_tail_is_not_a_completed_validation(tmp_path):
    path=tmp_path/'validation.jsonl';path.write_text('{"step":96}\n{"step":192')
    assert completed_lines(path)==[{'step':96}]


def test_only_matching_completed_epochs_produce_interim_comparisons(tmp_path):
    ids=[f'asset{i}' for i in range(32)]
    for name,steps,tp in [('p1-native-grid',[0,96,192],10),('p2-matcher-grid',[0,96],11)]:
        run=tmp_path/'retarget-native-v3/runs_ss64'/name;run.mkdir(parents=True)
        (run/'config.json').write_text(json.dumps({'validation':{'quick_metrics':{'sampling_steps':12}}}))
        rows=[{'step':step,'quick_per_asset':[{'sample_id':sid,'source_cluster_id':'source'+str(i),
            'seed':i,'tp':tp,'fp':5,'fn':20-tp} for i,sid in enumerate(ids)]} for step in steps]
        (run/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    outputs=record_epoch_contrasts(tmp_path)
    assert len(outputs)==1
    record=json.loads(outputs[0].read_text())
    assert record['step']==96 and record['assets']==32
    assert record['status']=='interim' and record['selection_use'] is False
    assert record['fixed_final128_three_noise'] is False
    assert record_epoch_contrasts(tmp_path)==[]


def test_seed101_epochs_require_hashes_and_same_execution_and_stay_in_own_namespace(tmp_path):
    folder=tmp_path/'ss-seed-confirmation-v1'
    paths=[]
    for name,tp in [('v1',10),('c1',11)]:
        run=folder/'runs_ss64'/name;run.mkdir(parents=True)
        config={'training_seed':101,'num_gpus':2,'trainer':{'args':{'batch_size_per_gpu':48,'batch_split':1}},
            'validation':{'quick_metrics':{'sampling_steps':12}}}
        (run/'config.json').write_text(json.dumps(config))
        row={'step':96,'quick_per_asset':[{'sample_id':f'asset{i}','source_cluster_id':f'source{i}',
            'seed':i,'tp':tp,'fp':5,'fn':20-tp,'target_voxels_sha256':f'target{i}',
            'target_voxel_shape':[64,64,64]} for i in range(32)]}
        path=run/'validation.jsonl';path.write_text(json.dumps(row)+'\n');paths.append(path)
    changed=json.loads(paths[1].read_text());changed['quick_per_asset'][0].pop('target_voxels_sha256')
    paths[1].write_text(json.dumps(changed)+'\n')
    with pytest.raises(ValueError,match='requires actual target voxel hashes'):record_epoch_contrasts(tmp_path)
    changed['quick_per_asset'][0]['target_voxels_sha256']='target0'
    paths[1].write_text(json.dumps(changed)+'\n')
    config_path=paths[1].parent/'config.json';config=json.loads(config_path.read_text())
    config['trainer']['args']['batch_split']=2;config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='requires matched execution'):record_epoch_contrasts(tmp_path)
    config['trainer']['args']['batch_split']=1;config_path.write_text(json.dumps(config))
    outputs=record_epoch_contrasts(tmp_path);assert len(outputs)==1
    result=json.loads(outputs[0].read_text());assert result['training_seed']==101 and result['global_batch']==96
    assert outputs[0].parent==folder/'epoch-comparisons'
    assert result['paired_target_hashes_verified']==32 and not result['selection_use']
