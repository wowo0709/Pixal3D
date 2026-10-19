import json

import pytest

from experiments.deformnext_retargeting.deterministic_ray_final import check_sources, verify_model
from experiments.deformnext_retargeting.deterministic_study_contract import validate_evaluation
from test_deterministic_study_contract import evaluation


def save(path, value):
    path.write_text(json.dumps(value));return str(path)


def model_files(tmp_path):
    models=[]; maps=[]
    for noise in range(3):
        record=evaluation(128)
        for index,row in enumerate(record['per_asset']):
            row.update(source_cluster_id='source-'+str(index%93),seed=index+noise*1000)
        models.append(save(tmp_path/f'noise{noise}.json',record))
        maps.append(save(tmp_path/f'seeds{noise}.json',{r['sample_id']:r['seed'] for r in record['per_asset']}))
    small=evaluation(32)
    small['per_asset']=json.loads(open(models[0]).read())['per_asset'][:32]
    return dict(final_paths=models,step=480,mode='edited-rgb',
        replay_path=save(tmp_path/'small.json',small),seed_maps=maps,
        panel128=save(tmp_path/'panel128.json',{'sample_ids':[str(i) for i in range(128)]}),
        panel32=save(tmp_path/'panel32.json',{'sample_ids':[str(i) for i in range(32)]}))


def test_final_requires_prescribed_noises_and_identical_saved_weights(tmp_path):
    model=model_files(tmp_path)
    assert verify_model(model)[1]['exact_counts_GT_noise_and_model_identity']
    path=tmp_path/'noise1.json';record=json.loads(path.read_text())
    record['per_asset'][0]['seed']+=1;save(path,record)
    with pytest.raises(ValueError,match='prescribed noise'):
        verify_model(model)
    model=model_files(tmp_path);record=json.loads(path.read_text())
    record['checkpoint']='/other/checkpoint.pt';save(path,record)
    with pytest.raises(ValueError,match='identity differs'):
        verify_model(model)


def test_final_rejects_a_different_gt_or_unrecorded_replay_panel(tmp_path):
    model=model_files(tmp_path);path=tmp_path/'noise1.json';record=json.loads(path.read_text())
    record['per_asset'][0]['target_voxels_sha256']='d'*64;save(path,record)
    with pytest.raises(ValueError,match='GT order'):
        verify_model(model)
    model=model_files(tmp_path);save(tmp_path/'panel32.json',{'sample_ids':[str(i) for i in range(1,33)]})
    with pytest.raises(ValueError,match='prescribed fixed32'):
        verify_model(model)


def test_wrong_address_provenance_cannot_be_used_as_a_normal_quality_result():
    record=evaluation(32);record.update(task_value_control='wrong-address',diagnostic_only=True)
    validate_evaluation(record,assets=32,step=480,value_control='wrong-address')
    with pytest.raises(ValueError,match='protocol'):
        validate_evaluation(record,assets=32,step=480)


def test_source_closure_does_not_accept_an_unclosed_archive(tmp_path):
    from hashlib import sha256
    runtime=tmp_path/'runtime.py';runtime.write_text('frozen source')
    archive=tmp_path/'archive.py';archive.write_bytes(runtime.read_bytes());digest=sha256(runtime.read_bytes()).hexdigest()
    contract=save(tmp_path/'contract.json',dict(runtime_components=[dict(runtime_path=str(runtime),archived_path=str(archive),sha256=digest)]))
    receipt=save(tmp_path/'receipt.json',dict(status='PASS',local_equals_actual_file4_SHA256=True,files=[]))
    plan=dict(source_contract=contract,source_receipt=receipt,receipt_root=str(tmp_path))
    with pytest.raises(ValueError,match='actual file4 receipt'):
        check_sources(plan)
    save(tmp_path/'receipt.json',dict(status='PASS',local_equals_actual_file4_SHA256=True,files=[dict(relative_path='archive.py',sha256=digest)]))
    check_sources(plan)
    runtime.write_text('changed')
    with pytest.raises(ValueError,match='frozen source changed'):
        check_sources(plan)
