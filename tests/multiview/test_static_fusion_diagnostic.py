import copy
import math

import pytest

from experiments.deformnext_retargeting.static_fusion_diagnostic import CONTROLS,jobs,summarize


def fixture():
    rows=[{'sample_id':str(i),'source_cluster_id':str(i),'seed':i,'tp':2,'fp':1,'fn':2,
        'target_voxels_sha256':'a'*64,'target_voxel_shape':[64,64,64]} for i in range(128)]
    common={'status':'PASS','stage':'ss64','step':480,'weights':'raw','sampling_steps':12,'guidance':3.,
        'generation_batch_size':8,'optimizer_updates_during_evaluation':0,'target_modality':'rgb',
        'conditioning_mode':'edited-rgb','transport_control':'normal','task_value_control':'normal',
        'checkpoint':'softSS480','conditioner_checkpoint':'softCond480','arm_config':'soft.json','manifest_sha256':'b'*64,
        'input_camera_ids':[8,9,10,11],'input_states':['M1','M0','M0','M0']}
    reference={**common,'per_asset':copy.deepcopy(rows)};records={}
    for name,mode in CONTROLS.items():
        events=[]
        if mode!='normal':
            top1=mode.startswith('top1-feature');dose=4 if mode.endswith('x4') else 1
            events=[{'mode':mode,'optimizer_updates':0,'pre_tanh_fusion_multiplier':dose,
                'actual_module_arm':'b2' if top1 else 'b3','selected_feature_count':1 if top1 else 64,
                'inference_valid_queries':64,'original_fusion_gate':.04,'actual_fusion_gate':math.tanh(dose*math.atanh(.04))}]
        records[name]={**common,'r9_path_control':mode,'diagnostic_only':mode!='normal',
            'per_asset':copy.deepcopy(rows[:32]),'r9_path_control_events':events}
    return records,reference


def test_completed_null_diagnostic_is_valid_without_becoming_model_success():
    records,reference=fixture();result=summarize(records,reference)
    assert result['status']=='PASS_COMPLETED_SAME_WEIGHT_AGGREGATION_FUSION_DIAGNOSTIC'
    assert result['full_goal_complete'] is False
    assert all(c['gain']==0 and c['paired_target_hashes_verified']==32 for c in result['contrasts'].values())


@pytest.mark.parametrize('field',['checkpoint','conditioner_checkpoint','manifest_sha256'])
def test_all_controls_must_use_the_same_actual_saved_model(field):
    records,reference=fixture();records['top1-gate4'][field]='another'
    with pytest.raises(ValueError,match='same saved'):summarize(records,reference)


def test_normal_generation_must_exactly_replay_fixed_final_subset():
    records,reference=fixture();records['soft-normal']['per_asset'][0]['tp']=3
    with pytest.raises(ValueError,match='exactly replay'):summarize(records,reference)


def test_recorded_higher_gate_must_obey_actual_tanh_parameter_dose():
    records,reference=fixture();records['soft-gate4']['r9_path_control_events'][0]['actual_fusion_gate']=.9
    with pytest.raises(ValueError,match='bounded gate'):summarize(records,reference)


def test_zero_queries_cannot_be_claimed_as_an_actual_intervention():
    records,reference=fixture();records['top1-normal']['r9_path_control_events'][0]['inference_valid_queries']=0
    with pytest.raises(ValueError,match='events'):summarize(records,reference)


def test_diagnostic_queue_uses_only_free_assigned_gpus_and_does_not_add_training(tmp_path):
    tasks=jobs(tmp_path,tmp_path/'work');assert len(tasks)==5
    for task in tasks[:4]:
        assert task['resource']=='evaluation' and 'gpu' not in task and 'gpus' not in task
        assert task['depends_on']==['train-static-c1','train-soft-topk']
        assert 'train.py' not in task['command']
        assert task['command'][task['command'].index('--generation-batch')+1]=='8'
    assert tasks[-1]['depends_on']==[t['name'] for t in tasks[:4]]
