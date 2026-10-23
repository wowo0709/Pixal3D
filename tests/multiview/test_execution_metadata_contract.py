import copy

import pytest

from experiments.deformnext_retargeting.execution_metadata_contract import validate_execution_metadata
from test_raw_correspondence_native import control as raw_control
from test_unified_design_controls import control as design_control


def inputs(arm):
    original,_=(raw_control(arm) if arm!='C_DECODER_NO_R' else design_control(arm))
    original['experiment'].update(assigned_GPUs=['6','7'],CPU_affinity_per_GPU={'6':[76,77],'7':[78,79]},recipe_path='/original.json')
    assignment=dict(authorized_GPUs=['0','1','2','3','6','7'],assigned_GPUs=['0','1','2','3'],
        CPU_affinity_per_GPU={'0':[84,85],'1':[86,87],'2':[88,89],'3':[90,91]},recipe_path='/actual.json')
    actual=copy.deepcopy(original)
    for k in ['assigned_GPUs','CPU_affinity_per_GPU','recipe_path']:actual['experiment'][k]=copy.deepcopy(assignment[k])
    return original,actual,assignment


@pytest.mark.parametrize('arm',['C_RAW','C_SPATIAL','C_D1_COS','C_DECODER_NO_R'])
def test_physical_device_assignment_preserves_every_scientific_configuration_field(arm):
    original,actual,assignment=inputs(arm);before=copy.deepcopy(original)
    value=validate_execution_metadata(original,actual,assignment)
    assert value['all_other_configuration_exact'] and original==before
    assert set(value['changed_fields'])=={'assigned_GPUs','CPU_affinity_per_GPU','recipe_path'}


@pytest.mark.parametrize('damage',['seed','microbatch','accumulation','LR','FFN','source-count','data',
    'checkpoint','sampling','output','extra-field','unauthorized-device','wrong-affinity','wrong-recipe'])
def test_execution_policy_cannot_hide_model_or_training_changes(damage):
    original,actual,assignment=inputs('C_RAW');t=actual['trainer']['args'];e=actual['experiment']
    if damage=='seed':actual['training_seed']=202
    elif damage=='microbatch':t['batch_size_per_gpu']=48
    elif damage=='accumulation':t['batch_split']=2
    elif damage=='LR':t['model_learning_rates']['deform_conditioner']=.001
    elif damage=='FFN':t['ffn_chunk_assets']=12
    elif damage=='source-count':e['actual_source_tensors']=19
    elif damage=='data':actual['data_dir']='other'
    elif damage=='checkpoint':t['continuation_ckpt']['denoiser']='other'
    elif damage=='sampling':actual['validation']['quick_metrics']['sampling_steps']=20
    elif damage=='output':actual['output_dir']='elsewhere'
    elif damage=='extra-field':e['additional_unreviewed_field']=True
    elif damage=='unauthorized-device':assignment['assigned_GPUs']=['4'];assignment['CPU_affinity_per_GPU']={'4':[84,85]};e.update(assigned_GPUs=['4'],CPU_affinity_per_GPU={'4':[84,85]})
    elif damage=='wrong-affinity':e['CPU_affinity_per_GPU']['0']=[1,2]
    else:e['recipe_path']='other'
    with pytest.raises(ValueError):validate_execution_metadata(original,actual,assignment)
