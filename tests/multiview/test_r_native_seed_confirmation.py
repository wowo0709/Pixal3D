import copy
import hashlib
import json

import pytest

from experiments.deformnext_retargeting.r_native_seed_confirmation import seed202_recipe,seed202_reference,validate_pair
from test_r_mediated_native_recipe import recipe
from test_r_top1_native_recipe import control


def inputs(tmp_path,phase):
    vanilla=recipe('vanilla');top1,reference=control()
    original=tmp_path/'original.json';original.write_text(json.dumps(reference))
    top1['experiment'].update(reference_science_config=str(original),
        reference_science_config_SHA256=hashlib.sha256(original.read_bytes()).hexdigest())
    ref202=seed202_reference(reference,output_dir='/recipe-only/R19',run_id='ref202')
    path=tmp_path/'reference202.json';path.write_text(json.dumps(ref202));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    a=seed202_recipe(vanilla,variant='vanilla',phase=phase,output_dir='/new202/vanilla/'+phase,run_id='vanilla202')
    b=seed202_recipe(top1,variant='R_top1_only',phase=phase,output_dir='/new202/Top1/'+phase,run_id='top1202',
        reference=ref202,reference_path=path,reference_SHA256=digest)
    return vanilla,top1,a,b,ref202


@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_changes_only_seed_output_logging_phase_and_pinned_reference(tmp_path,phase):
    olda,oldb,a,b,ref=inputs(tmp_path,phase)
    assert validate_pair(a,b,ref,phase=phase)['updates']==480
    for old,new in[(olda,a),(oldb,b)]:
        reverse=copy.deepcopy(new)
        for field in['training_seed','output_dir','wandb_id','wandb_name','use_wandb']:
            if field in old:reverse[field]=old[field]
            else:reverse.pop(field,None)
        reverse['experiment']['engineering_only']=old['experiment']['engineering_only']
        for field in['reference_science_config','reference_science_config_SHA256']:
            if field in old['experiment']:reverse['experiment'][field]=old['experiment'][field]
        assert reverse==old


@pytest.mark.parametrize('damage',['seed101','batch','teacher','learning-rate','data','reference-seed'])
def test_rejects_changed_confirmation_contract(tmp_path,damage):
    _,_,a,b,ref=inputs(tmp_path,'science')
    if damage=='seed101':b['training_seed']=101
    elif damage=='batch':b['trainer']['args']['batch_size_per_gpu']=48
    elif damage=='teacher':b['experiment']['source_teacher_updates']=112
    elif damage=='learning-rate':b['trainer']['args']['optimizer']['args']['lr']=1e-4
    elif damage=='data':b['data_dir']='different subset'
    elif damage=='reference-seed':ref['training_seed']=101
    with pytest.raises(ValueError):validate_pair(a,b,ref,phase='science')
