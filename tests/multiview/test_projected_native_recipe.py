import copy

import pytest

from experiments.deformnext_retargeting.projected_native_recipe import NAME,SCHEMA,VARIANTS,validate_config
from test_r_mediated_native_recipe import recipe


def control(arm='C_FIXED_TOPK',phase='science'):
    reference=recipe('R_raw_topk');cfg=copy.deepcopy(reference)
    cfg['trainer']['name']=NAME;cfg['trainer']['args'].pop('ray_feature_utility')
    cfg['use_wandb']=phase=='science'
    cfg['experiment'].update(native_variant=arm,conditioning_schema=SCHEMA,
        module_pipeline_schema='unified-module-pipeline-v1',readout_selection=VARIANTS[arm],
        actual_source_tensors=12,learned_projection_tensors=2,
        physical_deformation_query_channels=0,engineering_only=phase!='science')
    return cfg,reference


@pytest.mark.parametrize('arm',VARIANTS)
@pytest.mark.parametrize('phase',['proof','capacity','science'])
def test_projected_controls_share_native_model_training_and_data_contract(arm,phase):
    cfg,reference=control(arm,phase)
    assert validate_config(cfg,phase,reference)==arm


@pytest.mark.parametrize('change',['selection','source-lr','batch','data','teacher','head','extra-DQ','resume'])
def test_no_module_or_training_confound_is_silently_allowed(change):
    cfg,reference=control();t=cfg['trainer']['args'];e=cfg['experiment']
    if change=='selection':e['readout_selection']='uniform'
    elif change=='source-lr':t['model_learning_rates']['deform_conditioner']=.001
    elif change=='batch':t['batch_size_per_gpu']=48
    elif change=='data':cfg['data_dir']='other assets'
    elif change=='teacher':e['source_teacher_updates']=112
    elif change=='head':e['learned_projection_tensors']=9
    elif change=='extra-DQ':e['physical_deformation_query_channels']=2048
    else:cfg['load_dir']='old run'
    with pytest.raises(ValueError):validate_config(cfg,'science',reference)
