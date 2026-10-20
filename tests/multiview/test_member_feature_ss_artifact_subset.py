import copy
import pytest

from experiments.deformnext_retargeting.evaluate_member_feature_ss_artifact_subset import (
    validate_archive_arguments,validate_member_replay,RUNTIMES)
from test_ss_artifact_subset import reference_pair


def fixture(runtime):
    a,b,c=reference_pair();spec=RUNTIMES[runtime]
    source=dict(arm=spec['arm'],source_value_control='normal',physical_deformation_query_channels=spec['physical'],
        physical_retargeter_input_channels=spec['input_channels'],actual_QKV_feature_tensors=9,
        actual_optimizer_states=0,target_carrier_is_native_four_view_V1=True)
    a.update(step=480,diagnostic_only=False,native_prior_source_ray=source)
    b.update(step=480,diagnostic_only=False,native_prior_source_ray=copy.deepcopy(source))
    a['per_asset']=[dict(a['per_asset'][0],sample_id=f'asset-{i}') for i in range(32)]
    b['per_asset']=[dict(a['per_asset'][0],sample_id=f'asset-{i}') for i in range(128)]
    c=[dict(c[0],sample_id=f'asset-{i}') for i in range(32)]
    return a,b,c


@pytest.mark.parametrize('runtime',list(RUNTIMES))
def test_accepts_only_actual_matched_four_head_runtime_and_original128_reference(runtime):
    a,b,c=fixture(runtime)
    validate_member_replay(a,b,c,runtime)
    wrong=next(name for name in RUNTIMES if name!=runtime)
    with pytest.raises(ValueError,match='NoDQ/DQ archive identity'):
        validate_member_replay(a,b,c,wrong)


@pytest.mark.parametrize('field,value',[
    ('physical_retargeter_input_channels',4096),('physical_deformation_query_channels',2048),
    ('source_value_control','wrong-feature-values-fixed-use'),('actual_QKV_feature_tensors',5),
    ('actual_optimizer_states',1),('target_carrier_is_native_four_view_V1',False)])
def test_rejects_unchanged_prediction_records_with_wrong_model_or_intervention(field,value):
    a,b,c=fixture('member_feature_runtime');a['native_prior_source_ray'][field]=value
    with pytest.raises(ValueError,match='NoDQ/DQ archive identity'):
        validate_member_replay(a,b,c,'member_feature_runtime')


def test_rejects_source_only_or_changed_cam_order_even_when_both_records_agree():
    a,b,c=fixture('member_feature_runtime');a['input_states']=b['input_states']=['M0']*4
    with pytest.raises(ValueError):validate_member_replay(a,b,c,'member_feature_runtime')
    a,b,c=fixture('member_feature_runtime');a['input_camera_ids']=b['input_camera_ids']=[8,10,9,11]
    with pytest.raises(ValueError):validate_member_replay(a,b,c,'member_feature_runtime')


def test_duplicate_assets_cannot_release_a_complete_prediction_archive():
    a,b,c=fixture('member_feature_runtime')
    b['per_asset'][80]['sample_id']=b['per_asset'][79]['sample_id']
    with pytest.raises(ValueError,match='unique32'):validate_member_replay(a,b,c,'member_feature_runtime')


def test_archive_cannot_dispatch_science_phase_or_changed_sampler_budget():
    arguments=['--phase','evaluation','--stage','ss64','--step','480','--expected-assets','32',
        '--panel-size','32','--generation-batch','8','--source-value-control','normal']
    validate_archive_arguments(arguments)
    changed=list(arguments);changed[changed.index('--phase')+1]='science'
    with pytest.raises(ValueError):validate_archive_arguments(changed)
    with pytest.raises(ValueError):validate_archive_arguments(arguments+['--phase','science'])
    changed=list(arguments);changed[changed.index('--generation-batch')+1]='1'
    with pytest.raises(ValueError):validate_archive_arguments(changed)
