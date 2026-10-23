"""Scientific protocol consistency without data, models, or neural forwards."""
import pytest

from experiments.deformnext_retargeting.module_pipeline_protocol import protocol, validate_protocol


def test_all_module_contrasts_change_exactly_one_declared_factor():
    value = protocol()
    assert len(value['arms']) == 11 and len(value['module_contrasts']) == 9
    assert value['actual_new_training_started'] is False
    assert value['full_is_candidate_not_adopted'] is True


def test_old_direct_C_delivery_cannot_be_relabelled_as_this_full_pipeline():
    value = protocol();value['arms']['C_RAW']['delivery']='direct-C'
    with pytest.raises(ValueError):validate_protocol(value)


def test_decoder_comparison_cannot_also_change_query_or_retargeting():
    value = protocol();value['arms']['C_RAW']['query_unit']='individual-voxel'
    with pytest.raises(ValueError):validate_protocol(value)


def test_R_off_control_cannot_disconnect_a_full_attention_branch():
    value = protocol();value['arms']['C_DECODER_NO_R']['value_transform']=True
    with pytest.raises(ValueError):validate_protocol(value)


def test_no_score_control_is_top1_and_no_corr_uses_native_spatial_correspondence():
    value = protocol()
    assert value['arms']['C_TOP1_V']['selection']=='top1'
    assert value['arms']['C_SPATIAL']['correspondence']=='native-spatial'
