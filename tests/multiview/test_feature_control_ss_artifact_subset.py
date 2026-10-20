import copy

import pytest

from experiments.deformnext_retargeting.evaluate_feature_control_ss_artifact_subset import validate_control_replay, validate_runtime
from experiments.deformnext_retargeting.member_feature_control_recipe import NAME, SCHEMA, CONTROLS
from test_ss_artifact_subset import reference_pair


def fixture(kind):
    spec = dict(CONTROLS[kind], control=kind); actual, reference, captured = reference_pair()
    source = dict(arm=spec['arm'], feature_control=kind, source_value_control='normal',
        physical_deformation_query_channels=0, physical_retargeter_input_channels=2048,
        actual_feature_head_tensors=spec['head_tensors'], QK_score_parameters_present=spec['head_tensors'] == 9,
        original_correspondence_candidate_count=spec['candidates'], actual_optimizer_states=0, target_carrier_is_native_four_view_V1=True)
    for result, count in [(actual, 32), (reference, 128)]:
        result.update(step=480, diagnostic_only=False, native_prior_source_ray=copy.deepcopy(source))
        result['per_asset'] = [dict(result['per_asset'][0], sample_id=f'asset-{i}') for i in range(count)]
    captured = [dict(captured[0], sample_id=f'asset-{i}') for i in range(32)]
    marker = dict(status='PASS_COMPLETED_MEMBER_FEATURE_CONTROL_RUNTIME_PHASE', phase='evaluation', trainer=NAME, schema=SCHEMA,
        arm=spec['arm'], physical_control=kind, actual_feature_head_tensors=spec['head_tensors'],
        actual_source_tensors=spec['source_tensors'], actual_correspondence_candidates=spec['candidates'], physical_deformation_query_channels=0)
    return actual, reference, captured, marker, spec


@pytest.mark.parametrize('kind', list(CONTROLS))
def test_exact_replay_and_actual_physical_runtime_required(kind):
    actual, reference, captured, marker, spec = fixture(kind)
    validate_control_replay(actual, reference, captured, spec); validate_runtime(marker, spec)
    other = next(key for key in CONTROLS if key != kind)
    with pytest.raises(ValueError): validate_control_replay(actual, reference, captured, dict(CONTROLS[other], control=other))


@pytest.mark.parametrize('field,value', [
    ('actual_feature_head_tensors', 5), ('QK_score_parameters_present', True),
    ('source_value_control', 'wrong-feature-values-fixed-use'), ('actual_optimizer_states', 1),
    ('original_correspondence_candidate_count', 3), ('physical_retargeter_input_channels', 4096)])
def test_wrong_feature_control_identity_cannot_release_identical_counts(field, value):
    actual, reference, captured, marker, spec = fixture('top1-feature')
    actual['native_prior_source_ray'][field] = value
    with pytest.raises(ValueError): validate_control_replay(actual, reference, captured, spec)


def test_wrong_phase_source_count_or_candidate_count_marker_fails():
    _, _, _, marker, spec = fixture('spatial-feature-attention')
    for field, value in [('phase', 'science'), ('actual_source_tensors', 12), ('actual_correspondence_candidates', 64)]:
        with pytest.raises(ValueError): validate_runtime(dict(marker, **{field: value}), spec)


def test_duplicate_reference_or_source_only_protocol_cannot_release_archive():
    actual, reference, captured, _, spec = fixture('top1-feature')
    reference['per_asset'][80]['sample_id'] = reference['per_asset'][79]['sample_id']
    with pytest.raises(ValueError): validate_control_replay(actual, reference, captured, spec)
    actual, reference, captured, _, spec = fixture('top1-feature')
    actual['input_states'] = reference['input_states'] = ['M0'] * 4
    with pytest.raises(ValueError): validate_control_replay(actual, reference, captured, spec)
