"""The extension must check the non-screening suffix and reject leakage."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.dev128_pipeline_contract import (
    validate_archive128, validate_generation, validate_population,
)
from test_selected_pipeline_joint_v2 import inputs


def archive():
    bindings, _, _, _, _, _, kwargs = inputs()
    value = deepcopy(kwargs['archives']['E_LEARNED_RAW']['record'])
    rows, artifacts = [], []
    for i in range(128):
        rows.append(dict(sample_id=str(i), source_cluster_id=str(i % 93), seed=i,
            target_voxels_sha256='a'*64, tp=1, fp=2, fn=3))
        artifacts.append(dict(sample_id=str(i), seed=i, target_voxels_SHA256='a'*64,
            occupancy_counts=dict(tp=1, fp=2, fn=3)))
    value['per_asset'] = rows
    value['native_SS_artifacts'] = dict(archived_asset_count=128, artifacts=artifacts,
        same_forward_prediction_and_target_capture=True,
        predicted_SS_is_available_for_full_cascade=True, GT_SS_is_for_scoring_only=True)
    return value, deepcopy(value), bindings['SELECTED']['ss64']


def test_accepts_existing_full128_captures_without_new_SS_inference():
    value, reference, binding = archive()
    assert validate_archive128(value, 'E_LEARNED_RAW', reference, binding=binding) == list(map(str, range(128)))


@pytest.mark.parametrize('damage', ['suffix-target', 'suffix-noise', 'suffix-count', 'duplicate', 'missing', 'GT-generation'])
def test_damaging_suffix_beyond_original32_cannot_pass(damage):
    value, reference, binding = archive()
    if damage == 'suffix-target':
        value['per_asset'][127]['target_voxels_sha256'] = 'b'*64
    elif damage == 'suffix-noise':
        value['per_asset'][80]['seed'] += 1
    elif damage == 'suffix-count':
        value['native_SS_artifacts']['artifacts'][90]['occupancy_counts']['fp'] += 1
    elif damage == 'duplicate':
        value['per_asset'][127]['sample_id'] = '0'
    elif damage == 'missing':
        value['per_asset'].pop()
    else:
        value['native_SS_artifacts']['GT_SS_is_for_scoring_only'] = False
    with pytest.raises(ValueError):
        validate_archive128(value, 'E_LEARNED_RAW', reference, binding=binding)


@pytest.mark.parametrize('damage', ['test', 'depth'])
def test_extension_never_admits_test_rows_or_target_depth(damage):
    ids = list(map(str, range(128)))
    clusters = {sid:str(i % 93) for i, sid in enumerate(ids)}
    rows = [dict(sample_id=sid, source_cluster_id=clusters[sid], split='validation',
        conditioning_camera_ids=[8,9,10,11], target_anchor_kind='rgb', target_anchor='rgb',
        target_rgb_input='rgb', source_images=['a','b','c']) for sid in ids]
    validate_population(rows, ids, clusters)
    rows[127]['split' if damage == 'test' else 'target_anchor_kind'] = damage
    with pytest.raises(ValueError):
        validate_population(rows, ids, clusters)


@pytest.mark.parametrize('field', ['GT_geometry_or_coords_used_in_generation', 'per_output_alignment'])
def test_good_metrics_cannot_override_GT_generation_or_result_alignment(field):
    base = dict(status='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION', sample_id='x',
        SS_prediction_SHA256='a'*64, GT_geometry_or_coords_used_in_generation=False,
        GT_loaded_only_after_generation=True, per_output_alignment=False)
    replay = dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE', sample_id='x', mode='top1-only')
    validate_generation(base, replay, sample_id='x', SS_prediction_SHA256='a'*64)
    base[field] = True
    with pytest.raises(ValueError):
        validate_generation(base, replay, sample_id='x', SS_prediction_SHA256='a'*64)
