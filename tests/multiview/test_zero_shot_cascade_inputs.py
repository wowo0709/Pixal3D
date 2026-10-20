import copy
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from experiments.deformnext_retargeting.cascade_artifact_inputs import prepared_views
from experiments.deformnext_retargeting.evaluate_zero_shot_ss_artifact_subset import validate_arguments
from experiments.deformnext_retargeting.zero_shot_cascade_inputs import zero_shot_inputs, validate_zero_shot_cascade
from pixal3d.datasets.deformnext import _load_image


def row():
    return dict(conditioning_camera_ids=[8, 9, 10, 11], source_images=['s9.png', 's10.png', 's11.png'],
                target_anchor='edited8.png', source_anchor_rgb='source8.png', target_anchor_kind='rgb',
                transform_matrix=[np.eye(4).tolist()] * 4, source_anchor_transform_matrix=np.eye(4).tolist(),
                camera_angle_x=[1.] * 4, source_anchor_camera_angle_x=1.)


def test_zero_modes_change_only_original_target_camera_RGB():
    r = row(); paths = {p: '/cache/' + p for p in [r['target_anchor'], r['source_anchor_rgb'], *r['source_images']]}
    edited = zero_shot_inputs(r, 'edited-rgb', paths); source = zero_shot_inputs(r, 'source-only', paths)
    assert edited['input_states'] == ['M1', 'M0', 'M0', 'M0']
    assert source['input_states'] == ['M0'] * 4
    assert source['RGB_paths'] == ['/cache/source8.png', '/cache/s9.png', '/cache/s10.png', '/cache/s11.png']
    assert source['RGB_paths'][1:] == edited['RGB_paths'][1:]
    r['source_anchor_transform_matrix'][0][3] = 1
    with pytest.raises(ValueError, match='same camera'): zero_shot_inputs(r, 'source-only', paths)


@pytest.mark.parametrize('mutation', ['three-cameras', 'depth', 'missing-path', 'duplicate-local', 'fov'])
def test_wrong_modalities_cameras_or_undeclared_paths_fail(mutation):
    r = row(); paths = {p: '/cache/' + p for p in [r['target_anchor'], r['source_anchor_rgb'], *r['source_images']]}
    if mutation == 'three-cameras': r['conditioning_camera_ids'] = [9, 10, 11]
    elif mutation == 'depth': r['target_anchor_kind'] = 'depth'
    elif mutation == 'missing-path': del paths['source8.png']
    elif mutation == 'duplicate-local': paths['source8.png'] = paths['s9.png']
    elif mutation == 'fov': r['source_anchor_camera_angle_x'] = .8
    with pytest.raises((ValueError, KeyError)): zero_shot_inputs(r, 'source-only', paths)


def test_actual_native_dataset_and_cascade_RGBA_pixels_exact(tmp_path):
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (19, 27, 4), dtype=np.uint8))
    path = tmp_path / 'rgba.png'; image.save(path)
    for size in [16, 32]:
        assert torch.equal(_load_image(path, size), prepared_views([image], size, 'cpu')[0, 0])


def test_zero_cascade_reuses_exact_native_sampling_and_geometry_body():
    from experiments.deformnext_retargeting import cascade_from_ss_artifact as old
    from experiments.deformnext_retargeting import cascade_zero_shot_from_ss_artifact as zero
    original = Path(old.__file__).read_text(); current = Path(zero.__file__).read_text()
    start = '    from pixal3d import models\n'
    end = '    result=dict(status='
    original_body = original[original.index(start, original.index('def main():')):original.index(end)]
    current_body = current[current.index(start, current.index('def main():')):current.index(end)]
    assert current_body == original_body
    assert zero.load_stage is old.load_stage
    assert current_body.index('meshes,latents=pipeline.run(') < current_body.index("np.load(plan['source_geometry'])")


def test_deformation_PBR_or_mismatched_SS_protocol_rejected():
    r = row(); paths = {p: '/cache/' + p for p in [r['target_anchor'], r['source_anchor_rgb'], *r['source_images']]}
    protocol = zero_shot_inputs(r, 'source-only', paths)
    checkpoints = {name: f'/{name}/original.pt' for name in ['shape512', 'shape1024', 'pbr1024']}
    plan = dict(protocol, row=r, RGB_path_map=paths, original_SS_checkpoint='/ss4210.pt',
                original_stage_checkpoints=checkpoints, stages={name: dict(checkpoint=path,
                original_multiview_step=4210 if name == 'shape512' else 16850) for name, path in checkpoints.items()})
    evaluation = dict(protocol, stage='ss64', step=0, checkpoint='/ss4210.pt', conditioner_checkpoint=None,
                      optimizer_updates_during_evaluation=0, diagnostic_only=False,
                      native_SS_artifacts=dict(status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
                      original_prefix32_all_records_and_occupancy_counts_exact=True))
    assert validate_zero_shot_cascade(plan, evaluation) == protocol
    bad = copy.deepcopy(plan); bad['stages']['pbr1024']['checkpoint'] = '/deformation762.pt'
    with pytest.raises(ValueError, match='original multi-view'): validate_zero_shot_cascade(bad, evaluation)
    bad = copy.deepcopy(evaluation); bad['input_states'] = ['M1', 'M0', 'M0', 'M0']
    with pytest.raises(ValueError, match='protocol differ'): validate_zero_shot_cascade(plan, bad)


def test_archive_protocol_requires_original_SS_and_no_conditioner():
    args = ['--evaluation-module', 'evaluate_deterministic_checkpoint', '--stage', 'ss64', '--step', '0',
            '--panel-size', '32', '--generation-batch', '8', '--conditioning-mode', 'source-only']
    validate_arguments(args)
    for changed in [args + ['--step', '0'], args + ['--conditioner-checkpoint', '/head.pt'],
                    [a if a != '0' else '480' for a in args]]:
        with pytest.raises(ValueError): validate_arguments(changed)
