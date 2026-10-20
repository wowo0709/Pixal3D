"""Explicit calibrated RGB protocols for the original four-stage MV model."""
from pathlib import Path

import numpy as np


def zero_shot_inputs(row, mode, path_map):
    """Resolve only declared paths; the original anchor occupies camera8."""
    if mode not in {'edited-rgb', 'source-only'}:
        raise ValueError('explicit zero-shot RGB mode required')
    if (row['conditioning_camera_ids'] != [8, 9, 10, 11]
            or len(row['source_images']) != 3
            or len(row['transform_matrix']) != 4
            or len(row['camera_angle_x']) != 4
            or row['target_anchor_kind'] != 'rgb'):
        raise ValueError('four calibrated RGB inputs at cameras8/9/10/11 required')
    target_camera = np.asarray(row['transform_matrix'][0], dtype=np.float32)
    source_camera = np.asarray(row['source_anchor_transform_matrix'], dtype=np.float32)
    if (target_camera.shape != (4, 4) or source_camera.shape != (4, 4)
            or not np.isfinite(target_camera).all() or not np.isfinite(source_camera).all()
            or not np.allclose(target_camera, source_camera, rtol=1e-5, atol=1e-8)
            or abs(row['camera_angle_x'][0] - row['source_anchor_camera_angle_x']) > 1e-6):
        raise ValueError('original and edited anchors must have the same camera/FOV')
    anchor = row['source_anchor_rgb'] if mode == 'source-only' else row['target_anchor']
    declared = [anchor, *row['source_images']]
    if len(set(declared)) != 4 or row['source_anchor_rgb'] == row['target_anchor']:
        raise ValueError('four distinct declared RGB files and distinct anchor states required')
    paths = [str(Path(path_map[path])) for path in declared]
    if len(set(paths)) != 4 or any(Path(path).suffix.lower() not in {'.png', '.jpg', '.jpeg'} for path in paths):
        raise ValueError('four distinct explicit local RGB paths required')
    return dict(conditioning_mode=mode, input_camera_ids=[8, 9, 10, 11],
                input_states=['M0'] * 4 if mode == 'source-only' else ['M1', 'M0', 'M0', 'M0'],
                declared_RGB_paths=declared, RGB_paths=paths, target_modality='rgb',
                source_anchor_same_camera_as_target=True, visibility_mask_used=False)


def validate_zero_shot_cascade(plan, evaluation):
    protocol = zero_shot_inputs(plan['row'], plan['conditioning_mode'], plan['RGB_path_map'])
    for field in ['input_states', 'RGB_paths']:
        if plan[field] != protocol[field]:
            raise ValueError('actual cascade inputs differ from declared zero-shot ' + field)
    for field in ['conditioning_mode', 'input_states', 'input_camera_ids', 'target_modality']:
        if evaluation[field] != protocol[field]:
            raise ValueError('SS archive and downstream RGB protocol differ: ' + field)
    if (evaluation['stage'] != 'ss64' or evaluation['step'] != 0
            or evaluation['checkpoint'] != plan['original_SS_checkpoint']
            or evaluation['conditioner_checkpoint'] is not None
            or evaluation['optimizer_updates_during_evaluation'] != 0
            or evaluation['diagnostic_only']):
        raise ValueError('original unmodified zero-shot SS4210 archive required')
    archive = evaluation['native_SS_artifacts']
    if (archive['status'] != 'PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY'
            or not archive['original_prefix32_all_records_and_occupancy_counts_exact']):
        raise ValueError('exact original native SS reference replay required')
    expected_steps = {'shape512': 4210, 'shape1024': 16850, 'pbr1024': 16850}
    if set(plan['stages']) != set(expected_steps):
        raise ValueError('all three downstream original stages required')
    for name, step in expected_steps.items():
        spec = plan['stages'][name]
        if (spec['original_multiview_step'] != step
                or spec['checkpoint'] != plan['original_stage_checkpoints'][name]):
            raise ValueError('original multi-view base+LoRA checkpoint required: ' + name)
    return protocol
