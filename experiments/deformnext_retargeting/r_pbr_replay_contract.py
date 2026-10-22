"""Lineage contracts for new native source-PBR generated-geometry replay."""
from pathlib import Path

import numpy as np

from .generated_r_mediated_pbr import IMAGE_ROW_FIELDS, actual_target_patch_ids
from .pbr_replay_artifacts import PREDICTION_FIELDS


MODES = {'vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'}
KINDS = {'original16850-engineering', 'trained-native1900'}


def validate_lineage(plan, config, completion=None):
    mode, kind = plan.get('mode'), plan.get('checkpoint_kind')
    if (mode not in MODES or kind not in KINDS or type(plan.get('engineering_only')) is not bool
            or set(plan['row']) != set(IMAGE_ROW_FIELDS)
            or plan['input_states'] != ['M1', 'M0', 'M0', 'M0']
            or len(plan['RGB_paths']) != 4 or type(plan['PBR_replay_seed']) is not int
            or plan['PBR_replay_seed'] < 0 or plan['PBR_replay_seed'] >= 2**32
            or not 0 < plan['allocator_fraction'] <= 1):
        raise ValueError('explicit image-only four-RGB replay mode, lineage, seed and memory bound required')
    if (config['experiment']['PBR_source_mode'] != mode
            or config['experiment']['source_multiview_step'] != 16850):
        raise ValueError('checkpoint config must declare this own source mode and original PBR16850')
    source = plan.get('conditioner_checkpoint')
    spec = config['trainer']['args'].get('deform_conditioner')
    if mode == 'vanilla':
        if source is not None or spec is not None:
            raise ValueError('vanilla must physically omit source weights and module')
    elif spec is None or spec.get('appearance_mode') != mode:
        raise ValueError('new sparse source mode must match its own declared conditioner config')
    if kind == 'original16850-engineering':
        if (not plan['engineering_only'] or source is not None or completion is not None
                or plan['PBR_checkpoint'] != config['trainer']['args']['continuation_ckpt']['denoiser']):
            raise ValueError('engineering replay must use the original complete NN and a fresh cold source')
    else:
        native = Path(config['output_dir'])
        if (plan['engineering_only'] or plan['PBR_checkpoint'] != str(native/'ckpts/denoiser_step0001900.pt')
                or source != (None if mode == 'vanilla' else str(native/'ckpts/deform_conditioner_step0001900.pt'))
                or completion is None
                or completion.get('status') != 'PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE'
                or completion.get('mode') != mode or completion.get('actual_optimizer_updates') != 1900
                or completion.get('training_seed') != config['training_seed']
                or completion.get('source_multiview_step') != 16850
                or completion.get('source_tensors') != (0 if mode == 'vanilla' else 10 if mode == 'top1-only' else 19)
                or not completion.get('stage_training_complete')
                or not completion.get('frozen_base_matcher_and_anchors_exact')
                or not completion.get('post_training_DDP_parameters_exact')):
            raise ValueError('actual completed own native1900 NN/source pair and stage integrity required')
    return mode, kind


def validate_generated_shape(plan, report, prediction):
    if (report['sample_id'] != plan['row']['sample_id'] or report['actual_resolution'] != 1024
            or report['GT_geometry_or_coords_used_in_generation'] or report['per_output_alignment']
            or set(prediction) != set(PREDICTION_FIELDS)):
        raise ValueError('one unaligned real predicted1024 archive for this exact image input required')
    coords, features = prediction['shape_coords'], prediction['shape_feats']
    actual_target_patch_ids(plan['row'], coords)
    if (features.shape != (len(coords),32) or features.dtype.kind != 'f' or not np.isfinite(features).all()
            or prediction['texture_feats'].shape != features.shape
            or report['actual_HR_tokens'] != len(coords)):
        raise ValueError('finite actual generated native32-channel shape support in its stored order required')


def validate_cold_comparison(actual, reference):
    if (actual['checkpoint_kind'] != 'original16850-engineering'
            or reference['checkpoint_kind'] != 'original16850-engineering'
            or actual['PBR_checkpoint_SHA256'] != reference['PBR_checkpoint_SHA256']
            or actual['original_prediction_SHA256'] != reference['original_prediction_SHA256']
            or actual['sample_id'] != reference['sample_id'] or reference['PBR_arm'] != 'vanilla'):
        raise ValueError('cold parity requires the same original NN, actual generated shape and vanilla reference')
    for key in ['actual_initial_texture_noise_SHA256', 'actual_texture_latent_SHA256',
                'actual_decoded_coordinates_SHA256', 'actual_decoded_attributes_SHA256',
                'PBR_replay_seed', 'sampler_class', 'sampler_params', 'geometry_array_byte_identity']:
        if actual[key] != reference[key]:
            raise ValueError('actual cold generated-support replay differs: '+key)
