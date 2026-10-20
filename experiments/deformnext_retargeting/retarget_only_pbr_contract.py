"""Physical PBR Top1/R control and its own native continuation contracts."""
from __future__ import annotations

import hashlib

import torch


ARM = 'retarget-only-top1'


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_retarget_only(direct_config):
    """Consume the same Full constructor first and copy all30 common R states.

    The control has no attention module, RAW deformation channels or projection.
    Extra RNG consumed by its allocation is reset by the native stage seed.
    """
    from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
    from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner

    if ((direct_config['arm'], direct_config['value_dim'], direct_config['query_dim'],
         direct_config['deformation_dim']) != ('c2', 2048, 1024, 2048)
            or direct_config['candidate_aggregation'] != 'top1'
            or direct_config['projection_mode'] != 'nearest'
            or direct_config['implementation'] != 'r9-native-v5'):
        raise ValueError('the actual Full PBR2048/RAW2048 nearest Top1 carrier is required')
    prototype = DirectRayRetargetingConditioner(**direct_config)
    common = prototype.reference_state()
    config = dict(direct_config, arm='c1')
    config.pop('deformation_dim')
    module = DeformationAwareConditioner(**config)
    module.load_state_dict(common, strict=True)
    module.q2c.requires_grad_(False)
    module.retargeter.requires_grad_(True)
    module.fusion_gate.requires_grad_(True)
    if (len(common) != 30 or len(module.state_dict()) != 30
            or state_digest(common) != state_digest(module.state_dict())
            or sum(p.requires_grad for p in module.parameters()) != 6
            or module.retargeter.input_dim != 3072
            or hasattr(module, 'ray_readout')
            or hasattr(module.retargeter, 'deformation_projection')
            or module.task_attention is not None
            or float(module.fusion_gate.detach()) != 0
            or torch.count_nonzero(module.retargeter.net[-1].weight)
            or torch.count_nonzero(module.retargeter.net[-1].bias)):
        raise ValueError('exact common R30,only6 trainable tensors and physical3072 control required')
    return module, dict(common_R_state_tensors=30, common_R_SHA256=state_digest(common),
        common_R_exact_to_Full_constructor=True, physical_input_channels=3072,
        additional_deformation_channels=0, attention_module_physically_absent=True,
        matching_tensors_frozen=24, new_trainable_R_tensors=6)


def validate_profile(report, *, capacity):
    expected_status = ('PASS_ACTUAL_RETARGET_ONLY_PBR_DDP4_CAPACITY' if capacity
                       else 'PASS_ACTUAL_RETARGET_ONLY_PBR_MICRO1_PROFILE')
    updates = 4 if capacity else 3
    if (report.get('status') != expected_status or report.get('arm') != ARM
            or report.get('initial_optimizer_states') != 360
            or report.get('actual_optimizer_states') != 366
            or report.get('changed_trainable_tensors_by_group') != dict(default=360,retargeter=6)
            or report.get('final_optimizer_steps') != dict(default=[762+updates],retargeter=[updates])
            or not report.get('all_frozen_NN_and_matcher_states_exact')
            or not report.get('common_R_exact_to_Full_constructor')
            or report.get('physical_R_input_channels') != 3072
            or not report.get('attention_and_DQ_physically_absent')):
        raise ValueError('own native R-only366-state profile is required')
    if capacity:
        if ((report.get('world_size'), report.get('microbatch'),report.get('global_batch'),
             report.get('accumulation')) != (4,6,24,1)
                or not report.get('post_update_DDP_equal')
                or not report.get('post_validation_full_micro_update_passed')
                or not report.get('initial_native_prediction_exact_to_V1_capacity')
                or not report.get('first_loss_fields_exact_to_V1_capacity')):
            raise ValueError('own max-token world4/micro6 native capacity and V1 identity are required')
    elif (report.get('world_size'),report.get('microbatch'),report.get('global_batch')) != (1,1,1):
        raise ValueError('micro1 engineering proof must retain its scope')


def validate_initial32(report):
    if (report.get('status') != 'PASS_ACTUAL_RETARGET_ONLY_PBR_INITIAL32_IDENTITY'
            or report.get('assets') != 32 or report.get('optimizer_updates') != 0
            or not report.get('all32_baseline_metrics_records_and_render_images_exact')
            or report.get('conditioner_state_tensors') != 30):
        raise ValueError('own R-only native generated32 exact initial identity required')
