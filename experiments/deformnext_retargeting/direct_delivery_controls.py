"""Saved NoDQ interventions on delivery of C(F), preserving computed R and F.

This is a diagnostic consumer, never a training conditioner. The original
consumer runs once; normal returns its original result. Controls reconstruct
the same dense nearest consumer with only C's coefficient changed. In
particular, the semantic carrier is taken from the original evidence minus
the original residual, before any intervention or floating-point cancellation
can change it.
"""
from contextlib import contextmanager

import torch


CONTROLS = ('normal', 'direct-C-zero', 'direct-C-fusion-gated')


def controlled_residual(original_residual, masked_R, correction, gate, control):
    if control not in CONTROLS:
        raise ValueError('explicit saved-model direct-C control required')
    if original_residual.shape != masked_R.shape or correction.shape != masked_R.shape:
        raise ValueError('aligned original residual, R and C required')
    if gate.numel() != 1 or not torch.isfinite(gate).all():
        raise ValueError('actual finite scalar fusion gate required')
    if not torch.equal(original_residual, masked_R + correction):
        raise ValueError('captured R+C must exactly reconstruct original residual')
    if control == 'normal':
        return original_residual
    if control == 'direct-C-zero':
        return masked_R
    return masked_R + gate * correction


def nearest_dense_consumer(native, anchor, original_carrier, residual, valid, gate, patch_grid):
    """Exact original r9-native-v5 dense SS operation order; no model forward."""
    if (original_carrier.shape != residual.shape or original_carrier.ndim != 3
            or valid.shape != original_carrier.shape[:2] or valid.dtype != torch.bool
            or patch_grid.ndim != 4 or patch_grid.shape[0] != original_carrier.shape[0]
            or anchor.shape != native.shape or native.shape[-1] != original_carrier.shape[-1]):
        raise ValueError('explicit aligned dense nearest SS tensors required')
    patch_grid = patch_grid.long()
    if torch.any((patch_grid < 0) | (patch_grid >= original_carrier.shape[1])):
        raise ValueError('invalid target patch index')
    evidence = torch.cat((original_carrier, residual, valid[..., None].to(original_carrier.dtype)), -1)
    index = patch_grid.flatten(1)
    projected = torch.gather(evidence, 1, index[..., None].expand(-1, -1, evidence.shape[-1]))
    width = original_carrier.shape[-1]
    carrier, projected_residual, projected_valid = projected.split((width, width, 1), -1)
    if native.shape != carrier.shape:
        raise ValueError('native projection does not align with patch-grid delivery')
    semantic = (anchor + 3 * carrier.to(native.dtype)) / 4
    delta = semantic - native
    update = gate.to(native.dtype) * delta + (3 / 4) * projected_residual.to(native.dtype)
    return native + torch.where(projected_valid.bool(), update, torch.zeros_like(update))


@contextmanager
def direct_delivery_control(trainer_class, events, *, control, fingerprint):
    """Process-local consumer patch with actual per-forward invariant proofs."""
    if control not in CONTROLS:
        raise ValueError('unknown direct-C delivery intervention')
    original = trainer_class._inject_deform_transport
    prior = trainer_class.__dict__.get('_inject_deform_transport')

    def inject(self, condition, transport, coords, *, source_views=3, control='normal',
               anchor_condition=None, camera_info=None):
        from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
        # `control` here is the original transport API's input control, not
        # the enclosing delivery intervention.
        if (control != 'normal' or coords is not None or source_views != 3 or anchor_condition is None
                or self.deform_conditioner.implementation != 'r9-native-v5'
                or self.deform_conditioner.projection_mode != 'nearest'
                or self.deform_conditioner.value_dim != 1024
                or not isinstance(self.deform_conditioner, DirectRayRetargetingNoQueryConditioner)):
            raise ValueError('only the declared frozen physical-NoDQ dense SS consumer is supported')
        module = self.deform_conditioner
        if self.training_models.get('deform_conditioner', module) is not module:
            raise ValueError('single frozen evaluation conditioner required')
        if hasattr(module.retargeter, 'deformation_projection') or module.retargeter.input_dim != 2048:
            raise ValueError('physical absence of deformation channels required')
        key = 'proj_semantic' if 'proj_semantic' in condition else 'proj'
        native, anchor, global_object = condition[key], anchor_condition[key], condition['global']
        head, retarget, outputs = [], [], []
        handles = [module.ray_readout.register_forward_hook(lambda m, i, o: head.append((o[0], fingerprint(o[1])))),
                   module.retargeter.register_forward_hook(lambda m, i, o: retarget.append((o, fingerprint(i)))),
                   module.register_forward_hook(lambda m, i, o: outputs.append(o))]
        try:
            result = original(self, condition, transport, coords, source_views=source_views,
                              control=control, anchor_condition=anchor_condition, camera_info=camera_info)
        finally:
            for handle in handles:
                handle.remove()
        if len(retarget) != 1 or len(outputs) != 1:
            raise ValueError('one unmodified conditioner/R forward required')
        output = outputs[0]
        valid = output.query_valid
        selected = valid.flatten().nonzero().flatten()
        delta = output.evidence.new_zeros(valid.numel(), module.value_dim)
        if head:
            delta = delta.index_copy(0, selected, torch.cat([item[0] for item in head]))
        elif len(selected):
            raise ValueError('active rays did not execute the readout')
        F = delta.reshape(*valid.shape, module.value_dim)
        C = module.g_max * torch.tanh(F / module.g_max)
        R = retarget[0][0] * valid[..., None]
        original_residual = output.retarget_residual
        # Preserve the exact carrier actually consumed by the original method.
        carrier = output.evidence - original_residual
        normal_residual = controlled_residual(original_residual, R, C, output.fusion_scale, 'normal')
        reconstructed = nearest_dense_consumer(native, anchor, carrier, normal_residual,
                                               valid, output.fusion_scale, transport['patch_grid'])
        if not torch.equal(reconstructed, result[key]):
            raise ValueError('diagnostic reconstruction changed the original normal consumer')
        del reconstructed
        delivered = controlled_residual(original_residual, R, C, output.fusion_scale, intervention)
        if intervention != 'normal':
            result[key] = nearest_dense_consumer(native, anchor, carrier, delivered,
                                                 valid, output.fusion_scale, transport['patch_grid'])
        if result['global'] is not global_object or self._last_deform_condition is not output:
            raise ValueError('native global object or original conditioner output changed')
        events.append(dict(
            actual_CUDA_autocast_enabled=torch.is_autocast_enabled('cuda'),
            actual_CUDA_autocast_dtype=str(torch.get_autocast_dtype('cuda')),
            fixed_fields=fingerprint(dict(transport=transport, native_projected=native,
                anchor_projected=anchor, native_global=global_object, original_carrier=carrier,
                candidate_indices=output.candidate_indices, matching_logits=output.matching_logits,
                candidate_weights=output.candidate_weights, view_weights=output.view_weights,
                query_valid=valid, fusion_gate=output.fusion_scale, F=F, C=C, masked_R=R,
                R_input=retarget[0][1], attention_probabilities=[item[1] for item in head],
                original_evidence=output.evidence, original_residual=original_residual)),
            original_consumer_reconstruction_exact=True, original_R_plus_C_exact=True,
            native_global_object_preserved=True, original_module_output_unchanged=True,
            delivered_residual_SHA256=fingerprint(delivered), delivered_projected_SHA256=fingerprint(result[key]),
            actual_fusion_gate=float(output.fusion_scale.detach()),
            actual_C_coefficient=1. if intervention == 'normal' else (
                0. if intervention == 'direct-C-zero' else float(output.fusion_scale.detach())),
            C_RMS=float(C.square().mean().sqrt()), R_RMS=float(R.float().square().mean().sqrt())))
        return result

    intervention = control
    trainer_class._inject_deform_transport = inject
    try:
        yield
    finally:
        if prior is None:
            delattr(trainer_class, '_inject_deform_transport')
        else:
            trainer_class._inject_deform_transport = prior
