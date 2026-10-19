"""Mute transformed appearance only; retain correspondence and real DQ."""
from contextlib import contextmanager

from .ray_feature_controls import readout_control


@contextmanager
def appearance_path_off(conditioner, events):
    if not hasattr(conditioner.retargeter, 'deformation_projection'):
        raise ValueError('physical DQ projection must remain present')

    def observe(module, inputs, output):
        events.append(dict(control='appearance-path-off', DQ_projection_retained=True,
            real_deformation_input_RMS=float(inputs[0].detach().float().square().mean().sqrt()),
            DQ_projection_output_RMS=float(output.detach().float().square().mean().sqrt()),
            original_Top1_carrier_retained=True, source_RGB_Omega_and_matching_scores_retained=True))

    handle = conditioner.retargeter.deformation_projection.register_forward_hook(observe)
    try:
        with readout_control(conditioner.ray_readout, 'residual-off', events):
            yield
    finally:
        handle.remove()
