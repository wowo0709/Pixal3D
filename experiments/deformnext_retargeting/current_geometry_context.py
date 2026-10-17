"""Scoped native-voxel content adapter; query input is physically optional."""
from contextlib import contextmanager
import types

import torch


@contextmanager
def geometry_content_context(trainer, adapter, geometry, events):
    if adapter.with_query != (geometry is not None):
        raise ValueError('geometry input must be physically absent in the No-query arm')
    original = trainer._encode_deform_views
    had_override = '_encode_deform_views' in trainer.__dict__

    def encode(self, images, camera_info, transport, **kwargs):
        if transport is not None or images.ndim != 5 or images.shape[1] != 4:
            raise ValueError('native-voxel proof requires exactly four RGB views and no patch transport')
        if self.image_cond_model is None:
            self._init_image_cond_model()
        captures = []
        hook = self._image_cond_module.proj_grid.register_forward_hook(
            lambda module, arguments, output: captures.append(output.detach()))
        try:
            native, negative = original(images, camera_info, None, **kwargs)
        finally:
            hook.remove()
        if len(captures) != 4 or set(native) != {'global', 'proj'}:
            raise ValueError('expected exactly four native per-view projections')
        total = captures[0].float()
        for output in captures[1:]:
            total = total + output.float()
        if not torch.equal((total / 4).to(native['proj'].dtype), native['proj']):
            raise ValueError('per-view captures do not reproduce native projection')
        before = native['proj']
        changed, trace = adapter(captures[0], before, geometry)
        native = {**native, 'proj': changed}
        events.append({'native_projection_exact': torch.equal(changed, before),
            'max_projection_update': float((changed - before).detach().abs().max()),
            'global_and_negative_unmodified': True, 'geometry_channels': trace['geometry_channels'],
            'residual_input_channels': adapter.residual[0].normalized_shape[0]})
        return native, negative

    trainer._encode_deform_views = types.MethodType(encode, trainer)
    try:
        yield
    finally:
        if had_override:
            trainer._encode_deform_views = original
        else:
            del trainer._encode_deform_views
