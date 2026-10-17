"""Scoped three-source-camera slot transport around the native SS encoder."""
from __future__ import annotations

from contextlib import contextmanager
import types

import torch

from . import historical_reproduction as reference

MODES = ('native', 'source-off', 'gt-corr', 'wrong-corr')


@contextmanager
def native_slot_context(trainer, mode, read_grid, write_valid, sample_ids, events):
    """Freeze native global/negative, and only reconstruct per-view proj maps.

    Read/write GT metadata is provided by a separate diagnostic runner. Nothing
    in this context changes a training conditioner or an inference-safe cache.
    """
    if mode not in MODES:
        raise ValueError('unknown native-slot reproduction case')
    if read_grid.shape != (*write_valid.shape, 2) or write_valid.shape[1:] != (3, 32, 32):
        raise ValueError('expected three independent32x32 source-camera routes')
    if len(sample_ids) != len(read_grid):
        raise ValueError('route/sample batch identities differ')
    original = trainer._encode_deform_views
    had_instance_override = '_encode_deform_views' in trainer.__dict__

    def encode(self, images, camera_info, transport, **kwargs):
        if transport is not None or images.shape[:2] != (len(sample_ids), 4):
            raise ValueError('reproduction requires native four-view input without another transport module')
        if self.image_cond_model is None:
            self._init_image_cond_model()
        extractor = self._image_cond_module
        captures = []
        def capture(module, arguments, output):
            feature = arguments[0]
            if feature.ndim != 4 or feature.shape[1:3] != (32, 32):
                raise ValueError('SS native feature map must be BHWC32x32')
            captures.append((tuple(value.detach() for value in arguments), output.detach().clone()))
        hook = extractor.proj_grid.register_forward_hook(capture)
        try:
            native, negative = original(images, camera_info, None, **kwargs)
        finally:
            hook.remove()
        if len(captures) != 4 or set(native) != {'global', 'proj'}:
            raise ValueError('expected exactly four native SS per-view projections')
        # Match the production online mean's dtype and summation order exactly.
        total = captures[0][1].float()
        for _, projected in captures[1:]:
            total = total + projected.float()
        replay = (total / 4).to(native['proj'].dtype)
        if not torch.equal(replay, native['proj']):
            raise ValueError('native four-view condition does not replay captured per-view features')
        event = {'mode': mode, 'sample_ids': list(sample_ids), 'native_joint_proj_max_abs': 0.,
                 'source_write_views': 3, 'write_valid_per_view': write_valid.sum((2, 3)).tolist(),
                 'global_and_negative_preserved': True}
        if mode == 'native':
            events.append(event)
            return native, negative
        target = captures[0][1]
        if mode == 'source-off':
            native['proj'] = target
            events.append(event)
            return native, negative
        source = torch.stack([arguments[0] for arguments, _ in captures[1:]], dim=1)
        grid = read_grid
        if mode == 'wrong-corr':
            perm = torch.cat([reference.wrong_permutation(write_valid[i:i+1], pair_id=sid)
                              for i, sid in enumerate(sample_ids)])
            grid = read_grid.reshape(len(sample_ids), 3, 1024, 2).gather(
                2, perm.reshape(len(sample_ids), 3, 1024, 1).expand(-1, -1, -1, 2)).reshape_as(read_grid)
        selected = reference._sample_neighborhood(source, grid, radius=0)[..., 0, :]
        selected = torch.where(write_valid[..., None], selected, torch.zeros_like(selected))
        if mode == 'wrong-corr':
            correct = reference._sample_neighborhood(source, read_grid, radius=0)[..., 0, :]
            correct = torch.where(write_valid[..., None], correct, torch.zeros_like(correct))
            permuted = correct.reshape(len(sample_ids), 3, 1024, -1).gather(
                2, perm.reshape(len(sample_ids), 3, 1024, 1).expand(-1, -1, -1, correct.shape[-1]))
            if not torch.equal(selected, permuted.reshape_as(selected)):
                raise ValueError('wrong read changes evidence beyond the exact within-view permutation')
            event['source_value_multiset_exact'] = True
            event['active_address_changed_fraction'] = 1.
        source_args = [arguments for arguments, _ in captures[1:]]
        scales = source_args[0][3]
        if any(not torch.equal(arguments[3], scales) for arguments in source_args):
            raise ValueError('source views use different mesh scales')
        projected, mass = reference.project_static_source_maps(
            extractor, selected,
            camera_angle_x=torch.stack([arguments[1] for arguments in source_args], 1),
            distance=torch.stack([arguments[2] for arguments in source_args], 1),
            mesh_scale=scales,
            projection_matrices=torch.stack([arguments[4] for arguments in source_args], 1),
            write_valid=write_valid, mask_camera_fov=True)
        native['proj'] = reference.target_first_static_projection(target, projected, mass)
        event['projected_support_mean'] = float(mass.float().mean())
        events.append(event)
        return native, negative

    trainer._encode_deform_views = types.MethodType(encode, trainer)
    try:
        yield
    finally:
        if had_instance_override:
            trainer._encode_deform_views = original
        else:
            del trainer._encode_deform_views
