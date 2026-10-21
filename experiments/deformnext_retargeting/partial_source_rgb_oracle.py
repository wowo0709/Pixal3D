"""Diagnostic extra-GT RGB inputs; never a competitive model conditioner."""
import hashlib

import torch


CAMERAS = (8, 9, 10, 11)
STATES = ('M1', 'M1', 'M1', 'M0')


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def replace_source_rgb(cond, replacements):
    """Replace cameras9/10 only; preserve edited target8 and M0 source11."""
    if (cond.ndim != 4 or cond.shape[:2] != (4, 3)
            or not cond.is_floating_point() or set(replacements) != {9, 10}):
        raise ValueError('four ordered RGB views and exactly source cameras9/10 required')
    for value in [cond, *replacements.values()]:
        if (not torch.isfinite(value).all() or (value < 0).any()
                or (value > 1).any()):
            raise ValueError('finite RGB values in[0,1] required')
    if any(value.shape != cond[0].shape or value.dtype != cond.dtype
           or value.device != cond.device for value in replacements.values()):
        raise ValueError('replacement RGB shape/dtype/device must match the native input')
    result = cond.clone()
    result[1], result[2] = replacements[9], replacements[10]
    if not torch.equal(result[[0, 3]], cond[[0, 3]]):
        raise ValueError('target8 and original source11 changed')
    return result


def oracle_dataset_class(base, source_by_id, load_rgb, events, captures):
    """Process-local input adapter; every other native dataset field is shared."""
    class PartialSourceRGBOracleSS64(base):
        def __getitem__(self, index):
            item = super().__getitem__(index)
            row = self.rows[index]
            sid = row['sample_id']
            if (sid not in source_by_id or row['conditioning_camera_ids'] != list(CAMERAS)
                    or any(key.startswith('transport_') for key in item)):
                raise ValueError('allowlisted vanilla SS RGB input without stale transport required')
            replacement = {camera: load_rgb(self._resolve(path), self.image_size)
                           for camera, path in source_by_id[sid].items()}
            changed = replace_source_rgb(item['cond'], replacement)
            result = dict(item, cond=changed)
            if any(result[key] is not value for key, value in item.items() if key != 'cond'):
                raise ValueError('native cameras, scale or scoring target changed')
            event = dict(sample_id=sid, camera_ids=list(CAMERAS), input_states=list(STATES),
                original_view_SHA256=[tensor_sha(view) for view in item['cond']],
                actual_view_SHA256=[tensor_sha(view) for view in changed],
                teacher_view_SHA256={str(k): tensor_sha(v) for k, v in replacement.items()},
                native_non_RGB_fields_unchanged=True, transport_cache_used=False)
            if sid in events and events[sid] != event:
                raise ValueError('actual RGB inputs changed between repeated dataset reads')
            events[sid] = event
            if sid in captures:
                captures[sid] = changed.detach().cpu().clone()
            return result
    return PartialSourceRGBOracleSS64
