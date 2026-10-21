"""Select one actual native condition field from an extra-GT RGB encoding."""
from contextlib import contextmanager
import hashlib

import torch


def tensor_bytes_sha(value):
    raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def condition_field_oracle(mixin, resolve_original, field, events):
    if field not in {'global', 'proj'}:
        raise ValueError('one explicit native global/proj oracle field required')
    original = mixin._encode_deform_views
    direct = mixin.__dict__.get('_encode_deform_views')

    def encode(self, images, camera_info, transport, *, control='normal'):
        if (transport is not None or control != 'normal'
                or self.deform_conditioner is not None
                or images.ndim != 5 or images.shape[1:3] != (4, 3)
                or camera_info.get('coords') is not None):
            raise ValueError('native vanilla SS four-RGB input without transport required')
        baseline_images, ids = resolve_original(images)
        if (baseline_images.shape != images.shape or baseline_images.dtype != images.dtype
                or baseline_images.device != images.device or len(ids) != len(images)
                or not torch.equal(baseline_images[:, [0, 3]], images[:, [0, 3]])):
            raise ValueError('original inputs must preserve target8 and M0 source11 exactly')
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(images.device).clone() if images.is_cuda else None
        base, negative = original(self, baseline_images, camera_info, None, control='normal')
        base_sha = {k: tensor_bytes_sha(v) for k, v in base.items()}
        teacher, teacher_negative = original(self, images, camera_info, None, control='normal')
        if (set(base) != {'global', 'proj'} or set(teacher) != set(base)
                or set(negative) != set(teacher_negative)
                or base_sha != {k: tensor_bytes_sha(v) for k, v in base.items()}
                or any(not torch.equal(v, teacher_negative[k]) for k, v in negative.items())):
            raise ValueError('actual native condition schema, baseline or negative changed')
        if (not torch.equal(cpu_rng, torch.random.get_rng_state())
                or (cuda_rng is not None
                    and not torch.equal(cuda_rng, torch.cuda.get_rng_state(images.device)))):
            raise ValueError('extra frozen encoding consumed RNG and could change sampling')
        mixed = dict(base)
        mixed[field] = teacher[field]
        unselected = 'global' if field == 'proj' else 'proj'
        if mixed[unselected] is not base[unselected]:
            raise ValueError('unselected native condition must remain the exact original tensor')
        events.append(dict(sample_ids=ids, oracle_field=field,
            original_condition_SHA256=base_sha,
            actual_condition_SHA256={k: tensor_bytes_sha(v) for k, v in mixed.items()},
            teacher_condition_SHA256={k: tensor_bytes_sha(v) for k, v in teacher.items()},
            condition_shapes={k: list(v.shape) for k, v in mixed.items()},
            condition_dtypes={k: str(v.dtype) for k, v in mixed.items()},
            unselected_field_bit_exact=True, negative_bit_exact=True, RNG_unchanged=True,
            original_target8_source11_RGB_bit_exact=True, cached_feature_substitution=False))
        return mixed, negative

    mixin._encode_deform_views = encode
    try:
        yield
    finally:
        if direct is None:
            delattr(mixin, '_encode_deform_views')
        else:
            mixin._encode_deform_views = direct
