"""One bounded query-design change with unchanged full-ray feature delivery.

Only the source-camera ray's *utility* mean excludes voxels outside the input
target foreground. The original unmasked R target mean, member matching and
native ray projection/write support are preserved. This is not visibility
gating and it does not add trainable tensors. It can run during training:
checkpoint recomputation receives the already selected utility tensor.
"""
from contextlib import contextmanager

import torch


@contextmanager
def observed_member_utility():
    from pixal3d.modules import member_feature_source_ray as native
    selected = native.MemberFeatureSourceRayConditioner
    original_forward, original_pool = selected.forward, native.pool_ray_queries
    statistics = dict(forwards=0, utility_pool_calls=0, target_pool_calls=0,
                      utility_input_voxel_observations=0, observed_utility_voxel_observations=0,
                      target_context_remains_full_ray=True, full_ray_projection_and_write_unchanged=True,
                      trainable_tensors_added=0)

    def forward(self, **kwargs):
        if type(self) is not selected or native.pool_ray_queries is not original_pool:
            raise ValueError('one original NoDQ conditioner with unnested utility change required')
        calls = 0
        observed, ray_ids = kwargs['observed_query'], kwargs['ray_ids']
        if observed.dtype != torch.bool or observed.shape != kwargs['native_voxels'].shape[:2]:
            raise ValueError('original boolean per-voxel input-target support required')

        def pool(values, ids, *, num_rays):
            nonlocal calls
            if ids is not ray_ids or num_rays != 1024 or calls >= 2:
                raise ValueError('original two source-camera ray pool calls required')
            calls += 1
            if calls == 1:
                if values is not kwargs['native_voxels']:
                    raise ValueError('first original pool must be the utility native tensor')
                changed = torch.where(observed[:, None], ids, -1)
                result = original_pool(values, changed, num_rays=num_rays)
                statistics['utility_pool_calls'] += 1
                statistics['utility_input_voxel_observations'] += int((ids >= 0).sum())
                statistics['observed_utility_voxel_observations'] += int((changed >= 0).sum())
                return result
            if values is not kwargs['target_native_voxels']:
                raise ValueError('second original pool must retain the unmasked R target tensor')
            statistics['target_pool_calls'] += 1
            return original_pool(values, ids, num_rays=num_rays)

        try:
            native.pool_ray_queries = pool
            result = original_forward(self, **kwargs)
            if calls != 2:
                raise ValueError('both original utility and R target pools must complete')
            statistics['forwards'] += 1
            return result
        finally:
            native.pool_ray_queries = original_pool

    try:
        selected.forward = forward
        yield statistics
    finally:
        selected.forward = original_forward
        native.pool_ray_queries = original_pool
