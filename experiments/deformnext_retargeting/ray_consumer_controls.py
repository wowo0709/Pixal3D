"""Inference-only RGB-alpha controls after original source-ray projection."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from pixal3d.trainers.flow_matching.ray_group_matching import (
    RayGroupImageConditionedProjFlowMatchingCFGTrainer, project_ray_delta,
)


def restrict_consumer(native, updated, observed, mode, projected_R=None):
    if (native.shape!=updated.shape or observed.shape!=native.shape[:2]
            or observed.dtype!=torch.bool):
        raise ValueError('aligned native/updated SS features and RGB-alpha support required')
    if mode=='measure-only':return updated
    if mode=='mask-all':return torch.where(observed[...,None],updated,native)
    if mode=='mask-R':
        if projected_R is None or projected_R.shape!=updated.shape:
            raise ValueError('same-camera projected R component required')
        return torch.where(observed[...,None],updated,updated-projected_R.to(updated.dtype))
    raise ValueError('explicit consumer control required')


@contextmanager
def ray_consumer_control(mode,events):
    if mode not in {'measure-only','mask-all','mask-R'}:
        raise ValueError('explicit consumer control required')
    cls=RayGroupImageConditionedProjFlowMatchingCFGTrainer
    original=cls._inject_deform_transport

    def inject(self,condition,transport,coords,**kwargs):
        if torch.is_grad_enabled() or self.deform_conditioner.training or coords is not None:
            raise ValueError('consumer controls require frozen SS evaluation')
        if {'positive_indices','positive_visible','depth','ss_latent','target_vertices'}&transport.keys():
            raise ValueError('GT/teacher data cannot enter RGB-alpha consumer controls')
        key='proj_semantic' if 'proj_semantic' in condition else 'proj'
        native=condition[key]
        result=original(self,condition,transport,coords,**kwargs)
        updated=result[key];trace=self._last_deform_condition
        patch=transport['patch_grid'].flatten(1).long()
        observed=transport['target_valid'].gather(1,patch).bool()
        projected_R=None
        if mode=='mask-R':
            residual=trace.retarget_residual
            if residual.shape!=(len(native),3072,1024):
                raise ValueError('original three-camera ray R shape required')
            projected_R=project_ray_delta(self._image_cond_module.proj_grid,
                residual.reshape(len(native),3,1024,1024),kwargs['camera_info'])
        controlled=restrict_consumer(native,updated,observed,mode,projected_R)
        if not torch.equal(controlled[observed],updated[observed]):
            raise ValueError('consumer control changed inside-foreground condition')
        if mode=='mask-all' and not torch.equal(controlled[~observed],native[~observed]):
            raise ValueError('all-correction mask did not retain native outside support')
        if mode=='measure-only' and controlled is not updated:
            raise ValueError('measurement changed original condition')
        energy=(updated.float()-native.float()).square().sum(-1)
        after=(controlled.float()-native.float()).square().sum(-1)
        events.append(dict(mode=mode,batch_assets=len(native),same_candidate_addresses=True,
            same_query_support=True,matching_trace_unmodified=self._last_deform_condition is trace,
            inside_condition_byte_exact=True,GT_read_by_control=False,optimizer_updates=0,
            total_correction_energy=float(energy.sum()),outside_input_alpha_energy=float(energy[~observed].sum()),
            after_total_correction_energy=float(after.sum()),after_outside_input_alpha_energy=float(after[~observed].sum())))
        result[key]=controlled
        return result

    cls._inject_deform_transport=inject
    try:yield
    finally:cls._inject_deform_transport=original
