"""Also preserve the CPU RNG used by the native texture sampler's noise."""
from types import MethodType

import torch

from .generated_pbr_conditioning import load_generated_transport, encode_generated_pbr
from .generated_pbr_conditioning import install_generated_pbr_conditioning as _install


def install_generated_pbr_conditioning(pipeline, **kwargs):
    original = _install(pipeline, **kwargs)
    delegated = pipeline.get_proj_cond_shape
    events = kwargs['events']

    @torch.no_grad()
    def guarded(self, image_cond_model, *args, **call_kwargs):
        if image_cond_model is not self.image_cond_model_tex_1024:
            return delegated(image_cond_model, *args, **call_kwargs)
        cpu_rng = torch.get_rng_state().clone()
        result = delegated(image_cond_model, *args, **call_kwargs)
        if not torch.equal(cpu_rng, torch.get_rng_state()):
            raise ValueError('native PBR sampler draws CPU noise; conditioning must preserve CPU RNG too')
        events[-1]['sampler_CPU_RNG_unchanged'] = True
        return result

    pipeline.get_proj_cond_shape = MethodType(guarded, pipeline)
    return original
