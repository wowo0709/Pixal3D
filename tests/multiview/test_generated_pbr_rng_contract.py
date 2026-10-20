from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting import generated_pbr_conditioning_v2 as entry


@pytest.mark.parametrize('consume_cpu_noise',[False,True])
def test_texture_hook_rejects_consuming_native_cpu_sampler_noise(monkeypatch,consume_cpu_noise):
    texture=object()
    events=[]
    expected=object()
    pipeline=SimpleNamespace(image_cond_model_tex_1024=texture,get_proj_cond_shape=lambda *_:expected)
    saved_rng=torch.get_rng_state().clone()

    def install(pipe,**kwargs):
        original=pipe.get_proj_cond_shape

        def delegated(_extractor,*args,**kwargs):
            if consume_cpu_noise:
                torch.randn(32)
            events.append({})
            return expected

        pipe.get_proj_cond_shape=delegated
        return original

    monkeypatch.setattr(entry,'_install',install)
    try:
        entry.install_generated_pbr_conditioning(pipeline,events=events)
        if consume_cpu_noise:
            with pytest.raises(ValueError,match='CPU noise'):
                pipeline.get_proj_cond_shape(texture)
        else:
            assert pipeline.get_proj_cond_shape(texture) is expected
            assert torch.equal(saved_rng,torch.get_rng_state())
            assert events[-1]['sampler_CPU_RNG_unchanged']
    finally:
        torch.set_rng_state(saved_rng)
