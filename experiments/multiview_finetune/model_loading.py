"""Load a native flow checkpoint using its saved full-finetune or LoRA config."""
import torch

from pixal3d import models
from pixal3d.modules.lora import inject_lora


def load_run_denoiser(config, checkpoint):
    spec = config['models']['denoiser']
    model = getattr(models, spec['name'])(**spec['args'])
    lora = config['trainer']['args'].get('lora')
    model.convert_to(torch.bfloat16 if lora is not None else torch.float16)
    if lora is not None:
        inject_lora(model, **lora)
    incompatible = model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=False)
    if set(incompatible.missing_keys) - {'rope_phases'} or incompatible.unexpected_keys:
        raise RuntimeError(f'Run checkpoint does not match saved denoiser config: {incompatible}')
    return model.eval()
