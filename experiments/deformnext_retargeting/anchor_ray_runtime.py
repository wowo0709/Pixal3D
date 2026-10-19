"""Stage-local construction for a frozen positive SS/R plus six composer tensors."""
from pathlib import Path

import torch

from pixal3d.modules.anchor_ray_composition import AnchorRayCompositionConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer


class AnchorRayCompositionTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
    def init_models_and_more(self, *, anchor_reference_checkpoint, composer_chunk_size=256, **kwargs):
        if set(self.models) != {'denoiser'} or self.deform_conditioner is not None:
            raise ValueError('fresh original SS denoiser and declared reference conditioner required')
        if self.world_size > 1 and self.batch_split != 1:
            raise ValueError('frozen-backbone DDP composer currently requires accumulation1')
        denoiser = self.models.pop('denoiser')
        denoiser.requires_grad_(False)
        self.deform_conditioner = AnchorRayCompositionConditioner(
            **self.deform_conditioner_config, composer_chunk_size=composer_chunk_size).cuda()
        reference = torch.load(Path(anchor_reference_checkpoint), map_location='cpu', weights_only=True)
        self.deform_conditioner.load_reference(reference)
        self.models['deform_conditioner'] = self.deform_conditioner
        # BasicTrainer wraps/optimizes the six trainable tensors only. The
        # frozen denoiser still differentiates its input condition during loss.
        super().init_models_and_more(**kwargs)
        self.models['denoiser'] = denoiser
        self.training_models['denoiser'] = denoiser
        if len(self.model_params) != 6 or any(p.requires_grad for p in denoiser.parameters()):
            raise ValueError('only composer5 and projected gate may train')
        self._anchor_gate_projection_handle = self.deform_conditioner.install_gate_projection(self.optimizer)
