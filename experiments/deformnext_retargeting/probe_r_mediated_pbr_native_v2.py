"""Warm frozen NAF outside the micro1 cold-forward/RNG identity comparison.

The original proof and its failure stay immutable. Source/LoRA initialization,
optimizer and sampler are not altered. Lazy frozen-model initialization is
performed once after the native constructor, with CPU/CUDA RNG restored.
"""
from contextlib import contextmanager
import json
from pathlib import Path

import torch


def preload_frozen_naf(trainer):
    extractor = trainer.image_cond_model
    if (trainer.multiview_stage != 'pbr1024' or not extractor.use_naf_upsample
            or extractor.naf_model is not None or trainer.optimizer.state or trainer.step != 0):
        raise ValueError('fresh native PBR with uninitialized frozen NAF and optimizer0 required')
    cpu = torch.get_rng_state().clone()
    devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
    cuda = {d:torch.cuda.get_rng_state(d).clone()for d in devices}
    with torch.random.fork_rng(devices=devices):
        extractor._load_naf()
    if (not torch.equal(cpu,torch.get_rng_state())
            or any(not torch.equal(v,torch.cuda.get_rng_state(d))for d,v in cuda.items())
            or extractor.naf_model is None or extractor.naf_model.training
            or any(p.requires_grad for p in extractor.naf_model.parameters())
            or trainer.optimizer.state or trainer.step != 0):
        raise ValueError('frozen NAF preload must preserve RNG/optimizer and remain frozen eval')
    return dict(status='PASS_FROZEN_NAF_PRELOAD_BEFORE_COLD_FORWARD_WITH_RNG_RESTORED',
        CPU_RNG_exact=True, CUDA_RNG_exact=True, optimizer_updates=0,
        native_source_and_LoRA_initialization_not_reexecuted=True,
        NAF_frozen_eval=True, scientific_training_admitted=False, full_goal_complete=False)


@contextmanager
def warm_after_constructor(trainer_class):
    previous=trainer_class.__init__
    def initialize(self,*args,**kwargs):
        previous(self,*args,**kwargs)
        record=preload_frozen_naf(self)
        (Path(self.output_dir)/'frozen-NAF-preload.json').write_text(json.dumps(record,indent=2)+'\n')
    trainer_class.__init__=initialize
    try:
        yield
    finally:
        trainer_class.__init__=previous


def main():
    from pixal3d.trainers.basic import BasicTrainer
    from .probe_r_mediated_pbr_native import main as original_proof
    with warm_after_constructor(BasicTrainer):
        original_proof()


if __name__=='__main__':main()
