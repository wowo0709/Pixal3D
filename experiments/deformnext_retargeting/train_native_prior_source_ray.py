"""Fresh native SS source-ray training after actual V1 identity/capacity proof."""
import argparse
import copy
import json
import os
from pathlib import Path

from .train_target_first_observed import TRAINERS, validate_recipe


NAME='NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer'


def validate_native_prior_recipe(cfg):
    if (cfg['trainer']['name']!=NAME
            or cfg['trainer']['args'].get('ffn_chunk_assets')!=24
            or cfg['experiment'].get('conditioning_schema')!='native-prior-source-ray-v1'
            or cfg['experiment'].get('no_query_fallback')!='Native pretrained MV4 prior, zero source correction'
            or cfg['experiment'].get('initial_function')!='Exact native MV4 global/projection/negative; zero source correction'):
        raise ValueError('explicit native-MV4 initialization, source-ray correction and FFN24 recipe required')
    # Retain the existing fixed scientific optimizer/LoRA/data/noise/batch
    # protocol. Only validate a copied constructor name; never mutate recipe.
    adapted=copy.deepcopy(cfg)
    arm=adapted['experiment']['readout_arm']
    if arm not in TRAINERS:
        raise ValueError('declared same-carrier readout required')
    adapted['trainer']['name']=TRAINERS[arm]
    return validate_recipe(adapted)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());arm=validate_native_prior_recipe(cfg)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one exclusively assigned physical GPU0–3 required')
    from easydict import EasyDict
    import torch
    from pixal3d import trainers
    from pixal3d.trainers.flow_matching.native_prior_source_ray import NativePriorSourceRayImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from experiments.multiview_finetune.runtime import install_hooks
    import train as native

    class Scientific(Trainer):
        def run(self):
            names=[name for name,value in self.models['denoiser'].named_parameters() if value.requires_grad]
            source=sum(value.requires_grad for value in self.deform_conditioner.parameters())
            if ((self.world_size,self.batch_size,self.batch_size_per_gpu,self.batch_split,self.step)!=(1,96,96,1,0)
                    or len(names)!=360 or any('lora_' not in name for name in names)
                    or source!=(15 if arm in {'attention','spatial'} else 10)
                    or len(self.optimizer.state)!=0 or self._native_ffn_chunk_report is None
                    or self._native_ffn_chunk_report['chunk_assets']!=24):
                raise ValueError('fresh original LoRA/native source optimizer and SS96/FFN24 required')
            startup=dict(status='PASS_ACTUAL_FRESH_NATIVE_PRIOR_SOURCE_RAY_SS_STARTUP',arm=arm,
                global_batch=96,microbatch=96,accumulation=1,world_size=1,seed=cfg['training_seed'],
                trainable_NN_LoRA_tensors=len(names),trainable_source_tensors=source,optimizer_states=0,
                initial_function=cfg['experiment']['initial_function'],ffn_chunks=self._native_ffn_chunk_report,
                physical_deformation_query_channels=0,
                native_Adam_AMP_sampler_LoRA_and_epoch_validation_preserved=True,
                no_inference_GT_or_per_voxel_visibility_write_gate=True,
                trained_source_utility_or_final_RGB_preservation_not_proven=True)
            (Path(self.output_dir)/'startup.json').write_text(json.dumps(startup,indent=2)+'\n')
            return super().run()

    torch.cuda.set_device(0);install_hooks();torch.set_num_threads(2)
    setattr(trainers,NAME,Scientific)
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    native.main(0,EasyDict(cfg))


if __name__=='__main__':
    main()
