"""Fresh native SS training for the explicit four-head NoDQ source-ray arm."""
import argparse
import copy
import json
import os
from pathlib import Path

from .member_feature_control_recipe import NAME, validate_control_recipe, control_spec



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());arm=validate_control_recipe(cfg);control=control_spec(cfg)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one exclusively assigned physical GPU0–3 required')
    from easydict import EasyDict
    import torch
    from pixal3d import trainers
    from pixal3d.trainers.flow_matching.member_feature_controls import MemberFeatureControlImageConditionedProjFlowMatchingCFGTrainer as Trainer
    from experiments.multiview_finetune.runtime import install_hooks
    import train as native

    class Scientific(Trainer):
        def run(self):
            names=[name for name,value in self.models['denoiser'].named_parameters() if value.requires_grad]
            source=sum(value.requires_grad for value in self.deform_conditioner.parameters())
            if ((self.world_size,self.batch_size,self.batch_size_per_gpu,self.batch_split,self.step)!=(1,96,96,1,0)
                    or len(names)!=360 or any('lora_' not in name for name in names)
                    or source!=control['source_tensors']
                    or len(self.optimizer.state)!=0 or self._native_ffn_chunk_report is None
                    or self._native_ffn_chunk_report['chunk_assets']!=cfg['trainer']['args']['ffn_chunk_assets']):
                raise ValueError('fresh original LoRA/native source optimizer and SS96/FFN chunks required')
            startup=dict(status='PASS_ACTUAL_FRESH_MEMBER_FEATURE_CONTROL_SOURCE_RAY_SS_STARTUP',arm=arm,
                global_batch=96,microbatch=96,accumulation=1,world_size=1,seed=cfg['training_seed'],
                trainable_NN_LoRA_tensors=len(names),trainable_source_tensors=source,optimizer_states=0,
                initial_function=cfg['experiment']['initial_function'],ffn_chunks=self._native_ffn_chunk_report,
                ray_feature_utility=self._ray_feature_utility_initialization,actual_feature_head_tensors=control['head_tensors'],QK_score_parameters_present=control['head_tensors']==9,
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
