"""Instrument the immutable native PBR probe to distinguish DINO/NAF gradients."""
import hashlib
import json
from pathlib import Path

import torch


def main():
    from . import verify_ray_feature_pbr_gradient as entry
    original=entry.ProbeTrainer
    heads=[];retargeted=[];trainers=[];handles=[]
    sources={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             str(Path(entry.__file__).resolve()):hashlib.sha256(Path(entry.__file__).read_bytes()).hexdigest()}
    class ChannelProbeTrainer(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            trainers.append(self)
            def retain_output(module,args,output):
                if torch.is_grad_enabled() and output.requires_grad:
                    output.retain_grad();heads.append(output)
            handles.append(self.deform_conditioner.ray_readout.output.register_forward_hook(retain_output))
        def training_losses(self,*args,**kwargs):
            values=super().training_losses(*args,**kwargs)
            result=getattr(self,'_last_deform_condition',None)
            residual=None if result is None else result.retarget_residual
            if torch.is_grad_enabled() and residual is not None and residual.requires_grad:
                residual.retain_grad();retargeted.append(residual)
            return values
    entry.ProbeTrainer=ChannelProbeTrainer
    try:
        entry.main()
    finally:
        entry.ProbeTrainer=original
        for handle in handles:handle.remove()
    if len(trainers)!=1:raise ValueError('one actual native PBR trainer required')
    def statistics(tensors):
        rows=[]
        for value in tensors:
            if value.grad is not None:
                if value.shape[-1]!=2048:raise ValueError('DINO1024+NAF1024 tensor required')
                grad=value.grad.detach().float()
                rows.append(dict(DINO_gradient_L1=float(grad[...,:1024].abs().sum()),
                    NAF_gradient_L1=float(grad[...,1024:].abs().sum())))
        return rows
    head=statistics(heads);residual=statistics(retargeted)
    weight=trainers[0].deform_conditioner.ray_readout.output.weight.grad.detach().float()
    channels=dict(DINO_output_weight_gradient_L1=float(weight[:1024].abs().sum()),
                  NAF_output_weight_gradient_L1=float(weight[1024:].abs().sum()))
    if (len(head)<3 or len(residual)!=3 or head[-1]['NAF_gradient_L1']<=0
            or residual[-1]['NAF_gradient_L1']<=0 or min(channels.values())<=0):
        raise ValueError('native PBR high-resolution retargeting/feature gradient failed')
    for source,digest in sources.items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest()!=digest:
            raise ValueError('channel instrumentation source changed')
    report=dict(status='PASS_ACTUAL_PBR_DINO_AND_NAF_FEATURE_RETARGETER_GRADIENTS',
        source_SHA256=sources,feature_output_gradients=head,retarget_output_gradients=residual,
        output_projection_weight_gradients=channels,
        native_LR_semantic_delta_preserves_NAF=True,
        NAF_feature_path_reaches_loss_through_retargeter=True,
        numerical_forward_source_changed=False,process_local_instrumentation=True,
        temporary_updates=3,scientific_training_updates=0,PBR_LoRA_updates=0,
        scope='Same oneTRAINasset freshR correctness probe with retainedchannelgradients;not PBR validation quality.')
    output=Path(trainers[0].output_dir)
    (output/'channel-gradient-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
