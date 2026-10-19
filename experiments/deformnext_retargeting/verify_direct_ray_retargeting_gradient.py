"""Actual SS identity and gradient proof for physically added Omega query channels."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import models
from pixal3d.datasets.deformnext_deformation_query import DeformNeXtDeformationQuerySS64
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.utils.data_utils import recursive_to_device
from .anchor_ray_runtime import AnchorRayCompositionTrainer
from .verify_anchor_ray_ss_gradient import sha,tensor_digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--cache-index',type=Path,required=True)
    p.add_argument('--utility-head',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned physical GPU0-3 required')
    torch.set_num_threads(2);torch.cuda.set_device(0);torch.cuda.set_per_process_memory_fraction(.2,0)
    torch.use_deterministic_algorithms(True)
    source=[Path(__file__).resolve(),Path(__file__).parents[2]/'pixal3d/modules/ray_deformation_query.py',
        Path(__file__).parents[2]/'pixal3d/modules/direct_ray_retargeting.py',
        Path(__file__).parents[2]/'pixal3d/datasets/deformnext_deformation_query.py',args.config,args.utility_head]
    bound={str(path):sha(path) for path in source}
    cfg=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
    ds=DeformNeXtDeformationQuerySS64(cfg['data_dir'],deformation_query_channels=2048,**cfg['dataset']['args'])
    selected=[json.loads(line)['sample_id'] for line in args.cache_index.read_text().splitlines()
        if json.loads(line)['role']=='train' and json.loads(line)['usable_queries']>0][:2]
    index={row['sample_id']:i for i,row in enumerate(ds.rows)}
    pack=recursive_to_device(ds.collate_fn([ds[index[sid]] for sid in selected]),torch.device('cuda:0'))
    with torch.device('meta'):
        md={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kwargs=dict(cfg['trainer']['args']);kwargs.update(batch_size_per_gpu=2,batch_split=1,num_workers=0,
        prefetch_data=False,max_steps=1,p_uncond=0.)
    trainer=AnchorRayCompositionTrainer(md,ds,**kwargs,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    old=trainer.deform_conditioner;reference={name:value.detach().cpu().clone() for name,value in old.reference_state().items()}
    base_pack={key:value for key,value in pack.items() if key not in ['transport_target_context','transport_source_context']}
    def loss(batch):
        torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda',dtype=torch.bfloat16):values,_=trainer.training_losses(**batch)
        return values['loss']
    with torch.no_grad():
        baseline=float(loss(base_pack));evidence=trainer._last_deform_condition.evidence.clone()
    cfgargs=dict(cfg['trainer']['args']['deform_conditioner'],arm='c2',deformation_dim=2048)
    feature=DirectRayRetargetingConditioner(**cfgargs).cuda();feature.load_reference(reference)
    feature.ray_readout.score_selector.load_state_dict(torch.load(args.utility_head,map_location='cpu',weights_only=True),strict=True)
    trainer.deform_conditioner=feature;trainer.models['deform_conditioner']=feature
    trainer.training_models['deform_conditioner']=feature
    params=[parameter for parameter in feature.parameters() if parameter.requires_grad]
    if len(params)!=10 or old.retargeter.input_dim!=2048 or feature.retargeter.input_dim!=4096:
        raise ValueError('nine feature tensors plus physically present fresh deformation projection required')
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    before=tensor_digest(trainer.models['denoiser']);results=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for update in range(3):
        optimizer.zero_grad(set_to_none=True);value=loss(pack)
        if update==0:
            if float(value.detach())!=baseline or not torch.equal(trainer._last_deform_condition.evidence,evidence):
                raise ValueError('physical deformation channels changed initial actual SS loss/evidence')
        value.backward()
        gradients={name:float(parameter.grad.detach().float().abs().sum()) if parameter.grad is not None else None
            for name,parameter in feature.named_parameters() if parameter.requires_grad}
        if (any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values())
                or gradients['retargeter.deformation_projection.weight']<=0):
            raise ValueError('actual deformation-query gradient must learn on the first backward')
        if update==2 and any(v<=0 for v in gradients.values()):raise ValueError('all10 tensors must learn')
        if feature.last_feature_attention_stats['deformation_query_RMS']<=0:
            raise ValueError('real inferred deformation observations must not be replaced by zeros')
        torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        record=dict(update=update+1,loss=float(value.detach()),gradient_L1=gradients,
            statistics=feature.last_feature_attention_stats)
        results.append(record);print(json.dumps(record),flush=True)
    if tensor_digest(trainer.models['denoiser'])!=before or any(not torch.equal(value,feature.reference_state()[name].detach().cpu()) for name,value in reference.items()):
        raise ValueError('frozen NN or original R/matcher/fusion changed')
    if any(sha(path)!=digest for path,digest in bound.items()):raise ValueError('deformation-query proof source changed')
    result=dict(status='PASS_ACTUAL_SS_DIRECT_RAY_RETARGETING_IDENTITY_AND_TEN_TENSOR_GRADIENT',
        source_SHA256=bound,TRAIN_samples=selected,direct_feature_delivery='native retarget_residual outside semantic fusion gate',direct_feature_bound=.25,initial_loss=baseline,new_initial_loss=results[0]['loss'],
        initial_evidence_exact=True,no_DQ_reference_input_channels=2048,new_physical_input_channels=4096,
        deformation_channels=2048,deformation_observations_are_nonzero=True,
        common_R_parameter_shapes_unchanged=True,new_DQ_parameter_has_separate_fresh_Adam_state=True,
        all10_new_tensors_receive_actual_SS_gradients=True,complete_NN_R_matcher_fusion_exact=True,
        temporary_updates=3,scientific_training_updates=0,SS_LoRA_updates=0,R_updates=0,
        GT_correspondence_or_deformation_input=False,physical_GPU=visible,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,
        updates=results,prototype_discarded=True,
        scope='Two TRAIN assets,micro2 identity/learnability;not scientific training or quality improvement.')
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
