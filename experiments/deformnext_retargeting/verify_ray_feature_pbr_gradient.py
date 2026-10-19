"""Disposable actual PBR loss/gradient check with native2048 DINO+NAF values.

No positive trained PBR retargeter is assumed. This constructs fresh C1/R,
checks exact zero-output identity, and allows three temporary joint R/readout
updates. Frozen base+LoRA and D1 never update. This is not PBR fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import models
from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner
from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
from pixal3d.utils.data_utils import recursive_to_device
from .ray_feature_controls import DEFINITIONS,feature_control
from .verify_anchor_ray_ss_gradient import sha,tensor_digest


class ProbeTrainer(ImageConditionedProjSparseFlowMatchingCFGTrainer):
    def init_models_and_more(self,**kwargs):
        denoiser = self.models.pop('denoiser')
        denoiser.requires_grad_(False)
        reference = DeformationAwareConditioner(**self.deform_conditioner_config).cuda()
        reference.requires_grad_(False)
        feature = RayFeatureAttentionConditioner(**self.deform_conditioner_config).cuda()
        feature.load_reference(reference.state_dict())
        feature.retargeter.requires_grad_(True)
        feature.fusion_gate.requires_grad_(True)
        self.reference_conditioner = reference
        self.deform_conditioner = feature
        self.models['deform_conditioner'] = feature
        super().init_models_and_more(**kwargs)
        self.models['denoiser'] = denoiser
        self.training_models['denoiser'] = denoiser
        if len(self.model_params) != 15:
            raise ValueError('exactly6 original PBR R/fusion plus9 readout tensors required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--utility-head',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.32,0)
    torch.manual_seed(101);torch.cuda.manual_seed_all(101)
    sources = [Path(__file__).resolve(),Path(__file__).parents[2]/'pixal3d/modules/ray_feature_attention.py',
        Path(__file__).parents[2]/'pixal3d/datasets/deformnext_feature.py',
        Path(__file__).with_name('ray_feature_controls.py'),args.config,args.manifest,args.utility_head]
    bound = {str(path):sha(path) for path in sources}
    cfg = json.loads(args.config.read_text())
    dsargs = dict(cfg['dataset']['args'])
    dsargs.update(transport_arm='c1',task_matching_features=True)
    ds = DeformNeXtFeaturePBR1024(str(args.manifest),**dsargs)
    if len(ds) != 1 or ds.rows[0]['split'] != 'train':
        raise ValueError('one declared TRAIN asset required for disposable PBR probe')
    batch = ds.collate_fn([ds[0]])
    if ('transport_spatial_indices' in batch or 'transport_deformation' in batch
            or batch['transport_source_values'].shape != (1,3,1024,2048)
            or batch['transport_source_keys'].shape != (1,3,1024,1024)
            or batch['transport_target_query'].shape != (1,1024,1024)):
        raise ValueError('explicit native PBR DINO+NAF and inference-only Omega contract differs')
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.device('meta'):
        md={name:getattr(models,spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw = dict(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1,batch_split=1,num_workers=0,prefetch_data=False,
        max_steps=1,p_uncond=0.,ema_rate=[])
    kw['deform_conditioner']=dict(kw['deform_conditioner'],arm='c1',candidate_aggregation='top1',projection_mode='nearest')
    trainer=ProbeTrainer(md,ds,**kw,output_dir=str(args.output),load_dir=None,step=None,wandb_run=None)
    feature=trainer.deform_conditioner
    feature.ray_readout.score_selector.load_state_dict(torch.load(args.utility_head,map_location='cpu',weights_only=True),strict=True)
    reference=trainer.reference_conditioner
    pack=recursive_to_device(batch,torch.device('cuda:0'))
    def swap(module):
        trainer.deform_conditioner=module
        trainer.models['deform_conditioner']=module
        trainer.training_models['deform_conditioner']=module
    def loss():
        torch.manual_seed(101);torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            values,_=trainer.training_losses(**pack)
        return values['loss']
    with torch.no_grad():
        swap(reference);baseline=float(loss())
        evidence=trainer._last_deform_condition.evidence.clone()
        swap(feature);initial=float(loss())
        if initial!=baseline or not torch.equal(trainer._last_deform_condition.evidence,evidence):
            raise ValueError('zero-output native PBR loss/evidence differs from fresh C1/R')
    before=tensor_digest(trainer.models['denoiser'])
    matcher=tensor_digest(feature.q2c)
    params=[p for p in feature.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01)
    results=[];began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for update in range(3):
        optimizer.zero_grad(set_to_none=True);value=loss()
        if not torch.isfinite(value):raise ValueError('nonfinite actual PBR loss')
        value.backward()
        gradients={name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
            for name,p in feature.ray_readout.named_parameters()}
        if any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise ValueError('nonfinite or absent actual PBR feature gradient')
        if update==2 and any(v<=0 for v in gradients.values()):
            raise ValueError('all9 feature tensors must learn after fresh R/output warmup')
        torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        row=dict(update=update+1,loss=float(value.detach()),feature_gradient_L1=gradients,
            feature_stats=feature.last_feature_attention_stats)
        results.append(row);print(json.dumps(row),flush=True)
    diagnostics={}
    with torch.no_grad():
        normal_loss=float(loss());normal=trainer._last_deform_condition.evidence.clone()
        for name in DEFINITIONS:
            events=[]
            with feature_control(feature,name,events):
                controlled_loss=float(loss());changed=trainer._last_deform_condition.evidence
            if not events:raise ValueError('control did not execute:'+name)
            diagnostics[name]=dict(loss=controlled_loss,normal_loss=normal_loss,
                evidence_delta_RMS=float((changed-normal).square().mean().sqrt()),events=events,
                interpretation='Sensitivity of a discarded three-update TRAIN probe;not performance evidence.')
    if tensor_digest(trainer.models['denoiser'])!=before or tensor_digest(feature.q2c)!=matcher:
        raise ValueError('frozen complete PBR NN or matching weights changed')
    if any(sha(path)!=digest for path,digest in bound.items()):raise ValueError('PBR probe source changed')
    report=dict(status='PASS_ACTUAL_PBR2048_ZERO_OUTPUT_IDENTITY_AND_FEATURE_RETARGETER_GRADIENT',
        source_SHA256=bound,TRAIN_sample=ds.rows[0]['sample_id'],sparse_tokens=len(pack['x_0'].feats),
        reference='Fresh untrained native PBR C1/R;no trained positive PBR retargeter assumed',
        initial_loss=initial,reference_initial_loss=baseline,initial_evidence_exact=True,
        temporary_R_and_feature_optimizer_updates=3,scientific_training_updates=0,PBR_LoRA_updates=0,
        complete_frozen_PBR_NN_and_matching_exact=True,all_nine_feature_tensors_receive_PBR_gradients=True,
        allocator_fraction=.32,physical_GPU=visible,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began,updates=results,controls=diagnostics,
        GT_in_matching_query_or_correspondence=False,GT_Shape_is_standard_PBR_condition=True,
        prototype_discarded=True,scope='One TRAIN asset micro1 actual PBR correctness;not micro12 capacity or PBR performance.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
