"""Native PBR proof of physical Omega query and direct DINO/NAF learning.

One declared TRAIN asset; complete base+LoRA and matcher stay frozen. The
six original R tensors, nine attention tensors and one physical query tensor
receive only three discarded temporary updates. This is not PBR fine-tuning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import torch

from pixal3d import models
from pixal3d.datasets.deformnext_deformation_query_pbr import DeformNeXtDeformationQueryPBR1024
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
from pixal3d.utils.data_utils import recursive_to_device
from .ray_deformation_controls import DEFINITIONS, deformation_control
from .verify_anchor_ray_ss_gradient import sha, tensor_digest


class ProbeTrainer(ImageConditionedProjSparseFlowMatchingCFGTrainer):
    def init_models_and_more(self, **kwargs):
        denoiser = self.models.pop('denoiser')
        denoiser.requires_grad_(False)
        reference_config = dict(self.deform_conditioner_config, arm='c1')
        reference_config.pop('deformation_dim')
        reference = DeformationAwareConditioner(**reference_config).cuda()
        reference.requires_grad_(False)
        feature = DirectRayRetargetingConditioner(**self.deform_conditioner_config).cuda()
        feature.load_reference(reference.state_dict())
        feature.retargeter.requires_grad_(True)
        feature.fusion_gate.requires_grad_(True)
        self.reference_conditioner = reference
        self.deform_conditioner = feature
        self.models['deform_conditioner'] = feature
        super().init_models_and_more(**kwargs)
        self.models['denoiser'] = denoiser
        self.training_models['denoiser'] = denoiser
        if len(self.model_params) != 16 or feature.retargeter.input_dim != 5120:
            raise ValueError('PBR R6 + attention9 + physicalDQ1 and real5120 input required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--utility-head', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.25, 0)
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    root = Path(__file__).parents[2]
    sources = [Path(__file__).resolve(), root/'pixal3d/modules/direct_ray_retargeting.py',
        root/'pixal3d/modules/ray_deformation_query.py', root/'pixal3d/modules/ray_feature_attention.py',
        root/'pixal3d/datasets/deformnext_deformation_query_pbr.py',
        root/'pixal3d/datasets/deformnext_feature.py', Path(__file__).with_name('ray_deformation_controls.py'),
        Path(__file__).with_name('ray_feature_controls.py'), args.config, args.manifest, args.utility_head]
    bound = {str(path): sha(path) for path in sources}
    cfg = json.loads(args.config.read_text())
    ds = DeformNeXtDeformationQueryPBR1024(str(args.manifest),
        **dict(cfg['dataset']['args'], transport_arm='c1', task_matching_features=True,
               deformation_query_channels=2048))
    if len(ds) != 1 or ds.rows[0]['split'] != 'train':
        raise ValueError('one declared TRAIN asset required')
    batch = ds.collate_fn([ds[0]])
    if ('transport_spatial_indices' in batch or 'transport_deformation' in batch
            or batch['transport_source_values'].shape != (1,3,1024,2048)
            or batch['transport_source_keys'].shape != (1,3,1024,1024)
            or batch['transport_target_query'].shape != (1,1024,1024)
            or batch['transport_target_context'].shape != (1,1024,2048)
            or batch['transport_source_context'].shape != (1,3,1024,2048)):
        raise ValueError('native PBR appearance plus real inference-only RAW context required')
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name,spec in cfg['models'].items()}
    kw = dict(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1, batch_split=1, num_workers=0, prefetch_data=False,
              max_steps=1, p_uncond=0., ema_rate=[])
    kw['deform_conditioner'] = dict(kw['deform_conditioner'], arm='c2', deformation_dim=2048,
                                  candidate_aggregation='top1', projection_mode='nearest')
    trainer = ProbeTrainer(md, ds, **kw, output_dir=str(args.output), load_dir=None,
                           step=None, wandb_run=None)
    feature = trainer.deform_conditioner
    feature.ray_readout.score_selector.load_state_dict(
        torch.load(args.utility_head, map_location='cpu', weights_only=True), strict=True)
    pack = recursive_to_device(batch, torch.device('cuda:0'))
    base_pack = {key: value for key,value in pack.items()
                 if key not in ('transport_target_context', 'transport_source_context')}

    def swap(module):
        trainer.deform_conditioner = module
        trainer.models['deform_conditioner'] = module
        trainer.training_models['deform_conditioner'] = module

    def loss(data):
        torch.manual_seed(101)
        torch.cuda.manual_seed_all(101)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            values, _ = trainer.training_losses(**data)
        return values['loss']

    with torch.no_grad():
        swap(trainer.reference_conditioner)
        baseline = float(loss(base_pack))
        evidence = trainer._last_deform_condition.evidence.clone()
        swap(feature)
        initial = float(loss(pack))
        if initial != baseline or not torch.equal(trainer._last_deform_condition.evidence, evidence):
            raise ValueError('physical PBR query/direct delivery changed initial evidence or loss')
        if float(feature.fusion_gate) != 0 or torch.count_nonzero(feature.retargeter.net[-1].weight):
            raise ValueError('first-gradient proof requires a genuinely fresh zero-output R/gate')
    before = tensor_digest(trainer.models['denoiser'])
    matcher = tensor_digest(feature.q2c)
    params = [p for p in feature.parameters() if p.requires_grad]
    if len(params) != 16:
        raise ValueError('exact16 temporary trainable tensors required')
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=.01)
    results = []
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for update in range(3):
        optimizer.zero_grad(set_to_none=True)
        value = loss(pack)
        if not torch.isfinite(value):
            raise ValueError('nonfinite native PBR loss')
        value.backward()
        gradients = {name: float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
                     for name,p in feature.named_parameters() if p.requires_grad}
        if any(v is None or not torch.isfinite(torch.tensor(v)) for v in gradients.values()):
            raise ValueError('all16 native PBR gradients must be present and finite')
        if update == 2 and any(v <= 0 for v in gradients.values()):
            raise ValueError('all16 tensors must learn by temporary update3')
        weight = feature.ray_readout.output.weight.grad.detach().float()
        channels = dict(DINO_gradient_L1=float(weight[:1024].abs().sum()),
                        NAF_gradient_L1=float(weight[1024:].abs().sum()))
        if update == 0 and min(channels.values()) <= 0:
            raise ValueError('direct DINO AND NAF output must learn on first backward at gate0/R0')
        if feature.last_feature_attention_stats['deformation_query_RMS'] <= 0:
            raise ValueError('physical Omega observations must be real nonzero inferred features')
        torch.nn.utils.clip_grad_norm_(params, 1.)
        optimizer.step()
        row = dict(update=update+1, loss=float(value.detach()), gradient_L1=gradients,
                   output_weight_channel_gradients=channels, statistics=feature.last_feature_attention_stats)
        results.append(row)
        print(json.dumps(row), flush=True)
    diagnostics = {}
    with torch.no_grad():
        normal_loss = float(loss(pack))
        normal = trainer._last_deform_condition.evidence.clone()
        for name in DEFINITIONS:
            events = []
            with deformation_control(feature, name, events):
                controlled_loss = float(loss(pack))
                changed = trainer._last_deform_condition.evidence
            if not events:
                raise ValueError('native PBR control did not execute: ' + name)
            diagnostics[name] = dict(loss=controlled_loss, normal_loss=normal_loss,
                evidence_delta_RMS=float((changed-normal).square().mean().sqrt()), events=events,
                scope='Discarded3-update TRAIN probe sensitivity;not validation or standalone ablation.')
    if tensor_digest(trainer.models['denoiser']) != before or tensor_digest(feature.q2c) != matcher:
        raise ValueError('complete frozen PBR NN/base+LoRA or matching weights changed')
    if any(sha(path) != digest for path,digest in bound.items()):
        raise ValueError('Direct PBR proof source changed')
    report = dict(status='PASS_ACTUAL_DIRECT_PBR_PHYSICAL_QUERY_IDENTITY_FIRST_DINO_NAF_AND16_GRADIENTS',
        source_SHA256=bound, TRAIN_sample=ds.rows[0]['sample_id'], sparse_tokens=len(pack['x_0'].feats),
        reference='Fresh zero-output native C1/R on the complete multiview PBR base+LoRA;not a trained positive PBR R.',
        initial_loss=initial, reference_initial_loss=baseline, initial_evidence_exact=True,
        physical_deformation_channels=2048, physical_retargeter_input_channels=5120,
        first_head_gradient_reaches_both_DINO_and_NAF_with_zero_fusion_and_R=True,
        all16_tensors_receive_actual_PBR_gradients_by_update3=True,
        frozen_complete_PBR_NN_LoRA_and_matcher_exact=True, temporary_optimizer_updates=3,
        scientific_training_updates=0, PBR_LoRA_updates=0, prototype_discarded=True,
        physical_GPU=visible, allocator_fraction=.25, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began, updates=results, controls=diagnostics,
        GT_in_matching_query_or_correspondence=False, GT_Shape_is_standard_PBR_condition=True,
        scope='One TRAIN asset native micro1 learnability only;not PBR training,quality or fullbatch capacity.')
    (args.output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
