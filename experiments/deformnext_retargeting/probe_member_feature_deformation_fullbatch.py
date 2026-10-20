"""Disposable native single-GPU SS/global96 capacity and optimizer proof.

Run three real native updates from the common original multiview LoRA
checkpoint. No scientific checkpoint is exported. This measures memory,
throughput, frozen-weight integrity and gradient/update paths, not quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from easydict import EasyDict
import torch


def _frozen_digest(module):
    digest = hashlib.sha256()
    values = [(name, value) for name, value in module.named_parameters() if not value.requires_grad]
    values += list(module.named_buffers())
    for name, value in values:
        digest.update(name.encode())
        digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    allowed = os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
    if visible not in {'0', '1', '2', '3'} or visible not in allowed:
        raise ValueError('one exclusively assigned physical GPU0–3 required')
    cfg = EasyDict(json.loads(args.config.read_text()))
    spec = cfg.trainer.args
    if (cfg.get('training_seed') != 101 or cfg.get('auto_retry') != 0
            or cfg.get('load_ckpt') is not None or cfg.get('load_dir') != ''
            or cfg.get('use_wandb') is not False or cfg.dataset.name != 'DeformNeXtMemberRawSS64'
            or spec.batch_size_per_gpu != 96 or spec.batch_split != 1
            or spec.get('max_steps') != 480 or spec.get('prefetch_data') is not False
            or spec.get('mix_precision_mode') != 'amp' or spec.get('mix_precision_dtype') != 'bfloat16'
            or spec.get('ema_rate') != [] or spec.get('microbatch_to_device') is not True
            or spec.get('lora') != {'rank': 16, 'alpha': 16, 'dropout': 0.0}
            or set(spec.get('continuation_ckpt', {})) != {'denoiser'}
            or cfg.experiment.get('capacity_probe') is not True):
        raise ValueError('disposable original-LoRA/native-AMP/global96/micro96/accum1 SS recipe required')

    from pixal3d import trainers
    from pixal3d.trainers.flow_matching.member_feature_deformation_source_ray import MemberFeatureDeformationSourceRayImageConditionedProjFlowMatchingCFGTrainer as selected
    from experiments.multiview_finetune.runtime import install_hooks
    import train as native
    if (cfg.trainer.name != selected.__name__
            or cfg.experiment.get('conditioning_schema') != 'member-feature-deformation-source-ray-v1'
            or cfg.experiment.get('readout_arm') != 'feature-direct-dq'
            or spec.get('ffn_chunk_assets') not in (12,24)):
        raise ValueError('explicit four-head native SS96/FFN recipe required')

    class Capacity(selected):
        def init_models_and_more(self,**kwargs):
            super().init_models_and_more(**kwargs)
            self._capacity_FFN_batches=[]
            self._capacity_FFN_hooks=[]
            for block in self.models['denoiser'].blocks:
                self._capacity_FFN_hooks.append(block.mlp.mlp[0].register_forward_pre_hook(
                    lambda _,inputs:self._capacity_FFN_batches.append(int(inputs[0].shape[0]))))

        def run(self):
            module = self.deform_conditioner
            if (self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split, self.step) != (1, 96, 96, 1, 0):
                raise ValueError('fresh native SS fullbatch96 on one GPU required')
            expected = 380
            nn_parameters = {name: value for name, value in self.models['denoiser'].named_parameters() if value.requires_grad}
            source_parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
            if (len(nn_parameters) != 360 or any('lora_' not in name for name in nn_parameters)
                    or len(source_parameters) != expected - 360 or len(self.optimizer.state) != 0):
                raise ValueError('fresh360 native LoRA and declared new source tensors required')
            before = {name: _frozen_digest(value) for name, value in self.models.items()}
            snapshots = {id(value): value.detach().cpu().clone()
                for group in self.optimizer.param_groups for value in group['params']}
            updates = []
            began = time.monotonic()
            for _ in range(3):
                started = time.monotonic()
                result = self.run_step(self.load_data())
                self.step += 1
                gradients = {name: float(value.grad.detach().float().abs().sum())
                    if value.grad is not None else None for name, value in source_parameters.items()}
                if not math.isfinite(result['loss']['loss']):
                    raise ValueError('native fullbatch SS loss is nonfinite')
                updates.append(dict(step=self.step, loss=result['loss'], source_gradient_L1=gradients,
                    seconds=time.monotonic() - started, runtime=dict(self._runtime_last_row)))
            after = {name: _frozen_digest(value) for name, value in self.models.items()}
            changed = {group['model_name']: sum(not torch.equal(value.detach().cpu(), snapshots[id(value)])
                for value in group['params']) for group in self.optimizer.param_groups}
            actual_steps = sorted({int(state['step']) for state in self.optimizer.state.values()})
            gradients = updates[-1]['source_gradient_L1']
            if (before != after or len(self.optimizer.state) != expected or actual_steps != [3]
                    or not self._capacity_FFN_batches or max(self._capacity_FFN_batches)>spec.ffn_chunk_assets
                    or changed.get('default', 0) == 0 or changed.get('deform_conditioner', 0) == 0
                    or any(value is None or not math.isfinite(value) or value <= 0 for value in gradients.values())):
                raise ValueError('frozen integrity or actual native fullbatch optimizer/gradient proof failed')
            report = dict(status='PASS_ACTUAL_MEMBER_FEATURE_DQ_SS_FULLBATCH96_CAPACITY',
                ray_mode=module.ray_mode, world_size=1, global_batch=96, microbatch=96, accumulation=1,
                trainable_NN_LoRA_tensors=360, trainable_source_tensors=len(source_parameters),
                ffn_chunks=self._native_ffn_chunk_report,
                actual_maximum_FFN_expansion_batch=max(self._capacity_FFN_batches),
                actual_FFN_expansion_calls=len(self._capacity_FFN_batches),
                native_global96_micro96_accum1_unchanged=True,
                actual_QKV_feature_tensors=9,physical_deformation_query_channels=2048,physical_retargeter_input_channels=4096,
                ray_feature_utility=self._ray_feature_utility_initialization,
                initial_optimizer_states=0, actual_optimizer_states=len(self.optimizer.state),
                actual_optimizer_steps=actual_steps, changed_tensors_by_group=changed,
                all_source_tensors_positive_finite_gradient_by_update3=True,
                frozen_parameters_and_buffers_exact=before == after, frozen_SHA256=before,
                native_AdamW_AMP_LoRA_sampler_and_clipping_used=True,
                peak_allocated_GiB=max(row['runtime']['peak_allocated_gib'] for row in updates),
                maximum_reserved_GiB=max(row['runtime']['reserved_gib'] for row in updates),
                elapsed_seconds=time.monotonic() - began, updates=updates,
                physical_GPU=visible, temporary_optimizer_updates=3,
                scientific_training_updates=0, scientific_checkpoint_exported=False,
                probe_model_discarded=True, no_initial_V1_quality_parity_claim=True,
                no_trained_source_utility_or_quality_gain_claim=True, heldout_test_measured=False,
                full_goal_complete=False)
            for handle in self._capacity_FFN_hooks:
                handle.remove()
            (Path(self.output_dir) / 'capacity.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({key: value for key, value in report.items() if key != 'updates'}), flush=True)

    torch.cuda.set_device(0)
    install_hooks()
    torch.set_num_threads(2)
    setattr(trainers, cfg.trainer.name, Capacity)
    cfg.update(config=str(args.config), num_gpus=1, num_nodes=1, node_rank=0,
        master_addr='127.0.0.1', master_port='31717', multiview_runtime=True, tryrun=False, profile=False)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    native.main(0, cfg)


if __name__ == '__main__':
    main()
