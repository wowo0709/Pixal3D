"""Matched native LoRA/R continuation with transformed target-ray Top-K features.

Keep the already registered scalar-composer experiment immutable. Reuse its
audited epoch1 Adam/RNG helpers and native lifecycle, with an explicit third
optimizer group for the nine feature-attention tensors.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch

from pixal3d.modules.ray_feature_attention import RayFeatureAttentionConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .anchor_joint_runtime import AnchorJointTrainer, restore_rng, sha, transplant_adam_prefix


class RayFeatureJointTrainer(AnchorJointTrainer):
    def __init__(self, *args, anchor_stage, **kwargs):
        self.anchor_stage = dict(anchor_stage)
        spec = self.anchor_stage
        if (spec['arm'] != 'ray-feature-attention' or spec['reference_step'] != 96
                or kwargs['max_steps'] != 480 or kwargs.get('prefetch_data') is not False
                or kwargs.get('mix_precision_mode') != 'amp'
                or kwargs.get('mix_precision_dtype') != 'bfloat16' or kwargs.get('ema_rate') != []):
            raise ValueError('explicit matched feature-attention epoch1-to480 continuation required')
        self._verify_reference()
        # Call the native constructor; AnchorJointTrainer.__init__ specifically
        # implements the separate six-tensor scalar-composer experiment.
        ImageConditionedProjFlowMatchingCFGTrainer.__init__(self, *args, **kwargs)
        actual = self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split
        if not spec.get('diagnostic_profile', False) or spec.get('capacity_profile', False):
            if actual != (2, 96, 48, 1):
                raise ValueError('dedicated DDP2/global96/micro48/accumulation1 required')
        elif self.world_size != 1 or self.batch_size_per_gpu > 2 or self.batch_split != 1:
            raise ValueError('disposable cohost profile requires singleGPU/micro<=2')
        if self.step == 0:
            misc = torch.load(spec['misc_checkpoint'], map_location='cpu', weights_only=False)
            if misc['step'] != 96 or misc['lora'] != self.lora_config:
                raise ValueError('shared epoch1 step or LoRA settings differ')
            restored = transplant_adam_prefix(self.optimizer, misc['optimizer'], expected_step=96)
            if restored != 366:
                raise ValueError('all360 LoRA plus6 original R/fusion Adam states must continue')
            self.data_sampler.load_state_dict(misc['data_sampler'])
            self.grad_clip.load_state_dict(misc['grad_clip'])
            self.step = 96
            self._install_stage_schedule()
            seed = int(spec['stage_rng_seed']) + self.rank
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self._write_stage_initialization(restored, misc['data_sampler'])
            self._save_rank_rng()
        else:
            restore_rng(self._rng_path(self.step), step=self.step,
                sampler=self.data_sampler, loader_generator=self._anchor_loader_generator)

    def _verify_reference(self):
        super()._verify_reference()
        spec = self.anchor_stage
        done = json.loads(Path(spec['utility_completed']).read_text())
        if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
                or done['epochs'] != 7 or done['optimizer_updates'] != 112
                or done['pure_utility_export_SHA256'] != spec['utility_SHA256']
                or sha(spec['utility_checkpoint']) != spec['utility_SHA256']):
            raise ValueError('fixed final target-patch utility initializer required')
        proof_path = Path(spec['feature_identity_proof'])
        proof = json.loads(proof_path.read_text())
        if (proof['status'] != 'PASS_ACTUAL_SS_ZERO_FEATURE_OUTPUT_IDENTITY_AND_NINE_TENSOR_GRADIENT'
                or not proof['actual_initial_evidence_exact']
                or not proof['nine_feature_tensors_receive_gradient_by_update3']
                or proof['reference_initial_loss'] != proof['new_initial_loss']):
            raise ValueError('actual SS initial identity and feature learnability proof required')
        for path, digest in proof['source_SHA256'].items():
            if sha(path) != digest:
                raise ValueError('feature identity proof source changed: ' + path)
        if not spec.get('diagnostic_profile', False):
            closure = json.loads(Path(spec['feature_identity_receipt']).read_text())
            relative = str(proof_path.relative_to(Path(spec['campaign_root']).parent))
            if (closure['status'] != 'PASS' or not closure['local_equals_actual_file4_SHA256']
                    or not any(row['relative_path'] == relative and row['sha256'] == sha(proof_path)
                               for row in closure['files'])):
                raise ValueError('actual file4 feature identity closure required')
            capacity = json.loads(Path(spec['capacity_proof']).read_text())
            if (capacity['arm'] != 'ray-feature-attention'
                    or capacity['actual_optimizer_states'] != 375
                    or not capacity['all_nine_feature_tensors_learn_without_scalar_gate']):
                raise ValueError('capacity proof must exercise this375-tensor feature model')

    def init_models_and_more(self, **kwargs):
        spec = self.anchor_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser': spec['denoiser_checkpoint']}):
            raise ValueError('complete native NN continuation from shared oldR96 required')
        module = RayFeatureAttentionConditioner(**self.deform_conditioner_config).cuda()
        module.load_reference(torch.load(spec['conditioner_checkpoint'], map_location='cpu', weights_only=True))
        module.ray_readout.score_selector.load_state_dict(
            torch.load(spec['utility_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        module.retargeter.requires_grad_(True)
        module.fusion_gate.requires_grad_(True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        ImageConditionedProjFlowMatchingCFGTrainer.init_models_and_more(self, **kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name,p in module.named_parameters()
                    if p.requires_grad and not name.startswith('ray_readout.')]
        feature_params = list(module.ray_readout.parameters())
        if (len(nn_params), len(r_params), len(feature_params)) != (360, 6, 9):
            raise ValueError('exact360 native LoRA,6 original R/fusion and9 feature tensors required')
        self.optimizer = torch.optim.AdamW([
            dict(params=nn_params, model_name='default', lr=1e-5),
            dict(params=r_params, model_name='deform_conditioner', lr=1e-4),
            dict(params=feature_params, model_name='ray_readout', lr=1e-4),
        ], **self.optimizer_config['args'])
        self._install_stage_schedule()

    def _install_stage_schedule(self):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            [lambda _: 1., lambda _: 1., lambda step: min(1., (step+1)/5.)])

    def _write_stage_initialization(self, restored, sampler):
        super()._write_stage_initialization(restored, sampler)
        target = Path(self.output_dir)/f'feature-stage-initialization.rank{self.rank}.json'
        with target.open('x') as stream:
            json.dump(dict(arm='ray-feature-attention', zero_output_projection=True,
                scalar_mix_gate=False, target_patch_ray=True, original_nearest_carrier=True,
                matching_decoder_frozen=True, original_R_learns=True, native_LoRA_learns=True,
                new_feature_tensors=9, common_Adam_states=restored,
                reference_step=96, final_step=480, remaining_updates=384), stream, indent=2)
            stream.write('\n')

    def run_step(self, data_list):
        result = ImageConditionedProjFlowMatchingCFGTrainer.run_step(self, data_list)
        module = self.deform_conditioner
        statistics = dict(module.last_feature_attention_stats,
            feature_gradient_L1=sum(float(p.grad.detach().float().abs().sum())
                for p in module.ray_readout.parameters() if p.grad is not None))
        if self.is_master:
            with (Path(self.output_dir)/'ray-feature-attention.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.step+1, **statistics))+'\n')
            if getattr(self, 'wandb_run', None):
                self.wandb_run.log({'ray_feature/'+key:value for key,value in statistics.items()
                    if isinstance(value, (int,float))}, step=self.step+1)
        self._save_rank_rng(self.step+1)
        return result
