"""Matched feature-attention continuation with physical Omega query channels.

The common LoRA/R optimizer tensors keep their original shapes and moments.
Only the added 2048-to-256 query projection receives a fourth, fresh group.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .anchor_joint_runtime import AnchorJointTrainer, sha
from .ray_feature_joint_runtime import RayFeatureJointTrainer


class DirectRayJointTrainer(RayFeatureJointTrainer):
    def __init__(self, *args, anchor_stage, **kwargs):
        if (anchor_stage.get('variant') != 'direct-DQ2048'
                or kwargs['deform_conditioner'].get('arm') != 'c2'
                or kwargs['deform_conditioner'].get('deformation_dim') != 2048):
            raise ValueError('explicit direct-DQ2048 variant and C2 module required')
        # The inherited lifecycle uses the same epoch1 stage contract. The
        # variant and public experiment arm distinguish this additional arm.
        super().__init__(*args, anchor_stage=anchor_stage, **kwargs)

    def _verify_reference(self):
        AnchorJointTrainer._verify_reference(self)
        spec = self.anchor_stage
        done = json.loads(Path(spec['utility_completed']).read_text())
        if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
                or done['epochs'] != 7 or done['optimizer_updates'] != 112
                or done['pure_utility_export_SHA256'] != spec['utility_SHA256']
                or sha(spec['utility_checkpoint']) != spec['utility_SHA256']):
            raise ValueError('same fixed final seven-epoch utility initializer required')
        proof_path = Path(spec['DQ_identity_proof'])
        proof = json.loads(proof_path.read_text())
        if (proof['status'] != 'PASS_ACTUAL_SS_DIRECT_RAY_RETARGETING_IDENTITY_AND_TEN_TENSOR_GRADIENT'
                or not proof['initial_evidence_exact']
                or not proof['all10_new_tensors_receive_actual_SS_gradients']
                or not proof['deformation_observations_are_nonzero']
                or proof['new_physical_input_channels'] != 4096
                or proof['initial_loss'] != proof['new_initial_loss']):
            raise ValueError('actual physical query identity and learnability proof required')
        for path, digest in proof['source_SHA256'].items():
            if sha(path) != digest:
                raise ValueError('DQ identity proof source changed: ' + path)
        if not spec.get('diagnostic_profile', False):
            receipt = json.loads(Path(spec['DQ_identity_receipt']).read_text())
            relative = str(proof_path.relative_to(Path(spec['campaign_root']).parent))
            if (receipt['status'] != 'PASS' or not receipt['local_equals_actual_file4_SHA256']
                    or not any(row['relative_path'] == relative and row['sha256'] == sha(proof_path)
                               for row in receipt['files'])):
                raise ValueError('actual file4 DQ identity closure required')
            capacity = json.loads(Path(spec['capacity_proof']).read_text())
            if (capacity['arm'] != 'direct-ray-retargeting-dq'
                    or capacity['actual_optimizer_states'] != 376
                    or not capacity['all_ten_new_tensors_learn']):
                raise ValueError('exclusive capacity proof must exercise the376-tensor DQ model')

    def init_models_and_more(self, **kwargs):
        spec = self.anchor_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser': spec['denoiser_checkpoint']}):
            raise ValueError('complete native NN continuation from shared oldR96 required')
        module = DirectRayRetargetingConditioner(**self.deform_conditioner_config).cuda()
        module.load_reference(torch.load(spec['conditioner_checkpoint'], map_location='cpu', weights_only=True))
        module.ray_readout.score_selector.load_state_dict(
            torch.load(spec['utility_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        module.retargeter.requires_grad_(True)
        module.fusion_gate.requires_grad_(True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        ImageConditionedProjFlowMatchingCFGTrainer.init_models_and_more(self, **kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name, p in module.named_parameters() if p.requires_grad
                    and not name.startswith('ray_readout.')
                    and not name.startswith('retargeter.deformation_projection.')]
        attention_params = list(module.ray_readout.parameters())
        dq_params = list(module.retargeter.deformation_projection.parameters())
        if tuple(map(len, (nn_params, r_params, attention_params, dq_params))) != (360, 6, 9, 1):
            raise ValueError('exact360 LoRA,6 original R,9 attention and1 new query tensor required')
        self.optimizer = torch.optim.AdamW([
            dict(params=nn_params, model_name='default', lr=1e-5),
            dict(params=r_params, model_name='deform_conditioner', lr=1e-4),
            dict(params=attention_params, model_name='ray_readout', lr=1e-4),
            dict(params=dq_params, model_name='deformation_query', lr=1e-4),
        ], **self.optimizer_config['args'])
        self._install_stage_schedule()

    def _install_stage_schedule(self):
        warmup = lambda step: min(1., (step + 1) / 5.)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, [lambda _: 1., lambda _: 1., warmup, warmup])

    def _write_stage_initialization(self, restored, sampler):
        AnchorJointTrainer._write_stage_initialization(self, restored, sampler)
        report = dict(arm='direct-ray-retargeting-dq', variant='direct-DQ2048',
            target_patch_ray=True, original_nearest_carrier=True, scalar_mix_gate=False,
            physical_input_channels=self.deform_conditioner.retargeter.input_dim,
            additional_deformation_channels=2048, zero_initial_projection_weights=True,
            actual_input_observations_nonzero=True, common_R_shapes_unchanged=True,
            common_Adam_states=restored, new_attention_tensors=9, new_DQ_tensors=1,
            reference_step=96, final_step=480, remaining_updates=384,
            GT_correspondence_or_deformation_input=False,
            correspondence_feature_delivery='native retarget_residual outside semantic fusion gate',
            transformed_feature_correction_bound=self.deform_conditioner.g_max)
        target = Path(self.output_dir) / f'DQ-stage-initialization.rank{self.rank}.json'
        with target.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')

    def run_step(self, data_list):
        result = super().run_step(data_list)
        parameter = self.deform_conditioner.retargeter.deformation_projection.weight
        gradient = float(parameter.grad.detach().float().abs().sum()) if parameter.grad is not None else None
        if self.is_master:
            row = dict(step=self.step + 1, deformation_projection_gradient_L1=gradient,
                       **self.deform_conditioner.last_feature_attention_stats)
            with (Path(self.output_dir) / 'deformation-query.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if getattr(self, 'wandb_run', None) and gradient is not None:
                self.wandb_run.log({'deformation_query/gradient_L1': gradient}, step=self.step + 1)
        return result
