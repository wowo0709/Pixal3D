"""Matched374-state continuation for geometry-only matching-aware retargeting."""
import json
from pathlib import Path

import torch

from pixal3d.modules.ray_geometry_retargeting import RayGeometryRetargetingConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .anchor_joint_runtime import AnchorJointTrainer, sha
from .direct_ray_joint_runtime import DirectRayJointTrainer
from .ray_feature_joint_runtime import RayFeatureJointTrainer


class RayGeometryJointTrainer(DirectRayJointTrainer):
    def __init__(self, *args, anchor_stage, **kwargs):
        if (anchor_stage.get('variant') != 'ray-geometry-only-prior-anchored'
                or kwargs['deform_conditioner'].get('arm') != 'c2'
                or kwargs['deform_conditioner'].get('deformation_dim') != 2048):
            raise ValueError('explicit geometry-only prior-anchored physical-query variant required')
        RayFeatureJointTrainer.__init__(self, *args, anchor_stage=anchor_stage, **kwargs)

    def _verify_reference(self):
        AnchorJointTrainer._verify_reference(self)
        spec = self.anchor_stage
        done = json.loads(Path(spec['utility_completed']).read_text())
        if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
                or done['epochs'] != 7 or done['optimizer_updates'] != 112
                or done['pure_utility_export_SHA256'] != spec['utility_SHA256']
                or sha(spec['utility_checkpoint']) != spec['utility_SHA256']):
            raise ValueError('same fixed final utility initializer required')
        proof_path = Path(spec['geometry_identity_proof'])
        proof = json.loads(proof_path.read_text())
        if (proof['status'] != 'PASS_ACTUAL_SS_GEOMETRY_ONLY_IDENTITY_AND_EIGHT_TENSOR_GRADIENT'
                or not proof['appearance_VO_physically_absent']
                or not proof['initial_evidence_exact']
                or not proof['all_eight_new_tensors_receive_actual_SS_gradients']
                or not proof['actual_deformation_observations_nonzero']
                or proof['initial_loss'] != proof['new_initial_loss']):
            raise ValueError('actual native geometry-only identity and learnability proof required')
        for path, digest in proof['source_SHA256'].items():
            if sha(path) != digest:
                raise ValueError('geometry identity source changed: ' + path)
        closure = json.loads(Path(spec['geometry_identity_receipt']).read_text())
        relative = str(proof_path.relative_to(Path(spec['campaign_root']).parent))
        if (closure['status'] != 'PASS' or not closure['local_equals_actual_file4_SHA256']
                or not any(row['relative_path'] == relative and row['sha256'] == sha(proof_path)
                           for row in closure['files'])):
            raise ValueError('actual file4 geometry-only proof closure required')
        if not spec.get('diagnostic_profile', False):
            capacity = json.loads(Path(spec['capacity_proof']).read_text())
            if (capacity['arm'] != 'ray-geometry-retargeting'
                    or capacity['actual_optimizer_states'] != 374
                    or not capacity['all_eight_new_tensors_learn']
                    or not capacity['appearance_VO_physically_absent']):
                raise ValueError('own374-state full-batch capacity proof required')

    def init_models_and_more(self, **kwargs):
        spec = self.anchor_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser': spec['denoiser_checkpoint']}):
            raise ValueError('shared full native NN/R96 continuation required')
        module = RayGeometryRetargetingConditioner(**self.deform_conditioner_config).cuda()
        module.load_reference(torch.load(spec['conditioner_checkpoint'], map_location='cpu', weights_only=True))
        module.ray_readout.score_selector.load_state_dict(
            torch.load(spec['utility_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        if (hasattr(module.ray_readout, 'value') or hasattr(module.ray_readout, 'output')
                or torch.count_nonzero(module.ray_readout.query.weight)):
            raise ValueError('physical VO absence and intact initial matching prior required')
        module.retargeter.requires_grad_(True)
        module.fusion_gate.requires_grad_(True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        ImageConditionedProjFlowMatchingCFGTrainer.init_models_and_more(self, **kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name, p in module.named_parameters() if p.requires_grad
                    and not name.startswith('ray_readout.')
                    and not name.startswith('retargeter.deformation_projection.')]
        matching = list(module.ray_readout.parameters())
        geometry = list(module.retargeter.deformation_projection.parameters())
        if tuple(map(len, (nn_params, r_params, matching, geometry))) != (360, 6, 7, 1):
            raise ValueError('exact360 LoRA,6 R,7 matching,1 physical query tensor required')
        self.optimizer = torch.optim.AdamW([
            dict(params=nn_params, model_name='default', lr=1e-5),
            dict(params=r_params, model_name='deform_conditioner', lr=1e-4),
            dict(params=matching, model_name='ray_readout', lr=1e-4),
            dict(params=geometry, model_name='deformation_query', lr=1e-4),
        ], **self.optimizer_config['args'])
        self._install_stage_schedule()

    def _write_stage_initialization(self, restored, sampler):
        AnchorJointTrainer._write_stage_initialization(self, restored, sampler)
        report = dict(arm='ray-geometry-retargeting', variant='ray-geometry-only-prior-anchored',
            target_patch_ray=True, original_nearest_carrier=True, original_Top1_appearance=True,
            appearance_VO_physically_absent=True, scalar_mix_gate=False,
            physical_input_channels=self.deform_conditioner.retargeter.input_dim,
            additional_deformation_channels=2048, actual_input_observations_nonzero=True,
            common_R_shapes_unchanged=True, common_Adam_states=restored,
            new_matching_tensors=7, new_DQ_tensors=1, total_optimizer_tensors=374,
            initial_matching_query_weights_zero=True, observations_zeroed=False,
            reference_step=96, final_step=480, remaining_updates=384,
            GT_correspondence_or_deformation_input=False)
        target = Path(self.output_dir)/f'geometry-stage-initialization.rank{self.rank}.json'
        with target.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
