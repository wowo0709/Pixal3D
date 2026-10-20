"""Matched367-state SS continuation; protect matching and predict availability."""
import json
from pathlib import Path

import torch

from pixal3d.modules.ray_consistent_retargeting import RayConsistentRetargetingConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .anchor_joint_runtime import AnchorJointTrainer, sha
from .ray_feature_joint_runtime import RayFeatureJointTrainer


class ConsistentRayJointTrainer(RayFeatureJointTrainer):
    def __init__(self, *args, anchor_stage, **kwargs):
        module = kwargs['deform_conditioner']
        if (anchor_stage.get('variant') != 'consistent-paired-reliability'
                or module.get('arm') != 'c2' or module.get('deformation_dim') != 2048
                or module.get('composition') != 'paired' or module.get('use_reliability') is not True):
            raise ValueError('explicit candidate-paired, predicted-reliability physical-query arm required')
        super().__init__(*args, anchor_stage=anchor_stage, **kwargs)

    def _verify_reference(self):
        AnchorJointTrainer._verify_reference(self)
        spec = self.anchor_stage
        done = json.loads(Path(spec['utility_completed']).read_text())
        if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
                or done['epochs'] != 7 or done['optimizer_updates'] != 112
                or done['pure_utility_export_SHA256'] != spec['utility_SHA256']
                or sha(spec['utility_checkpoint']) != spec['utility_SHA256']):
            raise ValueError('fixed existing utility initializer required')
        proof_path = Path(spec['consistent_identity_proof'])
        proof = json.loads(proof_path.read_text())
        if (proof['status'] != 'PASS_ACTUAL_SS_CONSISTENT_RAY_IDENTITY_AND_PROJECTOR_GRADIENT'
                or not proof['initial_evidence_exact']
                or not proof['all_frozen_conditioner_state_exact']
                or not proof['one_new_projector_receives_actual_SS_gradients']
                or proof['initial_loss'] != proof['new_initial_loss']):
            raise ValueError('actual candidate-paired SS identity and learnability proof required')
        for path, digest in proof['source_SHA256'].items():
            if sha(path) != digest:
                raise ValueError('candidate-paired identity source changed: '+path)
        self._closed(proof_path, spec['consistent_identity_receipt'])
        pilot_path = Path(spec['reliability_evaluation'])
        pilot = json.loads(pilot_path.read_text())
        self._closed(pilot_path, spec['reliability_receipt'])
        self._closed(Path(spec['reliability_checkpoint']), spec['reliability_receipt'])
        if (pilot['status'] != 'PASS_ACTUAL_GPU_RELIABILITY_HEAD_PILOT'
                or pilot['epochs'] != 7 or not pilot['heldout_head_gate_passed']
                or sha(spec['reliability_checkpoint']) != spec['reliability_SHA256']):
            raise ValueError('fixed7 reliability pilot must pass its declared diagnostic gate')
        port_path = Path(spec['reliability_port_proof'])
        port = json.loads(port_path.read_text())
        self._closed(port_path, spec['reliability_port_receipt'])
        if (port['status'] != 'PASS_ACTUAL_CUDA_DEPLOYED_RELIABILITY_PORT'
                or port['maximum_probability_difference'] > .001
                or port['GT_in_head_forward'] or not port['frozen_actual_checkpoint']):
            raise ValueError('actual CUDA reliability deployment proof required')
        if not spec.get('diagnostic_profile', False):
            capacity_path = Path(spec['capacity_proof'])
            capacity = json.loads(capacity_path.read_text())
            self._closed(capacity_path, spec['capacity_receipt'])
            if (capacity['arm'] != 'consistent-ray-retargeting'
                    or capacity['actual_optimizer_states'] != 367
                    or not capacity['new_projector_learns']
                    or not capacity['matching_and_reliability_frozen_exact']):
                raise ValueError('own367-state full-batch capacity proof required')

    def _closed(self, path, receipt):
        record = json.loads(Path(receipt).read_text())
        relative = str(Path(path).relative_to(Path(self.anchor_stage['campaign_root']).parent))
        if (record['status'] != 'PASS' or not record['local_equals_actual_file4_SHA256']
                or not any(r['relative_path'] == relative and r['sha256'] == sha(path) for r in record['files'])):
            raise ValueError('actual file4 closure differs: '+str(path))

    def init_models_and_more(self, **kwargs):
        spec = self.anchor_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser': spec['denoiser_checkpoint']}):
            raise ValueError('complete shared NN/R96 continuation required')
        module = RayConsistentRetargetingConditioner(**self.deform_conditioner_config).cuda()
        module.load_reference(torch.load(spec['conditioner_checkpoint'], map_location='cpu', weights_only=True))
        module.ray_readout.score_selector.load_state_dict(
            torch.load(spec['utility_checkpoint'], map_location='cpu', weights_only=True), strict=True)
        module.reliability.load_pilot(spec['reliability_checkpoint'], spec['reliability_SHA256'])
        module.ray_readout.requires_grad_(False)
        module.reliability.requires_grad_(False)
        module.retargeter.requires_grad_(True)
        module.fusion_gate.requires_grad_(True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        ImageConditionedProjFlowMatchingCFGTrainer.init_models_and_more(self, **kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name, p in module.named_parameters()
                    if p.requires_grad and not name.startswith('retargeter.deformation_projection.')]
        geometry = list(module.retargeter.deformation_projection.parameters())
        if tuple(map(len, (nn_params, r_params, geometry))) != (360, 6, 1):
            raise ValueError('exact360 LoRA,6 common R/fusion,1 geometry tensor required')
        self.optimizer = torch.optim.AdamW([
            dict(params=nn_params, model_name='default', lr=1e-5),
            dict(params=r_params, model_name='deform_conditioner', lr=1e-4),
            dict(params=geometry, model_name='deformation_query', lr=1e-4),
        ], **self.optimizer_config['args'])
        self._install_stage_schedule()

    def _write_stage_initialization(self, restored, sampler):
        AnchorJointTrainer._write_stage_initialization(self, restored, sampler)
        report = dict(arm='consistent-ray-retargeting', variant='consistent-paired-reliability',
            common_Adam_states=restored, total_optimizer_tensors=367,
            matching7_frozen=True, reliability4_frozen=True, query='edited-target-patch-ray',
            original_Top1_R_carrier=True, appearance_VO_physically_absent=True,
            geometry_correction_before_score_weighting=True, extra_path_R_weights_detached=True,
            physical_deformation_channels=2048, reference_step=96, final_step=480,
            remaining_updates=384, GT_in_neural_forward=False)
        with (Path(self.output_dir)/f'consistent-stage-initialization.rank{self.rank}.json').open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
