"""Native engineering continuation for bounded task attention on patch rays."""
from pathlib import Path
import json

import torch

from pixal3d.modules.task_ray_retargeting import TaskRayRetargetingConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
from .anchor_joint_runtime import AnchorJointTrainer
from .consistent_ray_joint_runtime import ConsistentRayJointTrainer


class TaskRayJointTrainer(ConsistentRayJointTrainer):
    def _verify_reference(self):
        # The old proof verifies the unchanged common R96 reference. This new
        # attention still needs its own native identity/369-state/capacity proof.
        if not self.anchor_stage.get('diagnostic_profile',False):
            raise ValueError('new task-attention science requires its own native369 gates first')
        super()._verify_reference()

    def init_models_and_more(self,**kwargs):
        spec = self.anchor_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser':spec['denoiser_checkpoint']}):
            raise ValueError('shared complete R96 NN continuation required')
        module = TaskRayRetargetingConditioner(**self.deform_conditioner_config).cuda()
        module.load_reference(torch.load(spec['conditioner_checkpoint'],map_location='cpu',weights_only=True))
        module.ray_readout.score_selector.load_state_dict(
            torch.load(spec['utility_checkpoint'],map_location='cpu',weights_only=True),strict=True)
        module.reliability.load_pilot(spec['reliability_checkpoint'],spec['reliability_SHA256'])
        module.ray_readout.requires_grad_(False)
        module.reliability.requires_grad_(False)
        module.retargeter.requires_grad_(True)
        module.fusion_gate.requires_grad_(True)
        module.task_attention.requires_grad_(True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        ImageConditionedProjFlowMatchingCFGTrainer.init_models_and_more(self,**kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name,p in module.named_parameters() if p.requires_grad
                    and not name.startswith(('retargeter.deformation_projection.','task_attention.'))]
        geometry = list(module.retargeter.deformation_projection.parameters())
        attention = list(module.task_attention.parameters())
        if tuple(map(len,(nn_params,r_params,geometry,attention))) != (360,6,1,2):
            raise ValueError('exact360 LoRA,6 commonR,1 geometry and2 task-attention tensors required')
        self.optimizer = torch.optim.AdamW([
            dict(params=nn_params,model_name='default',lr=1e-5),
            dict(params=r_params,model_name='deform_conditioner',lr=1e-4),
            dict(params=geometry,model_name='deformation_query',lr=1e-4),
            dict(params=attention,model_name='task_ray_attention',lr=1e-4),
        ],**self.optimizer_config['args'])
        self._install_stage_schedule()

    def _install_stage_schedule(self):
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            [lambda _:1.,lambda _:1.,lambda step:min(1.,(step+1)/5.),
             lambda step:min(1.,(step+1)/5.)])

    def training_losses(self,*args,**kwargs):
        terms,status = super().training_losses(*args,**kwargs)
        kl = self.deform_conditioner.task_matching_KL
        if kl is None or not torch.isfinite(kl):
            raise ValueError('actual finite task-attention prior divergence required')
        if self.anchor_stage.get('diagnostic_profile',False) and self.step == 98:
            parameters = [self.deform_conditioner.retargeter.deformation_projection.weight,
                          *self.deform_conditioner.task_attention.parameters()]
            gradients = torch.autograd.grad(terms['mse'],parameters,retain_graph=True,allow_unused=True)
            self._task_flow_only_gradient_L1 = [float(g.detach().float().abs().sum())
                                               if g is not None else None for g in gradients]
        terms['task_ray_KL'] = kl
        terms['loss'] = terms['loss']+.001*kl
        return terms,status

    def _write_stage_initialization(self,restored,sampler):
        AnchorJointTrainer._write_stage_initialization(self,restored,sampler)
        report = dict(status='INITIALIZED_BOUNDED_TASK_RAY_ATTENTION_ENGINEERING',
            common_Adam_states=restored,new_geometry_and_attention_states=0,
            total_optimizer_tensors_after_first_update=369,semantic7_frozen=True,reliability4_frozen=True,
            added_task_attention_tensors=2,query_channels=4096,actual_deformation_query_channels=2048,
            task_logit_bound=1.,task_KL_weight=.001,common_R96=True,
            source_identity_proof_covers_common_R_only=True,own_native369_proof_required=True,
            profile_only=True,scientific_training_updates=0)
        target = Path(self.output_dir)/f'task-ray-initialization.rank{self.rank}.json'
        target.write_text(json.dumps(report,indent=2)+'\n')
