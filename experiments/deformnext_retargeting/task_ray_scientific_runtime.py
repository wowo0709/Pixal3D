"""Scientific continuation with own TaskRay369 gates and unchanged native init."""
from pathlib import Path
from types import SimpleNamespace
import json

import torch

from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer

from .anchor_joint_runtime import AnchorJointTrainer
from .consistent_ray_joint_runtime import ConsistentRayJointTrainer
from .task_ray_joint_runtime import TaskRayJointTrainer
from .task_ray_science_contract import closed_json,digest,validate_capacity,validate_initial32


class TaskRayScientificTrainer(TaskRayJointTrainer):
    def run(self):
        # Own native32 identity is already a mandatory closed gate. Preserve
        # epoch validation while avoiding a second copy of initial R96 sampling.
        return ImageConditionedProjFlowMatchingCFGTrainer.run(self)

    def _verify_reference(self):
        spec=self.anchor_stage
        mode=spec.get('task_execution_mode')
        if (mode not in {'capacity','train'} or spec.get('task_KL_weight')!=.001
                or spec.get('diagnostic_profile')!=(mode=='capacity')
                or spec.get('capacity_profile')!=(mode=='capacity')):
            raise ValueError('explicit new TaskRay scientific/capacity recipe required')
        # Validate ONLY the inherited common R96/utility/reliability references
        # through a separate view. The real scientific declaration is unchanged;
        # old367 capacity never admits a new369 scientific model.
        common=dict(spec,diagnostic_profile=True)
        view=SimpleNamespace(anchor_stage=common,_closed=self._closed)
        ConsistentRayJointTrainer._verify_reference(view)
        probe=closed_json(spec['task_probe'],spec['task_probe_receipt'],spec['campaign_root'])
        native=closed_json(spec['task_probe_plan'],spec['task_probe_plan_receipt'],spec['campaign_root'])
        if (probe['status']!='PASS_ACTUAL_BOUNDED_TASK_RAY369_NATIVE_PROFILE'
                or probe['actual_optimizer_states']!=369
                or not probe['new_attention_receives_actual_SS_gradient_not_only_KL']
                or not probe['all_frozen_NN_semantic_decoder_and_reliability_exact']):
            raise ValueError('actual native TaskRay learnability proof required')
        for path,sha in native['source_SHA256'].items():
            if digest(path)!=sha:raise ValueError('native TaskRay model source changed: '+path)
        initial=closed_json(spec['task_initial32'],spec['task_initial32_receipt'],spec['campaign_root'])
        validate_initial32(initial)
        if mode=='train':
            capacity=closed_json(spec['task_capacity'],spec['task_capacity_receipt'],spec['campaign_root'])
            validate_capacity(capacity)
        self._task_scientific_gate_verified=True

    def training_losses(self,*args,**kwargs):
        # The engineering entry's autograd.grad diagnostic is single GPU only.
        # DDP capacity/science uses exactly one backward over the SS+KL loss.
        terms,status=ConsistentRayJointTrainer.training_losses(self,*args,**kwargs)
        kl=self.deform_conditioner.task_matching_KL
        if kl is None or not torch.isfinite(kl):raise ValueError('finite actual ray prior KL required')
        terms['task_ray_KL']=kl
        terms['loss']=terms['loss']+.001*kl
        return terms,status

    def _write_stage_initialization(self,restored,sampler):
        AnchorJointTrainer._write_stage_initialization(self,restored,sampler)
        if not self._task_scientific_gate_verified or restored!=366:
            raise ValueError('all common native Adam states and own gates required')
        report=dict(status='PASS_TASK_RAY369_NATIVE_STAGE_INITIALIZATION',
            common_Adam_states=restored,common_source_step=96,total_optimizer_tensors=369,
            new_geometry_and_attention_optimizer_states=0,
            semantic7_frozen=True,reliability4_frozen=True,query_channels=4096,
            actual_deformation_query_channels=2048,task_logit_bound=1.,task_KL_weight=.001,
            final_step=480,additional_updates=384,GT_in_forward=False,
            diagnostic_capacity=self.anchor_stage['task_execution_mode']=='capacity')
        (Path(self.output_dir)/f'task-ray-stage-initialization.rank{self.rank}.json').write_text(json.dumps(report,indent=2)+'\n')

    def run_step(self,data_list):
        result=super().run_step(data_list)
        module=self.deform_conditioner
        gradients={name:float(p.grad.detach().float().abs().sum()) if p.grad is not None else None
            for name,p in module.named_parameters() if name.startswith(('task_attention.','retargeter.deformation_projection.'))}
        if self.is_master:
            record=dict(step=self.step+1,gradients=gradients,**module.last_feature_attention_stats)
            with (Path(self.output_dir)/'task-ray-training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if getattr(self,'wandb_run',None):
                self.wandb_run.log({'task_ray/'+key:value for key,value in gradients.items() if value is not None},step=self.step+1)
        return result
