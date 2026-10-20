"""Matched PBR762-to1905 control with only the native Top1/R correction."""
from __future__ import annotations

import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
from .anchor_joint_runtime import restore_rng
from .pbr_continuation_state import convert_pbr_sampler, restore_v1_adam
from .pbr_joint_runtime import PBRJointTrainer
from .retarget_only_pbr_contract import ARM,build_retarget_only,validate_profile,validate_initial32
from .task_ray_science_contract import closed_json


class RetargetOnlyPBRTrainer(PBRJointTrainer):
    def __init__(self,*args,pbr_stage,**kwargs):
        self.pbr_stage=dict(pbr_stage)
        spec=self.pbr_stage
        if (spec['arm']!=ARM or spec['reference_step']!=762 or spec['stage_rng_seed']!=101
                or kwargs['max_steps']!=1905 or kwargs.get('prefetch_data') is not False
                or kwargs.get('mix_precision_mode')!='amp'
                or kwargs.get('mix_precision_dtype')!='bfloat16' or kwargs.get('ema_rate')!=[]):
            raise ValueError('own native Top1/R762-to1905 continuation is required')
        self._verify_reference()
        ImageConditionedProjSparseFlowMatchingCFGTrainer.__init__(self,*args,**kwargs)
        if spec['diagnostic_profile'] and not spec['capacity_profile']:
            if (self.world_size,self.batch_size_per_gpu,self.batch_split)!=(1,1,1):
                raise ValueError('own micro1 engineering profile only')
        elif (self.world_size,self.batch_size,self.batch_size_per_gpu,self.batch_split)!=(4,24,6,1):
            raise ValueError('same actual selected world4/micro6/global24 without accumulation')
        if len(self.dataset)!=9139 or self.dataset.transport_arm!='c1':
            raise ValueError('same9139 eligible PBR assets and C1 cache carrier required')
        if self.step==0:
            misc=torch.load(spec['misc_checkpoint'],map_location='cpu',weights_only=False)
            if misc['step']!=762 or misc['lora']!=self.lora_config:
                raise ValueError('common V1 LoRA state or source step differs')
            names=[name for name,p in self.models['denoiser'].named_parameters() if p.requires_grad]
            nn=torch.load(spec['denoiser_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
            checkpoint_names=[name for name in nn if name in set(names)]
            restored=restore_v1_adam(self.optimizer,misc['optimizer'],names=names,
                                    checkpoint_names=checkpoint_names,expected_step=762)
            if restored!=360:
                raise ValueError('all360 native NN Adam moments must continue')
            del nn
            sampler=convert_pbr_sampler(misc['data_sampler'],source_world=3,source_batch=8,target_world=self.world_size)
            self.data_sampler.load_state_dict(sampler)
            self.grad_clip.load_state_dict(misc['grad_clip'])
            self.step=762
            self._install_stage_schedule()
            seed=101+self.rank
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
            self._write_initialization(restored,misc['data_sampler'],sampler)
            self._save_stage_rng()
        else:
            restore_rng(self._rng_path(self.step),step=self.step,sampler=self.data_sampler,
                        loader_generator=self._pbr_loader_generator)

    def _verify_reference(self):
        spec=self.pbr_stage
        # The old validator covers ONLY the common V1 source input bytes.
        # The real arm remains R-only and its own gates are checked below.
        common=SimpleNamespace(pbr_stage=dict(spec,arm='v1-control',diagnostic_profile=True))
        PBRJointTrainer._verify_reference(common)
        mode=spec['R_only_mode']
        if (mode not in {'profile','capacity','train'}
                or spec['diagnostic_profile']!=(mode!='train')
                or spec['capacity_profile']!=(mode=='capacity')):
            raise ValueError('own explicit R-only execution mode required')
        def closed(path,receipt):return closed_json(path,receipt,Path(spec['campaign_root']))
        if mode in {'capacity','train'}:
            validate_profile(closed(spec['own_profile'],spec['own_profile_receipt']),capacity=False)
            validate_initial32(closed(spec['own_initial32'],spec['own_initial32_receipt']))
        if mode=='train':
            validate_profile(closed(spec['own_capacity'],spec['own_capacity_receipt']),capacity=True)

    def init_models_and_more(self,**kwargs):
        spec=self.pbr_stage
        if (set(self.models)!={'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt!={'denoiser':spec['denoiser_checkpoint']}):
            raise ValueError('same complete V1 NN checkpoint required')
        expected=dict(spec['direct_conditioner_config'],arm='c1');expected.pop('deformation_dim')
        if self.deform_conditioner_config!=expected:
            raise ValueError('physical C1 config must remove RAW deformation channels')
        module,self._R_only_initializer=build_retarget_only(spec['direct_conditioner_config'])
        self.deform_conditioner=module.cuda()
        self.models['deform_conditioner']=self.deform_conditioner
        ImageConditionedProjSparseFlowMatchingCFGTrainer.init_models_and_more(self,**kwargs)
        nn_params=[p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params=[p for p in self.deform_conditioner.parameters() if p.requires_grad]
        if (len(nn_params),len(r_params))!=(360,6):
            raise ValueError('only NN360 plus R6 may be trained')
        self.optimizer=torch.optim.AdamW([dict(params=nn_params,model_name='default',lr=5e-6),
            dict(params=r_params,model_name='retargeter',lr=1e-4)],**self.optimizer_config['args'])
        self._install_stage_schedule()

    def _install_stage_schedule(self):
        self.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,
            [lambda _:1.,lambda step:min(1.,(step+1)/5.)])

    def _write_initialization(self,restored,source_sampler,sampler):
        PBRJointTrainer._write_initialization(self,restored,source_sampler,sampler)
        out=Path(self.output_dir)
        with (out/f'R-only-initializer.rank{self.rank}.json').open('x') as f:
            json.dump(dict(arm=ARM,**self._R_only_initializer),f,indent=2);f.write('\n')
        # This is the untouched initial state, not a profile-updated checkpoint.
        torch.save({name:value.detach().cpu() for name,value in self.deform_conditioner.state_dict().items()},
                   out/f'R-only-initial-conditioner.rank{self.rank}.pt')

    def run_step(self,data_list):
        result=ImageConditionedProjSparseFlowMatchingCFGTrainer.run_step(self,data_list)
        if self.is_master:
            cond=self._last_deform_condition
            record=dict(step=self.step+1,ray_query='edited-target-patch',
                physical_R_input_channels=3072,additional_deformation_channels=0,
                attention_module_physically_absent=True,
                fusion_gate=float(self.deform_conditioner.fusion_gate.detach()),
                retarget_residual_RMS=float(cond.retarget_residual.detach().float().square().mean().sqrt()))
            with (Path(self.output_dir)/'R-only-conditioning.jsonl').open('a') as f:
                f.write(json.dumps(record)+'\n')
            if getattr(self,'wandb_run',None):
                self.wandb_run.log({'PBR_R_only/'+k:v for k,v in record.items()
                    if isinstance(v,(int,float))},step=self.step+1)
        self._save_stage_rng(self.step+1)
        return result
