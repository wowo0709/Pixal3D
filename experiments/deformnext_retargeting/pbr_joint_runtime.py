"""Matched native PBR continuation from the existing V1 step762.

Restore all360 NN Adam states. Original V1 and physical-query/direct attention
start with the same complete base+LoRA and global sampler offset48. New R6,
attention9 and DQ1 have fresh moments and a five-update warmup.
"""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch

from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
from .anchor_joint_runtime import restore_rng, save_rng, sha
from .pbr_continuation_state import convert_pbr_sampler, restore_v1_adam


class PBRJointTrainer(ImageConditionedProjSparseFlowMatchingCFGTrainer):
    def __init__(self, *args, pbr_stage, **kwargs):
        self.pbr_stage = dict(pbr_stage)
        spec = self.pbr_stage
        if (spec['arm'] not in {'v1-control','direct-retargeting-dq'}
                or spec['reference_step'] != 762 or spec['stage_rng_seed'] != 101
                or kwargs['max_steps'] != 1905
                or kwargs.get('prefetch_data') is not False
                or kwargs.get('mix_precision_mode') != 'amp'
                or kwargs.get('mix_precision_dtype') != 'bfloat16' or kwargs.get('ema_rate') != []):
            raise ValueError('explicit matched native PBR762-to1905 BF16 continuation required')
        self._verify_reference()
        super().__init__(*args, **kwargs)
        if spec.get('diagnostic_profile', False) and not spec.get('capacity_profile', False):
            if (self.world_size,self.batch_size_per_gpu,self.batch_split) != (1,1,1):
                raise ValueError('disposable cohost correctness profile is limited to micro1')
        elif (self.world_size not in {2,4} or self.batch_size != 24
                or self.batch_size_per_gpu != 24//self.world_size or self.batch_split != 1):
            raise ValueError('actual DDP2/micro12 or DDP4/micro6 global24 without accumulation required')
        if len(self.dataset) != 9139:
            raise ValueError('same native32768-token eligible TRAIN9139 subset as original V1 required')
        if self.step == 0:
            misc = torch.load(spec['misc_checkpoint'], map_location='cpu', weights_only=False)
            if misc['step'] != 762 or misc['lora'] != self.lora_config:
                raise ValueError('source PBR step/LoRA differs')
            names = [name for name,p in self.models['denoiser'].named_parameters() if p.requires_grad]
            source_nn = torch.load(spec['denoiser_checkpoint'],map_location='cpu',weights_only=True,mmap=True)
            checkpoint_names = [name for name in source_nn if name in set(names)]
            restored = restore_v1_adam(self.optimizer,misc['optimizer'],names=names,
                                      checkpoint_names=checkpoint_names,expected_step=762)
            if restored != 360:
                raise ValueError('all360 native NN LoRA moments must continue')
            del source_nn
            sampler = convert_pbr_sampler(misc['data_sampler'],source_world=3,
                                          source_batch=8,target_world=self.world_size)
            self.data_sampler.load_state_dict(sampler)
            self.grad_clip.load_state_dict(misc['grad_clip'])
            self.step = 762
            self._install_stage_schedule()
            seed = int(spec['stage_rng_seed']) + self.rank
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            self._write_initialization(restored,misc['data_sampler'],sampler)
            self._save_stage_rng()
        else:
            restore_rng(self._rng_path(self.step),step=self.step,sampler=self.data_sampler,
                        loader_generator=self._pbr_loader_generator)

    def _verify_reference(self):
        spec = self.pbr_stage
        proof_path = Path(spec['reference_input_audit'])
        audit = json.loads(proof_path.read_text())
        receipt = json.loads(Path(spec['reference_receipt']).read_text())
        key = str(proof_path.relative_to(Path(spec['campaign_root']).parent))
        if (audit['status'] != 'PASS_ACTUAL_ORIGINAL_FILE4_PBR_V1_REFERENCE'
                or not audit['local_equals_actual_original_file4_SHA256']
                or receipt['status'] != 'PASS' or not receipt['local_equals_actual_file4_SHA256']
                or not any(r['relative_path']==key and r['sha256']==sha(proof_path) for r in receipt['files'])):
            raise ValueError('actual original file4 V1 reference and closed source audit required')
        checked = {r['local_path']:r for r in audit['files']}
        for field in ['denoiser_checkpoint','misc_checkpoint','source_config']:
            path = spec[field]
            if path not in checked or sha(path) != checked[path]['sha256']:
                raise ValueError('source V1 bytes differ: '+field)
        original = json.loads(Path(spec['source_config']).read_text())
        if original['num_gpus'] != 3 or original['trainer']['args']['batch_size_per_gpu'] != 8:
            raise ValueError('declared source world3/batch8 differs')
        if spec['arm'] == 'direct-retargeting-dq':
            identity_path = Path(spec['native_PBR_identity_proof'])
            identity = json.loads(identity_path.read_text())
            if (identity['status'] != 'PASS_ACTUAL_DIRECT_PBR_PHYSICAL_QUERY_IDENTITY_FIRST_DINO_NAF_AND16_GRADIENTS'
                    or not identity['initial_evidence_exact']
                    or not identity['first_head_gradient_reaches_both_DINO_and_NAF_with_zero_fusion_and_R']
                    or identity['physical_retargeter_input_channels'] != 5120
                    or sha(spec['utility_checkpoint']) != spec['utility_SHA256']):
                raise ValueError('actual Direct PBR identity and first DINO/NAF gradient proof required')
            for path,digest in identity['source_SHA256'].items():
                if sha(path) != digest:
                    raise ValueError('PBR native proof source changed: '+path)
        if not spec.get('diagnostic_profile',False):
            for arm in ['v1-control','direct-retargeting-dq']:
                path = Path(spec['capacity_proofs'][arm])
                capacity = json.loads(path.read_text())
                close = json.loads(Path(spec['capacity_receipts'][arm]).read_text())
                relative = str(path.relative_to(Path(spec['campaign_root']).parent))
                if (capacity['status'] != 'PASS_ACTUAL_PBR_JOINT_CAPACITY'
                        or capacity['arm'] != arm or capacity['global_batch'] != 24
                        or capacity['world_size'] != spec['selected_world_size']
                        or capacity['accumulation'] != 1 or not capacity['post_update_DDP_equal']
                        or not capacity['post_validation_full_micro_update_passed']
                        or close['status'] != 'PASS' or not close['local_equals_actual_file4_SHA256']
                        or not any(r['relative_path']==relative and r['sha256']==sha(path) for r in close['files'])):
                    raise ValueError('actual closed matched fullbatch capacity for BOTH PBR arms required')

    def init_models_and_more(self, **kwargs):
        spec = self.pbr_stage
        if (set(self.models) != {'denoiser'} or self.deform_conditioner is not None
                or self.continuation_ckpt != {'denoiser':spec['denoiser_checkpoint']}):
            raise ValueError('complete common PBR V1 NN continuation required')
        if spec['arm'] == 'direct-retargeting-dq':
            config = self.deform_conditioner_config
            if (config['arm'],config['value_dim'],config['query_dim'],config['deformation_dim']) != ('c2',2048,1024,2048):
                raise ValueError('physical PBR query schema5120 required')
            module = DirectRayRetargetingConditioner(**config).cuda()
            module.ray_readout.score_selector.load_state_dict(
                torch.load(spec['utility_checkpoint'],map_location='cpu',weights_only=True),strict=True)
            module.retargeter.requires_grad_(True)
            module.fusion_gate.requires_grad_(True)
            if (torch.count_nonzero(module.ray_readout.output.weight)
                    or torch.count_nonzero(module.retargeter.net[-1].weight)
                    or torch.count_nonzero(module.retargeter.net[-1].bias)
                    or torch.count_nonzero(module.retargeter.deformation_projection.weight)
                    or float(module.fusion_gate) != 0):
                raise ValueError('untouched zero-output native V1 initial function required')
            self.deform_conditioner = module
            self.models['deform_conditioner'] = module
        elif self.deform_conditioner_config is not None:
            raise ValueError('vanilla control must physically omit the conditioner')
        super().init_models_and_more(**kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        if len(nn_params) != 360:
            raise ValueError('exact360 native PBR LoRA tensors required')
        groups = [dict(params=nn_params,model_name='default',lr=5e-6)]
        if spec['arm'] == 'direct-retargeting-dq':
            module = self.deform_conditioner
            r_params = [p for name,p in module.named_parameters() if p.requires_grad
                and not name.startswith('ray_readout.')
                and not name.startswith('retargeter.deformation_projection.')]
            attention = list(module.ray_readout.parameters())
            deformation = list(module.retargeter.deformation_projection.parameters())
            if tuple(map(len,(r_params,attention,deformation))) != (6,9,1):
                raise ValueError('fresh original R6,attention9,DQ1 groups required')
            groups += [dict(params=r_params,model_name='retargeter',lr=1e-4),
                       dict(params=attention,model_name='ray_readout',lr=1e-4),
                       dict(params=deformation,model_name='deformation_query',lr=1e-4)]
        self.optimizer = torch.optim.AdamW(groups,**self.optimizer_config['args'])
        self._install_stage_schedule()

    def _install_stage_schedule(self):
        functions = [lambda _:1.]
        if self.pbr_stage['arm'] == 'direct-retargeting-dq':
            functions += [lambda step:min(1.,(step+1)/5.)]*3
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,functions)

    def prepare_dataloader(self, **kwargs):
        super().prepare_dataloader(**kwargs)
        self._pbr_loader_generator = torch.Generator().manual_seed(101+self.rank)
        self.dataloader.generator = self._pbr_loader_generator

    def _rng_path(self, step):
        return Path(self.output_dir)/'stage_rng'/f'rank{self.rank}'/f'step{step:07d}.pt'

    def _save_stage_rng(self, step=None):
        step = self.step if step is None else step
        save_rng(self._rng_path(step),step=step,sampler=self.data_sampler,
                 loader_generator=self._pbr_loader_generator)

    def _write_initialization(self, restored, source_sampler, sampler):
        record = dict(status='PASS_MATCHED_PBR_V1_STEP762_NATIVE_INITIALIZATION',rank=self.rank,
            arm=self.pbr_stage['arm'],source_step=762,total_steps=1905,additional_updates=1143,
            common_NN_Adam_states=restored,common_state_step=762,
            source_sampler=source_sampler,source_world=3,source_global_offset=source_sampler['idx']*3,
            target_sampler=sampler,target_world=self.world_size,
            global_batch=self.batch_size,full_microbatch=self.batch_size_per_gpu,accumulation=self.batch_split,
            original_PBR_token_limit=32768,actual_eligible_training_assets=len(self.dataset),
            sampler_rank_assignment_changed=self.world_size!=3,
            stage_RNG_reset=101+self.rank,old_training_RNG_resume_claimed=False,
            new_optimizer_states=0,GT_matching_or_deformation_input=False,
            optimizer_groups=[dict(name=g['model_name'],tensors=len(g['params']),lr=g['lr']) for g in self.optimizer.param_groups])
        with (Path(self.output_dir)/f'PBR-stage-initialization.rank{self.rank}.json').open('x') as stream:
            json.dump(record,stream,indent=2)
            stream.write('\n')

    def run_step(self, data_list):
        result = super().run_step(data_list)
        if self.deform_conditioner is not None and self.is_master:
            stats = dict(self.deform_conditioner.last_feature_attention_stats)
            with (Path(self.output_dir)/'PBR-direct-attention.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=self.step+1,**stats))+'\n')
            if getattr(self,'wandb_run',None):
                self.wandb_run.log({'PBR_direct/'+k:v for k,v in stats.items()
                                   if isinstance(v,(int,float))},step=self.step+1)
        self._save_stage_rng(self.step+1)
        return result
