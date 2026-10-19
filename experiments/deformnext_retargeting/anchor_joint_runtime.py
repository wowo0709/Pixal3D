"""Matched SS/retargeter continuation, with optional target-ray Top-K utility.

The source epoch1 is shared, not an extra five-epoch continuation of oldR480.
Native training, checkpointing, validation and LoRA loading remain in use.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist

from pixal3d.modules.anchor_ray_composition import AnchorRayCompositionConditioner
from pixal3d.modules.deform_conditioner import DeformationAwareConditioner
from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def transplant_adam_prefix(optimizer, source, *, expected_step):
    """Restore common named groups by position, rejecting stale/mismatched moments."""
    current = optimizer.state_dict()
    names = [group.get('model_name') for group in current['param_groups']]
    old_names = [group.get('model_name') for group in source['param_groups']]
    if (len(names) != len(set(names)) or old_names != ['default', 'deform_conditioner']
            or not set(old_names).issubset(names) or current['state']):
        raise ValueError('fresh optimizer and exactly the two common native groups required')
    by_name = {group['model_name']: (group, live) for group, live
               in zip(current['param_groups'], optimizer.param_groups)}
    used = set()
    for old_group in source['param_groups']:
        new_group, live = by_name[old_group['model_name']]
        if len(old_group['params']) != len(new_group['params']):
            raise ValueError('common optimizer tensor count differs')
        for old_id, new_id, parameter in zip(old_group['params'], new_group['params'], live['params']):
            state = source['state'][old_id]
            if (set(state) != {'step', 'exp_avg', 'exp_avg_sq'}
                    or float(state['step']) != expected_step
                    or state['exp_avg'].shape != parameter.shape
                    or state['exp_avg_sq'].shape != parameter.shape
                    or not torch.isfinite(state['exp_avg']).all()
                    or not torch.isfinite(state['exp_avg_sq']).all()):
                raise ValueError('common Adam state shape, update count or finiteness differs')
            current['state'][new_id] = copy.deepcopy(state)
            used.add(old_id)
        ids = new_group['params']
        new_group.clear()
        new_group.update(copy.deepcopy(old_group), params=ids)
    if used != set(source['state']):
        raise ValueError('unmapped old optimizer state')
    optimizer.load_state_dict(current)
    return len(used)


def save_rng(path, *, step, sampler, loader_generator):
    payload = dict(step=step, sampler=sampler.state_dict(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
        loader_generator=loader_generator.get_state())
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_rng(path, *, step, sampler, loader_generator):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['step'] != step or state['sampler'] != sampler.state_dict():
        raise ValueError('rank RNG/sampler checkpoint does not match native checkpoint')
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    loader_generator.set_state(state['loader_generator'])


class AnchorJointTrainer(ImageConditionedProjFlowMatchingCFGTrainer):
    def __init__(self, *args, anchor_stage, **kwargs):
        self.anchor_stage = dict(anchor_stage)
        spec = self.anchor_stage
        if (spec['arm'] not in {'retargeting-control', 'topk-composer'}
                or spec['reference_step'] != 96 or kwargs['max_steps'] != 480
                or kwargs.get('prefetch_data') is not False
                or kwargs.get('mix_precision_mode') != 'amp'
                or kwargs.get('mix_precision_dtype') != 'bfloat16'
                or kwargs.get('ema_rate') != []):
            raise ValueError('declared matched epoch1-to480 native BF16 continuation required')
        self._verify_reference()
        super().__init__(*args, **kwargs)
        if not spec.get('diagnostic_profile', False):
            if (self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split) != (2, 96, 48, 1):
                raise ValueError('scientific stage requires DDP2/global96/micro48/accumulation1')
        elif spec.get('capacity_profile', False):
            if (self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split) != (2, 96, 48, 1):
                raise ValueError('reserved-pair capacity proof requires DDP2/global96/micro48/accumulation1')
        elif self.world_size != 1 or self.batch_size_per_gpu > 2 or self.batch_split != 1:
            raise ValueError('disposable cohost profile is limited to singleGPU/micro<=2')
        if self.step == 0:
            misc = torch.load(spec['misc_checkpoint'], map_location='cpu', weights_only=False)
            if misc['step'] != 96 or misc['lora'] != self.lora_config:
                raise ValueError('source epoch1 or LoRA settings differ')
            restored = transplant_adam_prefix(self.optimizer, misc['optimizer'], expected_step=96)
            if restored != 366:
                raise ValueError('all360 NN LoRA and6 R/fusion Adam states must continue')
            self.data_sampler.load_state_dict(misc['data_sampler'])
            self.grad_clip.load_state_dict(misc['grad_clip'])
            self.step = 96
            self._install_stage_schedule()
            # Native old checkpoints lack RNG. Declare the same stage reset for
            # both arms, after every model constructor and optimizer transition.
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
        if self.anchor_stage['arm'] == 'topk-composer':
            self._anchor_projection_handle = self.deform_conditioner.install_gate_projection(self.optimizer)

    def _verify_reference(self):
        spec = self.anchor_stage
        receipt = json.loads(Path(spec['reference_receipt']).read_text())
        if receipt['status'] != 'PASS' or not receipt['local_equals_actual_file4_SHA256']:
            raise ValueError('actual file4 epoch1 reference receipt required')
        closed = {row['relative_path']: row for row in receipt['files']}
        for key in ['denoiser_checkpoint', 'conditioner_checkpoint', 'misc_checkpoint']:
            path = Path(spec[key])
            relative = str(path.relative_to(Path(spec['campaign_root']).parent))
            if relative not in closed or sha(path) != closed[relative]['sha256']:
                raise ValueError('epoch1 reference bytes differ: ' + key)
        if not spec.get('diagnostic_profile', False):
            capacity = json.loads(Path(spec['capacity_proof']).read_text())
            closure = json.loads(Path(spec['capacity_receipt']).read_text())
            if (capacity['status'] != 'PASS_ACTUAL_MATCHED_ANCHOR_JOINT_SS_PROFILE'
                    or (capacity['world_size'], capacity['microbatch'], capacity['accumulation']) != (2, 48, 1)
                    or not capacity['post_update_DDP_equal']
                    or closure['status'] != 'PASS' or not closure['local_equals_actual_file4_SHA256']):
                raise ValueError('actual closed reserved-pair full-batch capacity proof required')
            key = str(Path(spec['capacity_proof']).relative_to(Path(spec['campaign_root']).parent))
            if not any(row['relative_path'] == key and row['sha256'] == sha(spec['capacity_proof'])
                       for row in closure['files']):
                raise ValueError('actual capacity proof bytes differ')
        if spec['arm'] == 'topk-composer':
            if sha(spec['utility_checkpoint']) != spec['utility_SHA256']:
                raise ValueError('declared fixed utility initializer differs')
            if not spec.get('diagnostic_profile', False):
                done = json.loads(Path(spec['utility_completed']).read_text())
                if (done['status'] != 'PASS_FIXED_SEVEN_EPOCH_TARGET_PATCH_UTILITY_TRAINING'
                        or done['epochs'] != 7 or done['optimizer_updates'] != 112
                        or done['pure_utility_export_SHA256'] != spec['utility_SHA256']):
                    raise ValueError('fixed final target-patch head required before scientific SS stage')

    def init_models_and_more(self, **kwargs):
        spec = self.anchor_stage
        if set(self.models) != {'denoiser'} or self.deform_conditioner is not None:
            raise ValueError('native complete NN continuation expected before conditioner construction')
        if self.continuation_ckpt != {'denoiser': spec['denoiser_checkpoint']}:
            raise ValueError('the two arms must initialize NN from exactly oldR96')
        config = dict(self.deform_conditioner_config)
        cls = AnchorRayCompositionConditioner if spec['arm'] == 'topk-composer' else DeformationAwareConditioner
        module = cls(**config).cuda()
        reference = torch.load(spec['conditioner_checkpoint'], map_location='cpu', weights_only=True)
        if spec['arm'] == 'topk-composer':
            module.load_reference(reference)
            utility = torch.load(spec['utility_checkpoint'], map_location='cpu', weights_only=True)
            module.ray_composer.load_state_dict(utility, strict=True)
            for parameter in module.retargeter.parameters():
                parameter.requires_grad_(True)
            module.fusion_gate.requires_grad_(True)
        else:
            module.load_state_dict(reference, strict=True)
        self.deform_conditioner = module
        self.models['deform_conditioner'] = module
        super().init_models_and_more(**kwargs)
        nn_params = [p for p in self.models['denoiser'].parameters() if p.requires_grad]
        r_params = [p for name, p in module.named_parameters() if p.requires_grad
                    and name != 'ray_mix' and not name.startswith('ray_composer.')]
        if len(nn_params) != 360 or len(r_params) != 6:
            raise ValueError('exact native360 LoRA and6 original R/fusion tensors required')
        groups = [dict(params=nn_params, model_name='default', lr=1e-5),
                  dict(params=r_params, model_name='deform_conditioner', lr=1e-4)]
        if spec['arm'] == 'topk-composer':
            groups.extend([dict(params=list(module.ray_composer.parameters()), model_name='ray_composer', lr=1e-4),
                dict(params=[module.ray_mix], model_name='ray_mix', lr=1e-2, weight_decay=0.)])
        self.optimizer = torch.optim.AdamW(groups, **self.optimizer_config['args'])
        self._install_stage_schedule()
        expected = 372 if spec['arm'] == 'topk-composer' else 366
        if sum(len(group['params']) for group in self.optimizer.param_groups) != expected:
            raise ValueError('joint optimizer tensor count differs')

    def _install_stage_schedule(self):
        # Common original groups have finished their25-update warmup. Only the
        # newly added groups receive a declared five-update stage warmup.
        functions = [lambda _: 1., lambda _: 1.]
        if self.anchor_stage['arm'] == 'topk-composer':
            functions += [lambda step: min(1., (step + 1) / 5.)] * 2
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, functions)

    def prepare_dataloader(self, **kwargs):
        super().prepare_dataloader(**kwargs)
        self._anchor_loader_generator = torch.Generator().manual_seed(101 + self.rank)
        # Iterator worker seeds must not consume the SS noise/CFG RNG on resume.
        self.dataloader.generator = self._anchor_loader_generator

    def _rng_path(self, step):
        return Path(self.output_dir) / 'stage_rng' / f'rank{self.rank}' / f'step{step:07d}.pt'

    def _save_rank_rng(self, step=None):
        step = self.step if step is None else step
        save_rng(self._rng_path(step), step=step, sampler=self.data_sampler,
            loader_generator=self._anchor_loader_generator)

    def _write_stage_initialization(self, restored, sampler):
        module = self.deform_conditioner
        report = dict(status='PASS_MATCHED_EPOCH1_NATIVE_JOINT_INITIALIZATION', rank=self.rank,
            arm=self.anchor_stage['arm'], reference_step=96, final_step=480, remaining_updates=384,
            restored_common_Adam_states=restored, common_state_updates=96,
            new_optimizer_states=0, sampler=sampler, original_R_learns=True, NN_LoRA_learns=True,
            matching_decoder_and_scalar_frozen=all(not p.requires_grad for p in module.q2c.parameters()),
            gate=float(module.ray_mix.detach()) if hasattr(module, 'ray_mix') else None,
            optimizer_groups=[dict(name=g['model_name'], tensors=len(g['params']), lr=g['lr'])
                              for g in self.optimizer.param_groups],
            stage_rng_reset=self.anchor_stage['stage_rng_seed']+self.rank,
            old_training_RNG_resume_claimed=False, auxiliary_cost_separate=True,
            scientific_training_configuration=not self.anchor_stage.get('diagnostic_profile', False))
        target = Path(self.output_dir) / f'anchor-stage-initialization.rank{self.rank}.json'
        with target.open('x') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')

    def run_step(self, data_list):
        result = super().run_step(data_list)
        if hasattr(self.deform_conditioner, 'ray_mix'):
            module = self.deform_conditioner
            weights = self._last_deform_condition.candidate_weights.detach().float()
            valid = self._last_deform_condition.query_valid
            active = weights[valid]
            diagnostics = dict(gate=float(module.ray_mix.detach()),
                utility_gradient_L1=sum(float(p.grad.detach().float().abs().sum())
                    for p in module.ray_composer.parameters() if p.grad is not None),
                gate_gradient=float(module.ray_mix.grad.detach()) if module.ray_mix.grad is not None else None)
            if len(active):
                diagnostics['effective_K'] = float((-(active*active.clamp_min(1e-30).log()).sum(-1)).exp().mean())
            if self.is_master:
                with (Path(self.output_dir)/'anchor-composition.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=self.step+1, **diagnostics))+'\n')
                if getattr(self, 'wandb_run', None):
                    self.wandb_run.log({'anchor/'+k: v for k,v in diagnostics.items() if v is not None}, step=self.step+1)
        self._save_rank_rng(self.step+1)
        return result

    def run(self):
        if self.step == 96 and not self.anchor_stage.get('diagnostic_profile', False):
            # The native hook's initial-evaluation branch only handles step0.
            from experiments.multiview_finetune.runtime import evaluate
            if self.is_master:
                evaluate(self)
            if self.world_size > 1:
                dist.barrier()
        return super().run()
