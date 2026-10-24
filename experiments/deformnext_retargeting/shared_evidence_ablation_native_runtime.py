"""Native flow proof/capacity/training for physical shared-evidence controls.

Lifecycle follows r_mediated_native_runtime; the active R19 runtime is unchanged.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from types import MethodType

import numpy as np
import torch

from .shared_evidence_ablation_recipe import NAME, SCHEMA, STUDY_SCHEMA, VARIANTS, SOURCE_COUNTS, build_trainer, validate_config
from contextlib import nullcontext


def cold_source_binding(variant):
    if variant not in VARIANTS:
        raise ValueError("consistent raw evidence variant required")
    return nullcontext()


def tensor_sha(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
    digest = hashlib.sha256((str(value.dtype)+repr(tuple(value.shape))).encode())
    digest.update(raw.tobytes())
    return digest.hexdigest()


def state_sha(items):
    digest = hashlib.sha256()
    for name, value in items:
        digest.update(name.encode())
        digest.update(tensor_sha(value).encode())
    return digest.hexdigest()


def rng_identity():
    return dict(CPU=tensor_sha(torch.random.get_rng_state()), CUDA=tensor_sha(torch.cuda.get_rng_state()),
        numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())


def dataset_identity(dataset):
    ids=[r['sample_id']for r in dataset.rows]
    if len(dataset)!=9149 or len(ids)!=9149 or len(set(ids))!=9149:
        raise ValueError('exact complete unique TRAIN9149 required, no transport-specific asset filtering')
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


@contextmanager
def native_condition_only(trainer):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    previous = trainer._encode_deform_views
    def native(self, images, camera, transport, *, control='normal'):
        return ImageConditionedProjMixin._encode_deform_views(self, images, camera, None)
    trainer._encode_deform_views = MethodType(native, trainer)
    try:
        yield
    finally:
        trainer._encode_deform_views = previous


def selected_trainer(variant):
    if variant not in VARIANTS:
        raise ValueError("consistent raw evidence trainer required")
    return build_trainer(variant)


def parameter_groups(trainer, variant):
    from pixal3d.modules.shared_evidence_ablation import SharedEvidenceAblationConditioner, UnprioritizedRawRayReadout
    if variant not in VARIANTS:
        raise ValueError('explicit own shared-evidence control required')
    parent, mode, control, count = VARIANTS[variant]
    lora = {n:p for n,p in trainer.models['denoiser'].named_parameters() if p.requires_grad}
    module = trainer.deform_conditioner
    source = {n:p for n,p in module.named_parameters() if p.requires_grad}
    if (len(lora) != 360 or any('lora_' not in n for n in lora) or len(source) != count
            or not isinstance(module, SharedEvidenceAblationConditioner)
            or module.ablation != control or module.evidence_mode != mode
            or (module.retargeter is None) != (control == 'without-retargeting')
            or any(p.requires_grad for p in module.q2c.parameters())):
        raise ValueError('own original360 LoRA, physical control and frozen matcher required')
    head = module.ray_readout
    if mode == 'fixed-prior':
        if head is not None:raise ValueError('fixed controls must physically omit learned QK')
    elif (head is None or len(list(head.parameters())) != 7
            or any(hasattr(head,n) for n in ['value','output','residual_logit_bound'])
            or isinstance(head,UnprioritizedRawRayReadout) != (control == 'without-logit-prior')):
        raise ValueError('same actual QK/utility7, identity V and declared logit-prior control required')
    return lora, source


def source_metadata(module):
    return dict(learned_projection_tensors=0,
        learned_attention_head_tensors=7 if module.ray_readout is not None else 0,
        actual_initial_retained_parameters_equal_parent=module._actual_initial_retained_parameters_equal_parent,
        parent_initial_core10_equal_Top1=module._parent_initial_core10_equal_Top1,
        shared_evidence_consumers=['carrier','source_use']+([] if module.retargeter is None else ['retargeter']),
        retargeter_present=module.retargeter is not None, residual_logit_bound=None)


def study_metadata(cfg, variant):
    return dict(conditioning_schema=SCHEMA, ablation_study_schema=STUDY_SCHEMA,
        ablation_control=VARIANTS[variant][2], ablation_parent_variant=VARIANTS[variant][0],
        ablation_source_SHA256=cfg['experiment']['ablation_source_SHA256'])


def validate_own_admission(record, cfg, variant, *, phase, initial):
    status = ('PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD' if phase == 'proof' else
              'PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY')
    if (record.get('status') != status or record.get('variant') != variant
            or record.get('source_tensors') != SOURCE_COUNTS[variant] or record.get('LoRA_tensors') != 360
            or record.get('actual_initial_retained_parameters_equal_parent') is not True
            or record.get('retargeter_present') is not (VARIANTS[variant][2] != 'without-retargeting')
            or record.get('learned_attention_head_tensors') != (7 if VARIANTS[variant][1] == 'learned-anchored' else 0)
            or any(record.get(k) != v for k,v in study_metadata(cfg,variant).items())
            or any(record.get(k) != initial[k] for k in ['initial_LoRA_SHA256','initial_source_SHA256'])):
        raise ValueError('own same-source-schema actual GPU proof/capacity required; parent proof cannot substitute')
    if phase == 'proof':
        if (record.get('optimizer_updates') != 0 or record.get('scientific_training_updates') != 0
                or not record.get('fields_equal_to_native') or not all(record['fields_equal_to_native'].values())
                or record.get('actual_initial_head_probability_matches_declared_rule') is not
                    (None if VARIANTS[variant][1] == 'fixed-prior' else True)
                or VARIANTS[variant][2] == 'uniform-k' and record.get('actual_UniformK_candidate_weights_exact') is not True):
            raise ValueError('actual unchanged native cold condition and declared probability rule required')
    elif (record.get('temporary_optimizer_updates') != 3 or record.get('scientific_training_updates') != 0
            or record.get('actual_optimizer_states') != 360+SOURCE_COUNTS[variant]
            or record.get('actual_optimizer_steps') != [3]
            or [record.get(k) for k in ['world_size','global_batch','microbatch','accumulation']] != [1,96,96,1]
            or record.get('all_source_gradients_positive_finite_at_update3') is not True
            or record.get('frozen_parameters_buffers_exact') is not True
            or record.get('actual_initial_native32_recorded_identity') is not True):
        raise ValueError('actual micro96/accum1 three-update capacity and frozen-base evidence required')


def proof(cfg, args, variant):
    from pixal3d import datasets, models
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.utils.data_utils import recursive_to_device
    from train import setup_rng
    setup_rng(0, cfg['training_seed'])
    torch.use_deterministic_algorithms(True)
    dataset = getattr(datasets, cfg['dataset']['name'])(cfg['data_dir'], **cfg['dataset']['args'])
    manifest_order=dataset_identity(dataset)
    indices = [i for i, r in enumerate(dataset.rows) if r['sample_id'] == args.sample_id]
    if (len(indices) != 1 or dataset.rows[indices[0]]['split'] != 'train'
            or dataset.rows[indices[0]]['conditioning_input_sha256'] != args.expected_conditioning_sha256):
        raise ValueError('predeclared exact TRAIN input required')
    with torch.device('meta'):
        md = {n: getattr(models, s['name'])(**s['args']) for n, s in cfg['models'].items()}
    kw = copy.deepcopy(cfg['trainer']['args'])
    kw.update(batch_size_per_gpu=1, batch_split=1, max_steps=1, num_workers=0, prefetch_data=False)
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    Trainer = selected_trainer(variant)
    with cold_source_binding(variant):
        trainer = Trainer(md, dataset, **kw, output_dir=str(out/'native-trainer'),load_dir=None,step=None,wandb_run=None)
    ctor_rng = rng_identity()
    lora, source = parameter_groups(trainer, variant)
    if trainer.step != 0 or trainer.world_size != 1 or trainer.optimizer.state:
        raise ValueError('fresh native optimizer required')
    before = state_sha(list(lora.items()) + list(source.items()))
    initial_lora = state_sha(lora.items())
    initial_source = state_sha(source.items()) if source else None
    for model in trainer.models.values(): model.eval()
    trainer.p_uncond = 0
    pack = dataset.collate_fn([dataset[indices[0]]])
    pack = recursive_to_device(pack if isinstance(pack,dict) else pack[0],trainer.device)
    pack.pop('_dataset_name',None);pack.pop('_sha256',None)
    target = pack.pop('x_0')
    camera = trainer._extract_camera_info(dict(pack))
    transport = trainer._pop_deform_transport(dict(pack))
    start_rng = rng_identity()
    with torch.no_grad(), torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
        actual, negative = trainer._encode_deform_views(pack['cond'],camera,transport)
        native, native_negative = ImageConditionedProjMixin._encode_deform_views(trainer,pack['cond'],camera,None)
    equality = {n:torch.equal(v,native[n]) for n,v in actual.items()}
    equality.update({'negative_'+n:torch.equal(v,native_negative[n]) for n,v in negative.items()})
    if not all(equality.values()) or rng_identity() != start_rng:
        raise ValueError('actual cold fields/global/negative/RNG must equal native')
    module = trainer.deform_conditioner
    if not torch.equal(module.last_R_input_appearance,
                       module.last_selected_evidence.flatten(0, 1)[module.last_source_reference_rows]):
        raise ValueError('actual selected source evidence and retargeting input differ')
    head_rule_exact = None
    if module.ray_readout is not None:
        head = module.ray_readout
        if VARIANTS[variant][2] == 'without-logit-prior':
            from pixal3d.modules.shared_evidence_ablation import uniform_probabilities
            head_rule_exact = bool(torch.equal(head.last_probability, uniform_probabilities(head.last_prior[:,0] > 0)))
        else:
            head_rule_exact = head.first_forward_prior_exact
        if head_rule_exact is not True:raise ValueError('actual initial head differs from its declared probability rule')
    uniform_exact = None
    if VARIANTS[variant][2] == 'uniform-k':
        from pixal3d.modules.shared_evidence_ablation import uniform_probabilities
        attn = trainer._last_deform_condition
        valid = transport['source_valid'].flatten(1).gather(1,attn.candidate_indices.flatten(1)).reshape_as(attn.candidate_indices)
        valid = valid & attn.query_valid[...,None]
        expected = uniform_probabilities(valid.flatten(0,1)).mean(1).reshape_as(attn.candidate_weights)
        uniform_exact = bool(torch.equal(attn.candidate_weights,expected))
        if not uniform_exact:raise ValueError('actual GPU Uniform-K changed the declared valid support')
    if module.retargeter is None and trainer._last_deform_condition.retarget_residual.count_nonzero():
        raise ValueError('physically absent R must not produce a correction')
    predictions, losses = [], []
    handle = trainer.models['denoiser'].register_forward_hook(lambda m, inp, out: predictions.append(tensor_sha(out)))
    try:
        for bypass in [False, True]:
            with torch.random.fork_rng(devices=[trainer.device]):
                torch.manual_seed(80217);torch.cuda.manual_seed(80217)
                context = native_condition_only(trainer) if bypass else _null_context()
                with context, torch.no_grad(), torch.autocast('cuda',dtype=trainer.mix_precision_dtype):
                    result,_ = trainer.training_losses(x_0=target,**pack)
                losses.append(float(result['loss']))
    finally:
        handle.remove()
    if len(predictions) != 2 or predictions[0] != predictions[1] or losses[0] != losses[1]:
        raise ValueError('same-noise actual native velocity and loss differ at cold initialization')
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        with torch.random.fork_rng(devices=[trainer.device]):
            torch.manual_seed(80217);torch.cuda.manual_seed(80217)
            with torch.autocast('cuda',dtype=trainer.mix_precision_dtype): result,_=trainer.training_losses(x_0=target,**pack)
            result['loss'].backward()
    finally:
        torch.use_deterministic_algorithms(deterministic)
    gradients = {n:None if p.grad is None else float(p.grad.float().abs().sum()) for n,p in source.items()}
    nn_gradient = sum(float(p.grad.float().abs().sum()) for p in lora.values() if p.grad is not None)
    if not math.isfinite(float(result['loss'])) or not nn_gradient > 0:
        raise ValueError('actual native flow loss must reach original LoRA')
    if source:
        if (not gradients['fusion_gate'] > 0 or (module.retargeter is not None and not gradients['retargeter.net.2.weight'] > 0)
                or any(v is None or not math.isfinite(v) for v in gradients.values())):
            raise ValueError('cold source R/fusion gradient must be positive')
        if any(v != 0. for n, v in gradients.items() if n.startswith('ray_readout.')):
            raise ValueError('cold zero-fusion/zero-R head gradient must be exactly zero')
    if trainer.optimizer.state or before != state_sha(list(lora.items())+list(source.items())):
        raise ValueError('proof must perform no optimizer update or weight mutation')
    report = dict(status='PASS_ACTUAL_COLD_NATIVE_SS_FLOW_IDENTITY_AND_BACKWARD', variant=variant,
        sample_id=args.sample_id,conditioning_input_sha256=args.expected_conditioning_sha256,
        TRAIN_assets=len(dataset),TRAIN_order_SHA256=manifest_order,probe_batch=1,source_tensors=len(source),LoRA_tensors=len(lora),
        fields_equal_to_native=equality,actual_CPU_CUDA_numpy_python_RNG_preserved=True,
        same_noise_velocity_SHA256=predictions[0],same_noise_native_flow_loss=losses[0],
        actual_native_loss=float(result['loss']),source_gradient_L1=gradients,LoRA_gradient_L1=nn_gradient,
        **source_metadata(trainer.deform_conditioner),**study_metadata(cfg,variant),initial_LoRA_SHA256=initial_lora,
        actual_selected_evidence_equals_R_appearance=None if module.retargeter is None else True,
        actual_initial_head_probability_matches_declared_rule=head_rule_exact,
        actual_UniformK_candidate_weights_exact=uniform_exact,
        actual_initial_head_probability_equals_frozen_prior=(None if module.ray_readout is None
            else module.ray_readout.first_forward_prior_exact),
        initial_source_SHA256=initial_source,post_constructor_RNG=ctor_rng,
        optimizer_updates=0,scientific_training_updates=0,
        actual_micro96_capacity_not_proven=True,quality_gain_not_proven=True,
        model_adopted=False,full_goal_complete=False)
    (out/'proof.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


@contextmanager
def _null_context():
    yield


def native_training(cfg, phase, variant):
    from pixal3d import trainers
    from experiments.multiview_finetune import runtime
    from .probe_member_feature_fullbatch import _frozen_digest
    from easydict import EasyDict
    import train as native
    Parent = selected_trainer(variant)

    class Scientific(Parent):
        def run(self):
            lora, source = parameter_groups(self,variant)
            if ((self.world_size,self.batch_size,self.batch_size_per_gpu,self.batch_split,self.step)!=(1,96,96,1,0)
                    or self.optimizer.state or self._native_ffn_chunk_report['chunk_assets']!=cfg['trainer']['args']['ffn_chunk_assets']):
                raise ValueError('fresh native world1/micro96/accum1 and declared FFN chunks required')
            out = Path(self.output_dir)
            initial = dict(initial_LoRA_SHA256=state_sha(lora.items()),initial_source_SHA256=state_sha(source.items())if source else None,
                post_constructor_RNG=rng_identity(),source_teacher_updates=0,TRAIN_order_SHA256=dataset_identity(self.dataset))
            own_proof=json.loads(Path(cfg['experiment']['cold_proof_json']).read_text())
            validate_own_admission(own_proof,cfg,variant,phase='proof',initial=initial)
            if phase=='science':
                own_capacity=json.loads(Path(cfg['experiment']['micro96_capacity_json']).read_text())
                validate_own_admission(own_capacity,cfg,variant,phase='capacity',initial=initial)
            self._paired_trace_active=False;self._paired_trace=[]
            def trace(module,inputs):
                if self._paired_trace_active:
                    self._paired_trace.append(dict(x_t_SHA256=tensor_sha(inputs[0]),t_SHA256=tensor_sha(inputs[1]),
                        global_SHA256=tensor_sha(inputs[2]['global'])))
            self.models['denoiser'].register_forward_pre_hook(trace)
            startup=dict(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS_STARTUP',variant=variant,**initial,
                source_tensors=len(source),**source_metadata(self.deform_conditioner),**study_metadata(cfg,variant),LoRA_tensors=len(lora),global_batch=96,microbatch=96,accumulation=1,
                actual_optimizer_states=0,ffn_chunks=self._native_ffn_chunk_report)
            (out/'startup.json').write_text(json.dumps(startup,indent=2)+'\n')
            self._paired_reference_trace=None
            if phase=='science':
                reference_root=Path(cfg['experiment']['paired_reference_native_root'])
                reference=json.loads((reference_root/'startup.json').read_text())
                for key in ['initial_LoRA_SHA256','post_constructor_RNG','TRAIN_order_SHA256']:
                    if initial[key]!=reference[key]:
                        raise ValueError('actual scientific startup differs from paired V1: '+key)
                for name in ['cold_proof_json','micro96_capacity_json']:
                    own=json.loads(Path(cfg['experiment'][name]).read_text())
                    if any(initial[key]!=own[key] for key in ['initial_LoRA_SHA256','initial_source_SHA256']):
                        raise ValueError('actual scientific initial weights differ from own GPU proof/capacity')
                self._paired_reference_trace=[json.loads(line) for line in
                    (reference_root/'training-noise-trace.jsonl').read_text().splitlines()[:5]]
                if len(self._paired_reference_trace)!=5:
                    raise ValueError('actual completed V1 five-update reference required')
                frozen_before={n:_frozen_digest(m) for n,m in self.models.items()}
                result=super().run()
                if self.step==480:
                    frozen_after={n:_frozen_digest(m) for n,m in self.models.items()}
                    if frozen_before!=frozen_after:
                        raise ValueError('science changed frozen base or matcher parameters/buffers')
                    (out/'final-frozen-base-matcher.json').write_text(json.dumps(dict(
                        status='PASS_ACTUAL_SHARED_EVIDENCE_ABLATION_NATIVE480_FROZEN_BASE_MATCHER',
                        variant=variant,step=480,actual_frozen_parameters_buffers_exact=True,
                        frozen_SHA256=frozen_after,source_tensors=len(source),
                        actual_optimizer_states=len(self.optimizer.state)),indent=2)+'\n')
                return result
            # Reproduce the scientific initial-validation allocation and RNG
            # lifecycle before measuring fullbatch training capacity.
            self._runtime_config=cfg
            self._runtime_total_steps=480
            self._runtime_validation_every=96
            self._runtime_checkpoint_step=None
            self._runtime_last_row=None
            runtime.evaluate(self)
            before={n:_frozen_digest(m)for n,m in self.models.items()}
            snapshots={id(p):p.detach().cpu().clone()for p in self.model_params}
            updates=[]
            for _ in range(3):
                began=time.monotonic();value=self.run_step(self.load_data());self.step+=1
                gradients={n:None if p.grad is None else float(p.grad.float().abs().sum())for n,p in source.items()}
                if not math.isfinite(value['loss']['loss']):raise ValueError('nonfinite native loss')
                updates.append(dict(step=self.step,loss=value['loss'],source_gradient_L1=gradients,
                    seconds=time.monotonic()-began,runtime=dict(self._runtime_last_row)))
            after={n:_frozen_digest(m)for n,m in self.models.items()}
            expected=360+len(source);steps=sorted({int(v['step'])for v in self.optimizer.state.values()})
            changed={group.get('model_name','default'):sum(not torch.equal(p.detach().cpu(),snapshots[id(p)])for p in group['params'])for group in self.optimizer.param_groups}
            if (before!=after or len(self.optimizer.state)!=expected or steps!=[3]
                    or changed.get('default',0)==0
                    or (source and (changed.get('deform_conditioner',0)==0 or any(v is None or not math.isfinite(v)or v<=0 for v in gradients.values())))):
                raise ValueError('actual native3-update gradients/optimizer/frozen-integrity proof failed')
            report=dict(status='PASS_ACTUAL_COLD_NATIVE_TASK_SS96_THREE_UPDATE_CAPACITY',variant=variant,**initial,
                global_batch=96,microbatch=96,accumulation=1,world_size=1,source_tensors=len(source),LoRA_tensors=360,
                **source_metadata(self.deform_conditioner),**study_metadata(cfg,variant),
                actual_initial_native32_recorded_identity=True,
                temporary_optimizer_updates=3,scientific_training_updates=0,scientific_checkpoint_exported=False,
                all_source_gradients_positive_finite_at_update3=True if source else None,
                actual_optimizer_states=len(self.optimizer.state),actual_optimizer_steps=steps,changed_tensors=changed,
                frozen_parameters_buffers_exact=True,peak_allocated_GiB=max(r['runtime']['peak_allocated_gib']for r in updates),
                peak_reserved_GiB=max(r['runtime']['reserved_gib']for r in updates),updates=updates,
                model_adopted=False,full_goal_complete=False)
            (out/'capacity.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({k:v for k,v in report.items()if k!='updates'}),flush=True)

        def run_step(self,packs):
            self._paired_trace_active=True;self._paired_trace=[]
            ids=[pack.get('_sha256',[])for pack in packs]
            try:result=super().run_step(packs)
            finally:self._paired_trace_active=False
            if len(self._paired_trace)!=1:
                raise ValueError('one native fullbatch forward per update required')
            record=dict(step=self.step+1,asset_ids=ids,**self._paired_trace[0])
            with(Path(self.output_dir)/'training-noise-trace.jsonl').open('a')as stream:stream.write(json.dumps(record)+'\n')
            if phase=='science' and self.step<5:
                if record!=self._paired_reference_trace[self.step]:
                    raise ValueError('actual scientific assets/noise/timestep/global input differ from paired V1')
                if self.step==4:
                    (Path(self.output_dir)/'paired-V1-startup-and-five-update-prefix.json').write_text(json.dumps(dict(
                        status='PASS_ACTUAL_V1_STARTUP_AND_FIVE_TRAIN_UPDATE_PREFIX',
                        actual_LoRA_RNG_TRAIN_order_equal=True, own_GPU_proof_capacity_weights_equal=True,
                        actual_first_five_assets_noise_timesteps_global_inputs_equal=True,
                        reference_native_root=cfg['experiment']['paired_reference_native_root'],
                        quality_gain_proven=False,full_goal_complete=False),indent=2)+'\n')
            if phase=='science'and variant!='vanilla'and(self.step+1)%10==0:
                module=self.deform_conditioner
                grad=lambda parameters:sum(float(p.grad.detach().float().abs().sum())for p in parameters if p.grad is not None)
                layers=[('source_use',module.evidence_use),('frozen_q2c',module.q2c)]
                if module.retargeter is not None:layers.append(('retargeter',module.retargeter))
                if module.ray_readout is not None:
                    layers.extend([('query',module.ray_readout.query),('key',module.ray_readout.key),
                                   ('utility',module.ray_readout.score_selector)])
                values={f'native_source/{name}_gradient_L1':grad(layer.parameters()) for name,layer in layers}
                head=module.ray_readout
                if head is not None and head.last_probability is not None:
                    p,q=head.last_probability,head.last_prior
                    plog=p.clamp_min(1e-30).log();qlog=q.clamp_min(1e-30).log()
                    values.update({
                        'native_source/attention_entropy_last_chunk':float(-(p*plog).sum(-1).mean()),
                        'native_source/attention_prior_KL_last_chunk':float((p*(plog-qlog)).sum(-1).mean()),
                        'native_source/attention_prior_Top1_switch_fraction_last_chunk':
                            float((p.argmax(-1)!=q.argmax(-1)).float().mean())})
                values['native_source/fusion_gradient_L1']=(0. if module.fusion_gate.grad is None
                    else float(module.fusion_gate.grad.detach().float().abs().sum()))
                with(Path(self.output_dir)/'source-native-gradients.jsonl').open('a')as stream:
                    stream.write(json.dumps(dict(step=self.step+1,**values))+'\n')
                if self.wandb_run is not None:self.wandb_run.log(values,step=self.step+1)
            return result

    # Initial validation is completed and checked BEFORE the native run loop.
    previous_evaluate=runtime.evaluate
    def evaluate_and_check(trainer):
        previous_evaluate(trainer)
        if trainer.step==0:
            value=json.loads((Path(trainer.output_dir)/'validation.jsonl').read_text().splitlines()[-1])
            reference=json.loads(Path(cfg['experiment']['initial_reference_evaluation']).read_text())
            if value['quick_per_asset']!=reference['per_asset'] or value['quick_metrics']!=reference['quick_metrics']:
                raise ValueError('initial32 GT/noise/counts/metrics differ from original MV4 before first update')
            (Path(trainer.output_dir)/'initial-native32-identity.json').write_text(json.dumps(dict(
                status='PASS_ACTUAL_INITIAL_NATIVE32_RECORDED_METRIC_IDENTITY',assets=32,
                raw_voxel_masks_not_archived=True,optimizer_updates=0,reference=cfg['experiment']['initial_reference_evaluation']),indent=2)+'\n')
    runtime.evaluate=evaluate_and_check
    previous=trainers.__dict__.get(NAME)
    setattr(trainers,NAME,Scientific)
    try:
        out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=False)
        (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
        native.main(0,EasyDict(cfg))
    finally:
        runtime.evaluate=previous_evaluate
        if previous is None:delattr(trainers,NAME)
        else:setattr(trainers,NAME,previous)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['proof','capacity','science'],required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--sample-id');p.add_argument('--expected-conditioning-sha256')
    args=p.parse_args();cfg=json.loads(args.config.read_text());variant=validate_config(cfg,args.phase)
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in cfg['experiment']['assigned_GPUs'] or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one assigned physical GPU required')
    if args.phase=='proof'and(not args.sample_id or not args.expected_conditioning_sha256):raise ValueError('predeclared TRAIN input required')
    os.sched_setaffinity(0,set(cfg['experiment']['CPU_affinity_per_GPU'][gpu]))
    from experiments.multiview_finetune.runtime import install_hooks
    torch.cuda.set_device(0);torch.set_num_threads(2);install_hooks()
    if args.phase=='proof':proof(cfg,args,variant)
    else:
        with cold_source_binding(variant):native_training(cfg,args.phase,variant)


if __name__=='__main__':main()
