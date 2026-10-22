"""Discarded micro1 native PBR identity and three-update learnability proof.

This is neither full-microbatch capacity nor scientific training admission.
All variants start from the declared original complete MV PBR16850+LoRA.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MethodType

import torch

from .cascade_view_roles import ClosedInputs, file_sha256
from .evaluate_pbr_streaming import tensor_sha
from .frozen_projection_streaming import install_frozen_projection_streaming
from .probe_member_feature_fullbatch import _frozen_digest
from .r_mediated_native_runtime import rng_identity, state_sha


MODES = {'vanilla', 'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'}


def validate_config(cfg, original, mode):
    t, d = cfg['trainer']['args'], cfg['dataset']
    if (mode not in MODES or cfg['models'] != original['models']
            or cfg['training_seed'] != 101 or cfg['use_wandb'] is not False
            or cfg['experiment']['scientific_training_admitted'] is not False
            or cfg['experiment']['source_multiview_step'] != 16850
            or cfg['experiment']['PBR_source_mode'] != mode
            or t['batch_size_per_gpu'] != 1 or t['batch_split'] != 1
            or t['max_steps'] != 1900 or t['num_workers'] != 0
            or t['prefetch_data'] is not False or t['microbatch_to_device'] is not True
            or t['ema_rate'] != [] or t['p_uncond'] != .1
            or t['optimizer'] != dict(name='AdamW', args=dict(lr=5e-6, weight_decay=.01,
                betas=[.9, .95], eps=1e-8))
            or t['lr_scheduler'] != dict(name='LinearWarmupLRScheduler', args=dict(warmup_steps=25))
            or t['mix_precision_mode'] != 'amp' or t['mix_precision_dtype'] != 'bfloat16'
            or t['lora'] != original['trainer']['args']['lora']
            or set(t['continuation_ckpt']) != {'denoiser'} or t['finetune_ckpt'] is not None
            or d['args']['max_tokens'] != 32768 or d['args']['expected_split'] != 'train'):
        raise ValueError('original complete PBR LoRA, fresh micro1 proof and explicit non-scientific scope required')
    if mode == 'vanilla':
        if (cfg['trainer']['name'] != 'ImageConditionedProjSparseFlowMatchingCFGTrainer'
                or d['name'] != 'DeformNeXtPBR1024' or d['args']['transport_arm'] != 'v1'
                or t.get('deform_conditioner') is not None or t.get('ray_feature_utility') is not None
                or t.get('model_learning_rates', {}) != {}):
            raise ValueError('vanilla must physically omit source/utility modules')
    else:
        from pixal3d.trainers.flow_matching.r_mediated_pbr_source import pbr_source_kwargs
        pbr_source_kwargs(t['deform_conditioner'])
        if (cfg['trainer']['name'] != 'RMediatedPBRNativeTrainer'
                or d['name'] != 'DeformNeXtFeaturePBR1024'
                or d['args']['transport_arm'] != 'c1' or d['args']['task_matching_features'] is not True
                or t['deform_conditioner']['appearance_mode'] != mode):
            raise ValueError('explicit sparse native PBR source feature dataset and own trainer required')


@contextmanager
def native_only(trainer):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    previous = trainer._encode_deform_views
    def encode(self, images, camera_info, transport, *, control='normal'):
        return ImageConditionedProjMixin._encode_deform_views(self, images, camera_info, None)
    trainer._encode_deform_views = MethodType(encode, trainer)
    try:
        yield
    finally:
        trainer._encode_deform_views = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {'0', '1', '2', '3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one exclusively assigned physical GPU0-3 required')
    plan = json.loads(args.plan.read_text())
    check = ClosedInputs(plan['canonical_local_root'])
    def guard():
        check.check(args.plan, args.plan.parent / 'file4-closure.json')
        for path, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
            if file_sha256(path) != digest:
                raise ValueError('declared source/input changed: ' + path)
        for path, receipt in plan['closed_inputs'].items():
            check.check(path, receipt)
    guard()
    cfg = json.loads(Path(plan['config']).read_text())
    original = json.loads(Path(plan['original_config']).read_text())
    mode = plan['mode']
    validate_config(cfg, original, mode)
    from pixal3d import models
    from pixal3d.datasets.deformnext import DeformNeXtPBR1024
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
    from pixal3d.utils.data_utils import recursive_to_device
    from train import setup_rng
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    setup_rng(0, cfg['training_seed'])
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    install_frozen_projection_streaming(max_assets=1,
        audit_directory=output/'projection-streaming-audit')
    Dataset = DeformNeXtPBR1024 if mode == 'vanilla' else DeformNeXtFeaturePBR1024
    dataset = Dataset(cfg['data_dir'], **cfg['dataset']['args'])
    ids = [r['sample_id'] for r in dataset.rows]
    if len(ids) != len(set(ids)) or len(ids) != 9139:
        raise ValueError('complete common eligible TRAIN9139 required')
    selected = [i for i, r in enumerate(dataset.rows) if r['sample_id'] == plan['sample_id']]
    if len(selected) != 1 or dataset.rows[selected[0]]['conditioning_input_sha256'] != plan['conditioning_input_sha256']:
        raise ValueError('one predeclared actual TRAIN RGB input required')
    with torch.device('meta'):
        md = {name: getattr(models, spec['name'])(**spec['args']) for name, spec in cfg['models'].items()}
    Trainer = ImageConditionedProjSparseFlowMatchingCFGTrainer if mode == 'vanilla' else RMediatedPBRNativeTrainer
    trainer = Trainer(md, dataset, **copy.deepcopy(cfg['trainer']['args']), output_dir=str(output/'native'),
        load_dir=None, step=None, wandb_run=None)
    lora = {n: p for n, p in trainer.models['denoiser'].named_parameters() if p.requires_grad}
    source = ({} if mode == 'vanilla' else
        {n: p for n, p in trainer.deform_conditioner.named_parameters() if p.requires_grad})
    expected_source = 0 if mode == 'vanilla' else (10 if mode == 'top1-only' else 19)
    if (len(lora) != 360 or len(source) != expected_source or trainer.optimizer.state
            or trainer.step != 0 or (trainer.world_size, trainer.batch_size, trainer.batch_split) != (1, 1, 1)):
        raise ValueError('fresh native360 LoRA/source0,10,19 optimizer without continuation of old Adam required')
    initial_lora = state_sha(lora.items())
    initial_common = state_sha((n, p) for n, p in source.items() if not n.startswith('ray_readout.')) if source else None
    constructor_rng = rng_identity()
    frozen = {name: _frozen_digest(model) for name, model in trainer.models.items()}
    item = dataset[selected[0]]
    if len(item['coords']) != 31900:
        raise ValueError('declared real31900-token PBR input required')
    pack = dataset.collate_fn([item])
    pack.pop('_dataset_name', None); pack.pop('_sha256', None)
    pack = recursive_to_device(pack, trainer.device)
    target = pack.pop('x_0')
    camera_pack = dict(pack, coords=target.coords)
    camera = trainer._extract_camera_info(camera_pack)
    transport = trainer._pop_deform_transport(dict(pack))
    # Conditional engineering proof only. Scientific/native capacity dropout
    # remains .1 in the config and must be tested separately at full batch.
    trainer.p_uncond = 0.
    for model in trainer.models.values(): model.eval()
    before_encode = rng_identity()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual, negative = trainer._encode_deform_views(pack['cond'], camera, transport)
        native, native_negative = ImageConditionedProjMixin._encode_deform_views(trainer, pack['cond'], camera, None)
    equal = {key: torch.equal(value, native[key]) for key, value in actual.items()}
    equal.update({'negative_'+key: torch.equal(value, native_negative[key]) for key, value in negative.items()})
    if not all(equal.values()) or rng_identity() != before_encode:
        raise ValueError('real PBR native fields/global/negative/RNG must remain exact at cold initialization')
    predictions, losses = [], []
    handle = trainer.models['denoiser'].register_forward_hook(lambda m, a, out: predictions.append(tensor_sha(out)))
    try:
        for bypass in [False, True]:
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(80217); torch.cuda.manual_seed(80217)
                with native_only(trainer) if bypass else nullcontext():
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                        result, _ = trainer.training_losses(x_0=target, **pack)
                    losses.append(float(result['loss']))
    finally:
        handle.remove()
    if len(predictions) != 2 or predictions[0] != predictions[1] or losses[0] != losses[1]:
        raise ValueError('cold native PBR actual same-noise velocity/loss differ')
    for model in trainer.models.values(): model.train()
    trace, updates = [], []
    handle = trainer.models['denoiser'].register_forward_pre_hook(lambda m, a: trace.append(dict(
        x_t_SHA256=tensor_sha(a[0]), t_SHA256=tensor_sha(a[1]), global_SHA256=tensor_sha(a[2]['global']))))
    torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(3):
            began = time.monotonic()
            result = trainer.run_step([dict(pack, x_0=target)])
            trainer.step += 1
            gradients = {n: None if p.grad is None else float(p.grad.float().abs().sum()) for n, p in source.items()}
            if any(v is None or not math.isfinite(v) for v in gradients.values()):
                raise ValueError('all declared PBR source gradients must be present and finite')
            updates.append(dict(step=step+1, seconds=time.monotonic()-began, loss=result['loss'], source_gradient_L1=gradients))
    finally:
        handle.remove()
    if (len(trace) != 3 or len(trainer.optimizer.state) != 360+expected_source
            or {int(s['step']) for s in trainer.optimizer.state.values()} != {3}
            or any(v <= 0 for v in updates[-1]['source_gradient_L1'].values())
            or frozen != {name: _frozen_digest(model) for name, model in trainer.models.items()}):
        raise ValueError('actual native optimizer3, all source learnability and frozen state integrity required')
    guard()
    report = dict(status='PASS_ACTUAL_NATIVE_PBR_COLD_IDENTITY_AND_MICRO1_LEARNABILITY',
        mode=mode, physical_GPU=visible, sample_id=plan['sample_id'], actual_TRAIN_assets=len(dataset),
        actual_TRAIN_order_SHA256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(),
        actual_PBR_tokens=len(item['coords']), source_multiview_step=16850,
        complete_original_PBR_LoRA_continuation=True, fresh_optimizer_states=0,
        initial_LoRA_SHA256=initial_lora, initial_common_R10_SHA256=initial_common,
        post_constructor_RNG=constructor_rng, native_fields_equal=equal,
        exact_native_fields_and_encode_RNG=True, exact_same_noise_native_velocity_and_loss=True,
        same_noise_velocity_SHA256=predictions[0], same_noise_native_loss=losses[0],
        input_SHA256={key: tensor_sha(value) for key, value in dict(pack, x_0=target).items()
            if key in {'cond', 'concat_cond', 'x_0'}},
        actual_microbatch=1, actual_global_batch=1, world_size=1, accumulation=1,
        actual_optimizer_states=len(trainer.optimizer.state), optimizer_updates=3,
        conditional_engineering_dropout=0., future_native_capacity_and_science_dropout=.1,
        all_frozen_NN_matcher_and_anchor_states_exact=True, source_tensors=expected_source,
        first_three_native_noise_timestep_global_traces=trace, updates=updates,
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        temporary_weights_discarded=True, scientific_training_updates=0,
        initial_generated32_identity_proven=False, global24_microbatch_capacity_proven=False,
        scientific_training_admitted=False, quality_gain_proven=False, full_goal_complete=False,
        verified_inputs=check.inputs)
    (output/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
