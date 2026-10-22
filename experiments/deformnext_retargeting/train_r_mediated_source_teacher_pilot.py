"""Cold19 teacher pilots: matching features reach the source consumer through R."""
from pathlib import Path
import argparse
import json
import math
import os
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from .r_mediated_source_pilot import (
    MARKERS, validate_plan, build_conditioner, initial_trainable_state, assert_initial_prior,
    assert_R_only,
)
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid
from pixal3d.utils.grad_clip_utils import AdaptiveGradClipper
from .cascade_view_roles import file_sha256
from .source_feature_teacher import source_feature_teacher_loss
from .train_source_feature_teacher_pilot import TeacherPilotDataset, module_inputs, tensor_state_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--phase', choices=['capacity', 'train'], required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdigit() or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one exclusively assigned physical GPU required')
    validate_plan(plan)
    for path, digest in plan['input_SHA256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Pilot input changed: {path}')
    cfg = json.loads(Path(plan['native_config']).read_text())
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(101); torch.cuda.manual_seed_all(101)
    dataset = TeacherPilotDataset(plan, cfg)
    module = build_conditioner(cfg['trainer']['args']['deform_conditioner'],
        bound=plan['residual_logit_bound'], appearance_mode=plan['appearance_mode']).cuda()
    utility = cfg['trainer']['args']['ray_feature_utility']; verify_utility(utility)
    module.load_utility(torch.load(utility['checkpoint'], map_location='cpu', weights_only=True))
    module.train(); module.q2c.eval()
    parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
    expected = 19
    if len(parameters) != expected:
        raise ValueError('exactly nineteen source parameters and a frozen contrast marker required')
    originals = {name: value.detach().clone() for name, value in parameters.items()}
    frozen = tensor_state_sha(module.q2c.state_dict())
    common_sha = tensor_state_sha(initial_trainable_state(module))
    anchors = {k: v for k, v in module.named_buffers() if k.startswith('ray_readout.')}
    anchor_sha = tensor_state_sha(anchors)
    cold_prior = None
    gradient_probe = None
    cold_head_gradient_zero = False
    grid = ProjGrid(grid_resolution=16, image_resolution=512).cuda()
    root = Path(plan[a.phase+'_output']); root.mkdir(parents=True, exist_ok=False)
    torch.save(module.state_dict(), root/'conditioner_step0000000.pt')
    generator = torch.Generator().manual_seed(101)
    loader = DataLoader(dataset, batch_size=96, shuffle=True, drop_last=False,
        num_workers=2, prefetch_factor=1, persistent_workers=True,
        multiprocessing_context='spawn', generator=generator, pin_memory=False)
    start = time.monotonic(); updates = 0
    optimizer = (torch.optim.AdamW(list(parameters.values()), lr=1e-4, betas=(.9, .95),
        weight_decay=.01, eps=1e-8) if a.phase == 'train' else None)
    clip = AdaptiveGradClipper(max_norm=1., clip_percentile=95.)
    run = None
    if optimizer is not None:
        import wandb
        run = wandb.init(entity=plan['wandb_entity'], project=plan['wandb_project'],
            id=plan['wandb_id'], resume='never', dir=str(root), job_type='module-teacher-pilot',
            config=dict(variant=plan['variant'], TRAIN_assets=1536, TRAIN_sources=1536,
                epochs=7, global_batch=96, microbatch=96, accumulation=1,
                source_LR=1e-4, frozen_original_SS4210_evaluated_separately=True,
                validation_supervision=False, actual_joint_SS_or_PBR_training=False,
                residual_logit_bound=plan['residual_logit_bound'],
                checkpoint_schema=MARKERS[plan['appearance_mode']],
                appearance_mode=plan['appearance_mode'], feature_delivery='R-only-no-head-bypass',
                parent_teacher_warm_migration=False))
    try:
        for epoch in range(7 if optimizer is not None else 1):
            for pack in loader:
                ids = pack.pop('sample_id'); assert len(ids) == 96
                pack = {name: value.cuda() for name, value in pack.items()}
                args = module_inputs(pack, grid)
                torch.cuda.synchronize(); step_start = time.monotonic()
                module.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = module(**args)
                    assert_R_only(module, output)
                    loss = source_feature_teacher_loss(output, args['source_values'], pack['teacher'],
                        split='train', camera_ids=[9, 10])
                if cold_prior is None:
                    cold_prior = assert_initial_prior(module, output)
                if updates == 0 and source_ray_delta(output.source_numerator,
                        output.semantic_weight, args['source_values']).count_nonzero():
                    raise ValueError('cold actual GPU source correction must be exactly zero')
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite native source-feature loss')
                loss.backward()
                if any(value.grad is None or not torch.isfinite(value.grad).all() for value in parameters.values()):
                    raise ValueError('every source parameter needs a finite connected gradient')
                norm = float(clip(list(parameters.values()), error_if_nonfinite=True))
                if updates == 0:
                    for name in ['fusion_gate', 'retargeter.net.2.weight']:
                        if not parameters[name].grad.abs().sum() > 0:
                            raise ValueError('actual teacher loss must reach cold semantic gate and R')
                    cold_head_gradient_zero = not bool(parameters['ray_readout.output.weight'].grad.any())
                    if not cold_head_gradient_zero:
                        raise ValueError('zero-output R must block the first head gradient without a bypass')
                if optimizer is not None and updates == 2:
                    if any(not p.grad.abs().sum() > 0 for p in parameters.values()):
                        raise ValueError('all19 paths must open by the third real TRAIN backward')
                if optimizer is not None:
                    lr = 1e-4*min(1., (updates+1)/math.ceil(112*.05))
                    optimizer.param_groups[0]['lr'] = lr
                    optimizer.step(); updates += 1
                torch.cuda.synchronize()
                record = dict(epoch=epoch+1, step=updates, training_feature_MSE=float(loss.detach()),
                    preclip_gradient_norm=norm, compute_seconds=time.monotonic()-step_start,
                    global_batch=96, microbatch_assets=[96], accumulation=1,
                    peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3, sample_ids=ids)
                with (root/'training.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
                (root/'progress.json').write_text(json.dumps({k:v for k,v in record.items() if k!='sample_ids'}, indent=2)+'\n')
                print(json.dumps({k:v for k,v in record.items() if k!='sample_ids'}), flush=True)
                if run is not None:
                    run.log({k:v for k,v in record.items() if k!='sample_ids'}, step=updates,
                            commit=updates % 16 != 0)
                if optimizer is None:
                    # Probe the deliberate delayed head gradient with a fresh
                    # disposable module. It never becomes a scientific warm
                    # start, changes the initial checkpoint or admits NN fit.
                    before = tensor_state_sha(module.state_dict())
                    norms = []
                    with torch.random.fork_rng(devices=[0]):
                        probe = build_conditioner(cfg['trainer']['args']['deform_conditioner'],
                            bound=plan['residual_logit_bound'], appearance_mode=plan['appearance_mode']).cuda()
                        probe.load_state_dict(module.state_dict(), strict=True)
                        probe.train(); probe.q2c.eval()
                        probe_params = {n:p for n,p in probe.named_parameters() if p.requires_grad}
                        probe_optimizer = torch.optim.AdamW(list(probe_params.values()), lr=1e-4,
                            betas=(.9,.95), weight_decay=.01, eps=1e-8)
                        for probe_step in range(3):
                            probe_optimizer.zero_grad(set_to_none=True)
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                probe_output = probe(**args)
                                assert_R_only(probe, probe_output)
                                probe_loss = source_feature_teacher_loss(probe_output, args['source_values'],
                                    pack['teacher'], split='train', camera_ids=[9,10])
                            if not torch.isfinite(probe_loss):
                                raise ValueError('finite disposable TRAIN probe loss required')
                            probe_loss.backward()
                            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in probe_params.values()):
                                raise ValueError('finite connected disposable gradients required')
                            norms.append({n:float(p.grad.abs().sum()) for n,p in probe_params.items()})
                            probe_optimizer.step()
                        if not all(v>0 for v in norms[-1].values()):
                            raise ValueError('actual GPU all19 gradients must be nonzero at third probe backward')
                        del probe_output, probe_loss, probe_params, probe_optimizer, probe
                    if tensor_state_sha(module.state_dict()) != before:
                        raise ValueError('disposable probe changed actual scientific initial weights')
                    gradient_probe = dict(separate_probe_updates=3, per_backward_gradient_L1=norms,
                        all19_nonzero_finite_at_third_backward=True, scientific_initial_state_unchanged=True,
                        used_as_training_warm_start=False, scientific_optimizer_updates=0)
                    if any(not torch.equal(value, originals[name]) for name, value in parameters.items()):
                        raise ValueError('capacity pass must perform no weight update')
                    break
            if optimizer is not None:
                assert updates == (epoch+1)*16
                checkpoint = root/f'conditioner_step{updates:07d}.pt'
                torch.save(module.state_dict(), checkpoint)
                # Parent optimization pauses; release batch graphs and the
                # allocator cache before the read-only native SS child runs.
                # There is no simultaneous training or independent GPU owner.
                del pack, args, output, loss
                module.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                validation_root = root/'native-validation'/f'step{updates:07d}'
                command = [value.format(checkpoint=str(checkpoint), output=str(validation_root), step=updates)
                           for value in plan['native_validation_command']]
                with (root/f'validation-step{updates:07d}.log').open('x') as log:
                    subprocess.run(command, env=dict(os.environ), stdout=log,
                        stderr=subprocess.STDOUT, check=True)
                evaluation = json.loads((validation_root/'evaluation.json').read_text())
                with (root/'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=updates, quick_metrics=evaluation['quick_metrics'],
                        per_asset=evaluation['per_asset'], native_evaluation=str(validation_root/'evaluation.json')))+'\n')
                if run is not None:
                    import wandb
                    run.log({**{'validation/'+k:v for k,v in evaluation['quick_metrics'].items()},
                        'validation/native_SS_panel':wandb.Image(str(validation_root/'panel.png'))}, step=updates)
        assert frozen == tensor_state_sha(module.q2c.state_dict())
        assert not any(p.grad is not None for p in module.q2c.parameters())
        assert anchor_sha == tensor_state_sha(anchors)
        if optimizer is not None:
            assert updates == 112 and len(optimizer.state) == expected
            torch.save(dict(optimizer=optimizer.state_dict(), clip=clip.state_dict(),
                sampler_generator=generator.get_state(), updates=updates), root/'optimizer_final.pt')
        audit = dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_'+a.phase.upper(), variant=plan['variant'],
            actual_source_parameters=expected, initial_trainable19_SHA256=common_sha,
            residual_logit_bound=plan['residual_logit_bound'],
            actual_cold_four_head_prior=cold_prior,
            frozen_function_anchors_unchanged=True, frozen_function_anchors_SHA256=anchor_sha,
            parent_teacher_warm_migration=False, checkpoint_schema=MARKERS[plan['appearance_mode']],
            appearance_mode=plan['appearance_mode'], feature_delivery='R-only-no-head-bypass',
            retargeter_input_mode=plan['appearance_mode']+'-minus-own-native-source',
            cold_head_gradient_zero_as_required_by_R_zero=cold_head_gradient_zero,
            disposable_gradient_probe=gradient_probe,
            no_additional_trainable_source_projection=True, checkpoint_marker_required=True,
            frozen_matcher_SHA256=frozen, matcher_unchanged=True, epochs=7 if optimizer is not None else 0,
            optimizer_updates=updates, TRAIN_assets=1536, source_clusters=1536, validation_source_overlap=0,
            global_batch=96, microbatch=96, accumulation=1, physical_GPU=gpu,
            actual_original_SS_or_PBR_training=False, denoiser_not_loaded_or_updated=True,
            cached_target_DINO_projection_not_live_BF16_RGB_reencoding=True,
            actual_cold_GPU_source_delta_exact_zero=True,
            native_SS_validation_after_every_epoch=optimizer is not None,
            source_parameter_changes={name:dict(changed=int((value.detach()!=originals[name]).sum()),
                delta_L2=float((value.detach()-originals[name]).norm())) for name,value in parameters.items()},
            actual_native_SS_quality_evaluation_required=True, full9149_scope_retained=True,
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
            elapsed_seconds=time.monotonic()-start, model_adopted=False, full_goal_complete=False)
        if run is not None: audit['wandb_URL'] = run.url
        (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        print(json.dumps(audit), flush=True)
    finally:
        if run is not None: run.finish()


if __name__ == '__main__':
    main()
