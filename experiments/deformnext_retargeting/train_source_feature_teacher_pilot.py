"""A fixed TRAIN-only conditioner fitting pilot, separate from joint SS training."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.source_referenced_member_feature import SourceReferencedMemberFeatureConditioner
from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.ray_group_query import native_ray_membership
from pixal3d.trainers.flow_matching.member_feature_source_ray import verify_utility
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
    ProjGrid, compute_multiview_projection_matrices,
)
from pixal3d.utils.grad_clip_utils import AdaptiveGradClipper
from .cascade_view_roles import file_sha256
from .source_feature_teacher import source_feature_teacher_loss, validate_teacher_records


class TeacherPilotDataset(Dataset):
    def __init__(self, plan, cfg):
        self.rows = [json.loads(line) for line in Path(plan['train_manifest']).read_text().splitlines()]
        validation = [json.loads(line) for line in Path(plan['validation_manifest']).read_text().splitlines()]
        validate_teacher_records(self.rows, {r['source_cluster_id'] for r in validation})
        self.root = Path(plan['teacher_cache'])
        self.index = json.loads((self.root/'index.json').read_text())
        if len(self.rows) != 1536 or [r['sample_id'] for r in self.rows] != [r['sample_id'] for r in self.index]:
            raise ValueError('same fixed TRAIN1536 and actual encoded teacher index required')
        self.prefix = Path(plan['local_mirror_prefix'])
        self.projection_root = Path(cfg['dataset']['args']['projection_root'])

    def __len__(self):
        return len(self.rows)

    def local(self, canonical):
        if not canonical.startswith('/file4/'):
            raise ValueError('explicit file4 canonical input required')
        return self.prefix/canonical.lstrip('/')

    def __getitem__(self, index):
        row, entry = self.rows[index], self.index[index]
        teacher_path = self.root/entry['relative_path']
        if file_sha256(teacher_path) != entry['SHA256']:
            raise ValueError('TRAIN teacher cache changed')
        with np.load(teacher_path) as z:
            if (str(z['schema']) != 'source-feature-teacher-TRAIN-v1' or str(z['split']) != 'train'
                    or bool(z['not_for_training']) or str(z['sample_id']) != row['sample_id']
                    or str(z['source_cluster_id']) != row['source_cluster_id']
                    or str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']
                    or z['camera_ids'].tolist() != [9, 10]):
                raise ValueError('TRAIN-only teacher identity must match original input row')
            teacher = torch.from_numpy(z['teacher_source9_10_FP16'].copy())
        with np.load(self.local(row['transport_cache'])) as z:
            if str(z['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('original native/Omega source cache identity changed')
            data = {name: torch.from_numpy(z[name].copy())
                    for name in ['source_values', 'source_valid', 'target_query']}
            # The verified2048-D layout is [FRAME,GLOBAL]; matcher uses FRAME.
            if z['target_keys'].shape != (1024, 2048) or z['source_keys'].shape != (3, 1024, 2048):
                raise ValueError('original FRAME+GLOBAL Omega descriptors required')
            data['target_keys'] = torch.from_numpy(z['target_keys'][..., :1024].copy())
            data['source_keys'] = torch.from_numpy(z['source_keys'][..., :1024].copy())
        with np.load(self.projection_root/(row['sample_id']+'.npz')) as z:
            if str(z['schema']) != 'deformnext-anchor-projection-v2':
                raise ValueError('world-v4 native projection override required')
            data['patch_grid'] = torch.from_numpy(z['patch_grid'].copy()).long().reshape(-1)
        with Image.open(self.local(row['target_anchor'])) as image:
            alpha = np.asarray(image.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
        data.update(target_valid=torch.from_numpy((alpha[8::16, 8::16] > 127).copy()).flatten(),
            teacher=teacher, matrices=torch.tensor(row['transform_matrix'], dtype=torch.float32),
            fovs=torch.tensor(row['camera_angle_x'], dtype=torch.float32),
            scale=torch.tensor(row['mesh_scale'], dtype=torch.float32), sample_id=row['sample_id'])
        return data


def module_inputs(pack, grid):
    """Same camera/query construction as the CPU replay, without a denoiser."""
    tq = pack['target_query'].float()
    tk, sk = pack['target_keys'].float(), pack['source_keys'].float()
    matrices, fov, scale = pack['matrices'], pack['fovs'], pack['scale']
    distance = matrices[:, :, :3, 3].norm(dim=-1)
    transforms, _ = compute_multiview_projection_matrices(matrices, distance, grid.front_view_transform_matrix)
    target_native = grid(tq.reshape(-1, 32, 32, 1024), fov[:, 0], distance[:, 0], scale, transforms[:, 0])
    rays = native_ray_membership(grid, transform_matrix=matrices, distance=distance,
        camera_angle_x=fov, mesh_scale=scale, views=(1, 2, 3))
    patch = pack['patch_grid']
    gather = patch[..., None].expand(-1, -1, 1024)
    return dict(native_voxels=target_native, target_omega_voxels=tk.gather(1, gather),
        target_native_voxels=tq.gather(1, gather), frame_target_keys=tk, target_patch_ids=patch,
        ray_ids=rays, observed_query=pack['target_valid'].gather(1, patch), source_keys=sk,
        source_values=pack['source_values'].float(), source_valid=pack['source_valid'])


def tensor_state_sha(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--phase', choices=['capacity', 'train'], required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not gpu.isdigit() or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one exclusively assigned physical GPU required')
    if (plan['variant'] not in ['original-NoDQ', 'source-reference-NoDQ']
            or plan['epochs'] != 7 or plan['global_batch'] != 96 or plan['seed'] != 101):
        raise ValueError('two declared same-budget7epoch96/m96/a1 pilots required')
    for path, digest in plan['input_SHA256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Pilot input changed: {path}')
    cfg = json.loads(Path(plan['native_config']).read_text())
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.manual_seed(101); torch.cuda.manual_seed_all(101)
    dataset = TeacherPilotDataset(plan, cfg)
    selected = (SourceReferencedMemberFeatureConditioner if plan['variant'] == 'source-reference-NoDQ'
                else MemberFeatureSourceRayConditioner)
    module = selected(**cfg['trainer']['args']['deform_conditioner']).cuda()
    utility = cfg['trainer']['args']['ray_feature_utility']; verify_utility(utility)
    module.load_utility(torch.load(utility['checkpoint'], map_location='cpu', weights_only=True))
    module.train(); module.q2c.eval()
    parameters = {name: value for name, value in module.named_parameters() if value.requires_grad}
    expected = 20 if plan['variant'] == 'source-reference-NoDQ' else 19
    if len(parameters) != expected:
        raise ValueError('explicit nineteen/twenty source parameters required')
    originals = {name: value.detach().clone() for name, value in parameters.items()}
    frozen = tensor_state_sha(module.q2c.state_dict())
    common_state = {k: v for k, v in module.state_dict().items() if k != 'retargeter.source_projection.weight'}
    common_sha = tensor_state_sha(common_state)
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
                validation_supervision=False, actual_joint_SS_or_PBR_training=False))
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
                    loss = source_feature_teacher_loss(output, args['source_values'], pack['teacher'],
                        split='train', camera_ids=[9, 10])
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
                    for name in ['fusion_gate', 'retargeter.net.2.weight', 'ray_readout.output.weight']:
                        if not parameters[name].grad.abs().sum() > 0:
                            raise ValueError('actual teacher loss must reach all three cold output paths')
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
        if optimizer is not None:
            assert updates == 112 and len(optimizer.state) == expected
            torch.save(dict(optimizer=optimizer.state_dict(), clip=clip.state_dict(),
                sampler_generator=generator.get_state(), updates=updates), root/'optimizer_final.pt')
        audit = dict(status='PASS_ACTUAL_SOURCE_FEATURE_TEACHER_'+a.phase.upper(), variant=plan['variant'],
            actual_source_parameters=expected, common_initial_state_SHA256=common_sha,
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
            elapsed_seconds=time.monotonic()-start, model_adopted=False, full_goal_complete=False)
        if run is not None: audit['wandb_URL'] = run.url
        (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        print(json.dumps(audit), flush=True)
    finally:
        if run is not None: run.finish()


if __name__ == '__main__':
    main()
