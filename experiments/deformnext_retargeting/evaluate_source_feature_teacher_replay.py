"""Read-only validation feature diagnostics of a trained source-teacher pilot.

Uses existing validation-only diagnostic labels, never calls a training loss,
creates no optimizer, and does not substitute for actual SS generation metrics.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np
from PIL import Image
import torch

from pixal3d.modules.native_prior_source_ray import source_ray_delta
from pixal3d.modules.member_feature_source_ray import MemberFeatureSourceRayConditioner
from pixal3d.modules.source_referenced_member_feature import SourceReferencedMemberFeatureConditioner
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ProjGrid
from .train_source_feature_teacher_pilot import module_inputs, tensor_state_sha


def moments(prediction, desired, mask):
    if prediction.shape != desired.shape or mask.shape != desired.shape[:-1] or mask.dtype != torch.bool:
        raise ValueError('aligned feature changes and explicit reporting mask required')
    p, t = prediction[mask].double(), desired[mask].double()
    return dict(elements=p.numel(), tokens=len(p), desired_squared=float(t.square().sum()),
        error_squared=float((p-t).square().sum()), movement_squared=float(p.square().sum()), dot=float((p*t).sum()))


def summarize(raw):
    n, e, p = raw['elements'], raw['desired_squared'], raw['movement_squared']
    return dict(**raw, original_MSE=e/n if n else None, corrected_MSE=raw['error_squared']/n if n else None,
        MSE_ratio=raw['error_squared']/e if e else None,
        norm_ratio=(p/e)**.5 if e else None, cosine=raw['dot']/(p*e)**.5 if p*e else None)


def validate_validation_teacher(z, row):
    if (str(z['schema']) != 'diagnostic-validation32-source-projection-replay-v1'
            or str(z['split']) != 'validation' or not bool(z['not_for_training'])
            or str(z['sample_id']) != row['sample_id'] or row['split'] != 'validation'
            or z['teacher_source9_10_FP16'].shape != (2, 1024, 1024)
            or z['teacher_alpha9_10'].shape != (2, 1024)):
        raise ValueError('explicit read-only validation32 diagnostic labels required')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args(); plan = json.loads(a.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if (not gpu.isdigit() or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(',')
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'):
        raise ValueError('one explicitly assigned GPU and pinned CUBLAS workspace required')
    cfg = json.loads(Path(plan['native_config']).read_text())
    rows = [json.loads(line) for line in Path(plan['validation32_manifest']).read_text().splitlines()]
    if len(rows) != 32 or plan['variant'] not in ['original-NoDQ', 'source-reference-NoDQ']:
        raise ValueError('two declared final112 pilots on the original validation32 required')
    torch.set_num_threads(2); torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    selected = (MemberFeatureSourceRayConditioner if plan['variant'] == 'original-NoDQ'
                else SourceReferencedMemberFeatureConditioner)
    module = selected(**cfg['trainer']['args']['deform_conditioner']).cuda().eval()
    module.load_state_dict(torch.load(plan['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    before = tensor_state_sha(module.state_dict())
    grid = ProjGrid(grid_resolution=16, image_resolution=512).cuda()
    prefix = Path(plan['local_mirror_prefix'])
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=False)
    local = lambda raw: prefix/raw.lstrip('/')
    records, totals = [], {}
    with torch.inference_mode():
        for row in rows:
            label = Path(plan['validation_replays'])/(hashlib.sha256(row['sample_id'].encode()).hexdigest()+'.npz')
            with np.load(label) as z:
                validate_validation_teacher(z, row)
                teacher = torch.from_numpy(z['teacher_source9_10_FP16'].copy()).cuda().float()
                alpha = torch.from_numpy(z['teacher_alpha9_10'].copy()).cuda().bool()
            with np.load(local(row['transport_cache'])) as z:
                assert str(z['conditioning_input_sha256']) == row['conditioning_input_sha256']
                pack = {name: torch.from_numpy(z[name].copy())[None].cuda()
                        for name in ['source_values', 'source_valid', 'target_query']}
                pack['target_keys'] = torch.from_numpy(z['target_keys'][..., :1024].copy())[None].cuda()
                pack['source_keys'] = torch.from_numpy(z['source_keys'][..., :1024].copy())[None].cuda()
            with np.load(Path(cfg['dataset']['args']['projection_root'])/(row['sample_id']+'.npz')) as z:
                assert str(z['schema']) == 'deformnext-anchor-projection-v2'
                pack['patch_grid'] = torch.from_numpy(z['patch_grid'].copy()).long().reshape(1, -1).cuda()
            with Image.open(local(row['target_anchor'])) as image:
                observed = np.asarray(image.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
            pack.update(target_valid=torch.from_numpy((observed[8::16, 8::16]>127).copy()).flatten()[None].cuda(),
                matrices=torch.tensor(row['transform_matrix'], dtype=torch.float32)[None].cuda(),
                fovs=torch.tensor(row['camera_angle_x'], dtype=torch.float32)[None].cuda(),
                scale=torch.tensor([row['mesh_scale']], dtype=torch.float32).cuda())
            args = module_inputs(pack, grid)
            with torch.autocast('cuda', dtype=torch.bfloat16): result = module(**args)
            delta = source_ray_delta(result.source_numerator, result.semantic_weight, args['source_values'])[0]
            desired = teacher-args['source_values'][0, :2]
            foreground = alpha | pack['source_valid'][0, :2]
            active = result.attention.query_valid.reshape(3, 1024)
            assert not delta[~active].count_nonzero()
            masks = dict(source_all=torch.ones_like(foreground), source_foreground_union=foreground,
                source_active_foreground=foreground&active[:2], source_inactive_foreground=foreground&~active[:2])
            raw = {name: moments(delta[:2], desired, mask) for name,mask in masks.items()}
            records.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
                statistics={name:summarize(value)for name,value in raw.items()}))
            for name, values in raw.items():
                if name not in totals: totals[name] = {key:0 for key in values}
                for key,value in values.items(): totals[name][key] += value
            print(json.dumps(dict(completed_validation_assets=len(records), total=32)), flush=True)
    assert before == tensor_state_sha(module.state_dict()) and not any(p.grad is not None for p in module.parameters())
    audit = dict(status='PASS_ACTUAL_READ_ONLY_TEACHER_PILOT_VALIDATION_FEATURE_REPLAY32',
        variant=plan['variant'], checkpoint=plan['checkpoint'], source_state_SHA256=before,
        per_asset=records, summed_element_weighted_metrics={name:summarize(value)for name,value in totals.items()},
        validation_assets=32, optimizer_updates=0, all_parameters_unchanged=True,
        physical_GPU=gpu,
        labels_not_for_training=True, source9_10_only=True, source11_teacher_not_available=True,
        CUDA_BF16_module_with_FP16_cached_target_native_projection=True,
        actual_live_RGB_native_SS_generation_not_replayed_here=True,
        feature_MSE_not_a_generated_geometry_quality_metric=True, model_adopted=False, full_goal_complete=False)
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='per_asset'}), flush=True)


if __name__ == '__main__':
    main()
