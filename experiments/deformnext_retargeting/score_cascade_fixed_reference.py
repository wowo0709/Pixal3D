"""Score generated RGB against the SAME archived original-material GT pixels.

The reference was rendered independently from the original target mesh/UVs.
Reusing it keeps lighting, camera and material identical between model arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in {'0', '1', '2', '3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.2, 0)
    from pixal3d.utils.loss_utils import psnr, ssim, lpips
    output = Path(plan['output'])
    output.mkdir(exist_ok=False)
    proof = json.loads((Path(plan['fixed_GT_shared_root'])/'render-evaluation.json').read_text())
    if (proof['status'] != 'PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'
            or not proof['shared_renderer_lighting_camera_AO_seed']
            or proof['GT_VAE_reconstruction_not_used'] is not True
            or proof['maximum_GT_world_vertex_nearest_difference'] > 1e-5):
        raise ValueError('audited original-material shared-renderer reference required')
    rows = []
    with torch.inference_mode():
        for view in plan['views']:
            cid = view['camera_id']
            with np.load(Path(plan['prediction_render_root'])/f'camera{cid:02d}.npz') as pred_file, \
                 np.load(Path(plan['fixed_GT_render_root'])/f'camera{cid:02d}.npz') as camera_file, \
                 np.load(Path(plan['fixed_GT_shared_root'])/f'camera{cid:02d}.npz') as reference:
                for key in ['extrinsics', 'intrinsics']:
                    if not np.array_equal(pred_file[key], camera_file[key]):
                        raise ValueError('fixed GT and prediction cameras differ: '+key)
                pred_array = pred_file['shaded'].copy()
                pred_alpha_array = pred_file['alpha'].copy()
                gt_array = reference['GT_shared_renderer'].copy()
                gt_alpha_array = reference['GT_alpha'].copy()
            if (pred_array.shape != (3,512,512) or gt_array.shape != pred_array.shape
                    or pred_alpha_array.shape != (512,512) or gt_alpha_array.shape != pred_alpha_array.shape):
                raise ValueError('actual native512 RGB/alpha arrays required')
            pred = torch.from_numpy(pred_array).cuda()[None]
            gt = torch.from_numpy(gt_array).cuda()[None]
            pred_mask = torch.from_numpy(pred_alpha_array).cuda() > .5
            gt_mask = torch.from_numpy(gt_alpha_array).cuda() > .5
            union = pred_mask | gt_mask
            if not union.any():
                raise ValueError('empty common rendering foreground')
            rows.append(dict(camera_id=cid, PSNR=float(psnr(pred,gt)),
                SSIM=float(ssim(pred,gt)), LPIPS=float(lpips(pred,gt)),
                M_IoU=float((pred_mask&gt_mask).sum().double()/union.sum().double()),
                foreground_PSNR=float(-10*(pred-gt).square()[union[None,None].expand_as(pred)].mean().clamp_min(1e-12).log10()),
                fixed_GT_RGB_SHA256=hashlib.sha256(gt_array.tobytes()).hexdigest(),
                fixed_GT_alpha_SHA256=hashlib.sha256(gt_alpha_array.tobytes()).hexdigest(),
                identical_camera_matrices=True, source_material_identity_preserved=True))
            np.savez_compressed(output/f'camera{cid:02d}.npz', prediction=pred_array,
                prediction_alpha=pred_alpha_array, GT_shared_renderer=gt_array, GT_alpha=gt_alpha_array)
    result = dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE', sample_id=plan['sample_id'],
        per_view=rows, view_average={k:float(np.mean([r[k] for r in rows]))
            for k in ['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']},
        fixed_original_material_reference=str(plan['fixed_GT_shared_root']),
        GT_not_used_in_generation=True, original_materials_and_UVs=True,
        same_reference_pixels_for_every_model=True, reference_re_rendered=False,
        diagnostic_only=True, paper_evaluation_complete=False,
        DreamSim_CLIP_FID_KID_not_measured=True)
    (output/'render-evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],view_average=result['view_average'])),flush=True)


if __name__ == '__main__':
    main()
