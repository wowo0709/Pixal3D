"""Audit independent GT mesh raster versus scored GT RGBA on all TEST views.

This data-only diagnostic never loads a model prediction or quality result.
It uses exactly the original render camera/geometry raster settings and
records integer-pixel residuals before an evaluation-only QA correction.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def execute(manifest, file4_root, output):
    from pixal3d.representations import Mesh
    from pixal3d.renderers import MeshRenderer
    import utils3d

    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines() if line]
    if len(rows) != 508 or len({row['sample_id'] for row in rows}) != 508:
        raise ValueError('exact actual TEST508 manifest required')
    file4_root = Path(file4_root); output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(2); torch.cuda.set_device(0)
    renderer = MeshRenderer(dict(resolution=512, near=.01, far=100, ssaa=1, antialias=False))
    flip = torch.diag(torch.tensor([1.,-1.,-1.,1.], device='cuda'))
    records = []
    resolve = lambda path: file4_root/str(path).lstrip('/')
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            target_path = resolve(row['target_geometry'])
            with np.load(target_path) as arrays:
                mesh = Mesh(torch.from_numpy(arrays['V'].copy()).float().cuda(),
                    torch.from_numpy(arrays['F'].copy()).int().cuda())
            images = [row['target_anchor'], *row['evaluation_target_images']]
            if len(images) != 3:
                raise ValueError('target8 and source9/10 GT images required')
            for index, source in enumerate(images):
                source = resolve(source)
                rgba = np.asarray(Image.open(source).convert('RGBA'))
                if rgba.shape != (512,512,4):
                    raise ValueError('exact original 512px RGBA required')
                expected = torch.from_numpy((rgba[...,3] > 127).copy()).cuda()
                c2w = torch.tensor(row['transform_matrix'][index], dtype=torch.float32, device='cuda')
                ext = flip @ torch.linalg.inv(c2w)
                fov = torch.tensor(row['camera_angle_x'][index], dtype=torch.float32, device='cuda')
                intr = utils3d.torch.intrinsics_from_fov(fov_x=fov, fov_y=fov)
                actual = renderer.render(mesh, ext, intr, return_types=['mask'])['mask'] > .5
                dilated = F.max_pool2d(actual[None,None].float(),3,1,1)[0,0].bool()
                foreground = int(expected.sum()); outside = int((~dilated & expected).sum())
                containment = float((dilated & expected).sum()/expected.sum().clamp_min(1))
                iou = float((actual & expected).sum()/(actual | expected).sum().clamp_min(1))
                records.append(dict(sample_id=row['sample_id'],camera_id=8+index,
                    GT_alpha_pixels=foreground,GT_alpha_outside_one_pixel_raster=outside,
                    integer_ceil_one_percent_allowance=math.ceil(.01*foreground),
                    continuous_point99_pass=containment >= .99,
                    integer_quantized_one_percent_pass=outside <= math.ceil(.01*foreground),
                    GT_geometry_RGBA_containment=containment,GT_geometry_RGBA_mask_iou=iou,
                    GT_image=str(source),GT_image_SHA256=digest(source),
                    GT_geometry=str(target_path),GT_geometry_SHA256=digest(target_path)))
            if number % 50 == 0:
                print(json.dumps(dict(assets_checked=number,views_checked=len(records))),flush=True)
            del mesh
    with (output/'per-view.jsonl').open('x') as stream:
        for record in records:stream.write(json.dumps(record)+'\n')
    summary = dict(status='PASS_ACTUAL_TEST508_GT_RASTER_ALPHA_DATA_ONLY_AUDIT',
        assets=len(rows),views=len(records),manifest=str(manifest),manifest_SHA256=digest(manifest),
        continuous_point99_failures=sum(not r['continuous_point99_pass'] for r in records),
        integer_quantized_one_percent_failures=sum(not r['integer_quantized_one_percent_pass'] for r in records),
        smallest_containment=min(r['GT_geometry_RGBA_containment'] for r in records),
        largest_outside_pixels=max(r['GT_alpha_outside_one_pixel_raster'] for r in records),
        worst=[r for r in sorted(records,key=lambda r:r['GT_geometry_RGBA_containment'])[:20]],
        no_model_prediction_or_quality_result_loaded=True,metric_value_changes=0,
        full_goal_complete=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k!='worst'}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--file4-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    execute(args.manifest,args.file4_root,args.output)


if __name__=='__main__':main()
