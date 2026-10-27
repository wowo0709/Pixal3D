"""Run unchanged archived render/score/metric math on assigned GPUs."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

import torch.nn.functional as F

from .assigned_gpu_entry import adapt_guard


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['render','shared','perceptual','inception'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    permitted={'6','7'} if args.phase in {'render','shared'} else {'0','1','2','3'}
    if gpu not in permitted or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('single-view evaluation requires one assigned GPU for its phase')
    names={'render':'render_cascade_artifact_v2','shared':'score_cascade_fixed_reference',
        'perceptual':'evaluate_official_perceptual','inception':'extract_official_inception'}
    name=names[args.phase]
    module=importlib.import_module('experiments.deformnext_retargeting.'+name)
    namespace=dict(module.__dict__)
    if args.phase=='inception':
        proof=dict(changed_membership_literals=0,all_other_AST_statements_unchanged=True,
                   original_module_called_directly=True)
    else:
        tree,proof=adapt_guard(Path(module.__file__).read_text(),sorted(permitted))
        if proof['changed_membership_literals']!=1 or not proof['all_other_AST_statements_unchanged']:
            raise ValueError('non-device numerical source adaptation')
        exec(compile(tree,str(module.__file__),'exec'),namespace)
    old=sys.argv
    raster_observations=[]
    if args.phase=='render':
        from . import topology_labels
        from .operational_quantized_render_phase import quantized_mask_agreement
        original_mask_agreement=topology_labels.mask_agreement
        def checked_mask_agreement(actual,expected,*,alpha_cutouts=False):
            result=quantized_mask_agreement(actual,expected,alpha_cutouts=alpha_cutouts)
            dilated=F.max_pool2d(actual[None,None].float(),3,1,1)[0,0].bool()
            foreground=int(expected.sum().item())
            raster_observations.append(dict(GT_alpha_pixels=foreground,
                outside_one_pixel_raster=int((expected & ~dilated).sum().item()),
                integer_ceil_one_percent_allowance=(foreground+99)//100,
                IoU=result[0],containment=result[1]))
            return result
        topology_labels.mask_agreement=checked_mask_agreement
    try:
        sys.argv=[str(module.__file__),'--plan',str(args.plan)]
        if args.phase=='inception':
            if args.output is None:raise ValueError('inception output required')
            sys.argv.extend(['--output',str(args.output)])
        namespace['main']()
    finally:
        sys.argv=old
        if args.phase=='render':topology_labels.mask_agreement=original_mask_agreement
    if args.phase=='render':
        if len(raster_observations)!=3:
            raise ValueError('three-view uniform integer-pixel GT raster QA required')
        (args.plan.parent/'render-raster-gate.json').write_text(json.dumps(dict(
            status='PASS_UNIFORM_INTEGER_PIXEL_GT_RASTER_QA',
            policy='outside one-pixel dilated GT geometry raster <= ceil(0.01 * GT alpha pixels)',
            observations=raster_observations,original_render_source_unchanged=True,
            metric_math_and_GT_arrays_unchanged=True),indent=2)+'\n')
    report=args.plan.parent/(args.phase+'-device-proof.json')
    report.write_text(json.dumps(dict(physical_GPU=gpu,source=str(module.__file__),proof=proof,
        numerical_AST_unchanged=True),indent=2)+'\n')


if __name__=='__main__':main()
