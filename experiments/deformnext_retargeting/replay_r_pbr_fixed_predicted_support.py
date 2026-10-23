"""Explicit fixed-geometry PBR replay using an archived predicted-support guide.

Sampling is the original replay. Only the texture decoder's support hierarchy
is supplied from the archived prediction instead of a new Shape VAE decoding.
Use the same policy for all models in an isolation comparison. This entry does
not claim to reproduce the unrecorded historical Shape subdivision hierarchy.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .cascade_view_roles import ClosedInputs, file_sha256
from .predicted_support_guides import POLICY, predicted_support_guides
from . import replay_r_pbr_on_generated_shape as native


def run(plan):
    if plan.get('texture_support_policy') != POLICY:
        raise ValueError('explicit archived predicted-support policy required')
    if plan.get('cold_reference'):
        raise ValueError('historical native-guide cold reference cannot validate this different support policy')
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    root = Path(plan['generated_shape_root'])
    with np.load(root/'prediction.npz') as z:
        roots, leaves = z['shape_coords'].copy(), z['coords'].copy()
    previous = Pixal3DImageTo3DPipeline.decode_shape_slat
    events = []

    def fixed_guide(self, shape, resolution):
        if resolution != 1024 or not torch.equal(shape.coords, torch.from_numpy(roots).to(shape.coords)):
            raise ValueError('original Shape latent coordinates and native1024 resolution required')
        # These are model-predicted coordinates, already produced and archived.
        # The original replay copies evaluation-only geometry after prediction.
        raw = predicted_support_guides(shape.coords, torch.from_numpy(leaves).to(shape.coords))
        guides = [SparseTensor(feats=mask, coords=coords) for coords, mask in raw]
        events.append(dict(parent_counts=[len(c) for c,m in raw], child_counts=[int(m.sum()) for c,m in raw],
            guide_SHA256=[hashlib.sha256(c.cpu().numpy().tobytes()+m.cpu().numpy().tobytes()).hexdigest() for c,m in raw]))
        return [], guides

    Pixal3DImageTo3DPipeline.decode_shape_slat = fixed_guide
    try:
        report = native.run(plan)
    finally:
        Pixal3DImageTo3DPipeline.decode_shape_slat = previous
    if len(events) != 1:
        raise ValueError('exactly one fixed predicted-support decode required')
    report.update(texture_support_policy=POLICY, shape_decoder_used_only_for_native_texture_guide_hierarchy=False,
        shape_decoder_recomputed=False, archived_prediction_coordinates_used_for_texture_guide=True,
        historical_unrecorded_subdivision_hierarchy_reproduced=False,
        intermediate_branches_without_final_predicted_leaves_omitted=True,
        native_PBR_sampling_math_unchanged=True, fixed_support_events=events,
        all_compared_textures_require_same_support_policy=True)
    (Path(plan['output'])/'generation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    a=p.parse_args();plan=json.loads(a.plan.read_text());check=ClosedInputs(plan['canonical_local_root'])
    def guard():
        check.check(a.plan,a.plan.parent/'file4-closure.json')
        for path,digest in {**plan['source_SHA256'],**plan['input_SHA256']}.items():
            if file_sha256(path)!=digest:raise ValueError('bound source/input changed: '+path)
        for path,receipt in plan['closed_inputs'].items():check.check(path,receipt)
    guard();run(plan);guard()


if __name__=='__main__':main()
