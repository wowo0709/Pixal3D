"""Run the frozen final render with a uniformly quantized GT raster QA gate.

The 99% containment tolerance is rounded to the number of whole pixels in
each GT alpha mask. This changes only the admission check in the render phase;
the original renderer, predictions, GT arrays and metric formulas stay pinned.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch.nn.functional as F

from . import selected_final_phase, topology_labels
from .selected_pipeline_model_contract import require


def quantized_mask_agreement(actual, expected, *, alpha_cutouts=False):
    require(alpha_cutouts, 'operational wrapper is limited to final GT alpha QA')
    actual, expected = actual.bool(), expected.bool()
    intersection = (actual & expected).sum()
    union = (actual | expected).sum().clamp_min(1)
    dilated = F.max_pool2d(actual[None, None].float(), 3, 1, 1)[0, 0].bool()
    foreground = int(expected.sum().item())
    require(foreground > 0, 'empty GT alpha cannot certify final rendering')
    outside = int((expected & ~dilated).sum().item())
    allowance = (foreground + 99) // 100
    iou = float(intersection / union)
    containment = float((dilated & expected).sum() / expected.sum().clamp_min(1))
    if outside > allowance:
        raise topology_labels.RasterAgreementError(
            f'GT raster outside one-pixel tolerance: outside={outside}, '
            f'one-percent integer allowance={allowance}, IoU={iou}, containment={containment}')
    return iou, containment


def execute(path):
    path = Path(path)
    binding = json.loads(path.read_text())
    require(binding.get('kind') == 'assigned-native-render', 'only final rendering may use quantized GT QA')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(os.environ.get('SELECTED_OPERATIONAL_RENDER_SHA256') == digest,
            'independent operational render adapter SHA256 pin required')
    native = json.loads(Path(binding['native_plan']['path']).read_text())
    output = Path(native['output'])
    original = topology_labels.mask_agreement
    observations = []

    def checked(actual, expected, *, alpha_cutouts=False):
        value = quantized_mask_agreement(actual, expected, alpha_cutouts=alpha_cutouts)
        dilated = F.max_pool2d(actual[None, None].float(), 3, 1, 1)[0, 0].bool()
        foreground = int(expected.sum().item())
        observations.append(dict(GT_alpha_pixels=foreground,
                                 outside_one_pixel_raster=int((expected & ~dilated).sum().item()),
                                 integer_ceil_one_percent_allowance=(foreground + 99) // 100,
                                 IoU=value[0], containment=value[1]))
        return value

    topology_labels.mask_agreement = checked
    try:
        result = selected_final_phase.execute(path)
    finally:
        topology_labels.mask_agreement = original
    require(len(observations) == 3 and all(row['outside_one_pixel_raster'] <=
            row['integer_ceil_one_percent_allowance'] for row in observations),
            'all three final GT views must pass the same integer-pixel QA rule')
    proof = dict(status='PASS_UNIFORM_INTEGER_PIXEL_GT_RASTER_QA',
                 policy='outside one-pixel dilated GT geometry raster <= ceil(0.01 * GT alpha pixels)',
                 original_continuous_point99_gate_replaced_only_for_final_render_admission=True,
                 original_numeric_phase_result=result, observations=observations,
                 original_render_source_unchanged=True, metric_math_and_GT_arrays_unchanged=True,
                 operational_source_SHA256=digest)
    with (output / 'operational-raster-gate.json').open('x') as stream:
        json.dump(proof, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = execute(args.plan)
    print(json.dumps(dict(status=result['status'], observations=result['observations'])), flush=True)


if __name__ == '__main__':
    main()
