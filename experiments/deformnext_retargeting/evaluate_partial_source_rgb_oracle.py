"""Native zero-shot SS32 with two additional GT source RGBs, diagnostic only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from .partial_source_rgb_oracle import CAMERAS, STATES, oracle_dataset_class


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--oracle-spec', type=Path, required=True)
    local, remaining = parser.parse_known_args()
    spec = json.loads(local.oracle_spec.read_text())
    for flag, expected in {'--stage': 'ss64', '--step': '0', '--panel-size': '32',
                           '--generation-batch': '8', '--conditioning-mode': 'edited-rgb'}.items():
        if remaining.count(flag) != 1 or remaining[remaining.index(flag) + 1] != expected:
            raise ValueError('fixed native zero-shot diagnostic protocol required: ' + flag)
    if '--conditioner-checkpoint' in remaining or len(spec['sample_ids']) != 32:
        raise ValueError('no deformation conditioner or training in the extra-GT diagnostic')
    config = Path(remaining[remaining.index('--arm-config') + 1])
    cfg = json.loads(config.read_text())
    if (cfg['dataset']['name'] != 'PartialSourceRGBOracleSS64'
            or cfg['validation']['dataset_args']['transport_arm'] != 'v1'
            or cfg['trainer']['args'].get('deform_conditioner') is not None):
        raise ValueError('explicit vanilla SS without stale Omega/matching caches required')
    source_by_id = {sid: {int(k): v for k, v in views.items()}
                    for sid, views in spec['source_by_id'].items()}
    if list(source_by_id) != spec['sample_ids']:
        raise ValueError('the original fixed32 input order is required')
    # Dataset imports initialize the native GPU backend. Use only the GPU
    # already assigned by the dispatcher, after its CUBLAS contract is set.
    import torch
    from .evaluate_deterministic_checkpoint import validate_environment
    validate_environment(os.environ)
    torch.cuda.set_device(0)
    from pixal3d import datasets
    from pixal3d.datasets.deformnext import _load_image
    from . import evaluate_deterministic_checkpoint as native
    events, captures = {}, dict.fromkeys(spec['sample_ids'][:4])
    cls = oracle_dataset_class(datasets.DeformNeXtSS64, source_by_id, _load_image,
                               events, captures)
    name = 'PartialSourceRGBOracleSS64'
    if name in datasets.__dict__:
        raise ValueError('fresh process-local dataset registration required')
    previous = sys.argv
    try:
        setattr(datasets, name, cls)
        sys.argv = [native.__file__, *remaining]
        native.main()
    finally:
        sys.argv = previous
        delattr(datasets, name)
    output = Path(remaining[remaining.index('--output') + 1])
    path = output / 'evaluation.json'
    raw = path.read_bytes()
    result = json.loads(raw)
    if ([r['sample_id'] for r in result['per_asset']] != spec['sample_ids']
            or set(events) != set(spec['sample_ids'])
            or result['optimizer_updates_during_evaluation'] != 0
            or result['conditioner_checkpoint'] is not None):
        raise ValueError('actual fixed32 native evaluation/extra-RGB coverage differs')
    for event in events.values():
        before, after = event['original_view_SHA256'], event['actual_view_SHA256']
        if (before[0] != after[0] or before[3] != after[3]
                or after[1] != event['teacher_view_SHA256']['9']
                or after[2] != event['teacher_view_SHA256']['10']):
            raise ValueError('actual native input tensor roles differ from the declared oracle')
    (output / 'native-evaluator-before-oracle-annotation.json').write_bytes(raw)
    result.update(diagnostic_only=True, conditioning_mode='partial-M1-source-RGB9-10',
        input_states=list(STATES), input_camera_ids=list(CAMERAS),
        original_multiview_checkpoint_step=4210, deformation_training_updates=0,
        partial_source_RGB_oracle=dict(extra_GT_source_RGB_cameras=[9, 10],
            unchanged_edited_target_camera=8, unchanged_M0_source_camera=11,
            full_four_M1_oracle=False, transport_cache_used=False,
            native_non_RGB_inputs_preserved=True, actual_tensor_events=events,
            pre_oracle_annotation_evaluation_SHA256=hashlib.sha256(raw).hexdigest(),
            competitive_model_or_quality_gain_claim=False))
    result['deterministic_evaluation']['scope'] = (
        'Extra-GT diagnostic: exactly source RGB9/10 replaced; native sampler, '
        'weights, scoring GT and noise unchanged; not a competitive model.')
    path.write_text(json.dumps(result, indent=2) + '\n')
    # The generic evaluator labels its default source states as M0. Preserve
    # that raw contact sheet and publish the actual oracle tensor roles.
    (output / 'actual_inputs.png').rename(output / 'native-inputs-before-oracle-labels.png')
    from PIL import Image, ImageDraw
    contact = Image.new('RGB', (4 * 128, 4 * 148))
    draw = ImageDraw.Draw(contact)
    for line, sid in enumerate(spec['sample_ids'][:4]):
        cond = captures[sid]
        if cond is None:
            raise ValueError('capture the actual four native input tensors')
        for view, frame in enumerate(cond):
            pixels = (frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            contact.paste(Image.fromarray(pixels).resize((128, 128)), (view * 128, line * 148 + 20))
            draw.text((view * 128, line * 148),
                      f'{STATES[view]} cam{CAMERAS[view]}'+(' EXTRA GT' if view in (1, 2) else ''),
                      fill='white')
    contact.save(output / 'actual_inputs.png')
    print(json.dumps(dict(status='PASS_ACTUAL_PARTIAL_SOURCE_RGB_ORACLE_SS32',
                          assets=32, diagnostic_only=True, input_states=list(STATES),
                          optimizer_updates=0, full_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
