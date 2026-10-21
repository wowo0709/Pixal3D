"""Separate native global/projection effects of the fixed extra-GT RGB oracle."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

from .partial_source_rgb_oracle import tensor_sha
from .source_condition_field_oracle import condition_field_oracle


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--condition-field-oracle', choices=['global', 'proj'], required=True)
    local, remaining = parser.parse_known_args()
    spec = json.loads(Path(remaining[remaining.index('--oracle-spec') + 1]).read_text())
    config = json.loads(Path(remaining[remaining.index('--arm-config') + 1]).read_text())
    rows = [json.loads(line) for line in Path(config['validation']['roots_file']).read_text().splitlines()]
    if [row['sample_id'] for row in rows] != spec['sample_ids']:
        raise ValueError('same original fixed32 manifest/order required')
    by_id = {row['sample_id']: row for row in rows}
    preflight = {row['sample_id']: row for row in spec['actual_CPU_tensor_events']}
    input_ids = {tuple(row['actual_view_SHA256']): row['sample_id']
                 for row in spec['actual_CPU_tensor_events']}
    if len(input_ids) != 32:
        raise ValueError('unambiguous actual four-RGB tensor identity required')
    from .evaluate_deterministic_checkpoint import validate_environment
    validate_environment(os.environ)
    torch.cuda.set_device(0)
    from pixal3d.datasets.deformnext import _load_image
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from . import evaluate_partial_source_rgb_oracle as rgb_oracle
    path_map = config['validation']['dataset_args']['path_map']

    def resolve(raw):
        for source, target in path_map.items():
            if raw.startswith(source.rstrip('/') + '/'):
                return Path(target + raw[len(source):])
        raise ValueError('explicit canonical RGB path mapping required')

    def original_images(images):
        if images.dtype != torch.float32:
            raise ValueError('native input RGB identity requires its original FP32 values')
        originals, ids = [], []
        for image in images:
            key = tuple(tensor_sha(view) for view in image)
            if key not in input_ids:
                raise ValueError('actual encoder input differs from the fixed extra-GT oracle')
            sid = input_ids[key]
            row = by_id[sid]
            original = torch.stack([_load_image(resolve(p), 512)
                                    for p in [row['target_anchor'], *row['source_images']]])
            if [tensor_sha(view) for view in original] != preflight[sid]['original_view_SHA256']:
                raise ValueError('original RGB changed since canonical preflight')
            originals.append(original)
            ids.append(sid)
        return torch.stack(originals).to(device=images.device, dtype=images.dtype), ids

    events, previous = [], sys.argv
    try:
        with condition_field_oracle(ImageConditionedProjMixin, original_images,
                                    local.condition_field_oracle, events):
            sys.argv = [rgb_oracle.__file__, *remaining]
            rgb_oracle.main()
    finally:
        sys.argv = previous
    if [sid for event in events for sid in event['sample_ids']] != spec['sample_ids']:
        raise ValueError('exactly the32 actual native image conditioning rows must execute')
    for event in events:
        field = event['oracle_field']
        other = 'global' if field == 'proj' else 'proj'
        if (event['actual_condition_SHA256'][other] != event['original_condition_SHA256'][other]
                or event['actual_condition_SHA256'][field] != event['teacher_condition_SHA256'][field]
                or event['condition_shapes']['proj'][1:] != [4096, 1024]
                or event['condition_shapes']['global'][1:] != [5, 1024]):
            raise ValueError('native SS16 projected grid/global token or field identity differs')
    output = Path(remaining[remaining.index('--output') + 1])
    path = output / 'evaluation.json'
    raw = path.read_bytes()
    (output / 'before-condition-field-annotation.json').write_bytes(raw)
    result = json.loads(raw)
    result['conditioning_mode'] = 'partial-M1-source-RGB9-10-' + local.condition_field_oracle + '-only'
    result['condition_field_oracle'] = dict(
        field=local.condition_field_oracle, events=events,
        actual_native_encodings_for_both_inputs=True,
        unselected_field_negative_and_RNG_exact=True,
        image_contact_shows_extra_GT_encoder_inputs_not_both_selected_fields=True,
        diagnostic_only=True, competitive_model=False)
    result['deterministic_evaluation']['scope'] = (
        'Extra-GT diagnostic: only native ' + local.condition_field_oracle
        + ' selected from M1 source9/10 encoding; original other field, negative, '
        'RNG, camera, checkpoint, scoring GT and noise preserved.')
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status='PASS_ACTUAL_NATIVE_CONDITION_FIELD_ORACLE_SS32',
                          field=local.condition_field_oracle, assets=32,
                          optimizer_updates=0, full_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()
