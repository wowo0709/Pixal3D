"""Prepare only frozen V1/Selected 1024 TEST cases with image-inferred scale."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_ROW_FIELDS = ('sample_id', 'camera_angle_x', 'transform_matrix', 'mesh_scale',
                    'conditioning_input_schema', 'conditioning_input_sha256',
                    'target_anchor_kind', 'conditioning_camera_ids', 'target_anchor',
                    'source_images', 'transport_cache', 'transport_cache_pbr')


def sha(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--scale-predictions', type=Path, required=True)
    parser.add_argument('--ss-root', type=Path, required=True)
    parser.add_argument('--shape-template', type=Path, required=True)
    parser.add_argument('--selected-pbr-template', type=Path, required=True)
    parser.add_argument('--file4-local-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or not args.file4_local_root.is_dir():
        raise ValueError('fresh plan root and local file4 mirror required')
    freeze = json.loads(args.freeze.read_text())
    if freeze['status'] != 'FROZEN_FROM_DEVELOPMENT_ONLY':
        raise ValueError('development-only model freeze required')
    population = freeze['population']
    manifest = Path(population['manifest']['path'])
    if sha(manifest) != population['manifest']['SHA256']:
        raise ValueError('frozen TEST manifest SHA differs')
    rows = read_jsonl(manifest)
    by_id = {row['sample_id']: row for row in rows}
    ids = population['sample_ids']
    if (len(ids) != 508 or len(rows) != 508 or len(by_id) != 508
            or set(ids) != set(by_id) or len(population['generation_seeds']) != 508):
        raise ValueError('frozen TEST508 identity/seed closure differs')
    predictions = read_jsonl(args.scale_predictions)
    by_pred = {row['sample_id']: row for row in predictions}
    receipt = json.loads(args.scale_predictions.with_suffix('.receipt.json').read_text())
    if (len(predictions) != 508 or len(by_pred) != 508 or set(ids) != set(by_pred)
            or receipt['status'] != 'PASS_IMAGE_ONLY_SCALE_PREDICTIONS'
            or receipt['output_sha256'] != sha(args.scale_predictions)
            or any(row['M0_or_M1_mesh_or_GT_scale_used_at_inference'] is not False
                   for row in predictions)):
        raise ValueError('frozen TEST image-only scale predictions differ')
    shape_template = json.loads(args.shape_template.read_text())
    pbr_template = json.loads(args.selected_pbr_template.read_text())
    if (shape_template['object_scale_policy'] != 'image-estimated-v1'
            or pbr_template['mode'] != 'top1-only'
            or pbr_template['checkpoint_kind'] != 'trained-native1900'):
        raise ValueError('actual tested Shape/PBR template required')
    for arm in ('V1', 'SELECTED'):
        for stage in ('shape512', 'shape1024'):
            if freeze['models'][arm][stage] != freeze['models']['V1'][stage]:
                raise ValueError('Shape stages must be the same fixed pretrained weights')
        pbr = freeze['models']['V1']['pbr1024']
        if (shape_template['stages']['pbr1024']['config'] != pbr['native_config']['path']
                or shape_template['stages']['pbr1024']['checkpoint'] != pbr['checkpoint']['path']):
            raise ValueError('Shape template must use fixed vanilla PBR texture')
    selected_pbr = freeze['models']['SELECTED']['pbr1024']
    if (pbr_template['PBR_config'] != selected_pbr['native_config']['path']
            or pbr_template['PBR_checkpoint'] != selected_pbr['checkpoint']['path']
            or pbr_template['conditioner_checkpoint'] != selected_pbr['conditioner_checkpoint']['path']):
        raise ValueError('selected PBR template differs from frozen Top-1 checkpoint')
    def local(path: str) -> str:
        if not path.startswith('/file4/'):
            raise ValueError('canonical file4 input path required: ' + path)
        result = args.file4_local_root / path[len('/file4/'):]
        if not result.is_file():
            raise FileNotFoundError(result)
        return str(result)

    plans = []
    for sid in ids:
        row = by_id[sid]
        scale = float(by_pred[sid]['predicted_mesh_scale'])
        if not .4 < scale < 1.6:
            raise ValueError('frozen image-only scale outside contract: ' + sid)
        bound = dict(row, mesh_scale=scale)
        images = [local(row['target_anchor']), *(local(path) for path in row['source_images'])]
        source = local(row['source_geometry'])
        target = local(row['target_vertices'])
        for arm in ('V1', 'SELECTED'):
            shape = dict(shape_template)
            shape.update(status='PREPARED_FROZEN_TEST508_IMAGE_SCALE_CASCADE',
                         row=bound, object_scale_policy='image-estimated-v1',
                         image_scale_predictions=str(args.scale_predictions),
                         SS_evaluation=str(args.ss_root / arm / 'evaluation.json'),
                         RGB_paths=images, source_geometry=source, target_vertices=target,
                         seed=population['generation_seeds'][sid],
                         output=str(args.output_root / 'cases' / arm / sid / 'shape'),
                         frozen_model_set=str(args.freeze), TEST508_seed_frozen=True,
                         M1_derived_scale_supplied_to_generation=False)
            shape.pop('pilot_rebased_from', None)
            shape_plan = args.output_root / 'plans' / 'shape' / arm / (sid + '.json')
            shape_plan.parent.mkdir(parents=True, exist_ok=True)
            shape_plan.write_text(json.dumps(shape, indent=2) + '\n')
            entry = {'sample_id': sid, 'arm': arm, 'shape_plan': str(shape_plan)}
            if arm == 'SELECTED':
                pbr = dict(pbr_template)
                for old in ('source_SHA256', 'input_SHA256', 'closed_inputs',
                            'canonical_local_root', 'cold_reference', 'native_guides'):
                    pbr.pop(old, None)
                pbr.update(row={key: bound[key] for key in IMAGE_ROW_FIELDS},
                           RGB_paths=images, generated_shape_root=shape['output'],
                           PBR_replay_seed=population['generation_seeds'][sid],
                           output=str(args.output_root / 'cases' / arm / sid / 'top1-pbr'),
                           diagnostic_scale_policy='image-estimated-v1',
                           validation_only_plan_without_old_TEST_closure=False,
                           frozen_TEST508_scale_provenance=str(args.scale_predictions))
                pbr_plan = args.output_root / 'plans' / 'top1-pbr' / (sid + '.json')
                pbr_plan.parent.mkdir(parents=True, exist_ok=True)
                pbr_plan.write_text(json.dumps(pbr, indent=2) + '\n')
                entry['top1_pbr_plan'] = str(pbr_plan)
            plans.append(entry)
    if len(plans) != 1016:
        raise ValueError('two-arm TEST508 plan closure failed')
    result = {'status': 'PREPARED_FROZEN_V1_SELECTED_TEST508_IMAGE_SCALE',
              'samples': 508, 'arms': ['V1', 'SELECTED'], 'plans': plans,
              'freeze_sha256': sha(args.freeze),
              'scale_prediction_sha256': sha(args.scale_predictions),
              'SS_root': str(args.ss_root), 'shape_stages_identical': True,
              'selected_texture': 'frozen trained-native1900 Top-1-only replay',
              'training_updates': 0}
    (args.output_root / 'campaign.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'samples', 'arms')}), flush=True)


if __name__ == '__main__':
    main()
