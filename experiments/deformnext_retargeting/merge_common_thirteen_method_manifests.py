"""Merge external seven, official RGB1 pair, and frozen Ours V1/Selected pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NINE = {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter', 'corradapter',
        'shapeup', 'spacecontrol', 'pixal3d_official_rgb1_raw',
        'pixal3d_official_rgb1_source_m0'}
OURS = {'ours_v1_raw', 'ours_v1_source_m0',
        'ours_selected_raw', 'ours_selected_source_m0'}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nine-binding', type=Path, required=True)
    parser.add_argument('--ours-binding', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or not 1 <= args.expected_samples <= 502:
        raise ValueError('fresh common binding and declared sample count required')
    nine_refs = rows(args.nine_binding / 'references.jsonl')
    ours_refs = rows(args.ours_binding / 'references.jsonl')
    nine_preds = rows(args.nine_binding / 'predictions.jsonl')
    ours_preds = rows(args.ours_binding / 'predictions.jsonl')
    nine_ref_map = {row['sample_id']: row for row in nine_refs}
    ours_ref_map = {row['sample_id']: row for row in ours_refs}
    if (len(nine_refs) != args.expected_samples or len(nine_ref_map) != args.expected_samples
            or len(ours_refs) != args.expected_samples or ours_ref_map != nine_ref_map):
        raise ValueError('all thirteen methods must use identical source references')
    nine_index = {(row['method'], row['sample_id']): row for row in nine_preds}
    ours_index = {(row['method'], row['sample_id']): row for row in ours_preds}
    if (len(nine_preds) != len(nine_index) or len(ours_preds) != len(ours_index)
            or len(nine_index) != args.expected_samples * len(NINE)
            or len(ours_index) != args.expected_samples * len(OURS)
            or {key[0] for key in nine_index} != NINE
            or {key[0] for key in ours_index} != OURS):
        raise ValueError('nine plus four complete method populations required')
    ids = [row['sample_id'] for row in nine_refs]
    preds = []
    for method in sorted(NINE | OURS):
        index = nine_index if method in NINE else ours_index
        for sid in ids:
            row = index.get((method, sid))
            if row is None or row['seed'] != 0 or row['status'] != 'ok':
                raise ValueError('missing or failed common prediction: ' + method + '/' + sid)
            preds.append(row)
    args.output_root.mkdir(parents=True)
    (args.output_root / 'references.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in nine_refs))
    (args.output_root / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in preds))
    result = {'status': 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING',
              'samples': args.expected_samples, 'predictions': len(preds),
              'methods': sorted(NINE | OURS), 'sample_ids': ids,
              'camera_metadata_availability': 'method-specific; official RGB1 and Ours use supplied camera',
              'object_GT_scale_given': False,
              'source_registration_reference': 'input_source_M0_only'}
    (args.output_root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'samples', 'predictions')}), flush=True)


if __name__ == '__main__':
    main()
