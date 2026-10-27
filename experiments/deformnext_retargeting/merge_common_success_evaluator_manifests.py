"""Merge external seven methods and official RGB1 on their exact common successes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXTERNAL = {'trellis2', 'pixal3d', 'reconviagen', 'mvadapter',
            'corradapter', 'shapeup', 'spacecontrol'}
OFFICIAL = {'pixal3d_official_rgb1_raw', 'pixal3d_official_rgb1_source_m0'}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external-binding', type=Path, required=True)
    parser.add_argument('--official-binding', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    args = parser.parse_args()
    if args.output_root.exists() or not 1 <= args.expected_samples <= 502:
        raise ValueError('fresh output and declared common-success count required')
    external_refs = read_jsonl(args.external_binding / 'references.jsonl')
    official_refs = read_jsonl(args.official_binding / 'references.jsonl')
    external_preds = read_jsonl(args.external_binding / 'predictions.jsonl')
    official_preds = read_jsonl(args.official_binding / 'predictions.jsonl')
    ext_ref_map = {row['sample_id']: row for row in external_refs}
    off_ref_map = {row['sample_id']: row for row in official_refs}
    if (len(external_refs) != 502 or len(ext_ref_map) != 502
            or len(official_refs) != args.expected_samples
            or len(off_ref_map) != args.expected_samples):
        raise ValueError('reference populations have duplicate or wrong counts')
    if (set(off_ref_map) - set(ext_ref_map)
            or any(ext_ref_map[sid] != row for sid, row in off_ref_map.items())):
        raise ValueError('official and external source-view references differ')
    ext_index = {(row['method'], row['sample_id']): row for row in external_preds}
    off_index = {(row['method'], row['sample_id']): row for row in official_preds}
    if (len(ext_index) != 502 * 7 or len(external_preds) != len(ext_index)
            or len(off_index) != args.expected_samples * 2
            or len(official_preds) != len(off_index)
            or {key[0] for key in ext_index} != EXTERNAL
            or {key[0] for key in off_index} != OFFICIAL):
        raise ValueError('external seven or official two method prediction closure differs')
    ids = [row['sample_id'] for row in external_refs if row['sample_id'] in off_ref_map]
    if len(ids) != args.expected_samples:
        raise ValueError('official success intersection differs')
    refs = [ext_ref_map[sid] for sid in ids]
    preds = []
    for method in sorted(EXTERNAL | OFFICIAL):
        index = ext_index if method in EXTERNAL else off_index
        for sid in ids:
            row = index.get((method, sid))
            if row is None or row['seed'] != 0 or row['status'] != 'ok':
                raise ValueError('missing or failed common prediction: ' + method + '/' + sid)
            preds.append(row)
    args.output_root.mkdir(parents=True)
    (args.output_root / 'references.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in refs))
    (args.output_root / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in preds))
    result = {'status': 'PASS_NINE_METHOD_COMMON_SUCCESS_EVALUATOR_BINDING',
              'samples': len(ids), 'methods': sorted(EXTERNAL | OFFICIAL),
              'predictions': len(preds), 'excluded_from_frozen_external_502': 502-len(ids),
              'sample_ids': ids}
    (args.output_root / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'samples', 'predictions')}), flush=True)


if __name__ == '__main__':
    main()
