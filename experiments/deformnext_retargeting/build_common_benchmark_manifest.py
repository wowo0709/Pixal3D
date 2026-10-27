"""Bind the six frozen TEST508 internal rows to their archived raw outputs.

This manifest is for post-hoc common-renderer evaluation only. It never
selects checkpoints, samples, or generation seeds from quality results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ALIASES = ('zero-source4', 'zero-edited4', 'V1', 'RETARGET', 'SELECTED')
MIRROR = '/root/dev/.cache/deformnext-file4/file4'


def file4(path: Path) -> str:
    value = str(path)
    if not value.startswith(MIRROR + '/'):
        raise ValueError('not an archived file4 path: ' + value)
    return '/file4' + value[len(MIRROR):]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def bound_case(root: Path, sample_id: str, alias: str) -> dict:
    generation_root = root / ('geometry' if alias.startswith('zero-') else 'PBR')
    if alias == 'RGB1_CALIBRATED':
        generation_root = root / 'geometry'
    render_plan = load(root / ('render-plan.json' if alias == 'RGB1_CALIBRATED'
                               else 'runtime-plans/render/native.json'))
    row = render_plan['row']
    if row['sample_id'] != sample_id or float(row['mesh_scale']) <= 0:
        raise ValueError('sample/scale mismatch in render plan')
    generation = load(generation_root / 'generation.json')
    if generation['sample_id'] != sample_id:
        raise ValueError('sample mismatch in generation')
    return {
        'sample_id': sample_id, 'arm': alias,
        'prediction': file4(generation_root / 'prediction.npz'),
        'render_geometry': file4(root / 'render/render-geometry.npz'),
        'anchor_evaluation': file4(generation_root / 'geometry-evaluation.npz'),
        'generation': file4(generation_root / 'generation.json'),
        'mesh_scale': float(row['mesh_scale']),
        'export_relpath': f'exports/{alias}/{sample_id}.glb',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-manifest', type=Path, required=True)
    parser.add_argument('--remaining-campaign', type=Path, required=True)
    parser.add_argument('--rgb1-campaign', type=Path, required=True)
    parser.add_argument('--external-references', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('manifest is immutable')
    refs = {json.loads(line)['sample_id'] for line in args.external_references.read_text().splitlines()}
    if len(refs) != 508:
        raise ValueError('external benchmark is not TEST508')
    reuse = load(args.reuse_manifest)['groups']
    rows = []
    for sample_id, group in reuse.items():
        if sample_id not in refs or set(group['cases']) != set(ALIASES):
            raise ValueError('reused group differs from declared five arms')
        for alias in ALIASES:
            root = Path(group['cases'][alias]['audit']['path']).parent
            rows.append(bound_case(root, sample_id, alias))
    remaining_ids = set()
    for group_root in sorted((args.remaining_campaign / 'cases').iterdir()):
        if not group_root.is_dir():
            continue
        sample_id = load(group_root / 'SELECTED/audit.json')['sample_id']
        if sample_id in remaining_ids or sample_id in reuse or sample_id not in refs:
            raise ValueError('duplicate or unexpected remaining sample')
        remaining_ids.add(sample_id)
        for alias in ALIASES:
            rows.append(bound_case(group_root / alias, sample_id, alias))
    for sample_id in sorted(refs):
        root = args.rgb1_campaign / 'cases' / sample_id
        rows.append(bound_case(root, sample_id, 'RGB1_CALIBRATED'))
    if len(reuse) != 277 or len(remaining_ids) != 231 or len(rows) != 508 * 6:
        raise ValueError('expected 277 reused + 231 new, six rows each')
    index = {(row['arm'], row['sample_id']) for row in rows}
    if len(index) != len(rows) or any({sid for arm, sid in index if arm == alias} != refs
                                  for alias in (*ALIASES, 'RGB1_CALIBRATED')):
        raise ValueError('incomplete six-arm TEST508 manifest')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n'
                                   for row in sorted(rows, key=lambda r: (r['sample_id'], r['arm']))))
    print(json.dumps({'status': 'PASS_FROZEN_SIX_ARM_TEST508_EXPORT_MANIFEST',
                      'samples': len(refs), 'arms': len(ALIASES) + 1, 'rows': len(rows),
                      'reused': len(reuse), 'remaining': len(remaining_ids),
                      'output': str(args.output)}))


if __name__ == '__main__':
    main()
