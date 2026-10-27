"""Select a representative qualitative case from the final fair TEST subset.

The rule is fixed before reading the final scores: among cases where Selected
improves both source LPIPS and target mask IoU over V1, choose the case closest
to the two-dimensional median improvement. If none improve both, use the case
closest to the median of all paired differences. This script selects an ID;
the figure builder must still verify the exact common-render receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LEFT = 'ours_v1_source_m0'
RIGHT = 'ours_selected_source_m0'


def digest(path: Path) -> str:
    state = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            state.update(block)
    return state.hexdigest()


def method_rows(directory: Path, name: str) -> dict[str, dict]:
    matches = []
    for path in directory.glob('*-results.json'):
        report = json.loads(path.read_text())
        if report['method'] == name:
            matches.append((path, report))
    if len(matches) != 1:
        raise ValueError(f'exactly one completed evaluator result required: {name}')
    path, report = matches[0]
    rows = report['per_deformation_seed']
    indexed = {row['sample_id']: row for row in rows}
    if (len(indexed) != len(rows) or any(row['seed'] != 0 for row in rows)
            or not report['summary']['full_metric_computation']):
        raise ValueError('one fully scored seed-zero row per case required')
    return indexed, digest(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ours-evaluation', type=Path, required=True)
    parser.add_argument('--ours-target8', type=Path, required=True)
    parser.add_argument('--common-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable teaser choice already exists')
    common = json.loads(args.common_manifest.read_text())
    target = json.loads(args.ours_target8.read_text())
    ids = common['sample_ids']
    if (common['status'] != 'PASS_THIRTEEN_METHOD_COMMON_SUCCESS_BINDING'
            or target['status'] != 'PASS_OURS_V1_SELECTED_RAW_AND_M0_TARGET8_PAIRED'
            or not 1 <= len(ids) == common['samples'] == target['samples'] <= 502
            or len(ids) != len(set(ids))):
        raise ValueError('final common TEST population required')
    left, left_hash = method_rows(args.ours_evaluation, LEFT)
    right, right_hash = method_rows(args.ours_evaluation, RIGHT)
    target_rows = {row['sample_id']: row for row in target['per_sample']}
    if (set(left) != set(ids) or set(right) != set(ids)
            or set(target_rows) != set(ids)):
        raise ValueError('paired qualitative population differs')
    improvements = []
    for sid in ids:
        a, b, t = left[sid], right[sid], target_rows[sid]
        if not a['source_cluster_id'] == b['source_cluster_id'] == t['source_cluster_id']:
            raise ValueError('source identity differs: ' + sid)
        source_gain = float(a['metrics']['LPIPS'] - b['metrics']['LPIPS'])
        target_gain = float(t['methods'][RIGHT]['M_IoU'] - t['methods'][LEFT]['M_IoU'])
        if not np.isfinite((source_gain, target_gain)).all():
            raise ValueError('nonfinite qualitative ranking score: ' + sid)
        improvements.append((sid, source_gain, target_gain,
                             a['source_cluster_id']))
    favorable = [row for row in improvements if row[1] > 0 and row[2] > 0]
    population = favorable if favorable else improvements
    points = np.asarray([(row[1], row[2]) for row in population], dtype=np.float64)
    center = np.median(points, axis=0)
    scale = np.median(np.abs(points - center), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    selected = min(population, key=lambda row: (
        float(np.sum(np.abs((np.asarray(row[1:3]) - center) / scale))), row[0]))
    output = {
        'status': 'PASS_FAIR_COMMON_TEST_TEASER_CASE_SELECTED',
        'sample_id': selected[0], 'source_cluster_id': selected[3],
        'common_samples': len(ids), 'jointly_favorable_candidates': len(favorable),
        'selection_rule': ('median_jointly_favorable_source_LPIPS_target_MIoU'
                           if favorable else 'median_all_paired_source_LPIPS_target_MIoU'),
        'source_LPIPS_v1_minus_selected': selected[1],
        'target_MIoU_selected_minus_v1': selected[2],
        'evaluation_result_sha256': {LEFT: left_hash, RIGHT: right_hash},
        'target8_sha256': digest(args.ours_target8),
        'common_manifest_sha256': digest(args.common_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
