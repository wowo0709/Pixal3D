"""Audit archived TEST508 render meshes against untouched metric surfaces."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ALLOWED_BUDGETS = {250_000, 500_000, 1_000_000, 2_000_000, 5_000_000}
Q99_LIMIT = 0.01


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign.read_text())
    if campaign['samples'] != 508 or len(campaign['plans']) != 1016:
        raise ValueError('frozen paired TEST508 campaign required')
    keys = {(entry['arm'], entry['sample_id']) for entry in campaign['plans']}
    ids = {entry['sample_id'] for entry in campaign['plans']}
    if (len(keys) != 1016 or len(ids) != 508
            or keys != {(arm, sid) for sid in ids for arm in ('V1', 'SELECTED')}):
        raise ValueError('TEST508 arm/sample identity differs')
    counts = Counter()
    budgets = Counter()
    q99_max = 0.0
    q99_max_sample = None
    for entry in campaign['plans']:
        arm, sid = entry['arm'], entry['sample_id']
        if arm == 'V1':
            plan = json.loads(Path(entry['shape_plan']).read_text())
            root = Path(plan['output'])
            expected_arm, expected_stage = 'vanilla', 'shape'
        elif arm == 'SELECTED':
            plan = json.loads(Path(entry['top1_pbr_plan']).read_text())
            root = Path(plan['output'])
            expected_arm, expected_stage = 'top1-only', 'top1-pbr'
        else:
            raise ValueError('unexpected TEST arm')
        report_path = root / 'benchmark.json'
        receipt_path = root / 'archive-receipt.json'
        if not receipt_path.is_file():
            if report_path.is_file():
                counts['finished_export_not_archived'] += 1
            continue
        if not report_path.is_file():
            raise FileNotFoundError('archived render report missing: ' + str(root))
        report = json.loads(report_path.read_text())
        receipt = json.loads(receipt_path.read_text())
        budget = report['render_face_budget']
        q99 = report['render_geometry_registration_q99_world']
        if (report['status'] != 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT'
                or report['sample_id'] != sid or report['texture_arm'] != expected_arm
                or budget not in ALLOWED_BUDGETS or not 0 <= q99 <= Q99_LIMIT
                or report['surface_metrics_use_untouched_prediction'] is not True
                or report['GT_fitting_or_alignment'] is not False
                or report['remesh'] is not False
                or abs(report['image_estimated_scale'] - float(plan['row']['mesh_scale'])) > 1e-10
                or receipt['status'] != 'PASS_FILE4_ARCHIVE_CHECKSUM_VERIFIED'
                or not receipt['remote'].endswith('/' + arm + '/' + sid + '/' + expected_stage)
                or receipt['sha256']['benchmark.glb'] != report['glb_sha256']
                or receipt['sha256']['prediction.npz'] != report['prediction_sha256']
                or receipt['sha256']['benchmark.json'] != sha(report_path)):
            raise ValueError('render fidelity or file4 SHA receipt differs: ' + arm + '/' + sid)
        counts[arm] += 1
        budgets[budget] += 1
        if q99 > q99_max:
            q99_max, q99_max_sample = q99, arm + '/' + sid
    result = {
        'status': 'PASS_PARTIAL_RENDER_FIDELITY_AUDIT'
                  if counts['V1'] < 508 or counts['SELECTED'] < 508
                  else 'PASS_TEST508_TWO_ARM_RENDER_FIDELITY_AUDIT',
        'expected_per_arm': 508,
        'archived': {arm: counts[arm] for arm in ('V1', 'SELECTED')},
        'finished_export_not_archived': counts['finished_export_not_archived'],
        'actual_render_face_budget_counts': {str(k): budgets[k] for k in sorted(ALLOWED_BUDGETS)},
        'q99_world_max': q99_max,
        'q99_world_max_sample': q99_max_sample,
        'q99_world_limit': Q99_LIMIT,
        'geometry_metrics_use_untouched_prediction': True,
        'GT_fitting_or_alignment': False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + '.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.output)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
