"""Bind frozen V1/Selected raw/+M0 renders to the common source evaluator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = ('V1', 'SELECTED')
MODES = ('raw', 'source_m0')


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render-plan-root', type=Path, required=True)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--original-references', type=Path, required=True)
    parser.add_argument('--file4-output-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, required=True)
    args = parser.parse_args()
    if (args.output_root.exists() or not 1 <= args.expected_samples <= 502
            or not str(args.file4_output_root).startswith(
                '/file4/youngwoo/deformation/experiments/retargeting-v1/'
                'source-sim3-test508-20261028/ours-v1-selected-scale-ungiven-common502-v1')):
        raise ValueError('fresh file4 Ours binding and declared common count required')
    from deformnext_benchmark.io import canonical_hash, digest
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    if len(subset) != args.expected_samples or len(set(subset)) != len(subset):
        raise ValueError('declared final common-success sample IDs differ')
    all_scores = [row for path in args.scores for row in rows(path)]
    scores = {(row['sample_id'], row['method']): row for row in all_scores}
    if (len(all_scores) != 1004 or len(scores) != 1004
            or len({sid for sid, _ in scores}) != 502):
        raise ValueError('full frozen Ours 502×2 surface score closure required')
    originals = {row['sample_id']: row for row in rows(args.original_references)}
    if not set(subset) <= originals.keys():
        raise ValueError('original source-view reference population differs')
    jobs = {}
    chunks = sorted(args.render_plan_root.glob('chunk[0-9][0-9][0-9]'))
    for chunk in chunks:
        plan = json.loads((chunk / 'plan.json').read_text())
        complete = json.loads((chunk / 'completed.json').read_text())
        if (plan['status'] != 'PREPARED_OURS_V1_SELECTED_GT_FREE_RAW_AND_M0_RENDER'
                or plan['jobs'] != len(plan['sample_ids']) * 12
                or complete['status'] != 'PASS_RENDER_CHUNK_ARCHIVED_ON_FILE4'
                or complete['chunk'] != chunk.name):
            raise ValueError('Ours paired render chunk/archive differs')
        for job in rows(chunk / 'jobs.jsonl'):
            key = job['method'], job['sample_id'], job['camera_id']
            if key in jobs:
                raise ValueError('duplicate frozen Ours render job')
            jobs[key] = chunk.name, job
    if len(jobs) != 502 * 12:
        raise ValueError('Ours raw/+M0 camera8/9/10 render closure differs')
    refs, preds = [], []
    for sid in subset:
        ref = originals[sid]
        refs.append(ref)
        ref_views = {view['camera_id']: view for view in ref['views']}
        if set(ref_views) != {9, 10}:
            raise ValueError('source camera9/10 references required')
        for arm in ARMS:
            score = scores[sid, 'ours_' + arm.lower() + '_image_scale_1024']
            if score['registration_reference'] != 'input_source_M0_only':
                raise ValueError('M0-only scorer required')
            for mode in MODES:
                method = f'ours_{arm.lower()}_{mode}'
                aligned = [jobs[method, sid, camera] for camera in (8, 9, 10)]
                raw = np.asarray(score['to_world'], dtype=np.float64)
                correction = np.asarray(score['correction_sim3'], dtype=np.float64)
                expected_transform = raw if mode == 'raw' else correction @ raw
                if raw.shape != (4, 4) or correction.shape != (4, 4):
                    raise ValueError('Ours raw/+M0 transforms must be 4×4')
                views = []
                for camera, (chunk_name, job) in zip((8, 9, 10), aligned):
                    if (job['method'] != method or job['sample_id'] != sid
                            or job['camera_id'] != camera or job['frame_id'] != ref['frame_id']
                            or job['protocol_sha256'] != ref['protocol_sha256']
                            or not np.allclose(np.asarray(job['to_world']),
                                               expected_transform, rtol=0, atol=1e-8)):
                        raise ValueError('Ours camera/transform binding differs: ' + sid)
                    mesh = Path('/') / job['metric_mesh']
                    glb = Path('/') / job['mesh']
                    png = args.file4_output_root / chunk_name / job['output_rgba']
                    receipt_path = png.with_suffix('.receipt.json')
                    if not all(path.is_file() for path in (mesh, glb, png, receipt_path)):
                        raise FileNotFoundError('Ours prediction/render binding missing: ' + sid)
                    receipt = json.loads(receipt_path.read_text())
                    expected = {'metric_mesh_sha256': score['prediction_sha256'],
                                'input_sha256': digest(glb),
                                'to_world_sha256': canonical_hash(job['to_world']),
                                'output_sha256': digest(png),
                                'protocol_sha256': job['protocol_sha256']}
                    if any(receipt.get(key) != value for key, value in expected.items()):
                        raise ValueError('Ours render receipt/hash differs: ' + str(png))
                    if camera in ref_views:
                        view = ref_views[camera]
                        if (receipt['camera_sha256'] != view['camera_sha256']
                                or receipt['render_sha256'] != view['render_sha256']):
                            raise ValueError('Ours source camera/renderer differs: ' + sid)
                        views.append({'camera_id': camera, 'rgba': str(png).lstrip('/'),
                                      'camera_sha256': view['camera_sha256'],
                                      'render_sha256': view['render_sha256']})
                first = aligned[0][1]
                preds.append({'method': method, 'sample_id': sid, 'seed': 0,
                              'status': 'ok', 'mesh': first['metric_mesh'],
                              'to_world': first['to_world'], 'frame_id': first['frame_id'],
                              'protocol_sha256': first['protocol_sha256'], 'views': views})
    args.output_root.mkdir(parents=True)
    (args.output_root / 'references.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in refs))
    (args.output_root / 'predictions.jsonl').write_text(
        ''.join(json.dumps(row, sort_keys=True) + '\n' for row in preds))
    report = {'status': 'PASS_OURS_V1_SELECTED_RAW_AND_M0_EVALUATOR_BINDING',
              'samples': len(refs), 'predictions': len(preds),
              'methods': [f'ours_{arm.lower()}_{mode}' for arm in ARMS for mode in MODES],
              'source_views': [9, 10], 'target_camera8_rendered_separately': True,
              'source_registration_reference': 'input_source_M0_only',
              'target_M1_used_for_registration': False,
              'sample_ids': subset}
    (args.output_root / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'samples', 'predictions')}), flush=True)


if __name__ == '__main__':
    main()
