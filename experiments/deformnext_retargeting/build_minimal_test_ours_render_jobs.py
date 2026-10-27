"""Plan common raw/+source-M0 Cycles renders for frozen V1 and Selected."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ARMS = {'V1': 'shape', 'SELECTED': 'top1-pbr'}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative(path: Path) -> str:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('file4 artifact path must be absolute')
    return str(path).lstrip('/')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--common-subset', type=Path, required=True)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--target8-jobs', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--expected-samples', type=int, default=502)
    parser.add_argument('--chunk-samples', type=int, default=16)
    args = parser.parse_args()
    if args.output_root.exists() or args.chunk_samples < 1:
        raise ValueError('fresh render plan and positive chunk size required')
    subset = json.loads(args.common_subset.read_text())['sample_ids']
    if (len(subset) != args.expected_samples or len(set(subset)) != len(subset)
            or not 1 <= len(subset) <= 502):
        raise ValueError('declared common success subset required')
    scored = [row for path in args.scores for row in rows(path)]
    scores = {(row['sample_id'], row['method']): row for row in scored}
    if len(scores) != len(scored):
        raise ValueError('duplicate frozen surface scores')
    templates = {row['sample_id']: row for row in rows(args.target8_jobs)
                 if row['method'] == 'trellis2'}
    references = {row['sample_id']: row for row in rows(args.references)}
    if not set(subset) <= templates.keys() or not set(subset) <= references.keys():
        raise ValueError('frozen camera/reference population differs')
    args.output_root.mkdir(parents=True)
    total = 0
    for start in range(0, len(subset), args.chunk_samples):
        group = subset[start:start + args.chunk_samples]
        chunk = args.output_root / f'chunk{start // args.chunk_samples:03d}'
        chunk.mkdir()
        jobs, inputs = [], set()
        for sid in group:
            template, reference = templates[sid], references[sid]
            if (template['protocol_sha256'] != reference['protocol_sha256']
                    or template['frame_id'] != reference['frame_id']):
                raise ValueError('common camera protocol mismatch: ' + sid)
            refs = {view['camera_id']: view['camera'] for view in reference['views']}
            if set(refs) != {9, 10}:
                raise ValueError('source cameras 9/10 required')
            cameras = {8: template['camera'], 9: refs[9], 10: refs[10]}
            for arm, stage in ARMS.items():
                method = 'ours_' + arm.lower() + '_image_scale_1024'
                score = scores.get((sid, method))
                if score is None or score['registration_reference'] != 'input_source_M0_only':
                    raise ValueError('paired Ours surface score missing: ' + arm + '/' + sid)
                case = args.archive_root / arm / sid / stage
                export = json.loads((case / 'benchmark.json').read_text())
                receipt = json.loads((case / 'archive-receipt.json').read_text())
                glb, prediction = case / 'benchmark.glb', case / 'prediction.npz'
                if (export['status'] != 'PASS_OURS_FROZEN_GT_SCALE_FREE_1024_EXPORT'
                        or export['sample_id'] != sid
                        or export['prediction_sha256'] != score['prediction_sha256']
                        or receipt['sha256']['benchmark.glb'] != export['glb_sha256']
                        or not glb.is_file() or not prediction.is_file()):
                    raise ValueError('frozen Ours export/score binding differs: ' + arm + '/' + sid)
                raw = np.asarray(export['to_world'], dtype=np.float64)
                scored_world = np.asarray(score['to_world'], dtype=np.float64)
                correction = np.asarray(score['correction_sim3'], dtype=np.float64)
                if (raw.shape != (4, 4) or correction.shape != (4, 4)
                        or not np.allclose(raw, scored_world, atol=1e-8, rtol=0)
                        or not np.allclose(raw[3], [0, 0, 0, 1])
                        or not np.allclose(correction[3], [0, 0, 0, 1])):
                    raise ValueError('source-M0 correction or native world frame differs: ' + sid)
                inputs.update((relative(glb), relative(prediction)))
                for mode, transform in (('raw', raw), ('source_m0', correction @ raw)):
                    for camera_id in (8, 9, 10):
                        job = dict(template,
                                   method=f'ours_{arm.lower()}_{mode}',
                                   mesh=relative(glb), metric_mesh=relative(prediction),
                                   trusted_material_dump=None, camera_id=camera_id,
                                   camera=cameras[camera_id], to_world=transform.tolist(),
                                   output_rgba=f'renders/{arm.lower()}/{mode}/{sid}/view{camera_id:02d}.png')
                        jobs.append(job)
        if len(jobs) != len(group) * 12:
            raise ValueError('two arms × two conditions × three cameras required')
        (chunk / 'jobs.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in jobs))
        (chunk / 'input-files.txt').write_text(''.join(path + '\n' for path in sorted(inputs)))
        (chunk / 'plan.json').write_text(json.dumps({
            'status': 'PREPARED_OURS_V1_SELECTED_GT_FREE_RAW_AND_M0_RENDER',
            'sample_ids': group, 'jobs': len(jobs), 'input_files': len(inputs),
            'arms': ['V1', 'SELECTED'], 'conditions': ['raw', 'source_m0'],
            'cameras': [8, 9, 10]}, indent=2) + '\n')
        total += len(jobs)
    if total != args.expected_samples * 12:
        raise ValueError('declared paired Ours render population differs')
    print(json.dumps({'status': 'PASS_OURS_V1_SELECTED_RAW_AND_M0_RENDER_PLAN',
                      'samples': len(subset), 'jobs': total,
                      'chunks': (len(subset) + args.chunk_samples - 1) // args.chunk_samples}),
          flush=True)


if __name__ == '__main__':
    main()
