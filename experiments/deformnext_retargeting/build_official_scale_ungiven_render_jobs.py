"""Plan common Cycles raw/+M0 renders for official RGB1 no-scale Pixal3D."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def relative_to_remote_root(path: Path) -> str:
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('node16 artifact must have an absolute path')
    return str(path).lstrip('/')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, nargs='+', required=True)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--official-root', type=Path, required=True)
    parser.add_argument('--target8-jobs', type=Path, required=True)
    parser.add_argument('--references', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--chunk-samples', type=int, default=16)
    parser.add_argument('--expected-samples', type=int, default=502,
                        help='Use 1 only for a mechanical exporter/renderer pilot')
    args = parser.parse_args()
    if args.output_root.exists() or args.chunk_samples < 1:
        raise ValueError('new output root and positive chunk size required')
    scored = [row for path in args.scores for row in read_jsonl(path)]
    scores = {row['sample_id']: row for row in scored}
    if len(scored) != args.expected_samples or len(scores) != args.expected_samples or any(
            row['method'] != 'pixal3d_official_given_camera_scale_ungiven_1024_rgb1'
            or row['registration_reference'] != 'input_source_M0_only'
            or row['native_mesh_scale'] != 1.0 for row in scored):
        raise ValueError('complete declared official raw/+M0 scores required')
    target8 = {row['sample_id']: row for row in read_jsonl(args.target8_jobs)
               if row['method'] == 'trellis2'}
    references = {row['sample_id']: row for row in read_jsonl(args.references)}
    if not set(scores) <= set(target8) or not set(scores) <= set(references):
        raise ValueError('original target8/reference population differs')
    args.output_root.mkdir(parents=True)
    ids = sorted(scores)
    total_jobs = 0
    for start in range(0, len(ids), args.chunk_samples):
        group = ids[start:start + args.chunk_samples]
        chunk = args.output_root / f'chunk{start//args.chunk_samples:03d}'
        chunk.mkdir()
        jobs, inputs = [], set()
        for sid in group:
            score = scores[sid]
            case = args.official_root / 'cases' / sid
            generation = json.loads((case / 'generation.json').read_text())
            export = json.loads((args.exports / 'cases' / sid / 'prediction.json').read_text())
            glb = Path(export['glb'])
            prediction = case / 'prediction.npz'
            if (generation['status'] != 'PASS_OFFICIAL_PIXAL3D_GIVEN1024_SCALE_UNGIVEN'
                    or generation['mesh_scale_supplied']
                    or generation['mesh_scale_effective_native_default'] != 1.0
                    or generation['prediction_sha256'] != score['prediction_sha256']
                    or generation['prediction_sha256'] != export['prediction_sha256']
                    or export['status'] != 'PASS_OFFICIAL_RGB1_SCALE_UNGIVEN_RENDER_EXPORT'
                    or not export['pre_simplify'] or not glb.is_file() or not prediction.is_file()):
                raise ValueError('official prediction/export/score binding differs: ' + sid)
            raw = np.asarray(export['to_world'], dtype=np.float64)
            scored_world = np.asarray(score['to_world'], dtype=np.float64)
            correction = np.asarray(score['correction_sim3'], dtype=np.float64)
            if (raw.shape != (4, 4) or correction.shape != (4, 4)
                    or not np.allclose(raw, scored_world, atol=1e-8, rtol=0)
                    or not np.allclose(raw[3], [0, 0, 0, 1])
                    or not np.allclose(correction[3], [0, 0, 0, 1])):
                raise ValueError('official camera transform or M0 correction differs: ' + sid)
            template, reference = target8[sid], references[sid]
            if (template['protocol_sha256'] != reference['protocol_sha256']
                    or template['frame_id'] != reference['frame_id']):
                raise ValueError('camera/reference protocol differs: ' + sid)
            refs = {view['camera_id']: view['camera'] for view in reference['views']}
            if set(refs) != {9, 10}:
                raise ValueError('source camera9/10 references required')
            cameras = {8: template['camera'], 9: refs[9], 10: refs[10]}
            inputs.update((relative_to_remote_root(glb), relative_to_remote_root(prediction)))
            for mode, transform in [('raw', raw), ('source_m0', correction @ raw)]:
                for camera_id in (8, 9, 10):
                    job = dict(template, method=f'pixal3d_official_rgb1_{mode}',
                               mesh=relative_to_remote_root(glb),
                               metric_mesh=relative_to_remote_root(prediction),
                               trusted_material_dump=None, camera_id=camera_id,
                               camera=cameras[camera_id], to_world=transform.tolist(),
                               output_rgba=f'renders/{mode}/{sid}/view{camera_id:02d}.png')
                    jobs.append(job)
        if len(jobs) != len(group) * 6:
            raise ValueError('paired raw/+M0 × three-camera job count differs')
        (chunk / 'jobs.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in jobs))
        (chunk / 'input-files.txt').write_text(''.join(path + '\n' for path in sorted(inputs)))
        (chunk / 'plan.json').write_text(json.dumps({
            'status': 'PREPARED_OFFICIAL_RGB1_SCALE_UNGIVEN_RAW_AND_M0_RENDER',
            'sample_ids': group, 'jobs': len(jobs), 'input_files': len(inputs),
            'conditions': ['raw', 'source_m0'], 'cameras': [8, 9, 10]}, indent=2) + '\n')
        total_jobs += len(jobs)
    if total_jobs != args.expected_samples * 2 * 3:
        raise ValueError('declared samples × two conditions × three cameras required')
    print(json.dumps({'status': 'PASS_OFFICIAL_RGB1_RAW_AND_M0_RENDER_PLAN',
                      'samples': args.expected_samples, 'jobs': total_jobs,
                      'chunks': (args.expected_samples + args.chunk_samples - 1) // args.chunk_samples}), flush=True)


if __name__ == '__main__':
    main()
