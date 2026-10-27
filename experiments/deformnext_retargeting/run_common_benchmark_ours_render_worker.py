"""Render completed frozen ours exports at cameras 8/9/10 in common Cycles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


def as_relative_absolute(path: Path) -> str:
    return str(path.relative_to('/'))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--max-wait-seconds', type=int, default=259200)
    args = parser.parse_args()
    rows = read_jsonl(args.manifest)
    if len(rows) != 3048 or not 0 <= args.rank < args.world_size:
        raise ValueError('expected six frozen TEST508 arms and valid render shard')
    refs = {x['sample_id']: x for x in read_jsonl(args.prep_root / 'references.jsonl')}
    inputs = {x['sample_id']: x for x in read_jsonl(args.prep_root / 'inference-inputs-with-meshes.jsonl')}
    if len(refs) != 508 or len(inputs) != 508 or set(refs) != set(inputs):
        raise ValueError('reference/conditioning population mismatch')
    own = [(index, row) for index, row in enumerate(rows) if index % args.world_size == args.rank]
    blender = args.benchmark_repo / 'setup/tools/blender-4.2.3-linux-x64/blender'
    renderer = args.benchmark_repo / 'deformnext_benchmark/render_blender.py'
    protocol = args.prep_root / 'protocol.json'
    last_progress = time.monotonic()
    completed = set()
    while len(completed) < len(own):
        progressed = False
        for index, row in own:
            if index in completed:
                continue
            sid, arm = row['sample_id'], row['arm']
            output = args.root / row['export_relpath']
            report_path = output.with_suffix('.json')
            if not output.is_file() or not report_path.is_file():
                continue
            report = json.loads(report_path.read_text())
            if (report.get('status') != 'PASS_COMMON_BENCHMARK_MESH_EXPORT'
                    or report.get('prediction') != row['prediction']
                    or report.get('texture_size') != 2048):
                raise ValueError('invalid frozen export receipt: ' + str(report_path))
            record_root = args.root / 'render-records' / arm / sid
            record_root.mkdir(parents=True, exist_ok=True)
            ref = refs[sid]
            views = [(8, inputs[sid]['conditioning_cameras'][0])]
            views.extend((v['camera_id'], v['camera']) for v in ref['views'])
            if [v[0] for v in views] != [8, 9, 10]:
                raise ValueError('expected target8 and two source views')
            for camera_id, camera in views:
                image = args.root / 'renders' / arm / sid / f'view{camera_id:02d}.png'
                receipt = image.with_suffix('.receipt.json')
                if image.is_file() and receipt.is_file():
                    continue
                if image.exists() or receipt.exists():
                    raise FileExistsError('partial Cycles render: ' + str(image))
                job = {
                    'schema': 'deformnext-render-job-v1',
                    'sample_id': sid, 'method': arm, 'camera_id': camera_id,
                    'protocol_sha256': ref['protocol_sha256'],
                    'frame_id': ref['frame_id'], 'camera': camera,
                    'mesh': as_relative_absolute(output),
                    'metric_mesh': as_relative_absolute(Path(row['prediction'])),
                    'to_world': report['to_world'],
                    'trusted_material_dump': None,
                    'output_rgba': as_relative_absolute(image),
                }
                job_path = record_root / f'view{camera_id:02d}-job.json'
                if job_path.exists():
                    if json.loads(job_path.read_text()) != job:
                        raise ValueError('render job changed: ' + str(job_path))
                else:
                    job_path.write_text(json.dumps(job, sort_keys=True, indent=2) + '\n')
                command = [str(blender), '-b', '-t', '4', '--python-exit-code', '1',
                           '--python', str(renderer), '--', '--protocol', str(protocol),
                           '--job', str(job_path), '--root', '/', '--device', 'CPU']
                log_path = record_root / f'view{camera_id:02d}.log'
                before = time.monotonic()
                with log_path.open('w') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                proof = {'sample_id': sid, 'arm': arm, 'camera_id': camera_id,
                         'returncode': result.returncode, 'seconds': time.monotonic() - before,
                         'image': str(image), 'receipt': str(receipt),
                         'job': str(job_path), 'log': str(log_path)}
                (record_root / f'view{camera_id:02d}-terminal.json').write_text(
                    json.dumps(proof, indent=2) + '\n')
                print(json.dumps({'rank': args.rank, 'index': index, 'arm': arm,
                                  'camera': camera_id, 'returncode': result.returncode,
                                  'seconds': round(proof['seconds'], 2)}), flush=True)
                if result.returncode or not image.is_file() or not receipt.is_file():
                    raise RuntimeError('Cycles render failed: ' + str(log_path))
            completed.add(index)
            progressed = True
        if not progressed:
            if time.monotonic() - last_progress > args.max_wait_seconds:
                raise TimeoutError('frozen export workers did not complete all render inputs')
            time.sleep(15)
        else:
            last_progress = time.monotonic()
    print(json.dumps({'rank': args.rank, 'completed_exports': len(completed),
                      'status': 'PASS_COMMON_OURS_RENDER_SHARD'}), flush=True)


if __name__ == '__main__':
    main()
