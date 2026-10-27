"""Export one disjoint TEST508 shard on node17 using file4 and publish to node16."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time


NODE16 = ['ssh', '-o', 'BatchMode=yes', '-p', '55555', 'youngwoo@n16.unist.info']
FILE4 = ['ssh', '-S', '/tmp/file4-ctl-v2', '-o', 'BatchMode=yes',
         '-p', '55555', 'rvi@10.0.12.170']
ROOT16 = Path('/home/youngwoo/retargeting-unified-test508')
INPUT_KEYS = ('prediction', 'render_geometry', 'anchor_evaluation', 'generation')


def remote(script: str) -> str:
    result = subprocess.run(NODE16 + ['python3 -c ' + shlex.quote(script)],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()


def current(output: Path, report: Path, row: dict) -> bool:
    script = ('import json,pathlib\n'
              f'o=pathlib.Path({str(output)!r});p=pathlib.Path({str(report)!r})\n'
              'if o.is_file() and p.is_file():\n'
              f' d=json.loads(p.read_text());assert d["status"]=="PASS_COMMON_BENCHMARK_MESH_EXPORT" and d["prediction"]=={row["prediction"]!r} and d["texture_size"]==2048;print("COMPLETE")\n'
              'elif o.exists() or p.exists():raise RuntimeError("partial remote export")\n'
              'else:print("MISSING")\n')
    state = remote(script)
    if state not in ('COMPLETE', 'MISSING'):
        raise ValueError('unexpected remote export state: ' + state)
    return state == 'COMPLETE'


def pull_inputs(row: dict, local: Path) -> None:
    paths = [row[key] for key in INPUT_KEYS]
    command = 'tar -cf - -- ' + ' '.join(shlex.quote(path) for path in paths)
    source = subprocess.Popen(FILE4 + [command], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    target = subprocess.run(['tar', '-xf', '-', '-C', str(local)],
                            stdin=source.stdout, capture_output=True, text=True)
    source.stdout.close()
    stderr = source.stderr.read().decode(errors='replace')
    code = source.wait()
    if code or target.returncode:
        raise RuntimeError(f'file4 transfer failed {code}/{target.returncode}: '
                           + stderr[-500:] + target.stderr[-500:])
    if any(not (local / path.lstrip('/')).is_file() for path in paths):
        raise FileNotFoundError('file4 transfer omitted a frozen input')


def export(row: dict, local: Path, gpu: int, script: Path,
           pixal3d_repo: Path, benchmark_repo: Path) -> tuple[Path, Path, float]:
    output = local / 'export.glb'
    command = ['/opt/conda/envs/pixal3d/bin/python', str(script),
               '--pixal3d-repo', str(pixal3d_repo),
               '--benchmark-repo', str(benchmark_repo)]
    for key, flag in (('prediction', '--prediction'),
                      ('render_geometry', '--render-geometry'),
                      ('anchor_evaluation', '--anchor-evaluation'),
                      ('generation', '--generation')):
        command.extend([flag, str(local / row[key].lstrip('/'))])
    command.extend(['--mesh-scale', str(row['mesh_scale']),
                    '--output', str(output), '--texture-size', '2048'])
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1')
    start = time.monotonic()
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode or not output.is_file() or not output.with_suffix('.json').is_file():
        raise RuntimeError('local exporter failed: ' + result.stdout[-1500:] + result.stderr[-1500:])
    return output, output.with_suffix('.json'), time.monotonic() - start


def publish(row: dict, output: Path, report: Path, rank: int) -> None:
    destination = ROOT16 / row['export_relpath']
    receipt = destination.with_suffix('.json')
    doc = json.loads(report.read_text())
    doc.update(prediction=row['prediction'], render_geometry=row['render_geometry'],
               glb=str(destination), export_execution_host='node17',
               cross_node_pilot_rgb_mae=2.5436302166781388e-05,
               cross_node_pilot_mask_mismatch_pixels=0)
    report.write_text(json.dumps(doc, indent=2) + '\n')
    upload_glb = Path(str(destination) + f'.upload-{rank}')
    upload_report = Path(str(receipt) + f'.upload-{rank}')
    remote(f'from pathlib import Path;Path({str(destination.parent)!r}).mkdir(parents=True,exist_ok=True)')
    for source, target in ((output, upload_glb), (report, upload_report)):
        subprocess.run(['scp', '-q', '-P', '55555', str(source),
                        'youngwoo@n16.unist.info:' + str(target)], check=True)
    script = ('import hashlib,json,pathlib,os\n'
              f'o=pathlib.Path({str(destination)!r});p=pathlib.Path({str(receipt)!r})\n'
              f'to=pathlib.Path({str(upload_glb)!r});tp=pathlib.Path({str(upload_report)!r})\n'
              'assert not o.exists() and not p.exists() and to.is_file() and tp.is_file()\n'
              'd=json.loads(tp.read_text());h=hashlib.sha256()\n'
              'with to.open("rb") as f:\n'
              ' for block in iter(lambda:f.read(8388608),b""):h.update(block)\n'
              'assert h.hexdigest()==d["glb_sha256"]\n'
              'os.replace(to,o);os.replace(tp,p);print("PUBLISHED")\n')
    if remote(script) != 'PUBLISHED':
        raise RuntimeError('node16 did not publish the frozen export')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--exporter', type=Path, required=True)
    parser.add_argument('--pixal3d-repo', type=Path, required=True)
    parser.add_argument('--benchmark-repo', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--world-size', type=int, default=9)
    parser.add_argument('--min-index', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.world_size != 9 or not 3 <= args.rank <= 8:
        raise ValueError('local ranks 3-8 of the frozen nine-worker export only')
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    if len(rows) != 3048:
        raise ValueError('frozen export manifest must contain 3048 rows')
    completed = 0
    for index, row in enumerate(rows):
        if index % args.world_size != args.rank or index < args.min_index:
            continue
        destination = ROOT16 / row['export_relpath']
        if current(destination, destination.with_suffix('.json'), row):
            continue
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix=f'common-export-rank{args.rank}-') as dirname:
            local = Path(dirname)
            pull_inputs(row, local)
            output, report, seconds = export(row, local, args.gpu, args.exporter,
                                              args.pixal3d_repo, args.benchmark_repo)
            publish(row, output, report, args.rank)
        completed += 1
        print(json.dumps({'index': index, 'arm': row['arm'], 'gpu': args.gpu,
                          'export_seconds': round(seconds, 2),
                          'total_seconds': round(time.monotonic() - start, 2),
                          'new_exports': completed}), flush=True)
        if args.limit and completed >= args.limit:
            break
    print(json.dumps({'rank': args.rank, 'gpu': args.gpu,
                      'new_exports': completed, 'status': 'PASS_LOCAL_EXPORT_SHARD'}), flush=True)


if __name__ == '__main__':
    main()
