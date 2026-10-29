"""Index the public BenchUp release without inventing camera or target geometry."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'download-manifest.json').read_text())
    if manifest['repo'] != 'Inbar2344/BenchUp' or manifest['revision'] != 'a8b87439281c7f64788231f0935b6d6aedc3ba85':
        raise ValueError('unexpected BenchUp provenance')
    records = {row['path']: row for row in manifest['files']}
    if len(records) != 753:
        raise ValueError('unexpected file count or duplicate paths')
    for path, row in records.items():
        f = root / path
        if not f.is_file() or f.stat().st_size != row['size'] or digest(f) != row['sha256']:
            raise ValueError('file verification failed: ' + path)
    objects = sorted({path.split('/')[0] for path in records})
    result = []
    for obj in objects:
        mesh = f'{obj}/source/src.glb'
        pc = f'{obj}/source/pc.npz'
        sources = sorted(path for path in records if path.startswith(f'{obj}/source/mv/') and path.endswith('.webp'))
        edits = sorted(path for path in records if path.startswith(f'{obj}/edited_mv/') and path.endswith('.png'))
        if mesh not in records or pc not in records or not sources or not edits:
            raise ValueError('incomplete object: ' + obj)
        for edit in edits:
            name = Path(edit).stem
            if not re.fullmatch(r'render_\d{4}', name):
                raise ValueError('unexpected target name: ' + edit)
            same_index = f'{obj}/source/mv/{name}.webp'
            result.append({'sample_id': f'{obj}__{name}', 'object_id': obj,
                           'target_image': edit, 'source_mesh': mesh,
                           'source_points': pc, 'source_images': sources,
                           'same_index_source_candidate': same_index if same_index in records else None,
                           'source_camera_metadata': None, 'target_camera_metadata': None,
                           'target_mesh': None, 'camera_status': 'unavailable_requires_calibration'})
    if len(objects) != 23 or len(result) != 109 or len({r['sample_id'] for r in result}) != 109:
        raise ValueError('unexpected published release composition')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in result))
    print(json.dumps({'status': 'PASS', 'objects': len(objects), 'samples': len(result),
                      'same_index_source_candidates': sum(r['same_index_source_candidate'] is not None for r in result),
                      'manifest_sha256': digest(args.output), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
