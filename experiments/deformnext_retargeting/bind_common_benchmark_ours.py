"""Bind six complete saved ours outputs to the frozen node16 9/10 scorer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = {'RGB1_CALIBRATED', 'zero-source4', 'zero-edited4', 'V1', 'RETARGET', 'SELECTED'}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def abs_relative(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError('absolute path required')
    return str(path.relative_to('/'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--prep-root', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path,
                        help='saved export/render root when writing an exploratory binding elsewhere')
    parser.add_argument('--sample-ids', type=Path,
                        help='optional frozen exploratory subset; omitted for full TEST508 closure')
    args = parser.parse_args()
    artifact_root = args.artifact_root or args.output_root
    ref_output = args.output_root / 'source910-references.jsonl'
    pred_output = args.output_root / 'source910-ours-predictions.jsonl'
    if ref_output.exists() or pred_output.exists():
        raise FileExistsError('immutable common benchmark bindings already exist')
    refs = read_jsonl(args.prep_root / 'references.jsonl')
    if args.sample_ids:
        subset = json.loads(args.sample_ids.read_text())
        if not isinstance(subset, list) or len(subset) != len(set(subset)):
            raise ValueError('exploratory sample IDs must be a unique JSON list')
        refs = [ref for ref in refs if ref['sample_id'] in set(subset)]
        if len(refs) != len(subset):
            raise ValueError('exploratory sample IDs differ from TEST references')
    index = {x['sample_id']: x for x in refs}
    rows = [row for row in read_jsonl(args.manifest) if row['sample_id'] in index]
    expected = 6 * len(refs)
    if (not refs or len(index) != len(refs) or len(rows) != expected
            or (not args.sample_ids and (len(refs) != 508 or expected != 3048))):
        raise ValueError('expected complete TEST sample population for six methods')
    rebound = []
    for ref in refs:
        r = dict(ref)
        raw = args.prep_root / ref['mesh']
        if not raw.is_file():
            raise FileNotFoundError(raw)
        r['mesh'] = abs_relative(raw)
        r['views'] = [dict(view, rgba=abs_relative(args.baseline_root / 'shared/gt-a02' / view['rgba']))
                      for view in ref['views']]
        for view in r['views']:
            if not (Path('/') / view['rgba']).is_file():
                raise FileNotFoundError(view['rgba'])
        rebound.append(r)
    predictions = []
    for row in rows:
        sid, arm = row['sample_id'], row['arm']
        if sid not in index or arm not in ARMS:
            raise ValueError('unexpected frozen method or sample')
        export = artifact_root / row['export_relpath']
        report_path = export.with_suffix('.json')
        if not export.is_file() or not report_path.is_file():
            raise FileNotFoundError('incomplete frozen export: ' + str(export))
        report = json.loads(report_path.read_text())
        if (report['status'] != 'PASS_COMMON_BENCHMARK_MESH_EXPORT'
                or report['prediction'] != row['prediction'] or report['texture_size'] != 2048):
            raise ValueError('export provenance differs from frozen raw prediction')
        views = []
        for original in index[sid]['views']:
            cid = original['camera_id']
            png = artifact_root / 'renders' / arm / sid / f'view{cid:02d}.png'
            if not png.is_file() or not png.with_suffix('.receipt.json').is_file():
                raise FileNotFoundError('incomplete frozen Cycles render: ' + str(png))
            views.append({'camera_id': cid, 'rgba': abs_relative(png),
                          'camera_sha256': original['camera_sha256'],
                          'render_sha256': original['render_sha256']})
        predictions.append({'method': arm, 'sample_id': sid, 'seed': 0, 'status': 'ok',
                            'mesh': abs_relative(Path(row['prediction'])),
                            'to_world': report['to_world'],
                            'frame_id': index[sid]['frame_id'],
                            'protocol_sha256': index[sid]['protocol_sha256'],
                            'views': views})
    if (len(predictions) != expected
            or len({(x['method'], x['sample_id']) for x in predictions}) != expected):
        raise ValueError('incomplete or duplicate six-arm prediction population')
    args.output_root.mkdir(parents=True, exist_ok=True)
    ref_output.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in rebound))
    pred_output.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in predictions))
    print(json.dumps({'status': 'PASS_COMMON_OURS_SOURCE910_BINDING',
                      'references': len(rebound), 'predictions': len(predictions),
                      'reference_output': str(ref_output),
                      'prediction_output': str(pred_output)}))


if __name__ == '__main__':
    main()
