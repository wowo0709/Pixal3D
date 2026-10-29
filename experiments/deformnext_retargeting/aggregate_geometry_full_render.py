"""Aggregate completed geometry render shards without repeatedly decompressing features."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-root', type=Path, required=True)
    p.add_argument('--references', type=Path, required=True)
    p.add_argument('--benchmark-root', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    tasks = rows(a.score_root / 'tasks.jsonl')
    paths = sorted((a.score_root / 'shards').glob('rank*.jsonl'))
    if len(paths) != 12 or len(tasks) != 9018:
        raise ValueError('complete twelve-shard task population required')
    identity = lambda row: (row['arm'], row['sample_id'], row['camera_id'])
    expected = {identity(t) for t in tasks}
    scored = [r for path in paths for r in rows(path)]
    actual = [identity(r) for r in scored]
    if len(expected) != 9018 or len(actual) != 9018 or len(set(actual)) != 9018 or set(actual) != expected:
        raise ValueError('scored identities do not match tasks')
    refs = {r['sample_id'] for r in rows(a.references)}
    if len(refs) != 501:
        raise ValueError('expected 501 unique reference samples')
    feature_map = {}
    for path in paths:
        with np.load(path.with_suffix('.npz')) as z:
            keys, predictions, references = z['keys'], z['prediction'], z['reference']
            if not len(keys) == len(predictions) == len(references):
                raise ValueError('inception feature arrays differ in length')
            for i, raw in enumerate(keys):
                arm, sid, camera = json.loads(raw)
                key = (arm, sid, camera)
                if key in feature_map or key not in expected or camera not in (9, 10):
                    raise ValueError(f'invalid or duplicate inception feature {key}')
                feature_map[key] = (predictions[i].copy(), references[i].copy())
    arms = sorted({r['arm'] for r in scored})
    required = {(arm, sid, camera) for arm in arms for sid in refs for camera in (9, 10)}
    if len(arms) != 6 or set(feature_map) != required:
        raise ValueError('incomplete 6 x 501 x 2 inception population')
    protocol = json.loads(a.protocol.read_text())
    sys.path.insert(0, str(a.benchmark_root))
    from deformnext_benchmark.metrics import frechet_distance, kernel_distance
    result = {}
    for arm in arms:
        own = [r for r in scored if r['arm'] == arm]
        source = [r for r in own if r['camera_id'] in (9, 10)]
        target = [r for r in own if r['camera_id'] == 8]
        if len(source) != 1002 or len(target) != 501:
            raise ValueError(f'incomplete views for {arm}')
        summary = {k: float(np.mean([r['metrics'][k] for r in source])) for k in ('LPIPS', 'DreamSim', 'CLIP')}
        summary.update({k: float(np.mean([r['metrics'][k] for r in target])) for k in ('PSNR', 'SSIM', 'M_IoU')})
        ordered = sorted((sid, camera) for sid in refs for camera in (9, 10))
        prediction = np.stack([feature_map[(arm, sid, camera)][0] for sid, camera in ordered])
        reference = np.stack([feature_map[(arm, sid, camera)][1] for sid, camera in ordered])
        summary['FID'] = frechet_distance(prediction, reference)
        settings = protocol['distribution']
        summary.update(kernel_distance(prediction, reference, subset_size=settings['kid_subset_size'], subsets=settings['kid_subsets'], seed=settings['seed']))
        result[arm] = summary
    payload = {'status': 'PASS_COMMON501_GEOMETRY_SIX_ARM_RENDER_METRICS', 'samples': 501, 'rows': result}
    a.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload), flush=True)


if __name__ == '__main__':
    main()
