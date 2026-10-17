"""Freeze cluster-disjoint matcher calibration and common SS evaluation IDs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def freeze(manifest_root, teacher_root, output):
    output.mkdir(parents=True, exist_ok=True)
    read = lambda split: [json.loads(l) for l in (manifest_root / f'{split}.jsonl').read_text().splitlines() if l]
    train, validation = read('train'), read('validation')
    for field in ('source_uid', 'source_cluster_id'):
        if {r[field] for r in train} & {r[field] for r in validation}:
            raise ValueError('train/validation overlap: ' + field)
    clusters = sorted({r['source_cluster_id'] for r in train}, key=lambda k: digest('42017:' + k))
    calibration = set(clusters[:max(1, len(clusters) // 10)])
    certified = [r for r in train if (teacher_root / (r['sample_id'] + '.npz')).is_file()]
    groups = defaultdict(list)
    for row in validation:
        groups[row['category']].append(row)
    for rows in groups.values():
        rows.sort(key=lambda r: digest('panel:' + r['sample_id']))
    ordered, used = [], set()
    while len(ordered) < len(validation):
        added = False
        for category in sorted(groups):
            candidates = groups[category]
            if not candidates:
                continue
            pos = next((i for i, r in enumerate(candidates) if r['source_cluster_id'] not in used), 0)
            row = candidates.pop(pos); ordered.append(row); used.add(row['source_cluster_id']); added = True
        if not added:
            break
    records = {
        'head_train': [r for r in certified if r['source_cluster_id'] not in calibration],
        'head_calibration': [r for r in certified if r['source_cluster_id'] in calibration],
        'head_validation': validation,
    }
    for name, rows in records.items():
        (output / f'{name}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    seeds = {sid: 42 + i for i, sid in enumerate(sorted(r['sample_id'] for r in validation))}
    (output / 'seed-map.json').write_text(json.dumps(seeds, indent=2) + '\n')
    for size in (32, 128):
        ids = [r['sample_id'] for r in ordered[:size]]
        (output / f'panel-{size}.json').write_text(json.dumps({'sample_ids': ids, 'sha256': digest('\n'.join(ids))}, indent=2) + '\n')
    result = {'seed': 42017, 'calibration_clusters': len(calibration),
              'counts': {k: len(v) for k, v in records.items()},
              'manifest_sha256': {s: digest((manifest_root / f'{s}.jsonl').read_text()) for s in ('train', 'validation')}}
    (output / 'population.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest-root', 'teacher-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); print(json.dumps(freeze(a.manifest_root, a.teacher_root, a.output)), flush=True)


if __name__ == '__main__':
    main()
