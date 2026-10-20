"""Count supervision coverage without treating unknown patch centers as negatives.

Read an existing native validation cache and its loss-only GT teachers. No
images, features, weights or labels are rebuilt, and no model is evaluated.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    complete = json.loads((args.cache/'completed.json').read_text())
    rows = list(map(json.loads, (args.cache/'cache-index.jsonl').read_text().splitlines()))
    assert len(rows) == complete['assets'] == 32
    assert len({r['source_cluster_id'] for r in rows}) == 32
    assert all(r['role'] == r['split'] == 'validation' for r in rows)
    bindings = complete['source_SHA256']
    checked = {str(Path(__file__).resolve()): sha(__file__)}
    for name in ['completed.json', 'cache-index.jsonl']:
        checked[str(args.cache/name)] = sha(args.cache/name)
    records = []
    for row in rows:
        teachers = [Path(p) for p in bindings if '/topology_labels_' in p
                    and Path(p).name == row['sample_id']+'.npz']
        assert len(teachers) == 1
        teacher = teachers[0]
        assert sha(teacher) == bindings[str(teacher)]
        checked[str(teacher)] = bindings[str(teacher)]
        for field in ['inputs', 'labels']:
            path = args.cache/row[field+'_relative_path']
            assert sha(path) == row[field+'_SHA256']
            checked[str(path)] = row[field+'_SHA256']
        with np.load(args.cache/row['inputs_relative_path'], allow_pickle=False) as data:
            patches = data['query_patch_indices'].astype(np.int64)
            valid = data['candidate_valid'].copy()
        with np.load(args.cache/row['labels_relative_path'], allow_pickle=False) as data:
            relation = data['relation_query_indices'].copy()
            near = np.unpackbits(data['positive_near1_bits'], axis=-1,
                                 count=64, bitorder='little').astype(bool)
        with np.load(teacher, allow_pickle=False) as data:
            foreground = data['target_foreground'][patches].astype(bool)
            visible = data['positive_visible'][patches].astype(bool)
        observable = foreground & visible.any(-1)
        # Labels contain one row per visible source, rather than per target ray.
        assert np.array_equal(np.unique(relation), np.flatnonzero(observable))
        assert len(relation) == row['observable_relations']
        union = np.zeros((len(patches), 64), bool)
        np.logical_or.at(union, relation, near & valid[relation])
        hit = union.any(-1)
        counts = dict(native_queries=len(patches),
            foreground_center=int(foreground.sum()),
            unknown_background_center=int((~foreground).sum()),
            foreground_center_without_visible_source=int((foreground & ~visible.any(-1)).sum()),
            observable_rays=int(observable.sum()),
            observable_proposal_near1_hit=int((observable & hit).sum()),
            observable_proposal_near1_miss=int((observable & ~hit).sum()),
            observable_source_relations=len(relation))
        assert counts['native_queries'] == (counts['unknown_background_center']
            + counts['foreground_center_without_visible_source'] + counts['observable_rays'])
        assert counts['observable_rays'] == (counts['observable_proposal_near1_hit']
            + counts['observable_proposal_near1_miss'])
        records.append(dict(sample_id=row['sample_id'],
            source_cluster_id=row['source_cluster_id'], **counts))
    totals = {k: sum(r[k] for r in records) for k in counts}
    args.output.mkdir(exist_ok=False, parents=True)
    result = dict(status='PASS_NATIVE_MATCHING_OBSERVABILITY_PARTITION', assets=len(rows),
        totals=totals, per_asset=records,
        unknown_center_is_not_a_negative=True,
        visibility_scope='existing patch-center topology teacher; not full-patch coverage',
        no_source_means='GT foreground center, no source view marked visible by existing teacher',
        GT_used_only_for_analysis=True, native_forward_changed=False,
        rendering_or_latent_or_Omega_rebuilt=False, optimizer_updates=0,
        source_SHA256=checked)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], totals=totals)))


if __name__ == '__main__':
    main()
