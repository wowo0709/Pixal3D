"""Derive a small head dataset from existing ray caches and loss-only teachers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .ray_correspondence_reliability import inference_features


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['aux-cache', 'val-cache', 'teacher-root', 'output']:
        parser.add_argument('--'+flag, type=Path, required=True)
    args = parser.parse_args()
    bound = {str(Path(__file__).resolve()): sha(__file__),
        str(Path(__file__).with_name('ray_correspondence_reliability.py')):
            sha(Path(__file__).with_name('ray_correspondence_reliability.py'))}
    arrays, records, clusters = {}, [], {}
    for cache, allowed in [(args.aux_cache, {'train', 'calibration'}),
                           (args.val_cache, {'validation'})]:
        index = cache/'cache-index.jsonl'
        bound[str(index)] = sha(index)
        rows = list(map(json.loads, index.read_text().splitlines()))
        for row in rows:
            role = row['role']
            assert role in allowed
            cluster = row['source_cluster_id']
            assert cluster not in clusters or clusters[cluster] == role
            clusters[cluster] = role
            for kind in ['inputs', 'labels']:
                path = cache/row[kind+'_relative_path']
                assert sha(path) == row[kind+'_SHA256']
                bound[str(path)] = row[kind+'_SHA256']
            with np.load(cache/row['inputs_relative_path'], allow_pickle=False) as data:
                patches = data['query_patch_indices'].astype(np.int64)
                valid = data['candidate_valid'].copy()
                features = inference_features(data['native_query'], data['context_query'],
                    data['scalar_prior_logits'], valid, data['candidate_indices'])
            with np.load(cache/row['labels_relative_path'], allow_pickle=False) as data:
                relation = data['relation_query_indices'].copy()
                near = np.unpackbits(data['positive_near1_bits'], axis=-1,
                                     count=64, bitorder='little').astype(bool)
                fingerprint = str(data['teacher_input_sha256'])
            teacher = args.teacher_root/(row['sample_id']+'.npz')
            bound[str(teacher)] = sha(teacher)
            with np.load(teacher, allow_pickle=False) as data:
                assert str(data['input_sha256']) == fingerprint == row['GT_teacher_fingerprint']
                foreground = data['target_foreground'][patches].astype(bool)
                visible = data['positive_visible'][patches].astype(bool).any(-1)
                assert str(data['split']) == ('validation' if role == 'validation' else 'train')
            assert np.array_equal(np.unique(relation), np.flatnonzero(foreground & visible))
            hit = np.zeros((len(patches), 64), bool)
            np.logical_or.at(hit, relation, near & valid[relation])
            # Unknown background patch centers are excluded, rather than labeled negative.
            labels = hit.any(-1)[foreground]
            types = np.where(~visible[foreground], 0, np.where(labels, 2, 1))
            prefix = f'asset{len(records):04d}'
            arrays[prefix+'_x'] = features[foreground]
            arrays[prefix+'_y'] = labels
            arrays[prefix+'_type'] = types.astype(np.int8)
            records.append(dict(sample_id=row['sample_id'], source_cluster_id=cluster, role=role,
                array_prefix=prefix, native_queries=len(patches), known_rays=int(foreground.sum()),
                excluded_unknown_centers=int((~foreground).sum()), positives=int(labels.sum()),
                no_visible_source=int((types == 0).sum()), proposal_miss=int((types == 1).sum())))
    counts = {role: sum(r['role'] == role for r in records) for role in ['train', 'calibration', 'validation']}
    assert counts == dict(train=512, calibration=32, validation=32)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez(args.output/'observations.npz', **arrays)
    result = dict(status='PASS_RELIABILITY_HEAD_DERIVED_CACHE', assets=counts,
        input_width=2060, type_names={'0': 'no-visible-source', '1': 'proposal-miss', '2': 'usable-near1'},
        inference_fields=['native_query', 'context_query', 'scalar_prior_logits',
                          'candidate_valid', 'candidate_indices'],
        GT_only_labels_and_unknown_center_exclusion=True,
        all_asset_roles_retained=True, independent_source_clusters_between_roles=True,
        rendering_or_latent_or_Omega_rebuilt=False, model_weights_updated=False,
        observations_SHA256=sha(args.output/'observations.npz'), records=records, source_SHA256=bound)
    (args.output/'cache.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], assets=counts,
        totals={role: {k: sum(r[k] for r in records if r['role'] == role)
        for k in ['known_rays', 'positives', 'no_visible_source', 'proposal_miss', 'excluded_unknown_centers']}
        for role in counts}, bytes=(args.output/'observations.npz').stat().st_size)))


if __name__ == '__main__':
    main()
