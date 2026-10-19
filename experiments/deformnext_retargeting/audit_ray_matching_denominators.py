"""Resolve ray versus visible-source-relation denominators in matching metrics.

A single categorical ray distribution is evaluated against up to three
disjoint source-view positive sets. Relation coverage and probability mass
therefore have different attainable maxima. Derive ray-union metrics and
oracle ceilings from actual labels, without model forward or optimization.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['cache-root', 'metrics-root', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    index = args.cache_root/'cache-index.jsonl'
    rows = list(map(json.loads, index.read_text().splitlines()))
    if len(rows) != 32 or any(row['split'] != 'validation' for row in rows):
        raise ValueError('the exact actual SS validation32 matching cache required')
    bound = {str(Path(__file__).resolve()): sha(__file__), str(index): sha(index)}
    assets = []
    for row in rows:
        path = args.cache_root/row['labels_relative_path']
        if sha(path) != row['labels_SHA256']:
            raise ValueError('actual cached label bytes changed')
        bound[str(path)] = row['labels_SHA256']
        with np.load(path, allow_pickle=False) as archive:
            relation = archive['relation_query_indices'].astype(np.int64)
            kinds = {kind: np.unpackbits(archive['positive_'+kind+'_bits'], axis=1,
                count=64, bitorder='little').astype(bool) for kind in ['exact', 'near1']}
        n = row['queries']
        used = np.unique(relation)
        if (len(relation) and ((relation < 0).any() or (relation >= n).any())):
            raise ValueError('label relation lies outside native ray queries')
        metrics = {}
        for kind, positives in kinds.items():
            multiplicity = np.zeros((n, 64), dtype=np.uint8)
            np.add.at(multiplicity, relation, positives.astype(np.uint8))
            if (multiplicity > 1).any():
                raise ValueError('source-view positive sets overlap; mass cannot be converted by a ray denominator')
            covered = int((multiplicity > 0).any(-1).sum())
            metrics[kind] = dict(covered_queries=covered,
                covered_visible_source_relations=int(positives.any(-1).sum()))
        assets.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
            observable_rays=len(used), visible_source_relations=len(relation), native_target_rays=n,
            metrics=metrics))
    observable_rays = sum(row['observable_rays'] for row in assets)
    visible_relations = sum(row['visible_source_relations'] for row in assets)
    references = {row['sample_id']: row for row in assets}
    models = {}
    for name in ['cold-random-QK', 'pretrained-utility7', 'Direct192', 'Direct192-Qzero', 'Direct288', 'Direct288-Qzero']:
        path = args.metrics_root/(name+'.json')
        bound[str(path)] = sha(path)
        values = json.loads(path.read_text())
        if {r['sample_id'] for r in values['per_asset']} != set(references):
            raise ValueError('different actual matching diagnostic panel')
        model = {}
        for kind in ['exact', 'near1']:
            mass = hits = 0
            for row in values['per_asset']:
                old = references[row['sample_id']]
                if (row['observable_relations'] != old['visible_source_relations']
                        or row['source_cluster_id'] != old['source_cluster_id']):
                    raise ValueError('matching source/denominator changed')
                mass += row['metrics'][kind]['mass_sum']
                hits += row['metrics'][kind]['hit1']
            covered = sum(row['metrics'][kind]['covered_queries'] for row in assets)
            model[kind] = dict(reported_per_relation_mass=mass/visible_relations,
                per_observable_ray_union_mass=mass/observable_rays,
                per_observable_ray_Top1_union_coverage=hits/observable_rays,
                per_observable_ray_Top64_union_coverage=covered/observable_rays,
                per_relation_mass_or_Top1_oracle_upper_bound=covered/visible_relations)
            if mass > covered+1e-3 or hits > covered:
                raise ValueError('disjoint-set probability mass exceeds the actual ray oracle ceiling')
        models[name] = model
    counts = {name: sum(row[name] for row in assets)
              for name in ['observable_rays', 'visible_source_relations', 'native_target_rays']}
    if any(sha(path) != digest for path, digest in bound.items()):
        raise ValueError('read-only denominator analysis source/label/metrics changed')
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='PASS_ACTUAL_RAY_UNION_AND_RELATION_DENOMINATOR_AUDIT', assets=32,
        source_clusters=32, split='actual SS validation', counts=counts, models=models, per_asset=assets,
        source_view_positive_sets_are_disjoint=True,
        relation_probability_mass_is_not_expected_to_reach_Top64_relation_coverage=True,
        initial_matching_gap_must_be_interpreted_with_ray_union_ceiling=True,
        no_new_model_forward=True, optimizer_updates=0, GT_only_metric_denominators=True,
        source_SHA256=bound, model_adopted=False, full_goal_complete=False)
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(counts=counts, models=models)), flush=True)


if __name__ == '__main__':
    main()
