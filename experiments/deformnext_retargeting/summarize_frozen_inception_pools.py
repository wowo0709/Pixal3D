"""Final FID/KID for a development-frozen model set, including declared controls.

Model names, complete compositions and contrasts come from the pre-test freeze.
The historical three-model evaluator remains unchanged. Each pool must use all
508 prescribed assets and cameras8/9/10, with the same official Inception weights.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .evaluate_bound_population import pinned
from .inception_distribution_metrics import official_distribution_metrics
from .summarize_inception_pools import load_pool


def validate_frozen_models(freeze, prediction_names):
    population = freeze['population']
    ids = population['sample_ids']
    clusters = population['source_clusters_by_sample']
    models = freeze['models']
    roles = freeze['comparison_roles']
    contrasts = [tuple(pair) for pair in freeze['contrasts']]
    if (freeze['status'] != 'FROZEN_FROM_DEVELOPMENT_ONLY'
            or freeze['heldout_quality_read_before_freeze'] is not False
            or not freeze['development_evidence']
            or len(ids) != 508 or len(set(ids)) != 508
            or any(not isinstance(sid, str) or not sid for sid in ids)
            or set(clusters) != set(ids)
            or any(not isinstance(value, str) or not value for value in clusters.values())
            or len(set(clusters.values())) != 134
            or population['rendered_cameras'] != [8, 9, 10]):
        raise ValueError('complete prespecified test508/134-source population and pre-test freeze required')
    if (tuple(prediction_names) != tuple(models) or len(models) < 3
            or any(not isinstance(binding, dict) or not binding for binding in models.values())
            or not isinstance(freeze['reference_binding'], dict) or not freeze['reference_binding']):
        raise ValueError('every frozen model composition and one shared reference binding required')
    required = ['vanilla', 'retargeting', 'candidate']
    if (any(role not in roles for role in required)
            or len({roles[role] for role in required}) != 3
            or any(roles[role] not in models for role in required)
            or freeze['selected_candidate'] != roles['candidate']):
        raise ValueError('distinct frozen vanilla, retargeting and selected candidate compositions required')
    if (not contrasts or any(len(pair) != 2 for pair in contrasts)
            or len(contrasts) != len(set(contrasts))
            or any(a == b or a not in models or b not in models for a, b in contrasts)
            or any((roles['candidate'], roles[control]) not in contrasts
                   for control in ['vanilla', 'retargeting'])):
        raise ValueError('distinct declared contrasts must include candidate versus both matched controls')
    return ids, clusters, contrasts


def load_frozen_pool(spec, *, role, binding, sample_ids, clusters):
    audit_path, features_path = pinned(spec['audit']), pinned(spec['features'])
    features, audit, keys = load_pool(dict(audit=str(audit_path), features=str(features_path)),
                                    role=role, sample_ids=sample_ids)
    array = 'prediction' if role == 'generated-prediction' else 'GT_shared_renderer'
    expected = [(sid, cid, clusters[sid]) for sid in sample_ids for cid in (8, 9, 10)]
    if (keys != expected or audit['model_binding'] != binding
            or audit['package'] != 'torch-fidelity0.3.0'
            or audit['feature_layer'] != '2048' or audit['feature_dtype'] != 'float32'
            or not audit['same_batch_repeat_exact']
            or audit['GT_crop_or_CLIP_preprocessing_used']
            or audit['optimizer_updates'] != 0
            or any(row['array'] != array for row in audit['observations'])
            or not np.isfinite(features).all()):
        raise ValueError('actual ordered native feature pool differs from its frozen model/reference binding')
    return features, audit


def summarize(plan):
    if plan['scope'] != 'final-test508':
        raise ValueError('final-test508 scope required; no small development FID/KID proxy')
    freeze = json.loads(pinned(plan['model_freeze']).read_text())
    ids, clusters, contrasts = validate_frozen_models(freeze, plan['predictions'])
    for evidence in freeze['development_evidence']:
        pinned(evidence)
    package = Path(plan['package_root'])
    sys.path.insert(0, str(package))
    import torch_fidelity
    if (torch_fidelity.__version__ != '0.3.0'
            or not Path(torch_fidelity.__file__).resolve().is_relative_to(package.resolve())):
        raise ValueError('explicitly isolated official torch-fidelity0.3.0 package required')
    reference, ref_audit = load_frozen_pool(plan['reference'], role='shared-original-material-GT',
        binding=freeze['reference_binding'], sample_ids=ids, clusters=clusters)
    metrics, provenance = {}, {}
    for alias, spec in plan['predictions'].items():
        prediction, audit = load_frozen_pool(spec, role='generated-prediction',
            binding=freeze['models'][alias], sample_ids=ids, clusters=clusters)
        if audit['weights_SHA256'] != ref_audit['weights_SHA256']:
            raise ValueError('all prediction/reference pools must share actual official Inception weights')
        metrics[alias] = official_distribution_metrics(prediction, reference,
                                                       subset_size=1000, subsets=100, seed=42017)
        provenance[alias] = spec
    deltas = {a+'_minus_'+b: {metric: metrics[a][metric]-metrics[b][metric]
              for metric in ['frechet_inception_distance', 'kernel_inception_distance_mean']}
              for a, b in contrasts}
    # Check that a long official reduction did not race a changed source manifest.
    pinned(plan['model_freeze'])
    return dict(status='PASS_ACTUAL_FROZEN_MODEL_TEST508_OFFICIAL_FID_KID', scope='final-test508',
        assets=508, source_clusters=134, images_per_pool=1524, views=[8, 9, 10], metrics=metrics,
        point_contrasts=deltas, model_bindings=freeze['models'], reference_binding=freeze['reference_binding'],
        actual_feature_inputs=provenance, reference_input=plan['reference'], model_freeze=plan['model_freeze'],
        selected_candidate=freeze['selected_candidate'], comparison_roles=freeze['comparison_roles'],
        all_frozen_controls_retained=True, FID_finite_sample_estimate=True, KID_subset_std_is_not_CI=True,
        unbiased_KID_not_clamped=True, distribution_metrics_not_pairwise_preservation_metrics=True,
        global_naturalness_reference='same prescribed original-material M1 GT renders',
        images_weighted_equally=True, other_eight_metrics_still_required_for_complete_table=True,
        independent_training_replication_proven_by_this_evaluation=False, full_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing metric output')
    result = summarize(json.loads(args.plan.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], metrics=result['metrics'],
                          point_contrasts=result['point_contrasts'])), flush=True)


if __name__ == '__main__':
    main()
