"""Join a single frozen test508 evaluation into the requested ten-metric table."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .evaluate_bound_population import pinned
from .summarize_frozen_inception_pools import validate_frozen_models
from .metric_image_identity import validate_metric_image_join


# Prespecified camera roles follow the user's metric purposes. All-view scores
# remain a separate complete table, so this never chooses a favorable camera.
METRICS = (
    ('LPIPS', 'source9_10', 'LPIPS', 'source_preservation', 'lower', 'distance'),
    ('DreamSim', 'source9_10', 'DreamSim', 'source_preservation', 'lower', 'distance'),
    ('CLIP', 'source9_10', 'CLIP', 'source_preservation', 'higher', 'image_cosine'),
    ('FID', None, 'frechet_inception_distance', 'global_naturalness', 'lower', 'official_Inception2048'),
    ('KID', None, 'kernel_inception_distance_mean', 'global_naturalness', 'lower', 'signed_unbiased_raw'),
    ('PSNR', 'target8', 'PSNR', 'target_alignment', 'higher', 'dB'),
    ('SSIM', 'target8', 'SSIM', 'target_alignment', 'higher', 'unit_interval'),
    ('M-IoU', 'target8', 'M_IoU', 'target_alignment', 'higher', '2D_mask_fraction'),
    ('CD', 'all8_9_10', 'CD', 'target_alignment', 'lower', 'source_bbox_normalized_L1'),
    ('F-Score', 'all8_9_10', 'FScore', 'target_alignment', 'higher', 'fraction_at_normalized_distance_0.02'),
)


def assemble(eight, distribution, freeze, freeze_spec):
    ids, clusters, contrasts = validate_frozen_models(freeze, eight['model_bindings'])
    expected = len(freeze['models'])
    if (eight['status'] != 'PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS'
            or distribution['status'] != 'PASS_ACTUAL_FROZEN_MODEL_TEST508_OFFICIAL_FID_KID'
            or eight['scope'] != 'final-test508' or distribution['scope'] != 'final-test508'):
        raise ValueError('completed native final-test508 eight-metric and official distribution results required')
    if (eight['assets'] != 508 or distribution['assets'] != 508
            or eight['source_clusters'] != 134 or distribution['source_clusters'] != 134
            or eight['actual_completed_cases'] != expected*508
            or eight['actual_perceptual_views'] != expected*1524
            or distribution['images_per_pool'] != 1524 or distribution['views'] != [8, 9, 10]
            or eight['model_freeze'] != freeze_spec or distribution['model_freeze'] != freeze_spec
            or eight['manifest']['SHA256'] != freeze['population']['manifest_SHA256']
            or eight['model_bindings'] != distribution['model_bindings'] or eight['model_bindings'] != freeze['models']
            or distribution['reference_binding'] != freeze['reference_binding']
            or tuple(distribution['metrics']) != tuple(freeze['models'])
            or not eight['native_stage_weight_hashes_verified']
            or not distribution['unbiased_KID_not_clamped'] or not distribution['KID_subset_std_is_not_CI']):
        raise ValueError('all scores must bind the same frozen complete model compositions and population')
    image_identity = validate_metric_image_join(eight, distribution, sample_ids=ids, clusters=clusters,
                                                models=freeze['models'])
    for role in ['target8', 'source9_10', 'all8_9_10']:
        for model in freeze['models']:
            rows = eight['per_asset'][role][model]
            if ([row['sample_id'] for row in rows] != ids
                    or any(row['source_cluster_id'] != clusters[row['sample_id']] for row in rows)):
                raise ValueError('paired camera-role population differs')
    primary, allview, source_weighted, uncertainty = {}, {}, {}, {}
    for model in freeze['models']:
        primary[model], allview[model], source_weighted[model] = {}, {}, {}
        for label, role, metric, _, _, _ in METRICS:
            if role is None:
                value = distribution['metrics'][model][metric]
                primary[model][label] = allview[model][label] = value
            else:
                primary[model][label] = eight['roles'][role]['means'][model]['deformation_weighted'][metric]
                allview[model][label] = eight['roles']['all8_9_10']['means'][model]['deformation_weighted'][metric]
                source_weighted[model][label] = eight['roles'][role]['means'][model]['source_weighted'][metric]
        uncertainty[model] = dict(KID_subset_std=distribution['metrics'][model]['kernel_inception_distance_std'],
                                 KID_subset_std_is_not_CI=True)
    paired = {}
    for candidate, baseline in contrasts:
        key = candidate+'_minus_'+baseline; paired[key] = {}
        for label, role, metric, _, _, _ in METRICS:
            if role is None:
                paired[key][label] = dict(delta=distribution['point_contrasts'][key][metric], CI95=None,
                                          uncertainty='point contrast only; KID subset std is not a paired CI')
            else:
                effects = eight['roles'][role]['contrasts'][key]
                paired[key][label] = dict(delta=effects['deformation_weighted_delta'][metric],
                    CI95=effects['deformation_weighted_cluster_CI95'][metric],
                    source_weighted_delta=effects['source_weighted_delta'][metric],
                    source_weighted_CI95=effects['source_weighted_cluster_CI95'][metric])
    values = [v for table in [primary, allview, source_weighted] for row in table.values() for v in row.values()]
    values += [row['KID_subset_std'] for row in uncertainty.values()]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('complete finite metric tables required; no sample dropping')
    return dict(status='PASS_ACTUAL_FROZEN_TEST508_TEN_METRIC_TABLE', scope='final-test508',
        assets=508, source_clusters=134, views=[8, 9, 10], images_per_distribution_pool=1524,
        model_freeze=freeze_spec, model_bindings=freeze['models'], selected_candidate=freeze['selected_candidate'],
        primary_camera_role_table=primary, complete_allview_table=allview,
        equal_source_eight_metric_sensitivity=source_weighted, paired_contrasts=paired, distribution_uncertainty=uncertainty,
        metric_protocol={label:dict(camera_role=role or 'global_all_three_views', field=metric,
            purpose=purpose, better=direction, unit=unit) for label, role, metric, purpose, direction, unit in METRICS},
        source_preservation_reference='Original M0 material identity rendered on edited M1 geometry; not unchanged M0 geometry pixels',
        reference_binding=freeze['reference_binding'], global_distribution_estimates_not_pairwise_CIs=True,
        same_scored_image_identity=image_identity,
        source_cluster_intervals_condition_on_each_model_training_seed=True,
        allview_table_keeps_prior_reporting_convention=True, no_metric_rescaling_or_KID_clamping=True,
        statistical_table_does_not_itself_establish_module_attribution_or_training_replication=True,
        model_adopted=False, full_goal_complete=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve previous final metric tables')
    plan = json.loads(args.plan.read_text())
    loaded = {key: json.loads(pinned(plan[key]).read_text()) for key in ['model_freeze', 'eight_metrics', 'distribution_metrics']}
    result = assemble(loaded['eight_metrics'], loaded['distribution_metrics'], loaded['model_freeze'], plan['model_freeze'])
    result['input_bindings'] = plan
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for name, key in [('table1-camera-roles.csv', 'primary_camera_role_table'), ('table1-all-views.csv', 'complete_allview_table')]:
        with (args.output/name).open('x', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['model', *[m[0] for m in METRICS]])
            for model, row in result[key].items(): writer.writerow([model, *[row[m[0]] for m in METRICS]])
    print(json.dumps(dict(status=result['status'], models=list(result['model_bindings']), assets=508)), flush=True)


if __name__ == '__main__':
    main()
