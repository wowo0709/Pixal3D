import copy

import pytest

from experiments.deformnext_retargeting.assemble_final_metric_table import assemble, METRICS


def fixture():
    ids = [f'asset{i}' for i in range(508)]; clusters = {sid:'source'+str(i % 134) for i,sid in enumerate(ids)}
    models = {name:dict(ss64=dict(checkpoint=name+'-synthetic-only')) for name in ['V1', 'matched-R', 'feature-DQ', 'zero-source4']}
    pairs = [('feature-DQ', 'V1'), ('feature-DQ', 'matched-R'), ('V1', 'zero-source4')]
    spec = dict(path='/fixture-not-scientific/freeze.json', SHA256='fixture')
    freeze = dict(status='FROZEN_FROM_DEVELOPMENT_ONLY', heldout_quality_read_before_freeze=False,
        selected_candidate='feature-DQ', comparison_roles=dict(vanilla='V1', retargeting='matched-R', candidate='feature-DQ'),
        development_evidence=[dict(path='/fixture-not-scientific/development.json')], models=models,
        population=dict(sample_ids=ids, source_clusters_by_sample=clusters, rendered_cameras=[8,9,10], manifest_SHA256='manifest'),
        contrasts=pairs, reference_binding=dict(materials='M0', geometry='M1'))
    eight = dict(status='PASS_ACTUAL_BOUND_CASCADE_EIGHT_METRICS', scope='final-test508', assets=508, source_clusters=134,
        actual_completed_cases=4*508, actual_perceptual_views=4*1524, model_bindings=copy.deepcopy(models),
        model_freeze=spec, manifest=dict(SHA256='manifest'), native_stage_weight_hashes_verified=True, roles={}, per_asset={})
    for role,offset in [('target8',0.),('source9_10',1.),('all8_9_10',.5)]:
        means = {}
        for model in models:
            values = {metric:offset+.3 for metric in ['LPIPS','DreamSim','CLIP','PSNR','SSIM','M_IoU','CD','FScore']}
            means[model] = dict(deformation_weighted=values, source_weighted=copy.deepcopy(values))
        contrasts = {a+'_minus_'+b:dict(deformation_weighted_delta={key:.01 for key in next(iter(means.values()))['deformation_weighted']},
            deformation_weighted_cluster_CI95={key:[-.01,.03] for key in next(iter(means.values()))['deformation_weighted']},
            source_weighted_delta={key:.02 for key in next(iter(means.values()))['deformation_weighted']},
            source_weighted_cluster_CI95={key:[0.,.04] for key in next(iter(means.values()))['deformation_weighted']}) for a,b in pairs}
        eight['roles'][role] = dict(means=means, contrasts=contrasts)
        eight['per_asset'][role] = {name:[dict(sample_id=sid,source_cluster_id=clusters[sid]) for sid in ids] for name in models}
    distribution = dict(status='PASS_ACTUAL_FROZEN_MODEL_TEST508_OFFICIAL_FID_KID', scope='final-test508', assets=508,
        source_clusters=134, images_per_pool=1524, views=[8,9,10], model_freeze=spec, model_bindings=copy.deepcopy(models),
        unbiased_KID_not_clamped=True, KID_subset_std_is_not_CI=True,
        metrics={name:dict(frechet_inception_distance=2.,kernel_inception_distance_mean=-.002,kernel_inception_distance_std=.001) for name in models},
        point_contrasts={a+'_minus_'+b:dict(frechet_inception_distance=-.3,kernel_inception_distance_mean=-.0001) for a,b in pairs})
    return eight, distribution, freeze, spec


def test_ten_metrics_keep_requested_purposes_and_both_role_and_allview_tables():
    values = fixture(); before = copy.deepcopy(values); result = assemble(*values)
    primary = result['primary_camera_role_table']['feature-DQ']; allview = result['complete_allview_table']['feature-DQ']
    assert list(primary) == [row[0] for row in METRICS]
    assert primary['LPIPS'] == primary['DreamSim'] == primary['CLIP'] == 1.3
    assert primary['PSNR'] == primary['SSIM'] == primary['M-IoU'] == .3
    assert allview['LPIPS'] == allview['PSNR'] == .8
    assert primary['KID'] == -.002
    assert result['paired_contrasts']['feature-DQ_minus_V1']['KID']['CI95'] is None
    assert result['paired_contrasts']['feature-DQ_minus_V1']['LPIPS']['CI95'] == [-.01,.03]
    assert list(result['primary_camera_role_table']) == ['V1','matched-R','feature-DQ','zero-source4']
    assert not result['model_adopted'] and values == before


@pytest.mark.parametrize('problem', ['development', 'different-freeze', 'different-manifest', 'checkpoint', 'view-drop',
    'case-drop', 'asset-drop', 'cluster-mix', 'population-size', 'KID-clamped', 'KID-std-CI', 'nonfinite'])
def test_final_table_rejects_mixed_model_population_and_uncertainty_interpretations(problem):
    values = fixture(); eight, distribution, freeze, spec = values
    if problem == 'development': eight['scope'] = 'engineering-validation32'
    elif problem == 'different-freeze': distribution['model_freeze'] = dict(path='/different-freeze',SHA256='other')
    elif problem == 'different-manifest': eight['manifest']['SHA256'] = 'other-manifest'
    elif problem == 'checkpoint': distribution['model_bindings']['feature-DQ']['ss64']['checkpoint'] = 'other-weights'
    elif problem == 'view-drop': distribution['images_per_pool'] = 1016
    elif problem == 'case-drop': eight['actual_completed_cases'] -= 1
    elif problem == 'asset-drop': eight['per_asset']['source9_10']['feature-DQ'].pop()
    elif problem == 'cluster-mix': eight['per_asset']['target8']['feature-DQ'][0]['source_cluster_id'] = 'wrong-source'
    elif problem == 'population-size': eight['assets'] = 32
    elif problem == 'KID-clamped': distribution['unbiased_KID_not_clamped'] = False
    elif problem == 'KID-std-CI': distribution['KID_subset_std_is_not_CI'] = False
    else: distribution['metrics']['feature-DQ']['frechet_inception_distance'] = float('nan')
    with pytest.raises(ValueError): assemble(*values)
