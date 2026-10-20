"""Final508 FID/KID from completed matched official Inception feature pools."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .extract_official_inception import digest
from .inception_distribution_metrics import official_distribution_metrics
from .render_panel_statistics import MODELS, PAIRS


def load_pool(spec, *, role, sample_ids):
    audit = json.loads(Path(spec['audit']).read_text())
    if (audit['status']!='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL'
            or audit['scope']!='final-test508' or audit['pool_role']!=role
            or audit['images']!=1524 or audit['assets']!=508
            or audit['features_SHA256']!=digest(spec['features'])
            or audit['TF32'] or audit['autocast'] or not audit['fullframe']):
        raise ValueError('complete pinned native fullframe test508 official feature pool required')
    observations = audit['observations']
    keys = [(r['sample_id'],r['camera_id'],r['source_cluster_id']) for r in observations]
    if (len(keys)!=len(set(keys)) or {k[:2] for k in keys}!={(sid,cid) for sid in sample_ids for cid in (8,9,10)}):
        raise ValueError('no source/camera/sample dropping in distribution comparison')
    with np.load(spec['features']) as z:
        features = z['features']
        stored = list(zip(z['sample_id'].tolist(),z['camera_id'].tolist(),z['source_cluster_id'].tolist()))
    if features.shape!=(1524,2048) or features.dtype!=np.float32 or stored!=keys:
        raise ValueError('feature rows must match their actual rendered observations exactly')
    return features, audit, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if (plan['scope']!='final-test508' or tuple(plan['predictions'])!=MODELS
            or len(plan['expected_sample_ids'])!=508 or len(set(plan['expected_sample_ids']))!=508):
        raise ValueError('one prespecified complete test508 population and three ordered models required')
    package = Path(plan['package_root'])
    sys.path.insert(0,str(package))
    import torch_fidelity
    if not Path(torch_fidelity.__file__).resolve().is_relative_to(package.resolve()):
        raise ValueError('use the explicitly isolated official metric package')
    reference, ref_audit, keys = load_pool(plan['reference'],role='shared-original-material-GT',
                                          sample_ids=plan['expected_sample_ids'])
    metrics, bindings = {}, {}
    for model in MODELS:
        prediction, audit, actual_keys = load_pool(plan['predictions'][model],role='generated-prediction',
                                                  sample_ids=plan['expected_sample_ids'])
        if (actual_keys!=keys or audit['weights_SHA256']!=ref_audit['weights_SHA256']
                or audit['package']!=ref_audit['package']):
            raise ValueError('same ordered assets/cameras/sources and Inception weights required')
        metrics[model] = official_distribution_metrics(prediction,reference,subset_size=1000,subsets=100,seed=42017)
        bindings[model] = audit['model_binding']
    contrasts = {a+'_minus_'+b:{metric:metrics[a][metric]-metrics[b][metric]
        for metric in ['frechet_inception_distance','kernel_inception_distance_mean']} for a,b in PAIRS}
    result = dict(status='PASS_ACTUAL_FINAL_TEST508_OFFICIAL_FID_KID',scope='final-test508',
        assets=508,images_per_pool=1524,views=[8,9,10],metrics=metrics,point_contrasts=contrasts,
        model_bindings=bindings,reference_binding=ref_audit['model_binding'],
        actual_feature_inputs=plan,source_clusters=len({k[2] for k in keys}),
        FID_finite_sample_estimate=True,KID_subset_std_is_not_CI=True,
        unbiased_KID_not_clamped=True,global_naturalness_reference='same prescribed original-material M1 GT renders',
        distribution_metrics_not_pairwise_preservation_metrics=True,
        images_weighted_equally=True,all_models_same_population=True,
        independent_training_replication_proven_by_this_evaluation=False,
        other_eight_metrics_still_required_for_complete_table=True,full_goal_complete=False)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],metrics=metrics,point_contrasts=contrasts)),flush=True)


if __name__=='__main__':
    main()
