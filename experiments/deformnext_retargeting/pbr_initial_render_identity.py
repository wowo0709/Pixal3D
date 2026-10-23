"""Exact cold PBR latent identity with a separate declared renderer tolerance.

Same generated latent bytes are mandatory. Rendering is not a byte identity:
calibrate this operational bound on the recorded cold feature diagnostic and
confirm it on the unchanged vanilla model before admitting a fresh training.
"""
import copy
import math

POLICY = 'exact-input-noise-latent-bounded-render-v1'
RENDER_ABSOLUTE_LIMITS = {'psnr':0.01, 'ssim':0.0002, 'lpips':0.0003}


def validate_initial_latent_and_render(actual, reference):
    from .evaluate_r_mediated_pbr_initial import IDENTITY_FIELDS
    from .r_mediated_pbr_science_recipe import validate_initial4
    expected={r['sample_id']:r for r in reference['per_asset']}
    comparisons=[];adjusted=copy.deepcopy(reference)
    by_id={r['sample_id']:r for r in adjusted['per_asset']}
    for row in actual['quick_per_asset']:
        if row['sample_id']not in expected:raise ValueError('cold reference sample missing')
        prior=expected[row['sample_id']]
        if any(row[k]!=prior[k]for k in IDENTITY_FIELDS):
            raise ValueError('initial RGB/camera/GT/noise/PBR latent hashes must be exactly equal')
        if row['metrics'].keys()!=prior['metrics'].keys():raise ValueError('same explicit metric names required')
        deltas={}
        for key,value in row['metrics'].items():
            before=prior['metrics'][key]
            if not math.isfinite(value)or not math.isfinite(before):raise ValueError('finite initial metrics required')
            suffix=next((n for n in RENDER_ABSOLUTE_LIMITS if key.endswith(n)),None)
            limit=0. if suffix is None else RENDER_ABSOLUTE_LIMITS[suffix]
            delta=abs(value-before)
            if delta>limit:raise ValueError('initial rendering exceeds its explicit bound: '+key)
            deltas[key]=dict(absolute_delta=delta,absolute_limit=limit)
        comparisons.append(dict(sample_id=row['sample_id'],render_metrics=deltas))
        # Reuse the original schema, sample-identity and pooling checks after
        # separately proving each explicitly allowed renderer difference.
        by_id[row['sample_id']]['metrics']=copy.deepcopy(row['metrics'])
    validate_initial4(actual,adjusted)
    return dict(policy=POLICY,actual_inputs_noise_GT_and_PBR_latent_exact=True,
        original_render_metrics_exact=all(v['absolute_delta']==0 for r in comparisons for v in r['render_metrics'].values()),
        render_metrics_within_declared_absolute_bounds=True,render_absolute_limits=RENDER_ABSOLUTE_LIMITS,
        comparisons=comparisons,unlisted_metrics_require_exact_equality=True,
        no_model_or_optimizer_changes=True,training_quality_gain_not_implied=True)
