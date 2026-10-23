import copy
import pytest

from experiments.deformnext_retargeting.evaluate_r_mediated_pbr_initial import IDENTITY_FIELDS
from experiments.deformnext_retargeting.evaluate_pbr_streaming import aggregate_metrics
from experiments.deformnext_retargeting.pbr_initial_render_identity import validate_initial_latent_and_render


def pair():
    rows=[]
    for i in range(32):
        r={k:'same-'+str(i)for k in IDENTITY_FIELDS};r['metrics']={'render_psnr':20.,'base_color_psnr':20.,'foreground_psnr':20.,'foreground_base_color_psnr':20.,'foreground_fraction':.1,'render_ssim':.9,'crop_lpips':.1,'latent_mse':.2};rows.append(r)
    reference=dict(status='PASS_ACTUAL_NATIVE_PBR_COLD_GENERATED32',source_multiview_step=16850,per_asset=rows,optimizer_updates=0,original_and_source_states_unchanged=True)
    selected=copy.deepcopy(rows[:4]);actual=dict(step=0,quick_per_asset=selected,quick_panel_ids=[r['sample_id']for r in selected],quick_metrics=aggregate_metrics([r['metrics']for r in selected]))
    return actual,reference


def update(actual):actual['quick_metrics']=aggregate_metrics([r['metrics']for r in actual['quick_per_asset']])


def test_requires_exact_latent_but_separately_accepts_declared_small_render_differences():
    actual,reference=pair();before=copy.deepcopy(reference)
    actual['quick_per_asset'][0]['metrics']['render_psnr']+=.0025
    actual['quick_per_asset'][1]['metrics']['crop_lpips']+=.00015;update(actual)
    proof=validate_initial_latent_and_render(actual,reference)
    assert proof['actual_inputs_noise_GT_and_PBR_latent_exact']
    assert not proof['original_render_metrics_exact']
    assert reference==before


@pytest.mark.parametrize('field',IDENTITY_FIELDS[1:])
def test_never_tolerates_a_changed_actual_input_noise_GT_or_generated_latent(field):
    actual,reference=pair();actual['quick_per_asset'][0][field]='different'
    with pytest.raises(ValueError):validate_initial_latent_and_render(actual,reference)


@pytest.mark.parametrize('key,delta',[('render_psnr',.011),('render_ssim',.00021),('crop_lpips',.00031),('latent_mse',1e-9)])
def test_rejects_excess_renderer_drift_and_any_unlisted_metric_difference(key,delta):
    actual,reference=pair();actual['quick_per_asset'][0]['metrics'][key]+=delta;update(actual)
    with pytest.raises(ValueError):validate_initial_latent_and_render(actual,reference)
