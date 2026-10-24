"""Reuse the original bounded-render policy without changing actual metrics."""
from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_pbr_stage_summary import IDENTITY,aggregate
from experiments.deformnext_retargeting.pbr_confirmation_initial_identity import (
    compare_initial,initial_identity,POLICY,RENDER_ABSOLUTE_LIMITS,
)
from experiments.deformnext_retargeting.pbr_initial_render_identity import validate_initial_latent_and_render
from experiments.deformnext_retargeting.selected_pbr_seed_confirmation import summarize
from test_pbr_initial_render_identity import pair,update
from test_selected_pipeline_freeze_v2 import pbr_inputs


def recorded(drift=True):
    first,cold=pair();reference=deepcopy(first)
    if drift:
        first['quick_per_asset'][0]['metrics']['render_psnr']+=.0025
        first['quick_per_asset'][1]['metrics']['crop_lpips']+=.00015
        update(first)
    proof=deepcopy(validate_initial_latent_and_render(first,cold))
    return dict(validation_rows=[first],initial_render_identity=proof),reference,cold


def test_actual_producer_is_recomputed_and_metric_values_stay_unchanged():
    record,reference,cold=recorded();before=deepcopy((record,reference,cold))
    proof=initial_identity(record['validation_rows'][0],cold,
        producer=record['initial_render_identity'],declared_policy=POLICY)
    comparison=compare_initial(record,reference)
    assert proof==record['initial_render_identity'] and comparison['actual_inputs_noise_and_latents_exact']
    assert (record,reference,cold)==before


def test_legacy_exact_rendering_remains_valid_without_inventing_a_producer():
    record,reference,cold=recorded(False)
    proof=initial_identity(record['validation_rows'][0],cold)
    record.pop('initial_render_identity')
    assert proof['original_render_metrics_exact'] and compare_initial(record,reference)


@pytest.mark.parametrize('damage',['no-producer','wrong-policy','changed-proof','no-policy'])
def test_bounded_native_path_requires_its_own_recorded_policy_and_proof(damage):
    record,_,cold=recorded();producer=deepcopy(record['initial_render_identity']);policy=POLICY
    if damage=='no-producer':producer=None
    elif damage=='wrong-policy':policy='other-policy'
    elif damage=='no-policy':policy=None
    else:producer['render_metrics_within_declared_absolute_bounds']=False
    with pytest.raises(ValueError):initial_identity(record['validation_rows'][0],cold,
        producer=producer,declared_policy=policy)


@pytest.mark.parametrize('field',(*IDENTITY,'actual_prediction_PBR_latent_SHA256'))
def test_render_tolerance_never_applies_to_inputs_camera_GT_noise_or_latents(field):
    record,reference,_=recorded();record['validation_rows'][0]['quick_per_asset'][0][field]='different'
    with pytest.raises(ValueError):compare_initial(record,reference)


@pytest.mark.parametrize('key,delta',[('render_psnr',.011),('render_ssim',.00021),
    ('crop_lpips',.00031),('foreground_fraction',1e-9)])
def test_existing_renderer_limits_and_unlisted_exact_fields_are_not_loosened(key,delta):
    record,reference,_=recorded();record['validation_rows'][0]['quick_per_asset'][0]['metrics'][key]+=delta
    with pytest.raises(ValueError):compare_initial(record,reference)


@pytest.mark.parametrize('damage',['no-proof','loose-limit','false-exact-identity','NaN'])
def test_bounded_values_alone_do_not_stand_in_for_validated_native_proof(damage):
    record,reference,_=recorded()
    if damage=='no-proof':record.pop('initial_render_identity')
    elif damage=='loose-limit':record['initial_render_identity']['render_absolute_limits']['lpips']=.1
    elif damage=='false-exact-identity':record['initial_render_identity']['actual_inputs_noise_GT_and_PBR_latent_exact']=False
    else:record['validation_rows'][0]['quick_per_asset'][0]['metrics']['crop_lpips']=float('nan')
    with pytest.raises(ValueError):compare_initial(record,reference)


def test_complete_two_seed_summary_preserves_real_bounded_initial_curve_values():
    records,configs,hashes,kw=pbr_inputs();mode=kw['selected']
    record=records[202][mode];first=record['validation_rows'][0]
    cold_rows=deepcopy(first['quick_per_asset'])
    for index in range(4,32):
        row=deepcopy(cold_rows[0]);row['sample_id']=str(index);cold_rows.append(row)
    cold=dict(status='PASS_ACTUAL_NATIVE_PBR_COLD_GENERATED32',source_multiview_step=16850,
        optimizer_updates=0,original_and_source_states_unchanged=True,per_asset=cold_rows)
    first['quick_per_asset'][0]['metrics']['crop_lpips']+=.0001
    first['quick_metrics']=aggregate([r['metrics'] for r in first['quick_per_asset']])
    record['initial_render_identity']=validate_initial_latent_and_render(first,cold)
    value=first['quick_metrics']['crop_lpips']
    result=summarize(records,configs,hashes,**kw)
    assert result['curves'][202][mode][0]['pooled_metrics']['crop_lpips']==value
    assert result['actual_initial4_latents_exact'] and result['renderer_bit_identity_not_claimed']
    assert result['initial_renderer_absolute_limits']==RENDER_ABSOLUTE_LIMITS
    assert first['quick_metrics']['crop_lpips']==value
