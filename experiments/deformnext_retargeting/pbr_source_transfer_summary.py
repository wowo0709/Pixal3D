"""Paired descriptive readout of one trained PBR model, never new arm quality."""
import math
import statistics

from .native_pbr_stage_summary import IDENTITY, METRICS, aggregate, is_digest, require
from .pbr_source_transfer_controls import CONTROLS


KEYS=('normal','normal-repeat',*CONTROLS[1:])
SOURCE_FIELDS={'candidate_indices','matching_logits','candidate_weights','Top1_carrier',
               'computed_R','source_use','fusion_scale'}


def summarize(records):
    require(tuple(records)==KEYS,'two independent normal loads and all three saved-weight controls required')
    normal=records['normal'];reference=normal['per_asset'];curves={}
    require(len(reference)==4 and len({r['sample_id'] for r in reference})==4,'four distinct actual assets required')
    for key,record in records.items():
        require(record.get('status')=='PASS_ACTUAL_NATIVE_PBR1900_SOURCE_TRANSFER_READOUT_FIXED4'
            and record.get('diagnostic_key')==key
            and record.get('control')==('normal' if key=='normal-repeat' else key)
            and record.get('model_mode')=='top1-only'
            and record.get('actual_trained_model_updates')==1900
            and record.get('optimizer_updates_in_diagnostic')==0
            and record.get('unused_eval_optimizer_states')==0
            and record.get('actual_deterministic_algorithms') is True
            and record.get('all_weights_unchanged') is True
            and record.get('frozen_original_base_and_matcher_exact') is True
            and record.get('fixed_GT_target_shape') is True
            and record.get('full_cascade') is False and record.get('source_preservation') is False,
            'own actual1900 deterministic, read-only fixed-GT-shape protocol required')
        require(set(record['actual_model_files'])=={'denoiser','deform_conditioner'}
            and set(record['actual_model_state_SHA256'])=={'denoiser','deform_conditioner'}
            and record['actual_model_files']==normal['actual_model_files']
            and record['actual_model_state_SHA256']==normal['actual_model_state_SHA256']
            and record['actual_frozen_encoder_SHA256']==normal['actual_frozen_encoder_SHA256']
            and all(is_digest(v) for v in record['actual_model_state_SHA256'].values())
            and is_digest(record['actual_frozen_encoder_SHA256']), 'same actual model and frozen encoder weights required')
        rows=record['per_asset']
        require(len(rows)==4 and all(len(record[k])==4 for k in
            ('module_events','consumer_events','condition_events','panel_pixel_SHA256')), 'four complete actual generated events required')
        for i,(row,base) in enumerate(zip(rows,reference)):
            require(all(row[k]==base[k] for k in IDENTITY),'RGB/noise/GT/Shape/camera/input identity changed')
            fields=record['module_events'][i]['fixed_source_fields']
            require(set(fields)==SOURCE_FIELDS and all(is_digest(v) for v in fields.values())
                and fields==normal['module_events'][i]['fixed_source_fields'], 'candidate/score/carrier/R/use/gate changed before masking')
            require(record['condition_events'][i]==normal['condition_events'][i],
                    'native global, negative conditioning or GT Shape changed')
            consumer=record['consumer_events'][i];original=normal['consumer_events'][i]
            require(consumer['native_global_object_preserved'] is True
                and consumer['native_global_SHA256']==original['native_global_SHA256']
                and consumer['native_proj_SHA256']==original['native_proj_SHA256']
                and math.isfinite(consumer['actual_delta_RMS']) and consumer['actual_delta_RMS']>=0,
                'native consumer must be shared and source change finite')
            require(record['panel_pixel_SHA256'][i]['GT']==normal['panel_pixel_SHA256'][i]['GT'],
                    'actual GT render pixels changed between controls')
            if key=='normal-repeat':
                require(row['actual_prediction_PBR_latent_SHA256']==base['actual_prediction_PBR_latent_SHA256']
                    and row['metrics']==base['metrics']
                    and consumer==original
                    and record['panel_pixel_SHA256'][i]==normal['panel_pixel_SHA256'][i],
                    'independent normal loads must exactly reproduce predictions and rendering')
            if key=='both-source-contributions-off':
                require(consumer['actual_delta_RMS']==0,'both-off must deliver zero actual correction')
        pooled=aggregate([r['metrics'] for r in rows])
        require(set(record['quick_metrics'])==METRICS and all(math.isclose(
            v,record['quick_metrics'][k],rel_tol=0,abs_tol=1e-12) for k,v in pooled.items()), 'pooled PSNR aggregation differs')
        curves[key]=dict(pooled_metrics=pooled,
            mean_asset_metrics={k:statistics.mean(r['metrics'][k] for r in rows) for k in METRICS})
    contrasts={}
    for key in KEYS[2:]:
        contrasts[key+'_minus_normal']=dict(
            metric_deltas={kind:{m:curves[key][kind][m]-curves['normal'][kind][m] for m in METRICS}
                           for kind in ('pooled_metrics','mean_asset_metrics')},
            per_asset=[dict(sample_id=r['sample_id'],delta={m:r['metrics'][m]-b['metrics'][m] for m in METRICS})
                       for r,b in zip(records[key]['per_asset'],reference)])
    return dict(status='PASS_ACTUAL_NATIVE_PBR1900_SOURCE_TRANSFER_FIXED4_READOUT_ANALYSIS',
        assets=4,model_mode='top1-only',trained_updates=1900,diagnostic_optimizer_updates=0,
        independent_normal_repeats_exact=True,curves=curves,contrasts=contrasts,
        original_epoch_normal_prediction_exact=normal['actual_original_epoch_prediction_exact'],
        original_epoch_GT_pixels_exact=normal.get('actual_original_epoch_GT_pixels_exact'),
        protocol='new deterministic saved-weight readout, separately labelled from original epoch validation',
        original_epoch_metrics_not_used_for_control_contrasts=True,
        inference_interventions_not_retrained_ablation=True,fixed_GT_shape=True,
        source_preservation=False,full_cascade=False,inferential_CI_or_model_adoption=False,
        quality_gain_proven=False,full_goal_complete=False)
