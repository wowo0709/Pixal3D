"""Paired32-case reliance contrasts; not retrained module effects or adoption."""
import math
import numpy as np
from .spatial_readout_controls import CONTROLS


FIXED_FIELDS={'candidate_indices','matching_prior','native_spatial_values','Top1_feature',
              'R_target_context','source_use_probability','fusion_gate'}
IDENTITY=('sample_id','source_cluster_id','seed','target_voxels_sha256','target_voxel_shape','target_representation')
METRICS=('voxel_iou','voxel_precision','voxel_recall','voxel_f1')


def require(condition,message):
    if not condition: raise ValueError(message)


def summarize(records):
    require(tuple(records)==CONTROLS,'normal and all four spatial saved-weight controls required')
    normal=records['normal'];reference=normal['per_asset'];nd=normal['spatial_readout_diagnostic']
    require(nd['actual_original32_prediction_latents_and_voxels_exact'] is True,'actual original32 exact normal proof required')
    ids=[r['sample_id'] for r in reference]
    require(len(ids)==len(set(ids))==32,'complete unique32 development panel required')
    metrics={};arrays={}
    for name,r in records.items():
        d=r['spatial_readout_diagnostic']
        require(r['status']=='PASS' and r['stage']=='ss64' and r['step']==480 and r['weights']=='raw'
            and r['optimizer_updates_during_evaluation']==0 and len(r['per_asset'])==32
            and r['sampling_steps']==12 and r['guidance']==3. and r['generation_batch_size']==8
            and r['input_states']==['M1','M0','M0','M0'] and r['input_camera_ids']==[8,9,10,11]
            and r['deterministic_evaluation']['actual_deterministic_algorithms'] is True
            and d['status']=='PASS_ACTUAL_SAVED_SPATIAL480_READOUT_CONTROL32' and d['control']==name
            and d['actual_trained_updates']==480 and d['optimizer_updates']==0 and d['actual_unused_optimizer_states']==0
            and d['actual_model_states_unchanged'] is True,'actual frozen480 native spatial protocol required')
        require(d['reference_archive_SHA256']==nd['reference_archive_SHA256']
            and d['actual_loaded_model_state_SHA256']==nd['actual_loaded_model_state_SHA256']
            and d['actual_source_checkpoint_SHA256']==nd['actual_source_checkpoint_SHA256'], 'same actual model/reference required')
        require(set(d['actual_loaded_model_state_SHA256'])=={'denoiser','deform_conditioner'}
            and len(d['actual_source_checkpoint_SHA256'])==2
            and all(isinstance(v,str) and len(v)==64 for v in d['actual_loaded_model_state_SHA256'].values()),
            'complete actual loaded model identity required')
        source=r['native_prior_source_ray']
        require(source['feature_control']=='spatial-feature-attention' and source['actual_feature_head_tensors']==9
            and source['original_correspondence_candidate_count']==3 and source['physical_deformation_query_channels']==0,
            'physical native3 nine-tensor feature head and NoDQ required')
        require(all(len(d[k])==4 for k in ['events','consumer_events','encoder_events','sampler_events'])
            and len(d['actual_SS_artifacts'])==32,'complete four native batch8 events and32 predictions required')
        require(d['encoder_events']==nd['encoder_events'] and d['sampler_events']==nd['sampler_events'],
                'actual RGB/camera/feature/global/negative/noise/sampler identity changed')
        for row,base,artifact in zip(r['per_asset'],reference,d['actual_SS_artifacts']):
            require(all(row[k]==base[k] for k in IDENTITY),'paired case/GT/cluster/seed changed')
            require(artifact['sample_id']==row['sample_id'] and artifact['seed']==row['seed']
                and artifact['target_voxels_SHA256']==row['target_voxels_sha256']
                and all(artifact['occupancy_counts'][k]==row[k] for k in ['tp','fp','fn']), 'actual archived prediction does not match metrics')
            require(all(math.isfinite(row[m]) for m in METRICS),'nonfinite actual voxel metric')
        for i,event in enumerate(d['events']):
            require(set(event['fixed_source_fields'])==FIXED_FIELDS
                and event['fixed_source_fields']==nd['events'][i]['fixed_source_fields']
                and event['original_candidates_Top1_R_target_and_source_use_exact'] is True
                and event['same_forward_weights_unchanged'] is True,'native3 proposals/prior/carrier/use/gate/R target changed')
            c=d['consumer_events'][i];b=nd['consumer_events'][i]
            require(c['target_native_global_object_preserved'] is True
                and c['native_projected_SHA256']==b['native_projected_SHA256'], 'native projection/global changed')
        values=np.asarray([[row[m] for m in METRICS] for row in r['per_asset']],dtype=np.float64)
        arrays[name]=values
        sums={k:sum(row[k] for row in r['per_asset']) for k in ['tp','fp','fn']}
        metrics[name]=dict(mean_asset=dict(zip(METRICS,values.mean(0).tolist())),
            pooled_IoU=sums['tp']/sum(sums.values()),counts=sums)
    clusters={}
    for i,row in enumerate(reference):clusters.setdefault(row['source_cluster_id'],[]).append(i)
    groups=[clusters[k] for k in sorted(clusters)];sizes=np.asarray([len(g) for g in groups])
    resamples=np.random.default_rng(20261023).integers(0,len(groups),(4000,len(groups)))
    contrasts={}
    for name in CONTROLS[1:]:
        delta=arrays[name]-arrays['normal'];sums=np.stack([delta[g].sum(0) for g in groups])
        boot=sums[resamples].sum(1)/sizes[resamples].sum(1)[:,None]
        contrasts[name+'_minus_normal']=dict(mean_asset_delta=dict(zip(METRICS,delta.mean(0).tolist())),
            exploratory_cluster_CI95={m:np.quantile(boot[:,i],[.025,.975]).tolist() for i,m in enumerate(METRICS)},
            primary_IoU_family4_CI95=np.quantile(boot[:,0],[.05/(2*4),1-.05/(2*4)]).tolist())
    return dict(status='PASS_ACTUAL_SPATIAL480_SAVED_READOUT_CONTROL32_ANALYSIS',assets=32,
        source_clusters=len(groups),generation_noise_repeats=1,independent_training_seeds=1,
        metrics=metrics,contrasts=contrasts,primary_metric='mean-asset voxel IoU',
        bootstrap_draws=4000,bootstrap_seed=20261023,primary_IoU_family_comparisons=4,
        secondary_metrics_exploratory_unadjusted=True,actual_original_normal_latents_and_voxels_exact=True,
        readout_reliance_not_retrained_arms=True,quality_gain_proven=False,model_adopted=False,full_goal_complete=False)
