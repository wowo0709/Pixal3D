"""Reduce the prespecified twelve-model matched DEV study, without adoption."""
import numpy as np

from .module_pipeline_protocol_v2 import protocol


ALIASES = dict(V1='vanilla',C_DECODER='R_top1_only',FULL='R_raw_topk')
SOURCE_COUNTS = dict(V1=0,C_SPATIAL=10,C_RAW=10,C_D1_COS=10,C_DECODER=10,
    C_TOP1_V=12,C_FIXED_TOPK=12,FULL=19,FULL_QK_ONLY=19,FULL_VOXEL_Q=19,
    C_DECODER_NO_R=5,FULL_MEAN_MATCH=19)
METRICS = ('voxel_iou','voxel_precision','voxel_recall','voxel_f1')


def summarize_unified_modules(results, *, draws=4000, seed=42022):
    spec=protocol()
    if set(results)!=set(spec['arms']) or any(len(v)!=3 for v in results.values()):
        raise ValueError('all twelve physical models and three actual matched generation noises required')
    first=results['V1'][0]['per_asset']
    ids=[r['sample_id'] for r in first];clusters=[r['source_cluster_id'] for r in first]
    if len(ids)!=128 or len(set(ids))!=128 or any(not c for c in clusters):
        raise ValueError('unique128 assets and original nonempty source-cluster identities required')
    matrices={}
    for arm,noises in results.items():
        matrices[arm]=[]
        for j,result in enumerate(noises):
            rows=result['per_asset'];reference=results['V1'][j]['per_asset']
            if (result['stage']!='ss64' or result['step']!=480
                    or result['native_variant']!=ALIASES.get(arm,arm)
                    or result['evaluation_mode']!='final128'
                    or result['optimizer_updates_during_evaluation']!=0
                    or result['actual_source_tensors']!=SOURCE_COUNTS[arm]
                    or [r['sample_id'] for r in rows]!=ids
                    or [r['source_cluster_id'] for r in rows]!=clusters
                    or any((a['seed'],a['target_voxels_sha256'])!=(b['seed'],b['target_voxels_sha256'])
                           for a,b in zip(rows,reference))):
                raise ValueError('actual model, checkpoint step, source structure, asset/GT/noise pairing differs')
            matrices[arm].append([[r[m] for m in METRICS] for r in rows])
        matrices[arm]=np.asarray(matrices[arm],dtype=np.float64)
    if any(len({results['V1'][j]['per_asset'][i]['seed'] for j in range(3)})!=3 for i in range(128)):
        raise ValueError('three distinct generation seeds per asset required')
    if any(not np.isfinite(v).all() or ((v<0)|(v>1)).any() for v in matrices.values()):
        raise ValueError('finite voxel metrics in[0,1] required')
    assets={n:v.mean(0) for n,v in matrices.items()}
    groups=[np.flatnonzero(np.asarray(clusters)==c) for c in sorted(set(clusters))]
    sizes=np.asarray([len(g) for g in groups])
    resamples=np.random.default_rng(seed).integers(0,len(groups),(draws,len(groups)))
    comparisons={}
    contrasts=[dict(effect='pipeline_'+b,candidate=a,reference=b,family_size=2)
               for a,b in spec['primary_pipeline_contrasts']]
    contrasts += [dict(c,family_size=10) for c in spec['module_contrasts']]
    for c in contrasts:
        a,b,family=c['candidate'],c['reference'],c['family_size']
        delta=assets[a]-assets[b]
        sums=np.stack([delta[g].sum(0) for g in groups])
        boot=sums[resamples].sum(1)/sizes[resamples].sum(1)[:,None]
        source=(sums/sizes[:,None])[resamples].mean(1)
        comparisons[c['effect']]=dict(candidate=a,reference=b,
            mean_asset_delta=dict(zip(METRICS,delta.mean(0).tolist())),
            source_cluster_CI95={m:np.quantile(boot[:,k],[.025,.975]).tolist() for k,m in enumerate(METRICS)},
            mean_source_delta=dict(zip(METRICS,(sums/sizes[:,None]).mean(0).tolist())),
            equal_source_cluster_CI95={m:np.quantile(source[:,k],[.025,.975]).tolist() for k,m in enumerate(METRICS)},
            prespecified_IoU_family_size=family,
            primary_IoU_family_CI95=np.quantile(boot[:,0],[.025/family,1-.025/family]).tolist())
    return dict(status='PASS_ACTUAL_TWELVE_MODEL_PAIRED_MODULE_DEV128X3',
        assets=128,source_clusters=len(groups),generation_noises=3,independent_training_seeds=1,
        noise_is_not_independent_training_seed=True,
        primary_metric='mean asset voxel IoU, averaged over three paired generation noises',
        per_model_mean_asset={n:dict(zip(METRICS,v.mean((0,1)).tolist())) for n,v in matrices.items()},
        per_model_native_global_occupancy_metrics={n:[r['quick_metrics'] for r in noises] for n,noises in results.items()},
        comparisons=comparisons,module_family_size=10,pipeline_family_size=2,
        native_spatial3_to_semantic64_is_module_substitution_not_fixed_K_address_effect=True,
        retargeting_effect_measured_at_Top1_not_full_attention_interaction=True,
        repeated_development_search_is_not_independent_test=True,
        full_cascade_gain_proven=False,independent_training_seed_confirmation=False,
        held_out_test508_evaluated=False,automatic_model_adoption=False,full_goal_complete=False)
