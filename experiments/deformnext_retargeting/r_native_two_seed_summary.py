"""Separate training-seed replication from three-noise native SS evaluation."""
import copy

import numpy as np

from .r_mediated_native_recipe import common_contract


VARIANTS=('vanilla','R_top1_only')
METRICS=('voxel_iou','voxel_precision','voxel_recall','voxel_f1')


def validate_traces(traces):
    if set(traces)!={101,202}:raise ValueError('two actual training seeds required')
    for seed in (101,202):
        if set(traces[seed])!=set(VARIANTS):raise ValueError('both trained controls required')
        a,b=(traces[seed][v]for v in VARIANTS)
        if (len(a)!=480 or len(b)!=480 or a!=b
                or [r['step']for r in a]!=list(range(1,481))
                or any(len(r['asset_ids'])!=1 or len(r['asset_ids'][0])!=96 for r in a)):
            raise ValueError('complete paired480 actual asset/noise/timestep/global traces required')
    if traces[101]['vanilla']==traces[202]['vanilla']:
        raise ValueError('seed202 must change the actual training stream')


def summarize(results,configs,*,draws=4000,seed=20261022):
    if set(results)!={101,202}or set(configs)!={101,202}:
        raise ValueError('actual seed101 and seed202 results/configurations required')
    identities=None;matrices={}
    for training_seed in (101,202):
        if set(results[training_seed])!=set(VARIANTS)or set(configs[training_seed])!=set(VARIANTS):
            raise ValueError('own vanilla/physical Top1 results and configs required for both seeds')
        matrices[training_seed]={}
        for variant in VARIANTS:
            cfg=configs[training_seed][variant];root=cfg['output_dir']
            if cfg['training_seed']!=training_seed:raise ValueError('configuration belongs to another training seed')
            evaluations=results[training_seed][variant]
            if len(evaluations)!=3:raise ValueError('three fixed generation noises required')
            values=[];own=[]
            for result in evaluations:
                rows=result['per_asset']
                if (result.get('status')!='PASS'or result.get('stage')!='ss64'or result.get('step')!=480
                        or result.get('native_variant')!=variant or result.get('evaluation_mode')!='final128'
                        or result.get('optimizer_updates_during_evaluation')!=0
                        or result.get('actual_source_tensors')!=(0 if variant=='vanilla'else 10)
                        or not result.get('deterministic_algorithms')
                        or result.get('checkpoint')!=root+'/ckpts/denoiser_step0000480.pt'
                        or result.get('conditioner_checkpoint')!=(None if variant=='vanilla'else root+'/ckpts/deform_conditioner_step0000480.pt')
                        or result.get('arm_config')!=root+'/config.json'
                        or result.get('input_states')!=['M1','M0','M0','M0']
                        or result.get('input_camera_ids')!=[8,9,10,11]or len(rows)!=128):
                    raise ValueError('actual own native480 RGB checkpoint and deterministic128 records required')
                ids=[r['sample_id']for r in rows]
                if len(set(ids))!=128:raise ValueError('unique128 assets required')
                own.append([(r['sample_id'],r['source_cluster_id'],r['seed'],r['target_voxels_sha256'])for r in rows])
                values.append([[r[m]for m in METRICS]for r in rows])
            if identities is None:identities=own
            elif own!=identities:raise ValueError('paired assets/source clusters/generation noise/GT differ')
            matrix=np.asarray(values,dtype=np.float64)
            if matrix.shape!=(3,128,4)or not np.isfinite(matrix).all():raise ValueError('finite complete native metrics required')
            matrices[training_seed][variant]=matrix.mean(0)
    for variant in VARIANTS:
        contracts=[copy.deepcopy(common_contract(configs[s][variant]))for s in (101,202)]
        for c in contracts:c.pop('training_seed')
        if contracts[0]!=contracts[1]:raise ValueError('training recipe changed beyond the training seed')
    if any(len({identities[n][i][2]for n in range(3)})!=3 for i in range(128)):
        raise ValueError('generation noises must be distinct within each asset')
    clusters=np.asarray([r[1]for r in identities[0]])
    if any(not str(c)for c in clusters):raise ValueError('actual source-cluster IDs required')
    groups=[np.flatnonzero(clusters==c)for c in sorted(set(clusters))];sizes=np.asarray([len(g)for g in groups])
    resamples=np.random.default_rng(seed).integers(0,len(groups),(draws,len(groups)))
    def comparison(delta):
        sums=np.stack([delta[g].sum(0)for g in groups])
        boot=sums[resamples].sum(1)/sizes[resamples].sum(1)[:,None]
        return dict(mean_asset_delta=dict(zip(METRICS,delta.mean(0).tolist())),
            source_cluster_CI95={m:np.quantile(boot[:,i],[.025,.975]).tolist()for i,m in enumerate(METRICS)})
    differences={s:matrices[s]['R_top1_only']-matrices[s]['vanilla']for s in (101,202)}
    per_seed={s:dict(means={v:dict(zip(METRICS,matrices[s][v].mean(0).tolist()))for v in VARIANTS},
        Top1_R_minus_vanilla=comparison(differences[s]))for s in (101,202)}
    return dict(status='PASS_ACTUAL_NATIVE_TWO_TRAINING_SEEDS_PAIRED128_THREE_GENERATION_NOISES',
        training_seeds=[101,202],generation_noises=3,assets=128,source_clusters=len(groups),
        per_training_seed=per_seed,fixed_two_seed_average=comparison(np.stack(list(differences.values())).mean(0)),
        confidence_intervals_resample_asset_source_clusters_not_training_seed_population=True,
        Top1_R_IoU_direction_positive_in_both_training_seeds=all(d[:,0].mean()>0 for d in differences.values()),
        independent_decoder_or_semantic_vs_spatial_effect_identified=False,
        full_cascade_or_test508_evaluated=False,automatic_model_adoption=False,full_goal_complete=False)
