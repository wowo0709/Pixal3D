"""Final Fixed-context SS128x3 evidence and exact downstream table binding.

No model execution, selection or file mutation occurs here. Callers verify
each configuration/evaluation against its actual canonical file receipt.
"""
from copy import deepcopy
import math
import re

import numpy as np

from .final_fixed_ablation_cascade import ARMS, BASELINES, CONTRASTS, own_contract
from .selected_pipeline_model_contract import require, validate_ss_evaluation


METRICS = ('voxel_iou','voxel_precision','voxel_recall','voxel_f1')
STATUS = 'PASS_ACTUAL_FINAL_FIXED_SEVEN_MODEL_SS128X3'


def validate_series(arm, binding, evaluations, references):
    """Validate all three actual own checkpoints, samples, counts and noises."""
    require(arm in ARMS + BASELINES and len(evaluations) == len(references) == 3,
            'one required own model and three actual paired generation noises required')
    require(type(binding.get('training_seed')) is int and binding['training_seed'] == 101,
            'this ablation conditions on the actual training seed101')
    ids = [r['sample_id'] for r in references[0]['per_asset']]
    clusters = [r['source_cluster_id'] for r in references[0]['per_asset']]
    require(len(ids) == len(set(ids)) == 128 and len(set(clusters)) == 93
            and all(isinstance(c,str) and c for c in clusters),
            'the complete original128/93-source development panel required')
    matrices, identities, targets = [], [], None
    manifest = references[0].get('manifest_sha256')
    require(isinstance(manifest,str) and re.fullmatch('[0-9a-f]{64}', manifest),
            'explicit original validation manifest identity required')
    for result,reference in zip(evaluations,references):
        if arm in ARMS:
            own_contract(arm).validate_archive128(result,arm,reference,binding=binding)
        else:
            validate_ss_evaluation(result,binding,arm,seed=101)
        require(result.get('evaluation_mode') == 'final128'
                and result.get('manifest_sha256') == reference.get('manifest_sha256') == manifest,
                'same actual final128 validation manifest required')
        rows, refs = result['per_asset'], reference['per_asset']
        require(len(rows) == len(refs) == 128
                and [r['sample_id'] for r in rows] == [r['sample_id'] for r in refs] == ids
                and [r['source_cluster_id'] for r in rows] == [r['source_cluster_id'] for r in refs] == clusters,
                'all ordered asset/source rows must match, including the suffix')
        own, values, target_ids = [], [], []
        for row,ref in zip(rows,refs):
            require(type(row['seed']) is int and 0 <= row['seed'] < 2**31
                    and row['seed'] == ref['seed']
                    and isinstance(row['target_voxels_sha256'],str)
                    and re.fullmatch('[0-9a-f]{64}',row['target_voxels_sha256'])
                    and row['target_voxels_sha256'] == ref['target_voxels_sha256'],
                    'same actual generation noise and scoring-only target identity required')
            tp,fp,fn = (row[k] for k in ('tp','fp','fn'))
            require(all(type(n) is int and n >= 0 for n in (tp,fp,fn))
                    and tp + fp + fn <= 64**3
                    and tp + fn == ref['tp'] + ref['fn'],
                    'valid original64-grid occupancy counts and common GT count required')
            expected = (tp/max(1,tp+fp+fn),tp/max(1,tp+fp),
                        tp/max(1,tp+fn),2*tp/max(1,2*tp+fp+fn))
            actual = [row[m] for m in METRICS]
            require(all(isinstance(v,(int,float)) and not isinstance(v,bool)
                    and math.isfinite(v) and 0 <= v <= 1
                    and abs(v-e) <= 1e-12 for v,e in zip(actual,expected)),
                    'stored per-asset macro metrics must agree with original occupancy counts')
            own.append((row['sample_id'],row['source_cluster_id'],row['seed'],row['target_voxels_sha256']))
            target_ids.append((row['target_voxels_sha256'],tp+fn)); values.append(actual)
        require(targets is None or targets == target_ids,
                'GT identity/count must remain the same across generation noises')
        targets = target_ids; identities.append(own); matrices.append(values)
    require(all(len({identities[n][i][2] for n in range(3)}) == 3 for i in range(128)),
            'three distinct actual generation noises per asset required; no replication by duplication')
    return np.asarray(matrices,dtype=np.float64), identities


def summarize(results, bindings, *, draws=12000, seed=20261025):
    require(set(results) == set(bindings) == set(ARMS + BASELINES),
            'all five actual controls and two bound baseline series required')
    require(type(draws) is int and draws >= 12000,
            'at least12000 source-cluster draws for the six primary IoU contrasts required')
    reference = results['C_DECODER']; matrices = {}; paired = None
    for arm in ARMS + BASELINES:
        values, identities = validate_series(arm,bindings[arm],results[arm],reference)
        require(paired is None or identities == paired,
                'all three asset/source/noise/GT populations must pair across every model')
        paired = identities; matrices[arm] = values
    ids = [r[0] for r in paired[0]]; clusters = np.asarray([r[1] for r in paired[0]])
    groups = [np.flatnonzero(clusters == c) for c in sorted(set(clusters))]
    sizes = np.asarray([len(g) for g in groups]); n = len(groups)
    counts = np.random.default_rng(seed).multinomial(n,np.full(n,1/n),size=draws)
    denominators = counts @ sizes
    assets = {arm:v.mean(0) for arm,v in matrices.items()}; contrasts = {}
    for a,b in CONTRASTS:
        delta = assets[a] - assets[b]
        sums = np.stack([delta[g].sum(0) for g in groups])
        boot = (counts @ sums)/denominators[:,None]
        source_boot = (counts @ (sums/sizes[:,None]))/n
        family = len(CONTRASTS)
        contrasts[a+'_minus_'+b] = dict(candidate=a,reference=b,
            mean_asset_delta=dict(zip(METRICS,delta.mean(0).tolist())),
            source_cluster_CI95={m:np.quantile(boot[:,k],[.025,.975]).tolist() for k,m in enumerate(METRICS)},
            mean_source_delta=dict(zip(METRICS,(sums/sizes[:,None]).mean(0).tolist())),
            equal_source_cluster_CI95={m:np.quantile(source_boot[:,k],[.025,.975]).tolist() for k,m in enumerate(METRICS)},
            prespecified_IoU_family_size=family,
            primary_IoU_family_CI95=np.quantile(boot[:,0],[.025/family,1-.025/family]).tolist())
    return dict(status=STATUS,assets=128,source_clusters=n,generation_noises=3,
        independent_training_seeds=1,noise_is_not_independent_training_seed=True,
        primary_metric='macro per-asset voxel IoU averaged over three paired noises',
        per_model_mean_asset={a:dict(zip(METRICS,v.mean((0,1)).tolist())) for a,v in matrices.items()},
        per_model_mean_source={a:dict(zip(METRICS,np.stack([assets[a][g].mean(0) for g in groups]).mean(0).tolist())) for a in matrices},
        per_model_per_noise_macro={a:[dict(zip(METRICS,v.mean(0).tolist())) for v in matrix] for a,matrix in matrices.items()},
        per_model_native_global_occupancy_metrics={a:[r['quick_metrics'] for r in results[a]] for a in matrices},
        per_asset={a:[dict(sample_id=sid,source_cluster_id=str(clusters[i]),
                    **dict(zip(METRICS,assets[a][i].tolist()))) for i,sid in enumerate(ids)] for a in matrices},
        comparisons=contrasts,primary_IoU_family_size=len(CONTRASTS),
        source_cluster_bootstrap_draws=draws,bootstrap_seed=seed,
        actual_SS_bindings=deepcopy(bindings),sample_ids=ids,
        source_clusters_by_sample=dict(zip(ids,clusters.tolist())),
        repeated_development_search_is_not_independent_test=True,
        held_out_test508_evaluated=False,full_cascade_gain_proven=False,
        automatic_model_adoption=False,full_goal_complete=False)


def bind_cascade(ss, cascade):
    """Join stage diagnostics with actual outputs from the same physical models."""
    require(ss.get('status') == STATUS
            and cascade.get('status') == 'PASS_ACTUAL_FINAL_FIXED_CONTEXT_DEV128_EIGHT_METRICS'
            and ss['assets'] == cascade['assets'] == 128
            and ss['source_clusters'] == cascade['source_clusters'] == 93
            and ss['generation_noises'] == 3 and cascade['generation_noises'] == 1
            and ss['independent_training_seeds'] == cascade['independent_training_seeds'] == 1,
            'actual complete SS128x3 and cascade128x1 scopes required')
    require(set(ss['actual_SS_bindings']) == set(cascade['measured_compositions']) == set(ARMS + BASELINES),
            'all seven physical models required in both stages')
    for arm in ARMS + BASELINES:
        model = cascade['measured_compositions'][arm]
        require(ss['actual_SS_bindings'][arm] == model['ss64'],
                'SS and cascade must use the same actual checkpoint/config/source bytes')
        require(model['pbr1024'] == cascade['measured_compositions']['E_FIXED_RAW']['pbr1024'],
                'the common actual Top-1 R texture model must remain the same')
        for role in ('target8','source9_10','all8_9_10'):
            rows = cascade['per_asset'][role][arm]
            require([r['sample_id'] for r in rows] == ss['sample_ids']
                    and all(r['source_cluster_id'] == ss['source_clusters_by_sample'][r['sample_id']] for r in rows),
                    'every stage/role must retain identical full128 asset/source pairing')
    return [dict(arm=arm,SS_IoU=ss['per_model_mean_asset'][arm]['voxel_iou'],
        CD=cascade['roles']['all8_9_10']['means'][arm]['deformation_weighted']['CD'],
        FScore=cascade['roles']['all8_9_10']['means'][arm]['deformation_weighted']['FScore'],
        LPIPS=cascade['roles']['source9_10']['means'][arm]['deformation_weighted']['LPIPS'],
        DreamSim=cascade['roles']['source9_10']['means'][arm]['deformation_weighted']['DreamSim'])
        for arm in ('M_FIXED_SPATIAL','M_FIXED_RAW','M_FIXED_D1_COS','E_FIXED_RAW',
                    'C_DECODER','A_FIXED_UNIFORM','A_FIXED_NO_R')]
