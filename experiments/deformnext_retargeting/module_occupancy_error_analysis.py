"""Describe SS count differences without claiming a neural mechanism.

The symmetric two-factor decomposition is exact for IoU=TP/(GT+FP). Its
components describe missed hits versus extra occupancy in retrained models;
they are not inference interventions or new hypothesis tests.
"""
import numpy as np

from .module_pipeline_protocol_v2 import protocol
from .unified_module_summary import ALIASES,SOURCE_COUNTS


def require(value,message):
    if not value:raise ValueError(message)


def decompose_counts(candidate,reference):
    a,b=np.asarray(candidate),np.asarray(reference)
    require(a.ndim==3 and a.shape==b.shape and a.shape[-1]==3 and min(a.shape)>0
        and a.dtype.kind in 'iu' and b.dtype.kind in 'iu' and np.all(a>=0) and np.all(b>=0),
        'aligned nonnegative integer [noise,case,TP-FP-FN] arrays required')
    a,b=a.astype(np.float64),b.astype(np.float64);gt=a[:,:,0]+a[:,:,2]
    require(np.all(gt>0) and np.array_equal(gt,b[:,:,0]+b[:,:,2])
        and np.array_equal(gt,np.broadcast_to(gt[0],gt.shape)), 'identical positive actual GT occupancy for every model and noise required')
    tpa,fpa=a[:,:,0],a[:,:,1];tpb,fpb=b[:,:,0],b[:,:,1]
    ia,ib=tpa/(gt+fpa),tpb/(gt+fpb)
    tp=.5*(tpa-tpb)*(1/(gt+fpa)+1/(gt+fpb))
    fp=.5*(tpa+tpb)*(1/(gt+fpa)-1/(gt+fpb))
    require(np.allclose(tp+fp,ia-ib,rtol=0,atol=1e-14),'exact symmetric IoU decomposition failed')
    delta=(ia-ib).mean(0);edges=np.quantile(gt[0],[.25,.5,.75]);bins=np.searchsorted(edges,gt[0],side='right')
    strata=[]
    for index in range(4):
        selected=bins==index;n=int(selected.sum())
        strata.append(dict(GT_occupancy_quartile=index+1,cases=n,mean_case_IoU_delta=None if n==0 else float(delta[selected].mean())))
    def describe(values):
        tp,fp,fn=values.sum((0,1));target=tp+fn
        return dict(mean_case_IoU=float((values[:,:,0]/values.sum(2)).mean()),
            mean_noise_global_micro_IoU=float(np.mean(values[:,:,0].sum(1)/values.sum((1,2)))),
            all_noise_pooled_prediction_GT_volume_ratio=float((tp+fp)/target))
    return dict(candidate=describe(a),reference=describe(b),mean_case_IoU_delta=float(delta.mean()),
        exact_symmetric_TP_component=float(tp.mean()),exact_symmetric_FP_component=float(fp.mean()),
        mean_case_delta_TP_per_GT=float(((a[:,:,0]-b[:,:,0])/gt).mean()),
        mean_case_delta_FP_per_GT=float(((a[:,:,1]-b[:,:,1])/gt).mean()),
        mean_case_delta_FN_per_GT=float(((a[:,:,2]-b[:,:,2])/gt).mean()),
        improved_cases=int((delta>0).sum()),worsened_cases=int((delta<0).sum()),tied_cases=int((delta==0).sum()),
        GT_occupancy_quartile_edges_voxels=edges.tolist(),GT_occupancy_strata=strata,
        per_case_IoU_delta=delta.tolist(),per_case_TP_component=tp.mean(0).tolist(),per_case_FP_component=fp.mean(0).tolist(),
        decomposition_is_algebraic_not_mechanistic=True,strata_are_descriptive_posthoc=True)


def analyze(results):
    spec=protocol();require('V1' in results and len(results)>=2 and set(results)<=set(spec['arms']),
        'available actual original-study models including V1 required')
    reference=results['V1'];require(len(reference)==3,'three actual generation noises required')
    first=reference[0]['per_asset'];ids=[r['sample_id'] for r in first];clusters=[r['source_cluster_id'] for r in first]
    require(len(ids)==len(set(ids))==128 and len(set(clusters))==93,'actual128-case/93-source development population required')
    matrices={};max_error=0.
    for arm,noises in results.items():
        require(len(noises)==3,'same three generation noises required')
        matrix=[]
        for index,record in enumerate(noises):
            rows=record['per_asset'];base=reference[index]['per_asset']
            require(record.get('status')=='PASS' and record.get('stage')=='ss64' and record.get('step')==480
                and record.get('native_variant')==ALIASES.get(arm,arm) and record.get('actual_source_tensors')==SOURCE_COUNTS[arm]
                and record.get('evaluation_mode')=='final128' and record.get('optimizer_updates_during_evaluation')==0
                and len(rows)==128 and all(tuple(a[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256'])
                    ==tuple(b[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256']) for a,b in zip(rows,base)),
                'own actual480 saved model and complete paired source/GT/noise identities required')
            for r in rows:
                require(all(type(r[k]) is int and r[k]>=0 for k in ['tp','fp','fn']) and r['tp']+r['fn']>0,'actual nonnegative counts and nonempty GT required')
                difference=abs(r['voxel_iou']-r['tp']/(r['tp']+r['fp']+r['fn']));require(difference<=1e-7,'stored IoU differs from actual counts')
                max_error=max(max_error,difference)
            matrix.append([[r[k] for k in ['tp','fp','fn']] for r in rows])
        matrices[arm]=np.asarray(matrix,dtype=np.int64)
    require(all(len({reference[j]['per_asset'][i]['seed'] for j in range(3)})==3 for i in range(128)), 'distinct three generation seeds per case required')
    contrasts=[dict(c,family_size=10) for c in spec['module_contrasts']]
    contrasts += [dict(effect='pipeline_'+b,candidate=a,reference=b,family_size=2) for a,b in spec['primary_pipeline_contrasts']]
    comparisons={}
    for c in contrasts:
        a,b=c['candidate'],c['reference']
        if a not in matrices or b not in matrices:continue
        detail=decompose_counts(matrices[a],matrices[b])
        candidate_summary=detail.pop('candidate');reference_summary=detail.pop('reference')
        comparisons[c['effect']]=dict(candidate=a,reference=b,prespecified_IoU_family_size=c['family_size'],
            candidate_occupancy_summary=candidate_summary,reference_occupancy_summary=reference_summary,**detail)
    return dict(status='DESCRIPTIVE_ACTUAL_PAIRED_SS_OCCUPANCY_ERROR_DECOMPOSITION',assets=128,source_clusters=93,
        generation_noises=3,independent_training_seeds=1,available_models=list(results),
        missing_original_arms=[a for a in spec['arms'] if a not in results],all_twelve_arms_complete=set(results)==set(spec['arms']),
        comparisons=comparisons,case_ids=ids,source_cluster_ids=clusters,maximum_count_to_stored_IoU_difference=max_error,
        original_module_family_size=10,original_pipeline_family_size=2,new_significance_tests_performed=False,
        no_mechanism_or_inference_intervention_inferred_from_counts=True,GT_size_is_occupancy_not_deformation_severity=True,
        native_spatial3_to_semantic64_is_pathway_substitution=True,retargeting_contrast_is_conditional_on_Top1=True,
        no_new_neural_forward_or_optimizer_update=True,heldout_test_evaluated=False,full_goal_complete=False)
