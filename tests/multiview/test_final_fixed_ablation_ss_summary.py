"""Pair actual SS fields/counts across noises before joining cascade metrics."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.deformnext_retargeting import final_fixed_ablation_ss_summary as module
from experiments.deformnext_retargeting import final_fixed_ablation_cascade as cascade
from test_fixed_correspondence_pipeline_contract import inputs as matching_inputs
from test_shared_ablation_pipeline_contract import fixture as delivery_inputs
from test_final_fixed_ablation_cascade import population, pbr


def data(tmp_path):
    results,bindings={},{}
    for j,arm in enumerate(module.ARMS+module.BASELINES):
        if arm.startswith('M_'):binding,record,_=matching_inputs(tmp_path,arm)
        else:
            binding,record,_,_=delivery_inputs('A_FIXED_UNIFORM' if arm in module.BASELINES else arm)
        if arm in module.BASELINES:
            root='/runs/'+arm
            binding=dict(schema='unified-module-SS480-frozen-binding-v1',module_arm=arm,
                variant='R_top1_only' if arm=='C_DECODER' else arm,
                runtime_family='r_top1' if arm=='C_DECODER' else 'consistent_evidence',
                step=480,source_tensors=10,training_seed=101,conditioning_mode='edited-rgb',
                source_teacher_updates=0,native_task_objective='SS_flow_mse')
            for key,path in [('native_config','config.json'),('checkpoint','ckpts/denoiser_step0000480.pt'),
                             ('conditioner_checkpoint','ckpts/deform_conditioner_step0000480.pt')]:
                binding[key]=dict(path=root+'/'+path,SHA256='b'*64)
            if arm=='E_FIXED_RAW':
                binding.update(schema='consistent-raw-evidence-SS480-frozen-binding-v1',
                    checkpoint_bundle_schema='consistent-raw-ray-evidence-SS-checkpoint-pair-v1',
                    evidence_mode='fixed-prior',shared_carrier_use_retarget_evidence=True,
                    learned_projection_tensors=0,learned_attention_head_tensors=0)
                record.update(conditioning_schema='consistent-raw-ray-evidence-native-v1',
                    followup_study_schema='consistent-raw-ray-evidence-followup-v1',
                    shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
                    learned_attention_head_tensors=0,training_seed=101)
            record['native_variant']=binding['variant']
            record['actual_source_tensors']=10
            record['verified_inputs']={binding[k]['path']:binding[k]['SHA256']
                for k in ('native_config','checkpoint','conditioner_checkpoint')}
            for key,field in [('native_config','arm_config'),('checkpoint','checkpoint'),
                              ('conditioner_checkpoint','conditioner_checkpoint')]:record[field]=binding[key]['path']
        record['manifest_sha256']='f'*64;bindings[arm]=binding;results[arm]=[]
        for n in range(3):
            r=deepcopy(record)
            for i,row in enumerate(r['per_asset']):
                tp=20+i%11+j*3+n;fp=20+(128-i)%11;fn=80+i%13-tp
                row.update(seed=i+1000*n,target_voxels_sha256=f'{i:064x}',tp=tp,fp=fp,fn=fn,
                    voxel_iou=tp/(tp+fp+fn),voxel_precision=tp/(tp+fp),
                    voxel_recall=tp/(tp+fn),voxel_f1=2*tp/(2*tp+fp+fn))
                a=r['native_SS_artifacts']['artifacts'][i]
                a.update(seed=row['seed'],target_voxels_SHA256=row['target_voxels_sha256'],
                    occupancy_counts=dict(tp=tp,fp=fp,fn=fn))
            tp,fp,fn=(sum(row[k] for row in r['per_asset']) for k in ('tp','fp','fn'))
            r['quick_metrics']=dict(voxel_iou=tp/(tp+fp+fn),voxel_precision=tp/(tp+fp),
                voxel_recall=tp/(tp+fn),voxel_f1=2*tp/(2*tp+fp+fn))
            results[arm].append(r)
    return results,bindings


def test_seven_bound_models_average_noise_before_source_bootstrap(tmp_path):
    results,bindings=data(tmp_path);s=module.summarize(results,bindings)
    assert s['assets']==128 and s['source_clusters']==93
    assert s['generation_noises']==3 and s['independent_training_seeds']==1
    assert s['primary_IoU_family_size']==6 and not s['automatic_model_adoption']
    assert set(s['comparisons'])=={a+'_minus_'+b for a,b in cascade.CONTRASTS}
    for arm in module.ARMS+module.BASELINES:
        expected=np.mean([[r['voxel_iou'] for r in n['per_asset']] for n in results[arm]])
        assert s['per_model_mean_asset'][arm]['voxel_iou']==pytest.approx(expected)
        assert s['per_model_native_global_occupancy_metrics'][arm]==[n['quick_metrics'] for n in results[arm]]
    assert s['per_model_mean_asset']['E_FIXED_RAW']['voxel_iou']!=np.mean([r['quick_metrics']['voxel_iou'] for r in results['E_FIXED_RAW']])


@pytest.mark.parametrize('damage',['missing-model','missing-noise','duplicate-noise','suffix-ID','source',
    'GT','count','macro','nonfinite','manifest','foreign-checkpoint','physical-R','unknown-mode','seed202'])
def test_wrong_checkpoint_counts_or_population_cannot_enter_a_paper_row(tmp_path,damage):
    results,bindings=data(tmp_path);r=results['A_FIXED_NO_R'][2]
    if damage=='missing-model':results.pop('M_FIXED_SPATIAL')
    elif damage=='missing-noise':results['M_FIXED_RAW'].pop()
    elif damage=='duplicate-noise':results['M_FIXED_RAW'][2]=deepcopy(results['M_FIXED_RAW'][1])
    elif damage=='suffix-ID':r['per_asset'][-1]['sample_id']='wrong'
    elif damage=='source':r['per_asset'][-1]['source_cluster_id']='wrong'
    elif damage=='GT':r['per_asset'][-1]['target_voxels_sha256']='e'*64
    elif damage=='count':r['per_asset'][-1]['tp']+=1
    elif damage=='macro':r['per_asset'][-1]['voxel_iou']=r['quick_metrics']['voxel_iou']
    elif damage=='nonfinite':r['per_asset'][-1]['voxel_iou']=float('nan')
    elif damage=='manifest':r['manifest_sha256']='d'*64
    elif damage=='foreign-checkpoint':r['verified_inputs'][r['checkpoint']]='c'*64
    elif damage=='physical-R':r['retargeter_present']=True
    elif damage=='unknown-mode':results['M_FIXED_RAW'][0]['correspondence_mode']='D1-cosine'
    else:bindings['M_FIXED_RAW']['training_seed']=202
    with pytest.raises(ValueError):module.summarize(results,bindings)


def joined(tmp_path):
    results,bindings=data(tmp_path);ss=module.summarize(results,bindings)
    ids,clusters,records=population();downstream=cascade.summarize(records,sample_ids=ids,clusters=clusters,draws=20)
    downstream['measured_compositions']={a:dict(ss64=b,pbr1024=pbr()) for a,b in bindings.items()}
    return ss,downstream


def test_join_preserves_three_SS_noises_and_single_cascade_scope(tmp_path):
    ss,downstream=joined(tmp_path);rows=module.bind_cascade(ss,downstream)
    assert len(rows)==7 and rows[0]['arm']=='M_FIXED_SPATIAL'
    for row in rows:
        assert row['SS_IoU']==ss['per_model_mean_asset'][row['arm']]['voxel_iou']
        assert row['CD']==downstream['roles']['all8_9_10']['means'][row['arm']]['deformation_weighted']['CD']


@pytest.mark.parametrize('damage',['different-model','different-source-weight','suffix-source','missing-arm','noise-scope'])
def test_stage_join_rejects_cross_checkpoint_or_scope_mixups(tmp_path,damage):
    ss,downstream=joined(tmp_path)
    # Each object owns its bytes; mutate one stage only.
    downstream=deepcopy(downstream)
    if damage=='different-model':downstream['measured_compositions']['M_FIXED_RAW']['ss64']['checkpoint']['SHA256']='e'*64
    elif damage=='different-source-weight':downstream['measured_compositions']['M_FIXED_RAW']['pbr1024']['checkpoint']['SHA256']='e'*64
    elif damage=='suffix-source':downstream['per_asset']['source9_10']['M_FIXED_RAW'][-1]['source_cluster_id']='wrong'
    elif damage=='missing-arm':downstream['measured_compositions'].pop('A_FIXED_NO_R')
    else:downstream['generation_noises']=3
    with pytest.raises(ValueError):module.bind_cascade(ss,downstream)
