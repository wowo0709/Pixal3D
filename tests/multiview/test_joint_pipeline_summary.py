"""Joint composition evidence must not be inferred from separate stage gains."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import CASE_STATUS, GEOMETRY_FIELDS
from experiments.deformnext_retargeting.joint_pipeline_summary import compositions, summarize
from experiments.deformnext_retargeting.unified_module_summary import ALIASES, SOURCE_COUNTS
from experiments.deformnext_retargeting.unified_population_contract import SCHEMA, family
from test_common_pipeline_summary import inputs as common_inputs


def sha(value):return hashlib.sha256(str(value).encode()).hexdigest()
def spec(path):return dict(path=path,SHA256=sha(path))


def inputs(selected=('FULL','vanilla')):
    old_cases, old_g, old_r, old_official, old_kw = common_inputs()
    pairs, _ = compositions(selected)
    ids = old_kw['sample_ids']
    shapes = {a:dict(step=n,config=spec('/original/'+a+'/config.json'),
        checkpoint=spec('/original/'+a+f'/denoiser_step{n:07d}.pt'))for a,n in [('shape512',4210),('shape1024',16850)]}
    bindings, cases, generations, shared, bases, archives, pixels, gt = ({}for _ in range(8))
    official = deepcopy(old_official)
    official['per_view'] = []
    for alias,(arm,mode)in pairs.items():
        root='/science/SS/'+arm+'/native'
        ss=dict(schema=SCHEMA,module_arm=arm,variant=ALIASES.get(arm,arm),runtime_family=family(arm),
            step=480,training_seed=101,source_tensors=SOURCE_COUNTS[arm],conditioning_mode='edited-rgb',
            source_teacher_updates=0,native_task_objective='SS_flow_mse',native_config=spec(root+'/config.json'),
            checkpoint=spec(root+'/ckpts/denoiser_step0000480.pt'),
            conditioner_checkpoint=None if arm=='V1'else spec(root+'/ckpts/deform_conditioner_step0000480.pt'))
        p_root='/science/PBR/'+mode+'/native'
        pbr=dict(schema='native-cold-PBR1900-frozen-binding-v1',mode=mode,step=1900,training_seed=101,
            source_tensors=0 if mode=='vanilla'else 10 if mode=='top1-only'else 19,
            native_config=spec(p_root+'/config.json'),checkpoint=spec(p_root+'/ckpts/denoiser_step0001900.pt'),
            conditioner_checkpoint=None if mode=='vanilla'else spec(p_root+'/ckpts/deform_conditioner_step0001900.pt'))
        bindings[alias]=dict(ss64=ss,pbr1024=pbr)
        archive_spec=spec('/archives/'+arm+'/evaluation.json')
        ev=dict(status='PASS',step=480,weights='raw',native_variant=ALIASES.get(arm,arm),
            optimizer_updates_during_evaluation=0,arm_config=ss['native_config']['path'],checkpoint=ss['checkpoint']['path'],
            conditioner_checkpoint=None if arm=='V1'else ss['conditioner_checkpoint']['path'],
            verified_inputs={s['path']:s['SHA256']for s in ss.values()if isinstance(s,dict)},
            native_SS_artifacts=dict(artifacts=[dict(sample_id=s,prediction_voxels_SHA256=sha(('SS',arm,s)))for s in ids]))
        archives[arm]=dict(spec=archive_spec,record=ev)
        bases.setdefault(arm,{})
        cases[alias],generations[alias],shared[alias]={},{},{}
        for sid in ids:
            basepath='/base/'+arm+'/'+sid
            plan_spec=spec('/plans/'+arm+'/'+sid+'.json')
            bc=deepcopy(old_cases[arm][sid]);bc.update(SS_prediction_SHA256=sha(('SS',arm,sid)),new_Shape_cascade_calls=1,
                verified_inputs={archive_spec['path']:archive_spec['SHA256'],plan_spec['path']:plan_spec['SHA256']})
            bg=deepcopy(old_g[arm][sid]);bg.update(SS_prediction_SHA256=bc['SS_prediction_SHA256'],
                SS_source_evaluation=archive_spec['path'],actual_native_guide_archive_SHA256=sha(('guide',arm,sid)),
                original_stage_configs_and_weights_unchanged=True)
            plan=dict(stages={a:{k:s[k]['path']for k in ['config','checkpoint']}for a,s in shapes.items()},
                inputs_SHA256={s[k]['path']:s[k]['SHA256']for s in shapes.values()for k in ['config','checkpoint']})
            base=dict(case=bc,generation=bg,plan=plan,case_spec=spec(basepath+'/audit.json'),
                plan_spec=plan_spec,prediction_spec=spec(basepath+'/generation/prediction.npz'))
            bases[arm][sid]=base
            g=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',sample_id=sid,PBR_arm=mode,
                checkpoint_kind='trained-native1900',engineering_only=False,PBR_checkpoint=pbr['checkpoint']['path'],
                PBR_checkpoint_SHA256=pbr['checkpoint']['SHA256'],conditioner_checkpoint=None if mode=='vanilla'else pbr['conditioner_checkpoint']['path'],
                original_prediction_SHA256=base['prediction_spec']['SHA256'],SS_prediction_SHA256=bg['SS_prediction_SHA256'],
                SS_source_evaluation=bg['SS_source_evaluation'],metrics=deepcopy(bg['metrics']),
                geometry_array_byte_identity={k:True for k in GEOMETRY_FIELDS},GT_geometry_or_coords_used_in_generation=False,
                GT_sparse_latents_or_projection_cache_used=False,per_output_alignment=False,source_weights_unchanged=True,
                conditioning_preserves_CPU_and_CUDA_RNG=True,raw_surface_metrics_unchanged=True,
                actual_initial_texture_noise_SHA256=sha(('noise',arm,sid)),texture_support_policy=bg['texture_support_policy'],
                shape_decoder_recomputed=False,actual_native_guide_archive_SHA256=bg['actual_native_guide_archive_SHA256'],
                actual_native_guide_logits_order_scale_and_dead_branches_preserved=True,
                numerical_entry_AST_device_literal_only=dict(all_other_AST_statements_unchanged=True),
                PBR_replay_seed=20261020,sampler_class='native',sampler_params=dict(steps=12),
                geometry_evaluation_SHA256=sha(('surface',arm,sid)))
            case=dict(status=CASE_STATUS,mode=mode,sample_id=sid,source_cluster_id=old_kw['clusters'][sid],
                common_case=basepath,common_prediction_SHA256=g['original_prediction_SHA256'],
                native_guide_SHA256=g['actual_native_guide_archive_SHA256'],PBR_checkpoint_SHA256=pbr['checkpoint']['SHA256'],
                actual_four_rank_stage_completion_checked=True,new_SS_forwards=0,new_Shape_forwards=0,new_PBR_sampling_calls=1,
                actual_owned_terminals=[dict(returncode=0,actual_session_drained=True)for _ in range(3)],
                verified_inputs={s['path']:s['SHA256']for s in pbr.values()if isinstance(s,dict)})
            cases[alias][sid],generations[alias][sid],shared[alias][sid]=case,g,deepcopy(old_r[arm][sid])
            for cid in [8,9,10]:
                pixels[alias,sid,cid]=old_kw['pixel_SHA256'][arm,sid,cid]
                gt[alias,sid,cid]=old_kw['pixel_GT'][arm,sid,cid]
                row=deepcopy(next(r for r in old_official['per_view']if(r['arm'],r['sample_id'],r['camera_id'])==(arm,sid,cid)))
                row['arm']=alias;official['per_view'].append(row)
    kw=dict(selected=selected,sample_ids=ids,clusters=old_kw['clusters'],archives=archives,original_shapes=shapes,
        pixel_SHA256=pixels,pixel_GT=gt,metric_registry_SHA256='registry',draws=100)
    return bindings,cases,generations,bases,shared,official,kw


@pytest.mark.parametrize('selected,alias,count',[(None,None,2),(('V1','vanilla'),'V1',2),
    (('C_DECODER','top1-only'),'RETARGET',2),(('FULL','vanilla'),'SELECTED',3),
    (('V1','top1-only'),'SELECTED',3)])
def test_actual_joint_models_are_deduplicated_and_no_success_is_assumed(selected,alias,count):
    b,c,g,base,r,p,kw=inputs(selected)
    out=summarize(b,c,g,base,r,p,**kw)
    assert out['actual_official_views']==96*count and out['prospective_selected_alias']==alias
    assert out['actual_joint_SS_and_PBR_compositions_measured']
    assert not out['model_adopted'] and not out['full_goal_complete']
    assert len(out['roles']['target8']['contrasts'])==(1 if count==2 else 3)


@pytest.mark.parametrize('damage',['original-texture','SS-bytes','SS-archive','SS-prediction','Shape-weights',
    'PBR-bytes','source-bytes','source-absent','GT-geometry','guide','recomputed-guide','seed',
    'sampler','pixel-GT','pixel-score','camera','child-live','base-child-live','CPU-forward','missing-view'])
def test_stage_evidence_or_mismatched_execution_cannot_be_called_a_joint_pipeline(damage):
    b,c,g,base,r,p,kw=inputs()
    row=g['RETARGET']['0'];case=c['RETARGET']['0']
    if damage=='original-texture':row['status']='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
    elif damage=='SS-bytes':b['RETARGET']['ss64']['checkpoint']['SHA256']=sha('other')
    elif damage=='SS-archive':kw['archives']['C_DECODER']['record']['checkpoint']='/other.pt'
    elif damage=='SS-prediction':base['C_DECODER']['0']['generation']['SS_prediction_SHA256']=sha('other')
    elif damage=='Shape-weights':base['C_DECODER']['0']['plan']['inputs_SHA256'][kw['original_shapes']['shape512']['checkpoint']['path']]=sha('other')
    elif damage=='PBR-bytes':case['PBR_checkpoint_SHA256']=sha('other')
    elif damage=='source-bytes':case['verified_inputs'][b['RETARGET']['pbr1024']['conditioner_checkpoint']['path']]=sha('other')
    elif damage=='source-absent':b['RETARGET']['pbr1024']['conditioner_checkpoint']=None
    elif damage=='GT-geometry':row['GT_geometry_or_coords_used_in_generation']=True
    elif damage=='guide':row['actual_native_guide_archive_SHA256']=sha('other')
    elif damage=='recomputed-guide':row['shape_decoder_recomputed']=True
    elif damage=='seed':row['PBR_replay_seed']=12
    elif damage=='sampler':row['sampler_params']['steps']=8
    elif damage=='pixel-GT':kw['pixel_GT']['RETARGET','0',8]=('wrong','GT')
    elif damage=='pixel-score':p['per_view'][0]['shared_npz_SHA256']='wrong'
    elif damage=='camera':r['RETARGET']['0']['per_view'][0]['identical_camera_matrices']=False
    elif damage=='child-live':case['actual_owned_terminals'][0]['actual_session_drained']=False
    elif damage=='base-child-live':base['C_DECODER']['0']['case']['actual_owned_terminals'][0]['actual_session_drained']=False
    elif damage=='CPU-forward':p['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    else:p['per_view'].pop()
    with pytest.raises(ValueError):summarize(b,c,g,base,r,p,**kw)


def test_same_support_requires_same_actual_texture_noise_but_different_support_does_not():
    b,c,g,base,r,p,kw=inputs(('V1','top1-only'))
    summarize(b,c,g,base,r,p,**kw)
    g['SELECTED']['0']['actual_initial_texture_noise_SHA256']=sha('changed')
    with pytest.raises(ValueError,match='identical geometric support'):
        summarize(b,c,g,base,r,p,**kw)
