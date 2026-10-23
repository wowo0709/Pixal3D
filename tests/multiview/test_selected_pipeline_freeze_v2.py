"""Synthetic freeze/confirmation records only; no neural training or forward."""
from copy import deepcopy
import hashlib

import pytest

from experiments.deformnext_retargeting.common_geometry_pbr_contract import MODES
from experiments.deformnext_retargeting.module_pipeline_protocol_v2 import protocol
from experiments.deformnext_retargeting.native_final_freeze import METRIC_ROLES
from experiments.deformnext_retargeting.pbr_confirmation_recipe import make_confirmation
from experiments.deformnext_retargeting.pipeline_population_contract_v2 import ARMS
from experiments.deformnext_retargeting.selected_pbr_seed_confirmation import modes_for_selected,summarize
from experiments.deformnext_retargeting.selected_pipeline_freeze_v2 import METRIC_DIRECTIONS,make_freeze
from experiments.deformnext_retargeting.selected_pipeline_joint_summary_v2 import summarize as joint_summary
from test_native_pbr_stage_summary import snapshot
from test_r_mediated_pbr_science import recipe
from test_selected_pipeline_joint_v2 import inputs as joint_inputs


def spec(path):return dict(path=path,SHA256=hashlib.sha256(path.encode()).hexdigest())


def pbr_inputs(selected='raw-topk-plus-feature'):
    modes=modes_for_selected(selected);records,configs,hashes,bases=({s:{} for s in (101,202)} for _ in range(4))
    proofs={};mv=None
    for mode in modes:
        a,proof,original=recipe(mode,world=4);a['output_dir']='/native101/'+mode
        b=make_confirmation(a,proof=proof,original_multiview=original,output_dir='/native202/'+mode,
            run_id='pbr202-'+mode,recipe_path='/recipes/'+mode+'.json',master_port=47191)
        proofs[mode]=proof
        if mv is None:mv=original
        for seed,cfg in [(101,a),(202,b)]:
            record=snapshot(mode,.2 if seed==101 else -.1)
            record['completion_proof']['terminal']['actual_session_drained']=True
            for row in record['completion_proof']['ranks']:row['training_seed']=seed
            record['training_trace_digests']={str(rank):hashlib.sha256(f'{seed}:{rank}'.encode()).hexdigest() for rank in range(4)}
            records[seed][mode],configs[seed][mode]=record,cfg
            root=cfg['output_dir']
            hashes[seed][mode]={root+'/'+name:spec(root+'/'+name)['SHA256'] for name in
                ['config.json','ckpts/denoiser_step0001900.pt','ckpts/deform_conditioner_step0001900.pt']}
            bases[seed][mode]=dict(actual_original_frozen_NN_base_tensors_exact=True,LoRA_tensors=360,frozen_tensors=1)
    initializer=spec(configs[101]['vanilla']['trainer']['args']['continuation_ckpt']['denoiser'])
    return records,configs,hashes,dict(selected=selected,proofs=proofs,original_multiview=mv,
        original_checkpoint=initializer,frozen_base_proofs=bases)


@pytest.mark.parametrize('mode',MODES)
def test_independent_PBR_records_preserve_real_seed_and_fixed4_scope(mode):
    r,c,h,kw=pbr_inputs(mode);out=summarize(r,c,h,**kw)
    assert out['training_seeds']==[101,202] and out['measured_modes']==list(modes_for_selected(mode))
    assert out['model_bindings'][202][mode]['training_seed']==202
    assert out['assets']==4 and not out['source_preservation_measured']
    assert not out['generated_geometry_measured'] and not out['automatic_model_adoption']


@pytest.mark.parametrize('damage',['same-seed','rank-missing','rank-source','updates','initial','GTpixel',
    'same-stream','within-seed-stream','frozen','optimizer','LR','initial-weight','child-drain','fake-fullcascade'])
def test_false_PBR_seed_or_scope_cannot_release_freeze(damage):
    r,c,h,kw=pbr_inputs();mode=kw['selected'];row=r[202][mode]
    if damage=='same-seed':c[202][mode]['training_seed']=101
    elif damage=='rank-missing':row['completion_proof']['ranks'].pop()
    elif damage=='rank-source':row['completion_proof']['ranks'][0]['source_tensors']=10
    elif damage=='updates':row['completion_proof']['ranks'][0]['actual_optimizer_updates']=380
    elif damage=='initial':row['validation_rows'][0]['quick_per_asset'][0]['actual_prediction_PBR_latent_SHA256']='other'
    elif damage=='GTpixel':row['GT_PNG_pixel_SHA256']['1900']['0']='f'*64
    elif damage=='same-stream':
        for mode in r[202]:r[202][mode]['training_trace_digests']=deepcopy(r[101][mode]['training_trace_digests'])
    elif damage=='within-seed-stream':row['training_trace_digests']['0']='f'*64
    elif damage=='frozen':kw['frozen_base_proofs'][202][mode]['actual_original_frozen_NN_base_tensors_exact']=False
    elif damage=='optimizer':row['completion_proof']['ranks'][0]['actual_optimizer_states']=360
    elif damage=='LR':c[202][mode]['trainer']['args']['optimizer']['args']['lr']=1e-3
    elif damage=='initial-weight':c[202][mode]['trainer']['args']['continuation_ckpt']['denoiser']='/trained101.pt'
    elif damage=='child-drain':row['completion_proof']['terminal']['actual_session_drained']=False
    else:row['validation_rows'][-1]['full_cascade']=True
    with pytest.raises(ValueError):summarize(r,c,h,**kw)


def freeze_inputs(arm='E_LEARNED_RAW'):
    bindings,cases,generations,bases,shared,official,kw=joint_inputs(arm)
    joint=joint_summary(bindings,cases,generations,bases,shared,official,**kw)
    selected=(arm,'vanilla')
    ss={b['ss64']['module_arm']:deepcopy(b['ss64']) for b in bindings.values()}
    pbr={b['pbr1024']['mode']:deepcopy(b['pbr1024']) for b in bindings.values()}
    shapes=deepcopy(kw['original_shapes'])
    for name in shapes:
        step=shapes[name]['step'];shapes[name]['checkpoint']=spec('/original/'+name+f'/ckpts/denoiser_step{step:07d}.pt')
    joint['original_shapes']=deepcopy(shapes)
    stages=dict(shapes,ss64=dict(step=4210,config=spec('/original/ss64/config.json'),
        checkpoint=spec('/original/ss64/ckpts/denoiser_step0004210.pt')),
        pbr1024=dict(step=16850,config=spec('/original/pbr1024/config.json'),
        checkpoint=spec('/original/pbr1024/ckpts/denoiser_step0016850.pt')))
    initialclusters=[r['source_cluster_id'] for r in joint['per_asset']['target8']['V1']]
    unique=set(initialclusters);needed=93-len(unique)
    clusters=initialclusters+['newsource'+str(i) for i in range(needed)]
    clusters+=(initialclusters*128)[:128-len(clusters)]
    evaluations={};means={}
    for name,binding in ss.items():
        first=deepcopy(kw['archives'][name]['record'])
        evaluations[name]=[]
        for noise in range(3):
            ev=deepcopy(first);ev.update(evaluation_mode='final128',per_asset=[])
            for i in range(128):
                ev['per_asset'].append(dict(sample_id=str(i),source_cluster_id=clusters[i],seed=i+noise*1000,
                    target_voxels_sha256=hashlib.sha256(str(i).encode()).hexdigest(),
                    voxel_iou=.4,voxel_precision=.5,voxel_recall=.6,voxel_f1=.55))
            evaluations[name].append(ev)
        means[name]=dict(voxel_iou=.4,voxel_precision=.5,voxel_recall=.6,voxel_f1=.55)
    parities={a:dict(status='PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY',model_binding=b,
        independent_model_loads=2,reference_prefix32_records_exact=True,all36_records_and_archived_occupancy_exact=True,
        actual_native_prediction_and_target_latent_arrays_exact=True,tail_generation_batch=4,optimizer_updates=0) for a,b in ss.items()}
    ssmodels={seed:{} for seed in (101,202)}
    for seed in (101,202):
        for name,b in ss.items():
            binding=deepcopy(b);binding['training_seed']=seed
            if seed==202:
                for key in ['native_config','checkpoint','conditioner_checkpoint']:
                    if binding[key]:binding[key]['path']=binding[key]['path'].replace('/native/','/seed202/native/')
            ssmodels[seed][name]=dict(model_binding=binding,
                frozen_base=dict(actual_original_frozen_NN_base_tensors_exact=True,LoRA_tensors=360),
                terminal_proofs=[dict(actual_child_absent=True) for _ in range(4)])
    ssconf=dict(status='PASS_ACTUAL_CLOSED_SELECTED_PIPELINE_SS_TWO_SEED_DEV128X3',
        actual_frozen_base_and_model_bytes_verified=True,original_checkpoint=stages['ss64']['checkpoint'],
        statistics=dict(status='PASS_SELECTED_PIPELINE_SS_TWO_TRAINING_SEEDS_PAIRED_DEV128X3',
            selected_candidate=arm,measured_models=list(ss),training_seeds=[101,202],
            actual480_training_traces_paired_within_seed_and_distinct_between_seeds=True),model_evidence=ssmodels)
    pbrmodels={seed:{} for seed in (101,202)}
    for seed in (101,202):
        for mode,b in pbr.items():
            binding=deepcopy(b);binding['training_seed']=seed
            if seed==202:
                for key in ['native_config','checkpoint','conditioner_checkpoint']:
                    if binding[key]:binding[key]['path']=binding[key]['path'].replace('/native/','/seed202/native/')
            pbrmodels[seed][mode]=binding
    pbrconf=dict(status='PASS_ACTUAL_SELECTED_PBR_TWO_SEED_FIXED4_STAGE_CONFIRMATION',selected_mode='vanilla',
        measured_modes=list(pbr),training_seeds=[101,202],actual_checkpoint_bytes_verified=True,
        actual_original_frozen_base_tensors_exact_in_all_models=True,
        actual1900_per_rank_training_traces_paired_within_seed=True,actual_training_streams_differ_between_seeds=True,
        actual_initial4_predictions_exact=True,actual_original_checkpoint=stages['pbr1024']['checkpoint'],
        assets=4,fixed_GT_shape=True,source_preservation_measured=False,generated_geometry_measured=False,model_bindings=pbrmodels)
    core=dict(status='PASS_ACTUAL_TWELVE_MODEL_PAIRED_MODULE_DEV128X3',assets=128,source_clusters=93,
        generation_noises=3,independent_training_seeds=1,held_out_test508_evaluated=False,
        per_model_mean_asset={a:means['V1'] for a in protocol()['arms']})
    follow=dict(status='PASS_PAIRED_CONSISTENT_EVIDENCE_EXPLORATORY_DEV128_THREE_NOISE_ANALYSIS',assets=128,
        source_clusters=93,generation_noises=3,independent_training_seeds=1,held_out_test508_evaluated=False,
        per_model_mean_asset={a:means['V1'] for a in ['vanilla','R_top1_only','E_FIXED_RAW','E_LEARNED_RAW']})
    dev=dict(SS_core12=core,SS_followup=follow,joint=joint,SS_two_seeds=ssconf,PBR_two_seeds=pbrconf)
    evidence={k:spec('/evidence/'+k+'.json') for k in [*dev,'selection','cache_QA']}
    selection=dict(status='SELECTED_FROM_ACTUAL_DEVELOPMENT_ONLY',selected_SS_arm=arm,selected_PBR_mode='vanilla',
        heldout_quality_read=False,mandatory_all_modules=False,rationale='Actual measured pipeline, no forced gain.',measured_SS_arms=list(ARMS))
    def rows(split,n):return [dict(sample_id=split+str(i),split=split,source_uid=split+'uid'+str(i),
        source_cluster_id=split+'cluster'+str(i%134)) for i in range(n)]
    test=rows('test',508);manifest=spec('/populations/test.jsonl')
    qa=dict(status='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA',published_manifest=manifest['path'],
        published_manifest_SHA256=manifest['SHA256'],SS_PBR_cache_ready=True,all508_paired_coverage=True,
        no_test_asset_dropped=True,no_candidate_model_forward_or_test_quality_metric=True,
        matching_head_training_or_calibration_updates=0,Omega_completed_updates=23597)
    protocol_value=dict(schema='selected-pipeline-final-evaluation-protocol-v1',metric_roles=METRIC_ROLES,
        metric_directions=METRIC_DIRECTIONS,rendered_cameras=[8,9,10],input_cameras=[8,9,10,11],
        input_states=['M1','M0','M0','M0'],SS_sampling=dict(batch_size=8,steps=12,guidance=3.,deterministic=True),
        source_reference='original material on M1 geometry',per_output_alignment=False,GT_geometry_for_scoring_only=True,
        source_preservation_views=[9,10],target_alignment_view=8,distribution_views=[8,9,10],
        metric_registry=dict(path='/protocol/metrics.json',SHA256='registry'),
        distribution_metric_registry=spec('/protocol/distribution.json'),generation_template=spec('/protocol/geometry.json'),
        source_SHA256={'/source/evaluation.py':'a'*64})
    # The synthetic joint fixture uses the literal 'registry'; use a real
    # lowercase digest in both records, as real official registry files do.
    protocol_value['metric_registry']['SHA256']='b'*64;joint['metric_registry_SHA256']='b'*64
    return dict(selected_pair=selected,selection=selection,ss_models=ss,pbr_models=pbr,original_stages=stages,
        rows=test,train_rows=rows('train',9149),validation_rows=rows('validation',508),
        seeds={r['sample_id']:i for i,r in enumerate(test)},parities=parities,development=dev,
        development_evidence=evidence,ss_evaluations=evaluations,cache_QA=qa,test_manifest=manifest,
        evaluation_protocol=protocol_value,heldout_quality_read_before_freeze=False)


@pytest.mark.parametrize('arm',['FULL','E_FIXED_RAW','E_LEARNED_RAW'])
def test_freeze_actual_joint_primary_and_strong_controls_without_assuming_quality_gain(arm):
    value=make_freeze(**freeze_inputs(arm))
    assert value['status']=='FROZEN_FROM_DEVELOPMENT_ONLY' and len(value['models'])==3
    assert value['models']['SELECTED']['ss64']['module_arm']==arm
    assert set(value['primary_metric_families'])==set(METRIC_ROLES)
    assert value['independent_pipeline_seed_confirmation_not_proven']
    assert not value['final508_SS_generated'] and not value['all_ten_metrics_measured']


@pytest.mark.parametrize('damage',['test-read','missing-study','old-four','wrong-joint','other-weights',
    'other-Shape','missing202','SS-frozen','PBR-unverified','PBR-wrong-seed','PBR-fake-joint','one-load',
    'tail','test-overlap','test-dropped','QA-model-forward','wrong-metric-role','direction','registry',
    'source-M0-ref','SS-noise','SS-missing','SS-statistics','joint-sample','forced-modules','missing-rationale'])
def test_no_incomplete_or_mixed_evidence_can_freeze_a_different_model(damage):
    k=freeze_inputs();d=k['development']
    if damage=='test-read':k['heldout_quality_read_before_freeze']=True
    elif damage=='missing-study':d.pop('SS_followup')
    elif damage=='old-four':d['SS_core12']['status']='PASS_PAIRED_NATIVE_FOUR_MODEL_DEV128_THREE_NOISE_ANALYSIS'
    elif damage=='wrong-joint':d['joint']['measured_compositions']['SELECTED']['pbr1024']=deepcopy(k['pbr_models']['top1-only'])
    elif damage=='other-weights':k['ss_models']['E_LEARNED_RAW']['checkpoint']['SHA256']='f'*64
    elif damage=='other-Shape':k['original_stages']['shape1024']['checkpoint']['SHA256']='f'*64
    elif damage=='missing202':d['SS_two_seeds']['model_evidence'].pop(202)
    elif damage=='SS-frozen':d['SS_two_seeds']['model_evidence'][202]['E_LEARNED_RAW']['frozen_base']['actual_original_frozen_NN_base_tensors_exact']=False
    elif damage=='PBR-unverified':d['PBR_two_seeds']['actual_checkpoint_bytes_verified']=False
    elif damage=='PBR-wrong-seed':d['PBR_two_seeds']['model_bindings'][202]['vanilla']['training_seed']=101
    elif damage=='PBR-fake-joint':d['PBR_two_seeds']['generated_geometry_measured']=True
    elif damage=='one-load':k['parities']['V1']['independent_model_loads']=1
    elif damage=='tail':k['parities']['V1']['tail_generation_batch']=0
    elif damage=='test-overlap':k['rows'][0]['source_cluster_id']=k['train_rows'][0]['source_cluster_id']
    elif damage=='test-dropped':k['rows'].pop()
    elif damage=='QA-model-forward':k['cache_QA']['no_candidate_model_forward_or_test_quality_metric']=False
    elif damage=='wrong-metric-role':k['evaluation_protocol']['metric_roles']=dict(source_preservation=['PSNR'])
    elif damage=='direction':k['evaluation_protocol']['metric_directions']=dict(METRIC_DIRECTIONS,CLIP='lower')
    elif damage=='registry':k['evaluation_protocol']['metric_registry']['SHA256']='f'*64
    elif damage=='source-M0-ref':k['evaluation_protocol']['source_reference']='undeformed M0'
    elif damage=='SS-noise':k['ss_evaluations']['E_LEARNED_RAW'][1]['per_asset'][0]['seed']=0
    elif damage=='SS-missing':k['ss_evaluations']['E_LEARNED_RAW'][1]['per_asset'].pop()
    elif damage=='SS-statistics':k['ss_evaluations']['E_LEARNED_RAW'][1]['per_asset'][0]['voxel_iou']=.1
    elif damage=='joint-sample':d['joint']['per_asset']['target8']['SELECTED'][0]['sample_id']='other'
    elif damage=='forced-modules':k['selection']['mandatory_all_modules']=True
    else:k['selection']['rationale']=''
    with pytest.raises(ValueError):make_freeze(**k)
