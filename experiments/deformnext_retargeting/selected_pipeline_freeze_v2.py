"""Freeze the actually measured pipeline and strong controls before held-out quality.

No architecture is invented and no gain is assumed. Independent PBR evidence
retains its fixed-four/GT-shape scope; the joint DEV pipeline is a separate gate.
"""
from copy import deepcopy
import json
import math

from .evaluate_bound_population import validate_population
from .module_pipeline_protocol_v2 import protocol
from .native_final_freeze import METRIC_ROLES,validate_spec
from .pipeline_confirmation_recipe_v2 import models_for_selected
from .pipeline_population_contract_v2 import ARMS
from .selected_pbr_seed_confirmation import STATUS as PBR_CONFIRMATION,modes_for_selected
from .selected_pipeline_joint_summary_v2 import compositions
from .selected_pipeline_model_contract import require,validate_pbr_binding,validate_ss_binding,validate_ss_evaluation


STATUS='FROZEN_FROM_DEVELOPMENT_ONLY'
METRIC_DIRECTIONS=dict(LPIPS='lower',DreamSim='lower',CLIP='higher',FID='lower',KID='lower',
    PSNR='higher',SSIM='higher',M_IoU='higher',CD='lower',FScore='higher')


def validate_evaluation_protocol(value):
    require(value.get('schema')=='selected-pipeline-final-evaluation-protocol-v1'
        and exact(value.get('metric_roles'),METRIC_ROLES)
        and exact(value.get('metric_directions'),METRIC_DIRECTIONS)
        and value.get('rendered_cameras')==[8,9,10]
        and value.get('input_cameras')==[8,9,10,11]
        and value.get('input_states')==['M1','M0','M0','M0']
        and exact(value.get('SS_sampling'),dict(batch_size=8,steps=12,guidance=3.,deterministic=True))
        and value.get('source_reference')=='original material on M1 geometry'
        and value.get('per_output_alignment') is False
        and value.get('GT_geometry_for_scoring_only') is True
        and value.get('source_preservation_views')==[9,10]
        and value.get('target_alignment_view')==8
        and value.get('distribution_views')==[8,9,10],
        'explicit original RGB/sampler and all ten metric roles/directions/views required')
    for key in ['metric_registry','distribution_metric_registry','generation_template']:validate_spec(value[key])
    sources=value.get('source_SHA256',{})
    require(bool(sources),'actual final generation/render/metric source hashes required')
    for path,digest in sources.items():validate_spec(dict(path=path,SHA256=digest))


def exact(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def at_seed(value,seed):
    require(set(value) in ({101,202},{'101','202'}),'both actual seed101/202 keys required')
    return value[str(seed)] if str(seed) in value else value[seed]


def _parity(parity,binding):
    require(parity.get('status')=='PASS_ACTUAL_NATIVE_BOUND_VALIDATION36_REPEATABILITY'
        and exact(parity.get('model_binding'),binding) and parity.get('independent_model_loads')==2
        and parity.get('reference_prefix32_records_exact') is True
        and parity.get('all36_records_and_archived_occupancy_exact') is True
        and parity.get('actual_native_prediction_and_target_latent_arrays_exact') is True
        and parity.get('tail_generation_batch')==4 and parity.get('optimizer_updates')==0,
        'actual own saved-model validation36 two-load/tail4 parity required')


def make_freeze(*,selected_pair,selection,ss_models,pbr_models,original_stages,rows,
                train_rows,validation_rows,seeds,parities,development,development_evidence,
                ss_evaluations,cache_QA,test_manifest,evaluation_protocol,heldout_quality_read_before_freeze):
    require(heldout_quality_read_before_freeze is False,'held-out candidate quality must not precede model freeze')
    require(isinstance(selected_pair,tuple) and len(selected_pair)==2,'one explicit measured SS/PBR pair required')
    pairs,selected_alias=compositions(selected_pair)
    validate_evaluation_protocol(evaluation_protocol)
    selected_ss,selected_pbr=selected_pair
    ss_arms=models_for_selected(selected_ss);pbr_modes=modes_for_selected(selected_pbr)
    require(set(ss_models)==set(parities)==set(ss_evaluations)==set(ss_arms)
        and set(pbr_models)==set(pbr_modes),'actual selected model and deduplicated strong controls required')
    require(selection.get('status')=='SELECTED_FROM_ACTUAL_DEVELOPMENT_ONLY'
        and selection.get('selected_SS_arm')==selected_ss and selection.get('selected_PBR_mode')==selected_pbr
        and selection.get('heldout_quality_read') is False
        and selection.get('mandatory_all_modules') is False
        and isinstance(selection.get('rationale'),str) and bool(selection['rationale'].strip())
        and set(selection.get('measured_SS_arms',[]))==set(ARMS),
        'documented actual DEV selection; no forced all-modules or unmeasured composition required')
    panel=[r['sample_id'] for r in rows]
    validate_population(rows,scope='final-test508',panel=panel,seeds=seeds,
        train_rows=train_rows,validation_rows=validation_rows)
    clusters={r['sample_id']:r['source_cluster_id'] for r in rows}
    require(len(set(clusters.values()))==134,'actual complete508/134-source population required')
    validate_spec(test_manifest)
    require(cache_QA.get('status')=='PASS_ACTUAL_TEST508_NATIVE_CACHE_AND_LOADER_QA'
        and cache_QA.get('published_manifest')==test_manifest['path']
        and cache_QA.get('published_manifest_SHA256')==test_manifest['SHA256']
        and all(cache_QA.get(k) is True for k in ['SS_PBR_cache_ready','all508_paired_coverage',
            'no_test_asset_dropped','no_candidate_model_forward_or_test_quality_metric'])
        and cache_QA.get('matching_head_training_or_calibration_updates')==0
        and cache_QA.get('Omega_completed_updates')==23597,
        'actual complete test cache QA without candidate quality or test matching updates required')
    required={'SS_core12','SS_followup','joint','SS_two_seeds','PBR_two_seeds'}
    require(set(development)==required and set(development_evidence)==required|{'selection','cache_QA'},
        'actual module, joint and independent training evidence plus explicit pinned selection/QA required')
    for spec in development_evidence.values():validate_spec(spec)
    require(len({s['path'] for s in development_evidence.values()})==len(development_evidence),
        'different actual development records cannot share one placeholder path')
    core,follow=development['SS_core12'],development['SS_followup']
    require(core.get('status')=='PASS_ACTUAL_TWELVE_MODEL_PAIRED_MODULE_DEV128X3'
        and set(core.get('per_model_mean_asset',{}))==set(protocol()['arms'])
        and follow.get('status')=='PASS_PAIRED_CONSISTENT_EVIDENCE_EXPLORATORY_DEV128_THREE_NOISE_ANALYSIS'
        and set(follow.get('per_model_mean_asset',{}))=={'vanilla','R_top1_only','E_FIXED_RAW','E_LEARNED_RAW'},
        'actual complete original12 and separate exploratory follow-up results required')
    for study in [core,follow]:
        require((study.get('assets'),study.get('source_clusters'),study.get('generation_noises'))==(128,93,3)
            and study.get('independent_training_seeds')==1 and study.get('held_out_test508_evaluated') is False,
            'actual paired DEV128/93-source/three-noise scope required')
    joint=development['joint']
    expected={alias:dict(ss64=ss_models[a],pbr1024=pbr_models[m]) for alias,(a,m) in pairs.items()}
    require(joint.get('status')=='PASS_ACTUAL_SELECTED_DEV_PIPELINE_COMPOSITIONS32_EIGHT_METRICS_V2'
        and exact(joint.get('measured_compositions'),expected)
        and joint.get('actual_official_views')==32*3*len(pairs)
        and joint.get('actual_joint_SS_and_PBR_compositions_measured') is True
        and joint.get('stage_isolation_not_substituted_for_joint_measurement') is True
        and joint.get('original_cascade_texture_not_mixed_with_seed_reset_replays') is True
        and joint.get('metric_registry_SHA256')==evaluation_protocol['metric_registry']['SHA256']
        and joint.get('PBR_replay_seed')==20261020,
        'actual same selected SS+PBR joint pipeline and strong baselines must be measured before freeze')
    require(set(original_stages)=={'ss64','shape512','shape1024','pbr1024'},
        'all four user-specified original multiview stages/configs required including zero-shot SS')
    for stage,step in [('ss64',4210),('shape512',4210),('shape1024',16850),('pbr1024',16850)]:
        spec=original_stages[stage]
        require(type(spec.get('step')) is int and spec['step']==step,'original multiview stage step differs')
        for key in ['config','checkpoint']:validate_spec(spec[key])
        require(spec['checkpoint']['path'].endswith(f'/{stage}/ckpts/denoiser_step{step:07d}.pt'),
            'actual original stage checkpoint family and step required')
        if stage in ('shape512','shape1024'):
            require(exact(joint['original_shapes'][stage],spec),'actual joint Shape stages differ from frozen stages')
    ssconf=development['SS_two_seeds'];statistics=ssconf.get('statistics',{})
    require(ssconf.get('status')=='PASS_ACTUAL_CLOSED_SELECTED_PIPELINE_SS_TWO_SEED_DEV128X3'
        and ssconf.get('actual_frozen_base_and_model_bytes_verified') is True
        and ssconf.get('original_checkpoint',{}).get('SHA256')==original_stages['ss64']['checkpoint']['SHA256']
        and statistics.get('status')=='PASS_SELECTED_PIPELINE_SS_TWO_TRAINING_SEEDS_PAIRED_DEV128X3'
        and statistics.get('selected_candidate')==selected_ss
        and statistics.get('measured_models')==list(ss_arms) and statistics.get('training_seeds')==[101,202]
        and statistics.get('actual480_training_traces_paired_within_seed_and_distinct_between_seeds') is True,
        'actual chosen SS two-training-seed model/byte/frozen-base/complete480 trace evidence required')
    pbrconf=development['PBR_two_seeds']
    require(pbrconf.get('status')==PBR_CONFIRMATION and pbrconf.get('selected_mode')==selected_pbr
        and pbrconf.get('measured_modes')==list(pbr_modes) and pbrconf.get('training_seeds')==[101,202]
        and pbrconf.get('actual_checkpoint_bytes_verified') is True
        and pbrconf.get('actual_original_frozen_base_tensors_exact_in_all_models') is True
        and pbrconf.get('actual1900_per_rank_training_traces_paired_within_seed') is True
        and pbrconf.get('actual_training_streams_differ_between_seeds') is True
        and pbrconf.get('actual_initial4_predictions_exact') is True
        and pbrconf.get('actual_original_checkpoint',{}).get('SHA256')==original_stages['pbr1024']['checkpoint']['SHA256']
        and pbrconf.get('assets')==4 and pbrconf.get('fixed_GT_shape') is True
        and pbrconf.get('source_preservation_measured') is False
        and pbrconf.get('generated_geometry_measured') is False,
        'actual chosen PBR independent-stage confirmation required; do not relabel fixed4 as joint evidence')
    for arm in ss_arms:
        validate_ss_binding(ss_models[arm],arm);_parity(parities[arm],ss_models[arm])
        measured=ss_evaluations[arm]
        require(len(measured)==3,'all three actual SS generation-noise evaluations required')
        for noise,value in enumerate(measured):
            validate_ss_evaluation(value,ss_models[arm],arm)
            row=value.get('per_asset',[]);ref=ss_evaluations['V1'][noise]['per_asset']
            require(value.get('evaluation_mode')=='final128' and len(row)==128
                and len({r['sample_id'] for r in row})==128
                and len({r['source_cluster_id'] for r in row})==93
                and [(r['sample_id'],r['source_cluster_id'],r['seed'],r['target_voxels_sha256']) for r in row]
                    ==[(r['sample_id'],r['source_cluster_id'],r['seed'],r['target_voxels_sha256']) for r in ref],
                'actual paired128 assets/93 sources/GT/noise at every SS evaluation required')
        require(all(len({measured[n]['per_asset'][i]['seed'] for n in range(3)})==3 for i in range(128)),
            'three distinct actual SS generation seeds per asset required')
        reference=follow if arm in ('E_FIXED_RAW','E_LEARNED_RAW') else core
        name=arm
        for metric in ['voxel_iou','voxel_precision','voxel_recall','voxel_f1']:
            values=[float(row[metric]) for ev in measured for row in ev['per_asset']]
            require(all(math.isfinite(v) and 0<=v<=1 for v in values)
                and abs(sum(values)/len(values)-reference['per_model_mean_asset'][name][metric])<1e-12,
                'actual chosen SS measurements differ from reported development statistics')
        for seed in (101,202):
            proof=at_seed(ssconf['model_evidence'],seed)[arm]
            validate_ss_binding(proof['model_binding'],arm,seed=seed)
            require(proof['frozen_base']['actual_original_frozen_NN_base_tensors_exact'] is True
                and proof['frozen_base']['LoRA_tensors']==360
                and len(proof['terminal_proofs'])==4 and all(t['actual_child_absent'] is True for t in proof['terminal_proofs']),
                'actual selected SS frozen tensor bytes and all completed actual children required')
            if seed==101:require(exact(proof['model_binding'],ss_models[arm]),'SS confirmation uses another primary checkpoint')
    for mode in pbr_modes:
        validate_pbr_binding(pbr_models[mode],mode)
        for seed in (101,202):
            own=at_seed(pbrconf['model_bindings'],seed)[mode];validate_pbr_binding(own,mode,seed=seed)
            if seed==101:require(exact(own,pbr_models[mode]),'PBR confirmation uses another primary checkpoint')
    models={alias:dict(ss64=ss_models[a],shape512=original_stages['shape512'],
        shape1024=original_stages['shape1024'],pbr1024=pbr_models[m]) for alias,(a,m) in pairs.items()}
    for alias,(arm,_) in pairs.items():
        expected_ids=[(r['sample_id'],r['source_cluster_id']) for r in ss_evaluations[arm][0]['per_asset'][:32]]
        require([(r['sample_id'],r['source_cluster_id']) for r in joint['per_asset']['target8'][alias]]==expected_ids,
            'actual joint DEV32 must use the SS reference128 prefix32')
    return deepcopy(dict(status=STATUS,schema='selected-physical-pipeline-development-freeze-v2',
        selected_pair=list(selected_pair),selected_alias=selected_alias,models=models,selection=selection,
        heldout_quality_read_before_freeze=False,development_evidence=list(development_evidence.values()),
        development_evidence_roles=development_evidence,original_stages=original_stages,
        evaluation_protocol=evaluation_protocol,
        population=dict(sample_ids=panel,source_clusters_by_sample=clusters,rendered_cameras=[8,9,10],
            generation_seeds=seeds,manifest=test_manifest),primary_metric_families=METRIC_ROLES,
        input_cameras=[8,9,10,11],input_states=['M1','M0','M0','M0'],target_modality='rgb',
        original_zero_controls=['source4-M0-M0-M0-M0','edited4-M1-M0-M0-M0'],
        independent_PBR_confirmation_scope='fixed4 target-camera GT-shape stage trend only',
        independent_pipeline_seed_confirmation_not_proven=True,
        no_module_effect_or_gain_inferred_from_freeze=True,automatic_model_adoption=False,
        final508_SS_generated=False,full_cascade_generated=False,all_ten_metrics_measured=False,full_goal_complete=False))
