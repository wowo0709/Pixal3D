"""Admit actual final cases for the frozen physical SS/PBR compositions.

The final model can have any measured physical family. No alias is interpreted
as an architecture, and a shared geometry needs its real completed owner case.
"""
from .archived_native_guides import POLICY
from .native_final_case_integrity import paired_seed
from .pbr_replay_artifacts import GEOMETRY_FIELDS
from .selected_final_population import admit_ss, compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


CASE_STATUS = 'PASS_ACTUAL_SELECTED_PHYSICAL_TEST508_CASE'


def _digest(value):
    return isinstance(value,str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _terminals(terminals, names):
    require([t['phase'] for t in terminals] == names
        and all(type(t.get('PID')) is int and t['PID'] > 0
            and isinstance(t.get('birth'),(int,float)) and t['birth'] > 0
            and type(t.get('returncode')) is int and t['returncode'] == 0 for t in terminals)
        and len({(t['PID'],t['birth']) for t in terminals}) == len(names),
        'each required actual native child must terminate successfully with its own process identity')


def validate_geometry_reuse(case, model, owner, *, models, prediction_SHA256, guide_SHA256):
    ref = case['geometry_reference']; alias = ref['alias']
    require(alias in models and alias != case['alias'] and models[alias]['kind'] == 'selected-trained'
        and exact(models[alias]['ss64'],model['ss64'])
        and exact(models[alias]['geometry_stages'],model['geometry_stages'])
        and owner.get('status') == CASE_STATUS and owner.get('actual_completed_native_case') is True
        and owner.get('alias') == alias and owner.get('sample_id') == case['sample_id']
        and owner.get('source_cluster_id') == case['source_cluster_id']
        and exact(owner.get('frozen_model_binding'),models[alias])
        and exact(owner.get('source_SS_evaluation'),case['SS_evaluation'])
        and owner.get('geometry_source_root') == ref['generation_root'] == case['geometry_source_root']
        and owner.get('geometry_reference') is None
        and owner.get('geometry_group') == model['geometry_group']
        and owner.get('actual_SS_prediction_SHA256') == ref['SS_prediction_voxels_SHA256'] == prediction_SHA256
        and owner.get('actual_native_guide_archive_SHA256') == guide_SHA256,
        'reuse only an actual completed canonical owner of this exact frozen SS, original Shape and guide archive')
    _terminals(owner['terminals'],['geometry','PBR','render','shared'])


def validate_case_reports(case, *, parent, controls, SS_evaluation, geometry, final,
                          render, shared, surface, terminals, geometry_reference_audit=None):
    models = compose_models(parent, controls); alias = case['alias']; model = models[alias]
    require(exact(case['model'],model) and case.get('schema') == 'selected-physical-pipeline-TEST508-sample-plan-v1',
            'actual frozen own physical composition and sample plan required')
    artifacts = admit_ss(alias,model,SS_evaluation,parent=parent,controls=controls,
        manifest_SHA256=parent['population']['manifest']['SHA256'])
    sid = case['sample_id']; zero = model['kind'] == 'original-zero'; reuse = case['geometry_reference'] is not None
    require(sid in artifacts and case['source_cluster_id'] == parent['population']['source_clusters_by_sample'][sid],
            'one actual prescribed held-out sample and source identity required')
    names = ([] if reuse else ['geometry'])+([] if zero else ['PBR'])+['render','shared']
    _terminals(terminals,names)
    native = case['phases'][0]['plan']; guide = geometry.get('actual_native_guide_archive_SHA256')
    require(geometry.get('status') == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and geometry.get('sample_id') == sid
        and geometry.get('SS_source_evaluation') == case['SS_evaluation']['path']
        and geometry.get('SS_prediction_SHA256') == artifacts[sid]['prediction_voxels_SHA256']
        and geometry.get('GT_loaded_only_after_generation') is True
        and geometry.get('GT_geometry_or_coords_used_in_generation') is False
        and geometry.get('source_defined_normalization') is True
        and geometry.get('per_output_alignment') is False and geometry.get('actual_resolution') == 1024
        and geometry.get('fixed_shape_models') is True
        and geometry.get('original_stage_configs_and_weights_unchanged') is True
        and geometry.get('PBR_checkpoint') == model['geometry_stages']['pbr1024']['checkpoint']['path']
        and geometry.get('texture_support_policy') == POLICY and _digest(guide)
        and geometry.get('actual_native_guide_logits_order_scale_and_dead_branches_preserved') is True
        and geometry.get('shape_decoder_recomputed') is True
        and len(geometry.get('actual_native_guide_events',[])) == 1
        and geometry['actual_native_guide_events'][0].get('actual_guide_count') == 4
        and native['seed'] == paired_seed(sid,'cascade') and native['surface_seed'] == 101,
        'actual own predicted SS through the original Shape models and captured native hierarchy required')
    metrics = geometry['metrics']
    require(metrics.get('prediction_samples') == metrics.get('target_samples') == 20000
        and metrics.get('threshold') == .02 and metrics.get('per_output_alignment') is False
        and surface.get('status') == 'PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
        and surface.get('sample_id') == sid and surface.get('source_cluster_id') == case['source_cluster_id']
        and exact(surface.get('native_metrics'),metrics),
        'post-generation common GT surface join with the unchanged native20k/F@.02 scores required')
    require((geometry_reference_audit is None) == (not reuse), 'one actual geometry owner proof required only for reuse')
    if reuse:
        require(not zero,'original RGB input controls cannot share generated geometry')
        validate_geometry_reuse(case,model,geometry_reference_audit,models=models,
            prediction_SHA256=artifacts[sid]['prediction_voxels_SHA256'],guide_SHA256=guide)
    if zero:
        require(exact(final,geometry) and geometry.get('original_zero_shot_full_stack') is True
            and geometry.get('deformation_finetuning_updates') == 0
            and geometry.get('original_multiview_steps') == dict(ss64=4210,shape512=4210,shape1024=16850,pbr1024=16850),
            'original four-stage base+LoRA zero-shot stack required')
    else:
        pbr = model['pbr1024']
        require(final.get('status') == 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            and final.get('sample_id') == sid and final.get('PBR_arm') == model['PBR_mode']
            and final.get('checkpoint_kind') == 'trained-native1900' and final.get('engineering_only') is False
            and final.get('PBR_checkpoint') == pbr['checkpoint']['path']
            and final.get('PBR_checkpoint_SHA256') == pbr['checkpoint']['SHA256']
            and final.get('conditioner_checkpoint') == (None if pbr['conditioner_checkpoint'] is None else pbr['conditioner_checkpoint']['path'])
            and final.get('PBR_replay_seed') == paired_seed(sid,'texture')
            and _digest(final.get('actual_initial_texture_noise_SHA256'))
            and final.get('GT_geometry_or_coords_used_in_generation') is False
            and final.get('GT_sparse_latents_or_projection_cache_used') is False
            and final.get('GT_evaluation_artifact_copied_only_after_prediction_saved') is True
            and final.get('source_weights_unchanged') is True
            and final.get('conditioning_preserves_CPU_and_CUDA_RNG') is True
            and final.get('optimizer_updates') == 0 and final.get('raw_surface_metrics_unchanged') is True
            and set(final.get('geometry_array_byte_identity',{})) == set(GEOMETRY_FIELDS)
            and all(v is True for v in final['geometry_array_byte_identity'].values())
            and final.get('SS_source_evaluation') == geometry['SS_source_evaluation']
            and final.get('SS_prediction_SHA256') == geometry['SS_prediction_SHA256']
            and final.get('generated_shape_source') == case['geometry_source_root']
            and exact(final.get('metrics'),metrics)
            and final.get('texture_support_policy') == POLICY
            and final.get('actual_native_guide_archive_SHA256') == guide
            and final.get('actual_native_guide_logits_order_scale_and_dead_branches_preserved') is True
            and final.get('shape_decoder_recomputed') is False,
            'own actual native1900 PBR replay must preserve the exact generated geometry, sampler noise and captured guides')
    require(render.get('status') == 'PASS_ACTUAL_CASCADE_MULTIVIEW_RGB_ALPHA_AND_CAMERA_AUDIT'
        and render.get('sample_id') == sid and len(render.get('per_view',[])) == 3
        and {r['camera_id'] for r in render['per_view']} == {8,9,10}
        and all(r.get('GT_geometry_RGBA_containment',0) >= .99 for r in render['per_view'])
        and render.get('prediction_geometry_postprocess',{}).get('same_budget_for_all_predictions') is True
        and render['prediction_geometry_postprocess'].get('render_faces',1000001) <= 1000000,
        'same calibrated three-camera native rendering and geometry budget required')
    first = alias == 'zero-source4'
    require(shared.get('status') == ('PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE' if first else 'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE')
        and (shared.get('sample_id',sid) == sid if first else shared.get('sample_id') == sid)
        and len(shared.get('per_view',[])) == 3 and {v['camera_id'] for v in shared['per_view']} == {8,9,10},
        'same archived M1 original-material reference in each target/source camera required')
    if first:
        require(shared.get('original_materials_and_UVs') is True and shared.get('GT_VAE_reconstruction_not_used') is True
            and shared.get('shared_renderer_lighting_camera_AO_seed') is True
            and shared.get('GT_not_used_for_model_generation') is True
            and shared.get('maximum_GT_world_vertex_nearest_difference',float('inf')) <= 1e-5,
            'original materials and UVs on the independently verified edited geometry required')
    else:
        require(shared.get('same_reference_pixels_for_every_model') is True
            and all(v.get('identical_camera_matrices') is True for v in shared['per_view']),
            'identical archived reference pixels and camera matrices required')
    return dict(status=CASE_STATUS,alias=alias,sample_id=sid,source_cluster_id=case['source_cluster_id'],
        frozen_model_binding=model,source_SS_evaluation=case['SS_evaluation'],
        actual_SS_prediction_SHA256=geometry['SS_prediction_SHA256'],geometry_source_root=case['geometry_source_root'],
        geometry_group=model['geometry_group'],geometry_reference=case['geometry_reference'],
        actual_native_guide_archive_SHA256=guide,surface_reference_join=surface,raw_surface_metrics=metrics,
        terminals=terminals,actual_completed_native_case=True,actual_completed_child_phases=len(terminals),
        GT_not_used_in_generation=True,optimizer_updates=0,quality_gain_proven=False,full_goal_complete=False)
