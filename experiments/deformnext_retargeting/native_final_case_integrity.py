"""Admission and post-generation integrity for all six held-out native models."""
import hashlib

import numpy as np

from .cascade_remote_results import array_identities
from .cascade_view_roles import require
from .native_final_case_plans import ALIASES
from .native_fullcascade_contract import PAIRS
from .native_population_contract import validate_capture
from .original_zero_population import MODES


def paired_seed(sample_id, purpose):
    require(purpose in {'cascade','texture'}, 'explicit generation-noise purpose required')
    return int.from_bytes(hashlib.sha256(
        ('native-six-model-final508-20261022:'+purpose+':'+sample_id).encode()).digest()[:4], 'big') % 2**31


def admit_ss(alias, model, result, *, parent, manifest_SHA256):
    require(alias in ALIASES, 'one of six prespecified native/original models required')
    ids=parent['population']['sample_ids'];clusters=parent['population']['source_clusters_by_sample']
    seeds=parent['population']['generation_seeds'];binding=model['ss64'];zero=alias in MODES
    require(len(ids)==508 and len(set(ids))==508 and len(set(clusters.values()))==134
        and result.get('status')=='PASS' and result.get('stage')=='ss64'
        and result.get('population_scope')=='final-test508'
        and result.get('manifest_sha256')==manifest_SHA256
        and result.get('frozen_native_model_binding')==binding
        and result.get('checkpoint')==binding['checkpoint']['path']
        and result.get('step')==(0 if zero else 480)
        and result.get('conditioner_checkpoint')==(None if binding['conditioner_checkpoint']is None else binding['conditioner_checkpoint']['path'])
        and result.get('input_camera_ids')==[8,9,10,11]
        and result.get('input_states')==(['M0']*4 if alias=='zero-source4'else['M1','M0','M0','M0'])
        and result.get('target_modality')=='rgb'
        and result.get('optimizer_updates_during_evaluation')==0
        and result.get('generation_batch_size')==8 and result.get('sampling_steps')==12
        and result.get('guidance')==3. and result.get('quick_panel_ids')==ids,
        'actual same-model full508 SS archive, RGB protocol and sampler required')
    artifacts=result['native_SS_artifacts']
    expected=('PASS_ACTUAL_ORIGINAL_ZERO_SS_BOUND_POPULATION_ARCHIVE'if zero else
              'PASS_ACTUAL_NATIVE_BOUND_SS_POPULATION_ARCHIVE')
    require(artifacts.get('status')==expected and artifacts.get('archived_asset_count')==508,
            'complete actual bound508 predicted SS archives required')
    validate_capture(result,artifacts['artifacts'],508)
    require([r['sample_id']for r in result['per_asset']]==ids
        and all(r['source_cluster_id']==clusters[r['sample_id']]and r['seed']==seeds[r['sample_id']]
                for r in result['per_asset']), 'actual paired sample/cluster/noise identities differ')
    if zero:
        require(result.get('deformation_finetuning_updates')==0
            and result.get('original_multiview_SS_step')==4210
            and result.get('actual_original_full_base_and_LoRA_validation',{}).get(
                'original_full_base_and_LoRA_exact_before_evaluation')is True,
            'actual complete original zero-shot weights required')
    else:
        require(result.get('native_variant')==alias
            and result.get('actual_source_tensors')==(0 if alias=='vanilla'else 10 if alias=='R_top1_only'else 19),
            'own native source0/10/19 schema required')
    return {r['sample_id']:r for r in artifacts['artifacts']}


def local_surface_join(native, generated, *, geometry_path, references):
    """Read shared GT arrays only after a successful real native generation."""
    require(generated.get('status')=='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and generated.get('sample_id')==native['row']['sample_id']
        and generated.get('GT_loaded_only_after_generation')is True
        and generated.get('GT_geometry_or_coords_used_in_generation')is False
        and generated.get('source_defined_normalization')is True
        and generated.get('per_output_alignment')is False and generated.get('actual_resolution')==1024,
        'shared GT may only join completed native predictions, after generation')
    metrics=generated['metrics']
    require(metrics['prediction_samples']==metrics['target_samples']==20000
        and metrics['threshold']==.02 and metrics['per_output_alignment']is False,
        'native20k surface protocol and F@.02 without per-output alignment required')
    reference,reference_file,expected=references.expected(native)
    with np.load(geometry_path,allow_pickle=False)as arrays:
        actual=array_identities({k:arrays[k] for k in ['target_world','target_faces','source_center','source_extent']})
    require(actual==expected['native_GT_array_identities'],
            'actual producer used different GT arrays or source-defined world normalization')
    return dict(status='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN',
        sample_id=generated['sample_id'],source_cluster_id=reference['source_cluster_id'],
        GT_reference_audit=references.spec,GT_reference=reference_file,
        target_points_SHA256=expected['target_points_SHA256'],native_GT_array_identities=actual,
        native_metrics=metrics,actual_existing_native_scores_unchanged=True,
        GT_reference_not_used_for_model_conditioning=True)


def validate_case_reports(alias, model, geometry, final, shared, surface, terminals):
    """Validate real child reports before publishing a case or retiring raw IO."""
    zero=alias in MODES
    require(alias in ALIASES and len(terminals)==(3 if zero else 4)
        and [r['phase']for r in terminals]==(['geometry','render','shared']if zero else['geometry','PBR','render','shared'])
        and all(r['returncode']==0 and isinstance(r['PID'],int)and r['birth']>0 for r in terminals),
        'every actual native child process must terminate successfully')
    sid=geometry['sample_id']
    require(geometry.get('status')=='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and geometry.get('GT_loaded_only_after_generation')is True
        and geometry.get('GT_geometry_or_coords_used_in_generation')is False
        and geometry.get('per_output_alignment')is False
        and surface.get('status')=='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
        and surface.get('sample_id')==sid and surface.get('native_metrics')==geometry['metrics'],
        'completed native geometry and its common surface-reference join required')
    if zero:
        require(final==geometry and geometry.get('original_zero_shot_full_stack')is True
            and geometry.get('deformation_finetuning_updates')==0
            and geometry.get('PBR_checkpoint')==model['pbr1024']['checkpoint']['path'],
            'original four-stage zero-shot composition required')
    else:
        require(final.get('status')=='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            and final.get('PBR_arm')==PAIRS[alias] and final.get('checkpoint_kind')=='trained-native1900'
            and final.get('engineering_only')is False
            and final.get('PBR_checkpoint_SHA256')==model['pbr1024']['checkpoint']['SHA256']
            and final.get('conditioner_checkpoint')==(None if model['pbr1024']['conditioner_checkpoint']is None else
                model['pbr1024']['conditioner_checkpoint']['path'])
            and final.get('GT_geometry_or_coords_used_in_generation')is False
            and final.get('GT_sparse_latents_or_projection_cache_used')is False
            and final.get('source_weights_unchanged')is True
            and final.get('conditioning_preserves_CPU_and_CUDA_RNG')is True
            and final.get('raw_surface_metrics_unchanged')is True
            and final.get('geometry_array_byte_identity') and all(final['geometry_array_byte_identity'].values())
            and final.get('metrics')==geometry['metrics']
            and final.get('SS_prediction_SHA256')==geometry['SS_prediction_SHA256'],
            'own native1900 texture replay must preserve actual predicted geometry and its metrics')
    expected='PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'if alias=='zero-source4'else'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE'
    # The unchanged original shared-material entry predates sample_id in its
    # report; the case's immutable phase plan binds that first reference to sid.
    require(final['sample_id']==sid and shared.get('status')==expected
        and (shared.get('sample_id',sid)==sid if alias=='zero-source4'else shared.get('sample_id')==sid)
        and {v['camera_id']for v in shared['per_view']}=={8,9,10}and len(shared['per_view'])==3,
        'one same-renderer M1 original-material GT view per target/source camera required')
    if alias=='zero-source4':
        require(shared.get('original_materials_and_UVs')is True
            and shared.get('GT_VAE_reconstruction_not_used')is True
            and shared.get('shared_renderer_lighting_camera_AO_seed')is True
            and shared.get('GT_not_used_for_model_generation')is True
            and shared.get('maximum_GT_world_vertex_nearest_difference',float('inf'))<=1e-5,
            'actual independent original-material GT renderer reference required')
    else:
        require(shared.get('same_reference_pixels_for_every_model')is True
            and all(v.get('identical_camera_matrices')is True for v in shared['per_view']),
            'all models must use the same archived GT pixels and camera matrices')
    return dict(status='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE',
        alias=alias,sample_id=sid,original_shape_models=True,actual_completed_child_phases=len(terminals),
        actual_SS_checkpoint_SHA256=model['ss64']['checkpoint']['SHA256'],
        actual_final_PBR_checkpoint_SHA256=model['pbr1024']['checkpoint']['SHA256'],
        raw_surface_metrics=geometry['metrics'],surface_reference_join=surface,
        GT_not_used_in_generation=True,optimizer_updates=0,quality_gain_proven=False,full_goal_complete=False)
