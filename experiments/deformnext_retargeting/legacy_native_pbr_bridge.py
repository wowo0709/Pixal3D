"""Contracts for retexturing existing legacy SS geometry with common PBRs.

This is a deployment comparison across SS training protocols, not an isolated
QKV ablation. Existing geometry and its original PBR762 provenance are retained.
"""
from pathlib import Path


VARIANTS = ('retargeting', 'feature-spatial', 'feature-dq')
CASE_STATUS = 'PASS_ACTUAL_LEGACY_SS_GEOMETRY_COMMON_ORIGINAL_AND_NATIVE_PBR_PAIR'


def validate_geometry(generation, archive, original_plan, *, sample_id,
                      archive_path, expected_shapes, RGB_paths):
    g, p = generation, original_plan
    if (g.get('status') != 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
            or g.get('sample_id') != sample_id or g.get('actual_resolution') != 1024
            or g.get('GT_geometry_or_coords_used_in_generation') is not False
            or not g.get('GT_loaded_only_after_generation') or not g.get('fixed_shape_models')
            or not g.get('source_defined_normalization') or g.get('per_output_alignment') is not False
            or not g.get('original_stage_configs_and_weights_unchanged')
            or archive.get('step') != 480 or archive.get('optimizer_updates_during_evaluation') != 0
            or archive.get('input_states') != ['M1', 'M0', 'M0', 'M0']
            or archive.get('input_camera_ids') != [8, 9, 10, 11]
            or archive.get('target_modality') != 'rgb'
            or g.get('SS_source_evaluation') != archive_path or p.get('SS_evaluation') != archive_path
            or p.get('row', {}).get('sample_id') != sample_id
            or p.get('input_states') != ['M1', 'M0', 'M0', 'M0']
            or p.get('RGB_paths') != RGB_paths
            or p.get('GT_loaded_only_after_generation') is not True
            or g.get('generated_mesh') != str(Path(p['output'])/'prediction.npz')):
        raise ValueError('existing image-conditioned unaligned SS480 geometry and original producer required')
    entries = [r for r in archive['native_SS_artifacts']['artifacts'] if r['sample_id'] == sample_id]
    if len(entries) != 1 or entries[0]['prediction_voxels_SHA256'] != g['SS_prediction_SHA256']:
        raise ValueError('geometry must derive from this exact archived SS prediction')
    for stage, step in [('shape512', 4210), ('shape1024', 16850)]:
        actual, expected = p['stages'][stage], expected_shapes[stage]
        if (actual != expected['stage']
                or Path(actual['checkpoint']).name != f'denoiser_step{step:07d}.pt'
                or any(p['inputs_SHA256'].get(path) != digest
                       for path, digest in expected['SHA256'].items())):
            raise ValueError('fixed original Shape configs/checkpoint bindings differ')
    if (p['stages']['pbr1024']['checkpoint'] != g['PBR_checkpoint']
            or not g['PBR_checkpoint'].endswith('/denoiser_step0000762.pt')):
        raise ValueError('retain the original legacy PBR762 geometry-generation provenance')
    m = g['metrics']
    if (m.get('threshold') != .02 or m.get('prediction_samples') != 20000
            or m.get('target_samples') != 20000 or m.get('per_output_alignment') is not False):
        raise ValueError('original source-frame surface20k F@.02 protocol required')
    return entries[0]


def validate_pair(prior, matched, geometry, *, sample_id, original_PBR_SHA256,
                  trained_PBR_SHA256, original_prediction_SHA256):
    for kind, g in [('prior', prior), ('matched', matched)]:
        if (g.get('status') != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                or g.get('sample_id') != sample_id or g.get('PBR_arm') != 'vanilla'
                or g.get('checkpoint_kind') != ('original16850-engineering' if kind == 'prior' else 'trained-native1900')
                or g.get('engineering_only') is not (kind == 'prior')
                or g.get('conditioner_checkpoint') is not None
                or not g.get('source_weights_unchanged')
                or g.get('GT_geometry_or_coords_used_in_generation') is not False
                or not g.get('conditioning_preserves_CPU_and_CUDA_RNG')
                or not g.get('raw_surface_metrics_unchanged')
                or not g.get('geometry_array_byte_identity')
                or not all(g['geometry_array_byte_identity'].values())
                or g.get('original_prediction_SHA256') != original_prediction_SHA256
                or g.get('SS_prediction_SHA256') != geometry['SS_prediction_SHA256']
                or g.get('SS_source_evaluation') != geometry['SS_source_evaluation']
                or g.get('metrics') != geometry['metrics']
                or g.get('PBR_checkpoint_SHA256') != (original_PBR_SHA256 if kind == 'prior' else trained_PBR_SHA256)
                or g.get('PBR_replay_seed') != 20261020):
            raise ValueError('same legacy predicted geometry and explicit common vanilla PBR lineage required')
        if kind == 'matched' and not g['PBR_checkpoint'].endswith('/ckpts/denoiser_step0001900.pt'):
            raise ValueError('actual completed native1900 checkpoint required')
    for key in ['actual_initial_texture_noise_SHA256', 'PBR_replay_seed', 'sampler_class',
                'sampler_params', 'geometry_evaluation_SHA256', 'SS_prediction_SHA256', 'metrics']:
        if prior[key] != matched[key]:
            raise ValueError('within-geometry texture noise/sampler/support changed: '+key)


def validate_case(case):
    if (case.get('status') != CASE_STATUS or case.get('SS_variant') not in VARIANTS
            or case.get('PBR_mode') != 'vanilla'
            or not case.get('existing_geometry_reused_read_only')
            or case.get('new_SS_or_Shape_inferences') != 0
            or not case.get('original_Shape5124210_Shape102416850')
            or not case.get('paired_original_and_trained_texture_initial_noise_exact')
            or not case.get('paired_raw_geometry_and_native_predicted_LOD_identical')
            or not case.get('actual_prior_and_matched_three_camera_original_material_M1_shared_scores')
            or not case.get('original_generated_geometry_not_aligned_or_replaced')
            or case.get('optimizer_updates') != 0 or len(case.get('terminals', [])) != 6
            or any(t.get('returncode') != 0 or not t.get('actual_session_drained') for t in case['terminals'])):
        raise ValueError('actual completed six-child paired texture/render bridge required')
