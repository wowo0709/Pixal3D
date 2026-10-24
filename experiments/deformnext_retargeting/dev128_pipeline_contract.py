"""Explicit validation-only extension of three measured editing pipelines.

Numerical generation and scoring entries remain unchanged. These checks cover
all128 records, including the96 outside the historical screening prefix.
"""
from .selected_pipeline_model_contract import require, validate_ss_evaluation, validate_ss_binding, validate_pbr_binding


ARMS = ('C_DECODER', 'E_FIXED_RAW', 'E_LEARNED_RAW')
CASE_STATUS = 'PASS_ACTUAL_PAIRED_DEV128_NATIVE_PIPELINE_CASE'


def validate_archive128(result, arm, reference, *, binding):
    require(arm in ARMS, 'one of the three already measured SS candidates required')
    validate_ss_evaluation(result, binding, arm)
    archive = result.get('native_SS_artifacts', {})
    rows, refs = result.get('per_asset', []), reference.get('per_asset', [])
    artifacts = archive.get('artifacts', [])
    require(len(rows) == len(refs) == len(artifacts) == archive.get('archived_asset_count') == 128
        and archive.get('same_forward_prediction_and_target_capture') is True
        and archive.get('predicted_SS_is_available_for_full_cascade') is True
        and archive.get('GT_SS_is_for_scoring_only') is True,
        'actual full128 captured predictions, labels for scoring only, and complete reference required')
    ids = [r['sample_id'] for r in rows]
    require(len(set(ids)) == 128 and len({r['source_cluster_id'] for r in rows}) == 93,
        'the existing unique128/93-source validation panel is required')
    for row, ref, artifact in zip(rows, refs, artifacts):
        require(all(row[k] == ref[k] for k in ('sample_id', 'source_cluster_id', 'seed', 'target_voxels_sha256'))
            and artifact['sample_id'] == row['sample_id'] and artifact['seed'] == row['seed']
            and artifact['target_voxels_SHA256'] == row['target_voxels_sha256']
            and all(artifact['occupancy_counts'][k] == row[k] for k in ('tp', 'fp', 'fn')),
            'every128 prediction must retain its own sample, source, generation noise and target/count identity')
    return ids


def validate_population(rows, ids, clusters):
    require(len(rows) == len(ids) == len(set(ids)) == 128
        and [r['sample_id'] for r in rows] == ids
        and len(set(clusters.values())) == 93 and set(clusters) == set(ids),
        'exact ordered existing128/93-source development panel required')
    for row in rows:
        require(row['split'] == 'validation' and row['source_cluster_id'] == clusters[row['sample_id']]
            and row['conditioning_camera_ids'] == [8, 9, 10, 11]
            and row['target_anchor_kind'] == 'rgb' and row['target_anchor'] == row['target_rgb_input']
            and len(row['source_images']) == 3,
            'validation edited-target RGB and three undeformed sources only; no TEST or target-depth input')


def validate_models(bindings):
    require(set(bindings) == set(ARMS), 'extend only existing R, fixed and learned compositions')
    pbr = None
    for arm in ARMS:
        value = bindings[arm]
        validate_ss_binding(value['ss64'], arm)
        validate_pbr_binding(value['pbr1024'], 'top1-only')
        require(value['ss64']['training_seed'] == value['pbr1024']['training_seed'] == 101,
            'this extension measures the same primary training seed101 models')
        if pbr is None:
            pbr = value['pbr1024']
        require(value['pbr1024'] == pbr, 'the same actually trained Top-1 R PBR must serve all three SS models')


def validate_generation(base, replay, *, sample_id, SS_prediction_SHA256):
    """Reject GT-conditioned or realigned geometry, even if its scores look good."""
    require(base.get('status') == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and base.get('sample_id') == sample_id and base.get('SS_prediction_SHA256') == SS_prediction_SHA256
        and base.get('GT_geometry_or_coords_used_in_generation') is False
        and base.get('GT_loaded_only_after_generation') is True
        and base.get('per_output_alignment') is False,
        'own predicted SS, original cascade, scoring-only GT and unchanged coordinate frame required')
    require(replay.get('status') == 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
        and replay.get('sample_id') == sample_id
        and replay.get('mode') == 'top1-only', 'own measured Top-1 R texture replay required')
