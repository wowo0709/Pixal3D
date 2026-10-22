"""Narrow ownership contract for completed paired-cascade raw duplicates."""
from pathlib import Path

from .generated_artifact_cache_retirement import RAW_NAMES
from .native_fullcascade_contract import PAIRS


def validate_case_raw(path, *, case_root, owned_root, case, generation, canonical_entry,
                      actual_SHA256, readers):
    path,root,owned=Path(path),Path(case_root),Path(owned_root)
    role=path.parent.name
    variant=case.get('SS_variant')
    if (not owned.is_absolute() or not root.is_absolute() or root==owned
            or not root.is_relative_to(owned) or root.resolve()!=root
            or path.parent.parent!=root or role not in {'geometry','prior','matched'}
            or path.name not in RAW_NAMES or path.is_symlink() or not path.is_file()
            or path.resolve()!=path or readers
            or case.get('status')!='PASS_ACTUAL_NATIVE_SS480_AND_MATCHED_PBR1900_FULLCASCADE_CASE'
            or variant not in PAIRS or case.get('PBR_mode')!=PAIRS[variant]
            or not case.get('paired_original_and_trained_texture_initial_noise_exact')
            or not case.get('paired_raw_geometry_and_native_predicted_LOD_identical')
            or not case.get('actual_prior_and_matched_three_camera_original_material_M1_shared_scores')
            or len(case.get('terminals',[]))!=7
            or any(t.get('returncode')!=0 for t in case['terminals'])
            or generation.get('sample_id')!=case.get('sample_id')
            or generation.get('SS_prediction_SHA256')!=case.get('SS_prediction_SHA256')
            or generation.get('GT_geometry_or_coords_used_in_generation') is not False):
        raise ValueError('only own completed paired native cascade raw duplicates without readers may retire')
    declared=case.get({'geometry':'geometry_raw','prior':'prior_generation','matched':'matched_generation'}[role])
    if declared!=str(path.parent):raise ValueError('raw is not the exact case-owned generation directory')
    if role=='geometry':
        if (generation.get('status')!='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                or not generation.get('GT_loaded_only_after_generation') or not generation.get('fixed_shape_models')):
            raise ValueError('actual native predicted-SS geometry generation required')
    else:
        if (generation.get('status')!='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                or generation.get('checkpoint_kind')!=('original16850-engineering'if role=='prior'else'trained-native1900')
                or generation.get('engineering_only') is not(role=='prior')
                or generation.get('PBR_arm')!=('vanilla'if role=='prior'else PAIRS[variant])
                or not generation.get('source_weights_unchanged')
                or not all(generation['geometry_array_byte_identity'].values())):
            raise ValueError('actual paired original/trained PBR texture generation required')
    if canonical_entry.get('sha256')!=actual_SHA256 or canonical_entry.get('bytes')!=path.stat().st_size:
        raise ValueError('actual local raw bytes must equal the canonical file4 archive')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=actual_SHA256,local_duplicate_only=True)
