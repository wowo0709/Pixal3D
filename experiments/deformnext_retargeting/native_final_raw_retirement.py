"""Own completed final508 raw duplicates, after their real rendering consumers."""
from pathlib import Path

from .native_final_case_plans import ALIASES
from .generated_artifact_cache_retirement import RAW_NAMES


def validate_raw(path, *, case_root, owned_root, case, generation, canonical_entry,
                 actual_SHA256, readers):
    path,root,owned=Path(path),Path(case_root),Path(owned_root)
    role=path.parent.name;alias=case.get('alias');zero=alias in {'zero-source4','zero-edited4'}
    require=(owned.is_absolute() and root.is_absolute()and root!=owned and root.is_relative_to(owned)
        and root.resolve()==root and path.parent.parent==root and role in ({'geometry'}if zero else{'geometry','PBR'})
        and path.name in RAW_NAMES and path.is_file()and not path.is_symlink()and path.resolve()==path
        and not readers and alias in ALIASES
        and case.get('status')=='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE'
        and case.get('actual_completed_native_case')is True
        and case.get('surface_reference_join',{}).get('status')=='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
        and case.get('actual_shared_GT_material_pixel_identity_in_observations')is True
        and len(case.get('scored_image_observations',[]))==3
        and {r['camera_id']for r in case['scored_image_observations']}=={8,9,10}
        and len(case.get('terminals',[]))==(3 if zero else 4)
        and all(r.get('returncode')==0 for r in case['terminals'])
        and str(path.parent)in case.get('raw_archive_roots',[])
        and generation.get('sample_id')==case.get('sample_id')
        and generation.get('SS_prediction_SHA256')==case.get('actual_SS_prediction_SHA256')
        and generation.get('GT_geometry_or_coords_used_in_generation')is False)
    if not require:raise ValueError('only own completed final508 case raw duplicates without consumers may retire')
    if role=='geometry':
        if (generation.get('status')!='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
                or generation.get('GT_loaded_only_after_generation')is not True
                or generation.get('fixed_shape_models')is not True):
            raise ValueError('actual native geometry must precede retirement')
    elif (generation.get('status')!='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            or generation.get('checkpoint_kind')!='trained-native1900'
            or generation.get('engineering_only')is not False
            or generation.get('source_weights_unchanged')is not True
            or not generation.get('geometry_array_byte_identity')
            or not all(generation['geometry_array_byte_identity'].values())):
        raise ValueError('actual native texture replay must preserve raw geometry')
    if canonical_entry.get('sha256')!=actual_SHA256 or canonical_entry.get('bytes')!=path.stat().st_size:
        raise ValueError('actual local bytes must equal the complete canonical raw archive')
    return dict(path=str(path),sha256=actual_SHA256,bytes=path.stat().st_size,local_duplicate_only=True)
