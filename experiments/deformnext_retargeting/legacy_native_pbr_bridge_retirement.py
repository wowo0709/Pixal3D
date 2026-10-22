"""Retire only new paired texture duplicates, never existing legacy geometry."""
from pathlib import Path

from .generated_artifact_cache_retirement import RAW_NAMES
from .legacy_native_pbr_bridge import validate_case


def validate_case_raw(path, *, case_root, owned_root, case, generation,
                      canonical_entry, actual_SHA256, readers):
    validate_case(case)
    path, root, owned = Path(path), Path(case_root), Path(owned_root)
    role = path.parent.name
    original = Path(case['original_geometry_root'])
    if (not owned.is_absolute() or not root.is_absolute() or root == owned
            or not root.is_relative_to(owned) or root.resolve() != root
            or path.parent.parent != root or role not in {'prior', 'matched'}
            or path.name not in RAW_NAMES or path.is_symlink() or not path.is_file()
            or path.resolve() != path or readers
            or original == root or original.is_relative_to(owned)
            or case.get(role+'_generation') != str(path.parent)
            or generation.get('sample_id') != case.get('sample_id')
            or generation.get('SS_prediction_SHA256') != case.get('SS_prediction_SHA256')
            or generation.get('status') != 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            or generation.get('checkpoint_kind') != ('original16850-engineering' if role == 'prior' else 'trained-native1900')
            or generation.get('engineering_only') is not (role == 'prior')
            or generation.get('PBR_arm') != 'vanilla' or generation.get('conditioner_checkpoint') is not None
            or not generation.get('source_weights_unchanged')
            or generation.get('GT_geometry_or_coords_used_in_generation') is not False
            or not generation.get('geometry_array_byte_identity')
            or not all(generation['geometry_array_byte_identity'].values())):
        raise ValueError('only own completed bridge texture duplicates without consumers may retire')
    if (canonical_entry.get('sha256') != actual_SHA256
            or canonical_entry.get('bytes') != path.stat().st_size):
        raise ValueError('actual local bytes must equal the canonical remote archive')
    return dict(path=str(path), bytes=path.stat().st_size, sha256=actual_SHA256,
                local_duplicate_only=True, original_legacy_geometry_not_selected=True)
