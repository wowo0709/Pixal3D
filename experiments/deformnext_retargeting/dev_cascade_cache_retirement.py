"""Select three redundant DEV-case archives after all native consumers finish.

This module never deletes. Execution must freshly compare canonical remote
bytes, inspect actual readers and file identities, and journal local removal.
Geometry predictions, native guides, scored pixels and checkpoints stay local.
"""
from hashlib import sha256
from pathlib import Path
import re

from .final_fixed_ablation_cascade import ARMS, own_contract
from .selected_pipeline_model_contract import require


RETIRED_RELATIVE_PATHS = (
    'geometry/geometry-evaluation.npz',
    'PBR/prediction.npz',
    'PBR/geometry-evaluation.npz',
)


def candidates(case_root):
    return tuple(Path(case_root)/p for p in RETIRED_RELATIVE_PATHS)


def validate_completed_case(*, case_root, owned_root, case, geometry, replay):
    root, owned = Path(case_root), Path(owned_root)
    arm = case.get('SS_arm')
    require(arm in ARMS, 'one of the five explicit final-context DEV controls required')
    require(root.is_absolute() and owned.is_absolute() and owned.resolve() == owned
        and root.resolve() == root and root.parent.name == 'cases'
        and root.parent.parent.name == arm and root.parent.parent.parent == owned,
        'one directly owned final-control DEV case required')
    sample = case.get('sample_id')
    require(isinstance(sample,str) and sample
        and root.name == sha256(sample.encode()).hexdigest()[:16],
        'exact sample-owned case token required')
    require(case.get('status') == own_contract(arm).CASE_STATUS
        and case.get('new_training') is False and case.get('TEST_forward') is False
        and case.get('same_original_material_M1_reference') is True
        and case.get('all_original_numerical_source_files_unmodified') is True
        and case.get('reused_completed_native_generation') is None
        and case.get('new_Shape_cascade_calls') == 1
        and case.get('new_PBR_replay_calls') == 1
        and case.get('geometry_root') == str(root/'geometry')
        and case.get('final_generation_root') == str(root/'PBR')
        and case.get('shared_root') == str(root/'shared'),
        'actual own completed DEV generation, replay and shared-reference scoring required')
    terminals = case.get('actual_owned_terminals',[])
    require({r.get('phase') for r in terminals} >= {'geometry','PBR','render','shared'}
        and all(r.get('returncode') == 0 and r.get('actual_session_drained') is True for r in terminals),
        'all native rendering consumers must succeed and drain first')
    require(geometry.get('status') == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and geometry.get('sample_id') == sample
        and geometry.get('SS_prediction_SHA256') == case.get('SS_prediction_SHA256')
        and geometry.get('GT_geometry_or_coords_used_in_generation') is False
        and geometry.get('GT_loaded_only_after_generation') is True
        and geometry.get('per_output_alignment') is False,
        'actual predicted geometry with scoring-only GT required')
    require(replay.get('status') == 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
        and replay.get('sample_id') == sample and replay.get('PBR_arm') == 'top1-only'
        and replay.get('checkpoint_kind') == 'trained-native1900'
        and replay.get('engineering_only') is False
        and replay.get('source_weights_unchanged') is True
        and replay.get('geometry_array_byte_identity')
        and all(v is True for v in replay['geometry_array_byte_identity'].values()),
        'actual trained common-PBR replay on identical raw geometry required')


def validate_candidate(path, *, case_root, owned_root, case, geometry, replay,
                       canonical_entry, actual_local_SHA256, readers):
    validate_completed_case(case_root=case_root,owned_root=owned_root,
        case=case,geometry=geometry,replay=replay)
    path, root = Path(path), Path(case_root)
    require(path in candidates(root) and path.is_file() and not path.is_symlink()
        and path.resolve() == path and not readers,
        'only three explicit unused local raw duplicates without readers may retire')
    require(isinstance(actual_local_SHA256,str)
        and re.fullmatch('[0-9a-f]{64}',actual_local_SHA256)
        and canonical_entry.get('sha256') == actual_local_SHA256
        and type(canonical_entry.get('bytes')) is int
        and canonical_entry['bytes'] == path.stat().st_size,
        'actual local bytes must match the closed canonical archive')
    return dict(path=str(path),bytes=path.stat().st_size,sha256=actual_local_SHA256,
        local_duplicate_only=True,canonical_file4_retained=True)
