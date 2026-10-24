"""Own local raw duplicates only after every frozen geometry consumer completes.

This returns an admission token, not a deletion. The caller must verify actual
closed files, absent child/worker handles, current readers and fresh file4
checksums before retiring a local duplicate and releasing its disk reservation.
"""
from pathlib import Path

from .generated_artifact_cache_retirement import RAW_NAMES
from .native_final_freeze import validate_spec
from .pbr_replay_artifacts import GEOMETRY_FIELDS
from .selected_final_case_integrity import CASE_STATUS
from .selected_final_population import compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


def required_consumers(alias, role, *, parent, controls):
    models = compose_models(parent,controls)
    require(alias in models and role in ('geometry','PBR'), 'actual own frozen model and raw stage required')
    model = models[alias]
    require(role != 'PBR' or model['kind'] == 'selected-trained','original zero controls have no extra PBR replay')
    names = [name for name,m in models.items() if m['geometry_group'] == model['geometry_group']]
    require(role != 'geometry' or alias == names[0],'only the canonical producer owns shared raw Shape geometry')
    return names if role == 'geometry' else [alias]


def validate_raw(path, *, case_root, owned_root, case, generation, consumers,
                 parent, controls, parent_freeze, control_freeze, canonical_entry,
                 actual_SHA256, readers):
    path,root,owned = Path(path),Path(case_root),Path(owned_root)
    require(owned.is_absolute() and root.is_absolute() and root != owned
        and root.resolve() == root and root.is_relative_to(owned)
        and path.parent.parent == root and path.name in RAW_NAMES
        and path.is_file() and not path.is_symlink() and path.resolve() == path and not readers,
        'only case-owned raw duplicates without any current or pending reader may retire')
    role = path.parent.name; alias = case['alias']; models = compose_models(parent,controls)
    names = required_consumers(alias,role,parent=parent,controls=controls)
    for value in [parent_freeze,control_freeze]:validate_spec(value)
    require(set(consumers) == set(names) and alias in consumers and exact(consumers[alias],case),
            'every declared actual frozen geometry consumer must complete; no missing or unrelated case')
    for name,consumer in consumers.items():
        require(consumer.get('status') == CASE_STATUS and consumer.get('actual_completed_native_case') is True
            and consumer.get('alias') == name and consumer.get('sample_id') == case['sample_id']
            and consumer.get('source_cluster_id') == case['source_cluster_id']
            and exact(consumer.get('frozen_model_binding'),models[name])
            and exact(consumer.get('parent_freeze'),parent_freeze)
            and exact(consumer.get('control_freeze'),control_freeze)
            and consumer.get('actual_SS_prediction_SHA256') == case['actual_SS_prediction_SHA256']
            and consumer.get('geometry_source_root') == case['geometry_source_root']
            and consumer.get('actual_native_guide_archive_SHA256') == case['actual_native_guide_archive_SHA256']
            and consumer.get('actual_shared_GT_material_pixel_identity_in_observations') is True
            and consumer.get('surface_reference_join',{}).get('status') == 'PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
            and len(consumer.get('scored_image_observations',[])) == 3
            and {r['camera_id'] for r in consumer['scored_image_observations']} == {8,9,10}
            and consumer.get('terminals') and all(t.get('returncode') == 0 and t.get('actual_session_drained') is True
                for t in consumer['terminals']),
            'actual completed same-SS/source/sample/guide cases and all render consumers required')
    require(case.get('case_output') == str(root) and str(path.parent) in case.get('raw_archive_roots',[])
        and generation.get('sample_id') == case['sample_id']
        and generation.get('SS_prediction_SHA256') == case['actual_SS_prediction_SHA256']
        and generation.get('GT_geometry_or_coords_used_in_generation') is False,
        'raw stage must belong to this actual completed native generation')
    if role == 'geometry':
        require(generation.get('status') == 'PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
            and generation.get('GT_loaded_only_after_generation') is True and generation.get('fixed_shape_models') is True
            and case['geometry_source_root'] == str(path.parent), 'original native Shape producer required')
    else:
        require(generation.get('status') == 'PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
            and generation.get('checkpoint_kind') == 'trained-native1900' and generation.get('engineering_only') is False
            and generation.get('PBR_arm') == models[alias]['PBR_mode']
            and generation.get('source_weights_unchanged') is True
            and set(generation.get('geometry_array_byte_identity',{})) == set(GEOMETRY_FIELDS)
            and all(v is True for v in generation['geometry_array_byte_identity'].values()),
            'actual own trained PBR replay must preserve every raw geometry field')
    require(canonical_entry.get('sha256') == actual_SHA256 and canonical_entry.get('bytes') == path.stat().st_size,
            'actual local bytes must equal the complete canonical file4 raw archive')
    return dict(path=str(path),sha256=actual_SHA256,bytes=path.stat().st_size,local_duplicate_only=True,
        actual_frozen_raw_consumers=list(names),alias=alias,sample_id=case['sample_id'])
