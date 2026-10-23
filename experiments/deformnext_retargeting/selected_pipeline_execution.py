"""Bind development execution to the selected physical SS/PBR composition.

These are metadata checks around the original cascade/replay/metric entries.
No inference implementation, model selection or held-out release lives here.
"""
from pathlib import Path

from .archived_native_guides import POLICY
from .common_geometry_pbr_contract import digest
from .native_final_freeze import validate_spec
from .selected_pipeline_cascade_contract import CASE_STATUS, valid_case_status, validate_archive
from .selected_pipeline_joint_summary_v2 import compositions, validate_binding
from .selected_pipeline_model_contract import require


INDEX_STATUS='PREPARED_SELECTED_PHYSICAL_SS_PBR_DEV32_EXECUTION_V1'


def validate_index(index):
    require(index.get('status')==INDEX_STATUS,'explicit selected physical development execution index required')
    selected=index.get('selected_pair')
    require(selected is None or isinstance(selected,list) and len(selected)==2,
        'explicit selected SS/PBR pair or baseline-only index required')
    pairs,alias=compositions(None if selected is None else tuple(selected))
    require(index.get('prospective_selected_alias')==alias
        and set(index['models'])==set(index['compositions'])==set(pairs),
        'actual pair deduplication and physical bindings required')
    ids=index['sample_ids']
    require(len(ids)==len(set(ids))==32,'same unique32 development IDs required')
    arms={pair[0] for pair in pairs.values()}
    require(set(index['SS_archives'])==set(index['base_cases'])==arms,
        'one own SS archive and geometry set per distinct physical SS model required')
    clusters={}
    for name,pair in pairs.items():
        binding=index['compositions'][name]
        validate_binding(binding['ss64'],binding['pbr1024'],pair)
        rows=index['models'][name]
        require([r['sample_id'] for r in rows]==ids,
            'every actual composition must use the same ordered development32')
        require(all(isinstance(r.get('case_name'),str) and r['case_name'] for r in rows)
            and len({r['case_name'] for r in rows})==32,'actual distinct case jobs required')
        for row in rows:
            sid=row['sample_id'];cluster=row['source_cluster_id']
            require(isinstance(cluster,str) and bool(cluster)
                and clusters.setdefault(sid,cluster)==cluster,'source cluster differs between compositions')
            for key in ['case_output','generation','shared']:
                require(isinstance(row[key],str) and Path(row[key]).is_absolute(),
                    'explicit actual case/generation/pixel paths required')
            require(row['generation']==str(Path(row['case_output'])/'generation')
                and row['shared']==str(Path(row['case_output'])/'shared'),
                'each composition must read its own generated texture and rendered pixels')
    for arm in arms:
        rows=index['base_cases'][arm]
        require([r['sample_id'] for r in rows]==ids,'own ordered32 SS/Shape geometry cases required')
        for row in rows:
            require(row['source_cluster_id']==clusters[row['sample_id']],
                'base geometry belongs to another source cluster')
            require(all(isinstance(row.get(k),str) and Path(row[k]).is_absolute()
                for k in ['case_output','generation','geometry_plan'])
                and row['generation']==str(Path(row['case_output'])/'generation'),
                'actual own saved base case, geometry and bound original Shape plan required')
        archive=index['SS_archives'][arm]
        require(Path(archive['path']).is_absolute() and Path(archive['receipt']).is_absolute(),
            'explicit own SS archive and actual file4 receipt required')
    require(set(index['original_shapes'])=={'shape512','shape1024'},'original two frozen Shape stages required')
    for stage,step in [('shape512',4210),('shape1024',16850)]:
        value=index['original_shapes'][stage]
        require(type(value['step']) is int and value['step']==step,'original Shape stage step differs')
        for field in ['config','checkpoint']:validate_spec(value[field])
        require(value['checkpoint']['path'].endswith(f'denoiser_step{step:07d}.pt'),
            'original Shape checkpoint name differs')
    return pairs,alias


def validate_replay_base(case,generation,archive,reference,*,arm,binding,archive_spec,
                         prediction_SHA256,guide_SHA256):
    """Check the chosen model's actual SS occupancy and original captured guides."""
    validate_spec(archive_spec)
    ids=validate_archive(archive,arm,reference,binding=binding)
    sid=case['sample_id']
    require(sid in ids,'replay geometry is outside the actual matched DEV32 prefix')
    position=ids.index(sid);row=archive['per_asset'][position]
    artifact=archive['native_SS_artifacts']['artifacts'][position]
    require(valid_case_status(case['status'],arm) and case['SS_arm']==arm
        and case['source_cluster_id']==row['source_cluster_id']
        and case['SS_evaluation']==archive_spec['path']
        and case['verified_inputs'].get(archive_spec['path'])==archive_spec['SHA256']
        and case['SS_prediction_SHA256']==artifact['prediction_voxels_SHA256']
        and case['new_SS_neural_forwards']==0 and case['new_Shape_cascade_calls']==1
        and case['new_PBR_texture_calls']==1
        and len(case['actual_owned_terminals'])==3
        and all(t['returncode']==0 and t['actual_session_drained'] is True for t in case['actual_owned_terminals']),
        'actual own SS through original Shape and common PBR must complete and drain first')
    if case['status']==CASE_STATUS:
        require(case.get('actual_SS_model_binding')==binding,'selected base case used another physical SS model')
    require(generation['status']=='PASS_ACTUAL_SS_TO_SHAPE512_SHAPE1024_PBR1024_GENERATION'
        and generation['sample_id']==sid and generation['SS_source_evaluation']==archive_spec['path']
        and generation['SS_prediction_SHA256']==artifact['prediction_voxels_SHA256']
        and generation['GT_geometry_or_coords_used_in_generation'] is False
        and generation['GT_loaded_only_after_generation'] is True
        and generation['per_output_alignment'] is False
        and generation['source_defined_normalization'] is True
        and generation['original_stage_configs_and_weights_unchanged'] is True
        and generation['texture_support_policy']==POLICY
        and generation['actual_native_guide_logits_order_scale_and_dead_branches_preserved'] is True
        and generation['actual_native_guide_archive_SHA256']==guide_SHA256,
        'own predicted geometry and original native guides required, with no GT or support reconstruction')
    require(digest(prediction_SHA256) and digest(guide_SHA256),'actual geometry and guide SHA256 required')
    return dict(actual_physical_SS_arm=arm,own_SS_archive=archive_spec,
        actual_SS_prediction_voxels_SHA256=artifact['prediction_voxels_SHA256'],
        actual_generated_geometry_SHA256=prediction_SHA256,actual_native_guide_SHA256=guide_SHA256,
        new_SS_or_Shape_forwards_for_replay=0)
