"""Selected physical SS archive through the original Shape/common-PBR cascade."""
from .common_pipeline_contract import ARMS as ORIGINAL_CASE_ARMS
from .common_pipeline_contract import CASE_STATUS as ORIGINAL_CASE_STATUS
from .common_pipeline_contract import validate_geometry_plan, validate_pbr_ready
from .pipeline_population_contract_v2 import ARMS
from .selected_pipeline_model_contract import require,validate_ss_evaluation


CASE_STATUS='PASS_ACTUAL_SELECTED_SS_COMMON_TRAINED_V1_PBR_NATIVE_CASCADE_CASE_V2'


def valid_case_status(status,arm):
    return status==CASE_STATUS or arm in ORIGINAL_CASE_ARMS and status==ORIGINAL_CASE_STATUS


def validate_archive(result,arm,reference,*,binding):
    require(arm in ARMS,'an actual physical selected SS arm required')
    validate_ss_evaluation(result,binding,arm)
    archive=result.get('native_SS_artifacts',{})
    count=archive.get('archived_asset_count')
    require(count in (32,128) and len(archive.get('artifacts',[]))==count
        and len(result.get('per_asset',[]))==count
        and archive.get('predicted_SS_is_available_for_full_cascade') is True
        and (archive.get('original_prefix32_all_records_and_occupancy_counts_exact') is True
             if count==32 else archive.get('same_forward_prediction_and_target_capture') is True),
        'own actual original32 or same-forward original128 SS archive required')
    rows,refs=result['per_asset'][:32],reference['per_asset'][:32]
    ids=[r['sample_id'] for r in rows]
    require(len(set(ids))==len(refs)==32,'same unique DEV32 reference prefix required')
    for row,ref,artifact in zip(rows,refs,archive['artifacts'][:32]):
        require(all(row[k]==ref[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256'])
            and artifact['sample_id']==row['sample_id'] and artifact['seed']==row['seed']
            and artifact['target_voxels_SHA256']==row['target_voxels_sha256']
            and all(artifact['occupancy_counts'][k]==row[k] for k in ['tp','fp','fn']),
            'actual own predicted SS/GT/noise/cluster/count identity differs')
    return ids
