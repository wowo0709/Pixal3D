"""The five core SS controls with one actual trained vanilla PBR downstream."""
from .archived_native_guides import POLICY

ARMS = ('V1','C_DECODER','FULL','C_TOP1_V','C_FIXED_TOPK')
ALIASES = dict(V1='vanilla',C_DECODER='R_top1_only',FULL='R_raw_topk')
CASE_STATUS = 'PASS_ACTUAL_CORE_SS_COMMON_TRAINED_V1_PBR_NATIVE_CASCADE_CASE'


def validate_pbr_ready(ready, cfg):
    e,t=cfg['experiment'],cfg['trainer']['args']
    if (ready.get('status')!='PASS_ACTUAL_TRAINED_NATIVE_PBR1900_MODEL_READY'
            or ready.get('mode')!='vanilla' or ready.get('conditioner_checkpoint') is not None
            or not ready.get('checkpoint','').endswith('/ckpts/denoiser_step0001900.pt')
            or e.get('PBR_source_mode')!='vanilla' or e.get('engineering_only') is not False
            or e.get('source_multiview_step')!=16850 or e.get('target_modality')!='rgb'
            or e.get('training_assets')!=9139 or e.get('epochs')!=5
            or (e.get('world_size'),e.get('global_batch'),e.get('microbatch_per_gpu'),e.get('accumulation'))!=(4,24,6,1)
            or t.get('deform_conditioner') is not None or t.get('model_learning_rates',{})!={}
            or t.get('lora')!=dict(rank=16,alpha=16,dropout=0.)
            or t.get('max_steps')!=1900
            or set(ready.get('actual_four_rank_completion',{}))!={'0','1','2','3'}):
        raise ValueError('actual trained vanilla1900 native PBR with no source conditioner required')
    for rank,row in ready['actual_four_rank_completion'].items():
        if (row.get('rank')!=int(rank) or row.get('mode')!='vanilla'
                or row.get('actual_optimizer_updates')!=1900 or row.get('source_tensors')!=0
                or row.get('actual_optimizer_states')!=360 or not row.get('stage_training_complete')
                or not row.get('initial4_identity_before_first_update')
                or not row.get('frozen_base_matcher_and_anchors_exact')
                or not row.get('post_training_DDP_parameters_exact')):
            raise ValueError('all four actual final PBR ranks and unchanged frozen state required')
    return ready['checkpoint_SHA256']


def validate_archive(result, arm, reference):
    if arm not in ARMS:raise ValueError('one of the five fixed core SS variants required')
    archive=result.get('native_SS_artifacts',{})
    if (result.get('status')!='PASS' or result.get('stage')!='ss64' or result.get('step')!=480
            or result.get('native_variant')!=ALIASES.get(arm,arm)
            or result.get('input_states')!=['M1','M0','M0','M0']
            or result.get('input_camera_ids')!=[8,9,10,11]
            or result.get('target_modality')!='rgb'
            or result.get('optimizer_updates_during_evaluation')!=0
            or not result.get('deterministic_algorithms')
            or not archive.get('predicted_SS_is_available_for_full_cascade')):
        raise ValueError('same native480 edited RGB SS and actual original prediction capture required')
    count=archive.get('archived_asset_count')
    if (count not in (32,128) or len(archive.get('artifacts',[]))!=count
            or len(result.get('per_asset',[]))!=count
            or not (archive.get('original_prefix32_all_records_and_occupancy_counts_exact')
                    if count==32 else archive.get('same_forward_prediction_and_target_capture'))):
        raise ValueError('actual original32 archive or original128 same-forward capture required')
    rows=result['per_asset'][:32];refs=reference['per_asset'][:32]
    ids=[r['sample_id'] for r in rows]
    if len(set(ids))!=32 or len(refs)!=32:raise ValueError('unique common DEV32 prefix required')
    for row,ref,artifact in zip(rows,refs,archive['artifacts'][:32]):
        if (any(row[k]!=ref[k] for k in ['sample_id','source_cluster_id','seed','target_voxels_sha256'])
                or artifact['sample_id']!=row['sample_id'] or artifact['seed']!=row['seed']
                or artifact['target_voxels_SHA256']!=row['target_voxels_sha256']
                or any(artifact['occupancy_counts'][k]!=row[k] for k in ['tp','fp','fn'])):
            raise ValueError('actual SS/GT/noise/count/cluster pairing differs')
    return ids


def validate_geometry_plan(plan, template, ready):
    if (plan['input_states']!=['M1','M0','M0','M0'] or len(plan['RGB_paths'])!=4
            or plan['row']!=template['row'] or plan['RGB_paths']!=template['RGB_paths']
            or plan['row']['conditioning_camera_ids']!=[8,9,10,11]
            or plan['row']['split']!='validation'
            or plan['texture_support_policy']!=POLICY
            or any(plan[k]!=template[k] for k in ['seed','surface_seed','source_geometry',
                'target_vertices','decoders','pipeline_config','DINO_local_path','allocator_fraction'])
            or set(plan['stages'])!={'shape512','shape1024','pbr1024'}
            or any(plan['stages'][k]!=template['stages'][k] for k in ['shape512','shape1024'])
            or plan['stages']['pbr1024']!=dict(config=ready['config'],checkpoint=ready['checkpoint'])):
        raise ValueError('only SS archive, output and common actual trained V1 PBR may change')
    for stage,step in [('shape512',4210),('shape1024',16850)]:
        expected=f'/{stage}/ckpts/denoiser_step{step:07d}.pt'
        if not plan['stages'][stage]['checkpoint'].endswith(expected):
            raise ValueError('original dev-jaehyeok multi-view Shape stages required')
    return True
