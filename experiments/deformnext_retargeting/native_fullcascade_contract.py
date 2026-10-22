"""Bind the four native480 SS archives to actual matched native1900 PBR."""
import hashlib


PAIRS={'vanilla':'vanilla','R_top1_only':'top1-only','R_feature':'top1-plus-feature','R_raw_topk':'raw-topk-plus-feature'}


def validate_archives(records):
    if set(records)!=set(PAIRS):raise ValueError('all four actual native SS variants required')
    identities=None
    for variant,r in records.items():
        artifacts=r.get('native_SS_artifacts',{})
        if (r.get('status')!='PASS'or r.get('stage')!='ss64'or r.get('step')!=480
                or r.get('native_variant')!=variant or r.get('conditioning_mode')!='edited-rgb'
                or r.get('target_modality')!='rgb'or r.get('input_states')!=['M1','M0','M0','M0']
                or r.get('input_camera_ids')!=[8,9,10,11]
                or r.get('optimizer_updates_during_evaluation')!=0
                or not r.get('deterministic_algorithms')
                or artifacts.get('status')!='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY'
                or not artifacts.get('original_prefix32_all_records_and_occupancy_counts_exact')
                or not artifacts.get('predicted_SS_is_available_for_full_cascade')
                or artifacts.get('archived_asset_count')!=32 or len(r['per_asset'])!=32
                or len(artifacts['artifacts'])!=32):
            raise ValueError('actual deterministic native480 RGB predicted-SS32 exact-prefix archive required')
        ids=[p['sample_id']for p in r['per_asset']]
        if len(set(ids))!=32 or ids!=r['quick_panel_ids']:raise ValueError('fixed unique32 SS panel required')
        own=[]
        for p,a in zip(r['per_asset'],artifacts['artifacts']):
            if (a['sample_id']!=p['sample_id']or a['seed']!=p['seed']
                    or a['target_voxels_SHA256']!=p['target_voxels_sha256']
                    or any(a['occupancy_counts'][k]!=p[k]for k in['tp','fp','fn'])):
                raise ValueError('actual SS archive bytes/count/sample/noise identities differ')
            own.append(tuple(p[k]for k in['sample_id','source_cluster_id','seed','target_voxels_sha256']))
        if identities is None:identities=own
        elif identities!=own:raise ValueError('four SS archives do not share the same asset/GT/noise panel')
    return [x[0]for x in identities]


def validate_original_stages(stages,original_PBR_SHA256,*,sha):
    if set(stages)!={'shape512','shape1024','pbr1024'}:raise ValueError('explicit fixed native Shape512/1024/PBR prior required')
    suffixes=dict(shape512='/shape512/ckpts/denoiser_step0004210.pt',shape1024='/shape1024/ckpts/denoiser_step0016850.pt')
    for name,suffix in suffixes.items():
        if not stages[name]['checkpoint'].endswith(suffix):raise ValueError('the user-specified original MV Shape checkpoints are required')
    if sha(stages['pbr1024']['checkpoint'])!=original_PBR_SHA256:
        raise ValueError('SS-only cascade must retain the declared original complete PBR16850 prior')


def asset_token(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:16]
