import pytest

from experiments.deformnext_retargeting.native_fullcascade_contract import PAIRS,validate_archives,validate_original_stages


def archives():
    out={}
    for variant in PAIRS:
        rows=[dict(sample_id=str(i),source_cluster_id='cluster'+str(i),seed=42+i,
            target_voxels_sha256='GT'+str(i),tp=100,fp=10,fn=20)for i in range(32)]
        artifacts=[dict(sample_id=p['sample_id'],seed=p['seed'],target_voxels_SHA256=p['target_voxels_sha256'],
            occupancy_counts={k:p[k]for k in['tp','fp','fn']})for p in rows]
        out[variant]=dict(status='PASS',stage='ss64',step=480,native_variant=variant,conditioning_mode='edited-rgb',
            target_modality='rgb',input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],
            optimizer_updates_during_evaluation=0,deterministic_algorithms=True,per_asset=rows,
            quick_panel_ids=[p['sample_id']for p in rows],native_SS_artifacts=dict(
                status='PASS_ACTUAL_NATIVE_SS_ARCHIVE_AND_EXACT_REFERENCE_SUBSET_REPLAY',
                original_prefix32_all_records_and_occupancy_counts_exact=True,
                predicted_SS_is_available_for_full_cascade=True,archived_asset_count=32,artifacts=artifacts))
    return out


def test_valid_four_pair_archive_panel():
    assert validate_archives(archives())==list(map(str,range(32)))


@pytest.mark.parametrize('problem',['intermediate','depth','source4','wrong-variant','wrong-GT','wrong-count','not-deterministic'])
def test_rejects_wrong_native_cascade_lineage(problem):
    rows=archives();r=rows['R_feature']
    if problem=='intermediate':r['step']=384
    elif problem=='depth':r['target_modality']='depth'
    elif problem=='source4':r['input_states'][0]='M0'
    elif problem=='wrong-variant':r['native_variant']='R_raw_topk'
    elif problem=='wrong-GT':r['per_asset'][0]['target_voxels_sha256']='other GT'
    elif problem=='wrong-count':r['native_SS_artifacts']['artifacts'][0]['occupancy_counts']['tp']=101
    elif problem=='not-deterministic':r['deterministic_algorithms']=False
    with pytest.raises(ValueError):validate_archives(rows)


def test_original_shape_and_constant_original_texture_prior():
    stages=dict(shape512={'checkpoint':'/original/shape512/ckpts/denoiser_step0004210.pt'},
        shape1024={'checkpoint':'/original/shape1024/ckpts/denoiser_step0016850.pt'},pbr1024={'checkpoint':'/original/PBR16850.pt'})
    validate_original_stages(stages,'original',sha=lambda p:'original')
    with pytest.raises(ValueError):validate_original_stages(stages,'original',sha=lambda p:'trained1900')
    stages['shape512']['checkpoint']='/wrong/final.pt'
    with pytest.raises(ValueError):validate_original_stages(stages,'original',sha=lambda p:'original')
