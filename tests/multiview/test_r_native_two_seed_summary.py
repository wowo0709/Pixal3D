from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.r_native_two_seed_summary import summarize,validate_traces
from test_r_mediated_native_recipe import recipe
from test_r_top1_native_recipe import control


def inputs():
    results,configs={},{}
    for seed in (101,202):
        results[seed]={};configs[seed]={}
        for variant in ('vanilla','R_top1_only'):
            cfg=recipe('vanilla')if variant=='vanilla'else control()[0]
            cfg.update(training_seed=seed,output_dir=f'/synthetic/s{seed}/{variant}')
            configs[seed][variant]=cfg;results[seed][variant]=[]
            for noise in range(3):
                rows=[dict(sample_id=str(i),source_cluster_id='cluster'+str(i//2),seed=500+noise,
                    target_voxels_sha256='GT'+str(i),voxel_iou=.4+(variant!='vanilla')*.01,
                    voxel_precision=.5,voxel_recall=.6,voxel_f1=.55)for i in range(128)]
                root=cfg['output_dir']
                results[seed][variant].append(dict(status='PASS',stage='ss64',step=480,native_variant=variant,
                    evaluation_mode='final128',optimizer_updates_during_evaluation=0,
                    actual_source_tensors=0 if variant=='vanilla'else 10,deterministic_algorithms=True,
                    checkpoint=root+'/ckpts/denoiser_step0000480.pt',conditioner_checkpoint=None if variant=='vanilla'else root+'/ckpts/deform_conditioner_step0000480.pt',
                    arm_config=root+'/config.json',input_states=['M1','M0','M0','M0'],input_camera_ids=[8,9,10,11],per_asset=rows))
    return results,configs


def test_reports_training_seeds_separately_and_bootstraps_source_clusters_only():
    r,c=inputs();out=summarize(r,c)
    assert out['training_seeds']==[101,202]and out['generation_noises']==3
    assert out['Top1_R_IoU_direction_positive_in_both_training_seeds']
    assert out['fixed_two_seed_average']['mean_asset_delta']['voxel_iou']==pytest.approx(.01)
    assert out['confidence_intervals_resample_asset_source_clusters_not_training_seed_population']


def test_negative_replication_is_retained_instead_of_claiming_success():
    r,c=inputs()
    for result in r[202]['R_top1_only']:
        for row in result['per_asset']:row['voxel_iou']=.38
    out=summarize(r,c)
    assert not out['Top1_R_IoU_direction_positive_in_both_training_seeds']
    assert out['fixed_two_seed_average']['mean_asset_delta']['voxel_iou']==pytest.approx(-.005)


@pytest.mark.parametrize('damage',['seed','NN','GT','missing-noise','same-noise','lr','nondeterministic'])
def test_rejects_training_or_evaluation_confound(damage):
    r,c=inputs();value=r[202]['R_top1_only']
    if damage=='seed':c[202]['R_top1_only']['training_seed']=101
    elif damage=='NN':value[0]['checkpoint']='/other/run/ckpts/denoiser_step0000480.pt'
    elif damage=='GT':value[0]['per_asset'][0]['target_voxels_sha256']='other GT'
    elif damage=='missing-noise':value.pop()
    elif damage=='same-noise':
        for seed in (101,202):
            for variant in r[seed]:r[seed][variant][1]=deepcopy(r[seed][variant][0])
    elif damage=='lr':c[202]['R_top1_only']['trainer']['args']['optimizer']['args']['lr']=1e-4
    elif damage=='nondeterministic':value[0]['deterministic_algorithms']=False
    with pytest.raises(ValueError):summarize(r,c)


def test_actual_training_streams_must_match_within_seed_and_change_between_seeds():
    traces={seed:{v:[dict(step=i+1,asset_ids=[list(range(96))],x_t_SHA256=str(seed)+str(i))for i in range(480)]
        for v in ('vanilla','R_top1_only')}for seed in (101,202)}
    validate_traces(traces)
    other=deepcopy(traces);other[202]=deepcopy(other[101])
    with pytest.raises(ValueError,match='actual training stream'):validate_traces(other)
    other=deepcopy(traces);other[202]['R_top1_only'][0]['x_t_SHA256']='mismatch'
    with pytest.raises(ValueError,match='paired480'):validate_traces(other)
