import copy

import pytest

from experiments.deformnext_retargeting.consistent_evidence_summary import summarize_consistent_evidence
from test_r_top1_native_summary import four


def data():
    original = four()
    out = {n: original[n] for n in ['vanilla','R_top1_only']}
    for name,count,offset,head in [('E_FIXED_RAW',10,.002,0),('E_LEARNED_RAW',17,.006,7)]:
        out[name] = copy.deepcopy(out['R_top1_only'])
        for r in out[name]:
            r.update(native_variant=name,actual_source_tensors=count,
                conditioning_schema='consistent-raw-ray-evidence-native-v1',
                followup_study_schema='consistent-raw-ray-evidence-followup-v1',
                shared_carrier_use_retarget_evidence=True,learned_projection_tensors=0,
                learned_attention_head_tensors=head,training_seed=101)
            for row in r['per_asset']: row['voxel_iou'] += offset
    for results in out.values():
        for r in results:
            r.update(manifest_sha256='a'*64,target_modality='rgb',input_states=['M1','M0','M0','M0'],
                input_camera_ids=[8,9,10,11],generation_batch_size=8,sampling_steps=12,
                guidance=3.,deterministic_algorithms=True)
    return out


def test_two_exploratory_module_contrasts_preserve_cluster_and_noise_scope():
    results = data()
    report = summarize_consistent_evidence(results,{n:101 for n in results},draws=100)
    assert report['comparisons']['E_FIXED_RAW_minus_R_top1_only']['mean_asset_delta']['voxel_iou'] == pytest.approx(.002)
    assert report['comparisons']['E_LEARNED_RAW_minus_E_FIXED_RAW']['mean_asset_delta']['voxel_iou'] == pytest.approx(.004)
    assert all(v['prespecified_IoU_family_size']==2 for v in report['comparisons'].values())
    assert report['followup_is_exploratory_after_previous_module_results']
    assert report['original_twelve_arm_family_and_results_unchanged']
    assert report['independent_training_seeds']==1 and report['generation_noises']==3
    assert not report['automatic_model_adoption'] and not report['held_out_test508_evaluated']


@pytest.mark.parametrize('damage',['head','V','family','GT','noise','source','sampler','RGB','camera','missing','seed'])
def test_quality_gain_does_not_allow_changed_architecture_or_pairing(damage):
    results=data();r=results['E_LEARNED_RAW'][0];seeds={n:101 for n in results}
    if damage=='head':r['actual_source_tensors']=19
    elif damage=='V':r['learned_projection_tensors']=2
    elif damage=='family':r['followup_study_schema']='unified-module-pipeline-v2'
    elif damage=='GT':r['per_asset'][0]['target_voxels_sha256']='other'
    elif damage=='noise':r['per_asset'][0]['seed']+=1
    elif damage=='source':r['per_asset'][0]['source_cluster_id']='other'
    elif damage=='sampler':r['guidance']=1.
    elif damage=='RGB':r['target_modality']='depth'
    elif damage=='camera':r['input_camera_ids']=[9,10,11]
    elif damage=='missing':r.pop('manifest_sha256')
    else:seeds['E_LEARNED_RAW']=202
    with pytest.raises(ValueError):summarize_consistent_evidence(results,seeds,draws=100)
