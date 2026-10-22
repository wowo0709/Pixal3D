from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.native_fullcascade_contract import PAIRS
from experiments.deformnext_retargeting.native_fullcascade_summary import GROUPS,summarize
from test_r_pbr_generated_summary import inputs as pbr_inputs


def inputs():
    oldg,oldr,official,kw=pbr_inputs();ids=kw['sample_ids'];cases={};g={};r={};pixels={};scores=[]
    for variant,mode in PAIRS.items():
        cases[variant]={}
        for kind in ('prior','matched'):
            group=variant+'-'+kind;g[group]={};r[group]={}
            for sid in ids:
                item=deepcopy(oldg[mode][sid]);item.update(SS_prediction_SHA256='SS'+variant+sid,
                    SS_source_evaluation='/saved/'+variant,original_prediction_SHA256='shape'+variant+sid,
                    geometry_evaluation_SHA256='evaluation'+variant+sid,actual_initial_texture_noise_SHA256='noise'+variant+sid)
                if kind=='prior':item.update(checkpoint_kind='original16850-engineering',engineering_only=True,
                    PBR_arm='vanilla',PBR_checkpoint_SHA256='original',conditioner_checkpoint=None)
                else:item['PBR_checkpoint_SHA256']='trained'+mode
                g[group][sid]=item;r[group][sid]=deepcopy(oldr[mode][sid])
                for cid in (8,9,10):
                    pixel='pixels'+group+sid+str(cid);pixels[group,sid,cid]=pixel
                    source=next(x for x in official['per_view']if(x['arm'],x['sample_id'],x['camera_id'])==(mode,sid,cid))
                    scores.append(dict(source,arm=group,shared_npz_SHA256=pixel))
                cases[variant][sid]=dict(status='PASS_ACTUAL_NATIVE_SS480_AND_MATCHED_PBR1900_FULLCASCADE_CASE',
                    sample_id=sid,source_cluster_id=kw['clusters'][sid],SS_variant=variant,PBR_mode=mode,
                    original_Shape5124210_Shape102416850=True,paired_original_and_trained_texture_initial_noise_exact=True,
                    paired_raw_geometry_and_native_predicted_LOD_identical=True,
                    actual_prior_and_matched_three_camera_original_material_M1_shared_scores=True,
                    original_generated_geometry_not_aligned_or_replaced=True,terminals=[dict(returncode=0)]*7,
                    actual_original_stage_checkpoint_SHA256=dict(shape512='S512',shape1024='S1024',pbr1024='original'),
                    actual_matched_PBR_checkpoint_SHA256='trained'+mode,SS_prediction_SHA256='SS'+variant+sid,
                    source_SS_evaluation='/saved/'+variant)
    official['per_view']=scores;kw.update(pixel_SHA256=pixels,original_PBR_SHA256='original')
    return cases,g,r,official,kw


def test_same_synthetic_scores_have_zero_contrasts_and_allow_different_SS_supports():
    c,g,r,p,kw=inputs();result=summarize(c,g,r,p,**kw)
    assert len(result['models'])==8 and len(result['roles']['target8']['contrasts'])==16
    assert result['texture_noise_not_claimed_identical_across_different_generated_supports']
    assert all(value==0 for d in result['roles']['all8_9_10']['contrasts'].values()
        for value in d['deformation_weighted_delta'].values())


@pytest.mark.parametrize('error',['noise','SS','geometry','prior-NN','trained-NN','GT','pixels','missing-view','LOD','terminal','registry'])
def test_rejects_false_pairing_or_lineage(error):
    c,g,r,p,kw=inputs();group='R_raw_topk-matched';item=g[group]['0']
    if error=='noise':item['actual_initial_texture_noise_SHA256']='different'
    elif error=='SS':item['SS_prediction_SHA256']='different'
    elif error=='geometry':item['metrics']['chamfer_L1']=.2
    elif error=='prior-NN':g['vanilla-prior']['0']['PBR_checkpoint_SHA256']='legacy762'
    elif error=='trained-NN':item['PBR_checkpoint_SHA256']='other1900'
    elif error=='GT':r[group]['0']['per_view'][0]['fixed_GT_RGB_SHA256']='different'
    elif error=='pixels':p['per_view'][0]['shared_npz_SHA256']='different'
    elif error=='missing-view':p['per_view'].pop()
    elif error=='LOD':c['R_raw_topk']['0']['paired_raw_geometry_and_native_predicted_LOD_identical']=False
    elif error=='terminal':c['R_raw_topk']['0']['terminals']=[dict(returncode=1)]*7
    elif error=='registry':p['metric_registry_SHA256']='different'
    with pytest.raises(ValueError):summarize(c,g,r,p,**kw)
