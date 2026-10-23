from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.archived_native_guides import POLICY
from experiments.deformnext_retargeting.native_common_pbr_summary import CASE_STATUS, VARIANTS, summarize
from experiments.deformnext_retargeting.native_fullcascade_summary import summarize as native_summary
from test_native_fullcascade_summary import inputs as native_inputs


def inputs():
    nc,g,r,p,kw = native_inputs(); cases = {}; extra = deepcopy(p); extra['per_view'] = []
    for group, items in g.items():
        for sid, value in items.items():
            value.update(texture_support_policy=POLICY, shape_decoder_recomputed=False,
                actual_native_guide_archive_SHA256='guides'+group.rsplit('-',1)[0]+sid,
                GT_sparse_latents_or_projection_cache_used=False)
    previous = native_summary(nc,g,r,p,**kw)
    for variant in VARIANTS:
        group=variant+'-common-V1PBR'; g[group]=deepcopy(g[variant+'-matched']); r[group]=deepcopy(r[variant+'-matched']); cases[variant]={}
        for sid in kw['sample_ids']:
            value=g[group][sid]; value.update(PBR_arm='vanilla', PBR_checkpoint=g['vanilla-matched'][sid]['PBR_checkpoint'],
                PBR_checkpoint_SHA256='trainedvanilla', conditioner_checkpoint=None)
            cases[variant][sid]=dict(status=CASE_STATUS, SS_variant=variant,sample_id=sid,source_cluster_id=kw['clusters'][sid],
                actual_PBR_checkpoint_SHA256='trainedvanilla',original_native_geometry_and_guides_reused_read_only=True,
                actual_original_predicted_LOD_reused=True,new_SS_or_Shape_forward_calls=0,
                actual_owned_terminals=[dict(returncode=0,actual_session_drained=True)for _ in range(3)])
            for cid in (8,9,10):
                score=deepcopy(next(x for x in p['per_view']if(x['arm'],x['sample_id'],x['camera_id'])==(variant+'-matched',sid,cid)))
                score.update(arm=group,shared_npz_SHA256='pixels'+group+sid+str(cid));extra['per_view'].append(score)
                kw['pixel_SHA256'][group,sid,cid]=score['shared_npz_SHA256']
    kw['pixel_GT']={}
    for report in [p,extra]:
        report.update(model_weights=dict(official='weights'),fullframe_primary=True,separate_GT_only_crop_supplementary=True)
        for row in report['per_view']:
            key=row['arm'],row['sample_id'],row['camera_id'];view=next(v for v in r[key[0]][key[1]]['per_view']if v['camera_id']==key[2])
            kw['pixel_GT'][key]=(view['fixed_GT_RGB_SHA256'],view['fixed_GT_alpha_SHA256'])
            row['image_audit']=dict(original_float_GT_SHA256=view['fixed_GT_RGB_SHA256'],original_GT_alpha_SHA256=view['fixed_GT_alpha_SHA256'],
                GT_only_crop_box=[0,0,256,256],converted_image_SHA256=dict(reference_full='full'+key[1]+str(key[2]),reference_GTcrop='crop'+key[1]+str(key[2])))
    return nc,cases,g,r,p,extra,previous,kw


def test_common_PBR_separates_SS_and_texture_effects_without_changing_native8_statistics():
    *args, kw=inputs(); result=summarize(*args,**kw)
    assert len(result['models'])==11 and result['actual_official_views']==1056
    assert result['native8_previous_statistics_max_difference']==0
    assert result['SS_effect_comparisons_share_trained_V1_PBR']
    assert result['PBR_effect_comparisons_share_actual_geometry_and_guides']
    assert len(result['roles']['target8']['contrasts'])==28


@pytest.mark.parametrize('error',['noise','SS','geometry','guide','policy','PBR','source-weights','pixel','GT','missing-view','registry','CPU-forward','native-statistics','terminal'])
def test_rejects_false_component_attribution(error):
    nc,c,g,r,p,e,old,kw=inputs(); group='R_feature-common-V1PBR'; value=g[group]['0']
    if error=='noise':value['actual_initial_texture_noise_SHA256']='wrong'
    elif error=='SS':value['SS_prediction_SHA256']='wrong'
    elif error=='geometry':value['metrics']['chamfer_L1']+=1
    elif error=='guide':value['actual_native_guide_archive_SHA256']='wrong'
    elif error=='policy':value['texture_support_policy']='derived-fixed-support'
    elif error=='PBR':value['PBR_checkpoint_SHA256']='wrong'
    elif error=='source-weights':value['conditioner_checkpoint']='/wrong/adapter.pt'
    elif error=='pixel':e['per_view'][0]['shared_npz_SHA256']='wrong'
    elif error=='GT':kw['pixel_GT'][group,'0',8]=('wrong','wrong')
    elif error=='missing-view':e['per_view'].pop()
    elif error=='registry':e['metric_registry_SHA256']='wrong'
    elif error=='CPU-forward':e['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    elif error=='native-statistics':old['roles']['target8']['means']['vanilla-prior']['deformation_weighted']['PSNR']+=1
    else:c['R_feature']['0']['actual_owned_terminals'][0]['actual_session_drained']=False
    with pytest.raises(ValueError):summarize(nc,c,g,r,p,e,old,**kw)
