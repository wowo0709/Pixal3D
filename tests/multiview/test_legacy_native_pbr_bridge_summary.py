from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.legacy_native_pbr_bridge import CASE_STATUS,VARIANTS
from experiments.deformnext_retargeting.legacy_native_pbr_bridge_summary import summarize
from experiments.deformnext_retargeting.native_fullcascade_summary import GROUPS as NATIVE_GROUPS,summarize as native_summary
from test_native_fullcascade_summary import inputs as native_inputs


def inputs():
    nc,g,r,official,kw=native_inputs();legacy_cases={};originals={};legacy=deepcopy(official);legacy['per_view']=[]
    for groups in g.values():
        for value in groups.values():value['PBR_replay_seed']=20261020
    previous=native_summary(nc,g,r,official,**kw)
    for variant in VARIANTS:
        legacy_cases[variant]={};originals[variant]={}
        for kind in ['prior','matched']:
            group='legacy-'+variant+'-'+kind;g[group]=deepcopy(g['vanilla-'+kind]);r[group]=deepcopy(r['vanilla-'+kind])
            for sid in kw['sample_ids']:
                root='/old/'+variant+'/'+sid;own=g[group][sid]
                originals[variant][sid]=dict(SS_prediction_SHA256=own['SS_prediction_SHA256'],
                    SS_source_evaluation=own['SS_source_evaluation'],metrics=own['metrics'])
                legacy_cases[variant][sid]=dict(status=CASE_STATUS,SS_variant=variant,sample_id=sid,
                    source_cluster_id=kw['clusters'][sid],PBR_mode='vanilla',existing_geometry_reused_read_only=True,
                    new_SS_or_Shape_inferences=0,original_Shape5124210_Shape102416850=True,
                    paired_original_and_trained_texture_initial_noise_exact=True,
                    paired_raw_geometry_and_native_predicted_LOD_identical=True,
                    actual_prior_and_matched_three_camera_original_material_M1_shared_scores=True,
                    original_generated_geometry_not_aligned_or_replaced=True,optimizer_updates=0,
                    terminals=[dict(returncode=0,actual_session_drained=True)for _ in range(6)],
                    original_geometry_root=root,original_geometry_inputs_SHA256={root+'/prediction.npz':own['original_prediction_SHA256']},
                    actual_original_shape_checkpoint_SHA256=dict(shape512='S512',shape1024='S1024'),
                    actual_original_PBR_checkpoint_SHA256='original',actual_matched_PBR_checkpoint_SHA256='trainedvanilla')
                for cid in [8,9,10]:
                    p=deepcopy(next(x for x in official['per_view']if(x['arm'],x['sample_id'],x['camera_id'])==('vanilla-'+kind,sid,cid)))
                    p['arm']=group;p['shared_npz_SHA256']='pixels'+group+sid+str(cid);legacy['per_view'].append(p)
                    kw['pixel_SHA256'][group,sid,cid]=p['shared_npz_SHA256']
    kw['pixel_GT']={}
    for report in [official,legacy]:
        report.update(model_weights=dict(official='weights'),fullframe_primary=True,separate_GT_only_crop_supplementary=True)
        for p in report['per_view']:
            key=p['arm'],p['sample_id'],p['camera_id'];v=next(v for v in r[key[0]][key[1]]['per_view']if v['camera_id']==key[2])
            kw['pixel_GT'][key]=(v['fixed_GT_RGB_SHA256'],v['fixed_GT_alpha_SHA256'])
            p['image_audit']=dict(original_float_GT_SHA256=v['fixed_GT_RGB_SHA256'],original_GT_alpha_SHA256=v['fixed_GT_alpha_SHA256'],
                GT_only_crop_box=[0,0,256,256],converted_image_SHA256=dict(reference_full='full'+key[1]+str(key[2]),reference_GTcrop='crop'+key[1]+str(key[2])))
    return nc,legacy_cases,originals,g,r,official,legacy,previous,kw


def test_fourteen_group_join_reproduces_native_and_does_not_claim_module_causality():
    *args,kw=inputs();result=summarize(*args,**kw)
    assert result['actual_official_perceptual_views']==1344 and len(result['models'])==14
    assert result['native8_previous_statistics_max_difference']==0
    assert result['cross_SS_training_protocol_deployment_comparison_not_isolated_module_causality']
    assert len(result['roles']['source9_10']['contrasts'])==91


@pytest.mark.parametrize('error',['noise','common-PBR','shape','pixels','GT','missing-view','registry','CPU-forward','native-statistics'])
def test_rejects_false_native_legacy_comparison(error):
    nc,c,o,g,r,p,l,old,kw=inputs();group='legacy-feature-spatial-matched';item=g[group]['0']
    if error=='noise':item['actual_initial_texture_noise_SHA256']='other'
    elif error=='common-PBR':item['PBR_checkpoint_SHA256']='other'
    elif error=='shape':c['feature-spatial']['0']['actual_original_shape_checkpoint_SHA256']['shape512']='other'
    elif error=='pixels':l['per_view'][0]['shared_npz_SHA256']='other'
    elif error=='GT':kw['pixel_GT'][group,'0',8]=('different','alpha')
    elif error=='missing-view':l['per_view'].pop()
    elif error=='registry':l['metric_registry_SHA256']='other'
    elif error=='CPU-forward':l['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    else:old['roles']['target8']['means']['vanilla-prior']['deformation_weighted']['PSNR']+=1
    with pytest.raises(ValueError):summarize(nc,c,o,g,r,p,l,old,**kw)
