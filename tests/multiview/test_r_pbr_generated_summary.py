from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.r_pbr_generated_summary import MODES,summarize


def inputs():
    ids=[str(i)for i in range(32)];clusters={s:'cluster'+s for s in ids}
    generations={m:{}for m in MODES};renders=deepcopy(generations);pixels={};scores=[]
    for m in MODES:
        for sid in ids:
            generations[m][sid]=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',
                checkpoint_kind='trained-native1900',engineering_only=False,PBR_arm=m,sample_id=sid,
                PBR_checkpoint='/synthetic/ckpts/denoiser_step0001900.pt',source_weights_unchanged=True,
                conditioner_checkpoint=None if m=='vanilla'else'/synthetic/ckpts/deform_conditioner_step0001900.pt',
                GT_geometry_or_coords_used_in_generation=False,conditioning_preserves_CPU_and_CUDA_RNG=True,
                raw_surface_metrics_unchanged=True,geometry_array_byte_identity={'vertices':True},
                original_prediction_SHA256='shape'+sid,geometry_evaluation_SHA256='evaluation'+sid,
                actual_initial_texture_noise_SHA256='noise'+sid,PBR_replay_seed=42,sampler_class='native',
                sampler_params={'steps':12},metrics={'chamfer_L1':.01,'surface_FScore':.8})
            rows=[]
            for cid in[8,9,10]:
                pixel='pixels'+m+sid+str(cid);pixels[m,sid,cid]=pixel
                rows.append(dict(camera_id=cid,PSNR=24.,SSIM=.95,LPIPS=.05,M_IoU=.8,foreground_PSNR=20.,
                    identical_camera_matrices=True,source_material_identity_preserved=True,
                    fixed_GT_RGB_SHA256='GT'+sid+str(cid),fixed_GT_alpha_SHA256='alpha'+sid+str(cid)))
                scores.append(dict(arm=m,sample_id=sid,camera_id=cid,source_cluster_id=clusters[sid],shared_npz_SHA256=pixel,
                    scores=dict(DreamSim_fullframe=.1,CLIP_cosine_fullframe=.9,
                        DreamSim_GT_foreground_crop=.12,CLIP_cosine_GT_foreground_crop=.88)))
            renders[m][sid]=dict(status='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE',sample_id=sid,
                same_reference_pixels_for_every_model=True,per_view=rows)
    official=dict(status='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION',metric_registry_SHA256='registry',
        model_precision='float32,TF32off,noAMP',no_model_training=True,global_environment_changed=False,full_panel=True,
        GPU_sanity=dict(status='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY',CPU_model_forward_or_training_updates=0),per_view=scores)
    return generations,renders,official,dict(sample_ids=ids,clusters=clusters,pixel_SHA256=pixels,metric_registry_SHA256='registry')


def test_valid_synthetic_join_has_zero_paired_differences():
    g,r,p,kw=inputs();actual=summarize(g,r,p,**kw)
    assert actual['assets']==32 and actual['source_clusters']==32
    for role in actual['roles'].values():
        for contrast in role['contrasts'].values():
            assert all(v==0 for v in contrast['deformation_weighted_delta'].values())
            assert all(v==[0.,0.]for v in contrast['deformation_weighted_cluster_CI95'].values())


@pytest.mark.parametrize('problem',['cold','noise','GT','pixels','missing-view','registry','CPU-forward'])
def test_rejects_wrong_scientific_lineage_pixels_and_protocol(problem):
    g,r,p,kw=inputs()
    if problem=='cold':g['top1-only']['0']['checkpoint_kind']='original16850-engineering'
    elif problem=='noise':g['top1-only']['0']['actual_initial_texture_noise_SHA256']='other noise'
    elif problem=='GT':r['top1-only']['0']['per_view'][0]['fixed_GT_RGB_SHA256']='other GT'
    elif problem=='pixels':p['per_view'][0]['shared_npz_SHA256']='other pixels'
    elif problem=='missing-view':p['per_view'].pop()
    elif problem=='registry':p['metric_registry_SHA256']='other registry'
    elif problem=='CPU-forward':p['GPU_sanity']['CPU_model_forward_or_training_updates']=1
    with pytest.raises(ValueError):summarize(g,r,p,**kw)
