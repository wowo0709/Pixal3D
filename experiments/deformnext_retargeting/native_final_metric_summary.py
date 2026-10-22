"""All ten metrics from six immutable model compositions and the same pixels."""
import numpy as np

from .cascade_view_roles import require,summarize_view_roles
from .metric_image_identity import validate_metric_image_join
from .native_final_case_plans import ALIASES


CONTRASTS=(('zero-edited4','zero-source4'),('vanilla','zero-edited4'),
    ('R_top1_only','vanilla'),('R_feature','R_top1_only'),('R_raw_topk','R_top1_only'),
    ('R_raw_topk','R_feature'),('R_feature','vanilla'),('R_raw_topk','vanilla'))


def eight_metrics(*, models, population, cases, shared, perceptual, metric_registry_SHA256,
                  draws=4000,bootstrap_seed=20261022):
    ids=population['sample_ids'];clusters=population['source_clusters_by_sample']
    require(tuple(models)==ALIASES and len(ids)==508 and len(set(ids))==508
        and set(clusters)==set(ids)and len(set(clusters.values()))==134
        and population['rendered_cameras']==[8,9,10]
        and set(cases)==set(shared)==set(perceptual)==set(models),
        'all six prespecified frozen models and complete508/134-source population required')
    records={};observations={};input_hashes={};reference_pixels={};reference_surfaces={}
    for alias,model in models.items():
        require(list(cases[alias])==ids and list(shared[alias])==ids,
                'ordered completed case and shared-render coverage must retain every test sample')
        report=perceptual[alias]
        require(report.get('status')=='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            and report.get('metric_registry_SHA256')==metric_registry_SHA256
            and report.get('model_precision')=='float32,TF32off,noAMP'
            and report.get('full_panel')is True and report.get('no_model_training')is True
            and report.get('global_environment_changed')is False
            and report.get('GPU_sanity',{}).get('status')=='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            and report['GPU_sanity'].get('CPU_model_forward_or_training_updates')==0,
            'actual registered GPU-only official DreamSim and separate original CLIP required')
        scores={(r['sample_id'],r['camera_id']):r for r in report['per_view']}
        require(len(report['per_view'])==len(scores)==1524
            and set(scores)=={(sid,cid)for sid in ids for cid in [8,9,10]},
            'all1524 distinct official model views required')
        records[alias]=[];observations[alias]=[]
        for sid in ids:
            case=cases[alias][sid];s=shared[alias][sid]
            require(case.get('status')=='PASS_ACTUAL_FROZEN_NATIVE_SIX_MODEL_TEST508_CASE'
                and case.get('actual_completed_native_case')is True and case.get('alias')==alias
                and case.get('sample_id')==sid and case.get('source_cluster_id')==clusters[sid]
                and case.get('frozen_model_binding')==model
                and case.get('actual_SS_checkpoint_SHA256')==model['ss64']['checkpoint']['SHA256']
                and case.get('actual_final_PBR_checkpoint_SHA256')==model['pbr1024']['checkpoint']['SHA256']
                and case.get('GT_not_used_in_generation')is True
                and len(case['terminals'])==(3 if alias.startswith('zero-')else 4)
                and all(r['returncode']==0 for r in case['terminals']),
                'actual completed same-frozen-weight native case and child terminals required')
            surface=case['surface_reference_join']
            require(surface.get('status')=='PASS_ACTUAL_LOCAL_NATIVE_GEOMETRY_SHARED_GT_SURFACE_JOIN'
                and surface.get('sample_id')==sid and surface.get('source_cluster_id')==clusters[sid]
                and surface.get('native_metrics')==case['raw_surface_metrics']
                and surface.get('GT_reference_not_used_for_model_conditioning')is True,
                'actual post-generation common GT surface join required')
            identity=(surface['GT_reference']['SHA256'],surface['target_points_SHA256'])
            require(reference_surfaces.setdefault(sid,identity)==identity,
                    'all models must share the identical source-defined GT surface points')
            expected=('PASS_ACTUAL_SHARED_RENDERER_MATERIAL_REFERENCE'if alias=='zero-source4'else'PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE')
            require(s.get('status')==expected and len(s['per_view'])==3,
                    'actual original-material shared renderer reference required')
            per={r['camera_id']:r for r in s['per_view']}
            obs={r['camera_id']:r for r in case['scored_image_observations']}
            require(set(per)==set(obs)=={8,9,10},'exact target8/source9/10 cameras required')
            views={}
            for cid in [8,9,10]:
                value=per[cid];pixel=obs[cid];official=scores[sid,cid]
                require(pixel['sample_id']==sid and pixel['source_cluster_id']==clusters[sid]
                    and official['arm']==alias and official['source_cluster_id']==clusters[sid]
                    and official['shared_npz_SHA256']==pixel['archive_SHA256'],
                    'perceptual values must belong to the actual scored model pixels and camera')
                identity=tuple(pixel[k]for k in ['reference_float_RGB_SHA256','reference_CHW_uint8_SHA256','reference_alpha_SHA256'])
                require(reference_pixels.setdefault((sid,cid),identity)==identity,
                        'original-material RGB and alpha must be identical across every model')
                input_hashes[pixel['archive']]=pixel['archive_SHA256'];observations[alias].append(pixel)
                score=official['scores']
                views[cid]={k:value[k]for k in ['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']}
                views[cid].update(DreamSim=score['DreamSim_fullframe'],CLIP=score['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=score['DreamSim_GT_foreground_crop'],CLIP_GTcrop=score['CLIP_cosine_GT_foreground_crop'])
            metrics=case['raw_surface_metrics']
            require(metrics['prediction_samples']==metrics['target_samples']==20000
                and metrics['threshold']==.02 and metrics['per_output_alignment']is False,
                'native20k/F@.02 without per-output alignment required')
            records[alias].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,
                CD=metrics['chamfer_L1'],FScore=metrics['surface_FScore']))
    stats=summarize_view_roles(records,list(CONTRASTS),draws=draws,seed=bootstrap_seed)
    return dict(status='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_EIGHT_METRICS',scope='final-test508',
        assets=508,source_clusters=134,models=models,statistics=stats,
        scored_image_observations=observations,input_SHA256=input_hashes,
        actual_completed_cases=3048,actual_perceptual_views=9144,
        primary_metric_families=dict(source_preservation=['LPIPS','DreamSim','CLIP'],
            target_alignment=['PSNR','SSIM','M_IoU','CD','FScore']),
        source_cluster_CI_unadjusted=True,independent_cascade_training_seeds=1,
        model_selection_from_test=False,FID_KID_not_yet_measured=True,full_goal_complete=False)


def validate_feature_pool(audit, features, *, observations, model_binding, weights_SHA256):
    require(audit.get('status')=='PASS_ACTUAL_OFFICIAL_INCEPTION2048_FEATURE_POOL'
        and audit.get('scope')=='final-test508'and audit.get('images')==1524 and audit.get('assets')==508
        and audit.get('package')=='torch-fidelity0.3.0'and audit.get('feature_layer')=='2048'
        and audit.get('feature_dtype')=='float32'and audit.get('fullframe')is True
        and audit.get('TF32')is False and audit.get('autocast')is False
        and audit.get('same_batch_repeat_exact')is True and audit.get('optimizer_updates')==0
        and audit.get('model_binding')==model_binding and audit.get('weights_SHA256')==weights_SHA256,
        'actual same-model official Inception-v3-compat2048 FP32 pool required')
    values=np.asarray(features)
    require(values.shape==(1524,2048)and values.dtype==np.float32 and np.isfinite(values).all()
        and np.any(values),'complete finite1524x2048 official feature bank required')
    require(audit['observations']==observations,'actual extracted feature image observations differ')


def combine_ten(eight, distribution, *, population):
    models=eight['models'];ids=population['sample_ids'];clusters=population['source_clusters_by_sample']
    require(eight.get('status')=='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_EIGHT_METRICS'
        and distribution.get('status')=='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_FID_KID'
        and distribution.get('models')==models and tuple(models)==ALIASES
        and set(distribution['distribution_metrics'])==set(models),
        'all six actual full508 eight-metric and official distribution results required')
    join=validate_metric_image_join(eight,distribution,sample_ids=ids,clusters=clusters,models=models)
    table={}
    for alias in models:
        roles=eight['statistics']['roles']
        src=roles['source9_10']['means'][alias]['deformation_weighted']
        tgt=roles['target8']['means'][alias]['deformation_weighted']
        geom=roles['all8_9_10']['means'][alias]['deformation_weighted']
        d=distribution['distribution_metrics'][alias]
        require(d['images_per_pool']==1524 and d['feature_dimension']==2048
            and d['package']=='torch-fidelity0.3.0'and d['KID_subset_std_is_not_CI']is True
            and d['unbiased_KID_clamped']is False,'same-count official global fullframe FID/KID required')
        table[alias]=dict(LPIPS=src['LPIPS'],DreamSim=src['DreamSim'],CLIP=src['CLIP'],
            FID=d['frechet_inception_distance'],KID=d['kernel_inception_distance_mean'],
            PSNR=tgt['PSNR'],SSIM=tgt['SSIM'],M_IoU=tgt['M_IoU'],CD=geom['CD'],FScore=geom['FScore'])
        require(np.isfinite(list(table[alias].values())).all(),'all ten primary table values must be finite')
    return dict(status='PASS_ACTUAL_NATIVE_SIX_MODEL_TEST508_ALL_TEN_METRICS',scope='final-test508',
        assets=508,source_clusters=134,models=models,primary_ten_metric_table=table,
        same_image_join=join,role_statistics=eight['statistics'],distribution_metrics=distribution['distribution_metrics'],
        KID_subset_std_not_source_cluster_CI=True,model_selection_from_test=False,
        independent_cascade_training_seeds=1,automatic_model_adoption=False,full_goal_complete=False)
