"""Join actual native1900 PBR isolation renders and official GPU metrics."""
import json
from .cascade_view_roles import summarize_view_roles


MODES=('vanilla','top1-only','top1-plus-feature','raw-topk-plus-feature')


def summarize(generations,shared,perceptual,*,sample_ids,clusters,pixel_SHA256,metric_registry_SHA256):
    if (set(generations)!=set(MODES) or set(shared)!=set(MODES)
            or len(sample_ids)!=32 or len(set(sample_ids))!=32 or set(clusters)!=set(sample_ids)
            or perceptual.get('status')!='PASS_ACTUAL_OFFICIAL_DREAMSIM_CLIP_RENDER_EVALUATION'
            or perceptual.get('metric_registry_SHA256')!=metric_registry_SHA256
            or perceptual.get('model_precision')!='float32,TF32off,noAMP'
            or perceptual.get('full_panel') is not True
            or not perceptual.get('no_model_training') or perceptual.get('global_environment_changed')
            or perceptual.get('GPU_sanity',{}).get('status')!='PASS_ACTUAL_GPU_OFFICIAL_METRIC_SANITY'
            or perceptual['GPU_sanity'].get('CPU_model_forward_or_training_updates')!=0):
        raise ValueError('four actual native1900 models, fixed32 panel and pinned official GPU metric protocol required')
    scores={}
    for row in perceptual['per_view']:
        key=(row['arm'],row['sample_id'],row['camera_id'])
        if key in scores:raise ValueError('duplicate official metric image')
        scores[key]=row
    if set(scores)!={(m,s,c)for m in MODES for s in sample_ids for c in (8,9,10)}:
        raise ValueError('complete four-model32 three-camera official384 coverage required')
    expected_GT={};expected_geometry={};records={}
    for mode in MODES:
        if set(generations[mode])!=set(sample_ids)or set(shared[mode])!=set(sample_ids):
            raise ValueError('complete matching generation/render identities required')
        records[mode]=[]
        for sid in sample_ids:
            g=generations[mode][sid];r=shared[mode][sid]
            if (g.get('status')!='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE'
                    or g.get('checkpoint_kind')!='trained-native1900' or g.get('engineering_only')is not False
                    or g.get('PBR_arm')!=mode or g.get('sample_id')!=sid
                    or not g['PBR_checkpoint'].endswith('/ckpts/denoiser_step0001900.pt')
                    or not g.get('source_weights_unchanged')or g.get('GT_geometry_or_coords_used_in_generation')is not False
                    or not g.get('conditioning_preserves_CPU_and_CUDA_RNG')
                    or not g.get('raw_surface_metrics_unchanged')or not all(g['geometry_array_byte_identity'].values())
                    or g.get('conditioner_checkpoint')!=(None if mode=='vanilla'else g['PBR_checkpoint'].replace(
                        '/ckpts/denoiser_step0001900.pt','/ckpts/deform_conditioner_step0001900.pt'))
                    or r.get('status')!='PASS_ACTUAL_FIXED_SHARED_RENDER_REFERENCE' or r.get('sample_id')!=sid
                    or not r.get('same_reference_pixels_for_every_model')or len(r['per_view'])!=3):
                raise ValueError('actual trained own1900 pair, predicted geometry and shared material reference required')
            identity=tuple(g[k]for k in['original_prediction_SHA256','geometry_evaluation_SHA256',
                'actual_initial_texture_noise_SHA256','PBR_replay_seed','sampler_class'])
            identity+=(json.dumps(g['sampler_params'],sort_keys=True),g['metrics']['chamfer_L1'],g['metrics']['surface_FScore'])
            if expected_geometry.setdefault(sid,identity)!=identity:
                raise ValueError('PBR isolation changed geometry, initial noise, sampler or surface metrics')
            views={}
            for view in r['per_view']:
                cid=view['camera_id'];key=(mode,sid,cid)
                if cid not in (8,9,10)or cid in views or key not in scores:
                    raise ValueError('one exact target8/source9/source10 camera required')
                per=scores[key]
                if (per['source_cluster_id']!=clusters[sid]or per['shared_npz_SHA256']!=pixel_SHA256[key]
                        or not view.get('identical_camera_matrices')or not view.get('source_material_identity_preserved')):
                    raise ValueError('official metric belongs to different pixels, source or camera')
                gt=(view['fixed_GT_RGB_SHA256'],view['fixed_GT_alpha_SHA256'])
                if expected_GT.setdefault((sid,cid),gt)!=gt:
                    raise ValueError('reference RGB/alpha differs between modes')
                views[cid]={k:view[k]for k in['PSNR','SSIM','LPIPS','M_IoU','foreground_PSNR']}
                p=per['scores'];views[cid].update(DreamSim=p['DreamSim_fullframe'],CLIP=p['CLIP_cosine_fullframe'],
                    DreamSim_GTcrop=p['DreamSim_GT_foreground_crop'],CLIP_GTcrop=p['CLIP_cosine_GT_foreground_crop'])
            records[mode].append(dict(sample_id=sid,source_cluster_id=clusters[sid],views=views,
                CD=g['metrics']['chamfer_L1'],FScore=g['metrics']['surface_FScore']))
    pairs=[['top1-only','vanilla'],['top1-plus-feature','vanilla'],['raw-topk-plus-feature','vanilla'],
           ['top1-plus-feature','top1-only'],['raw-topk-plus-feature','top1-only'],['raw-topk-plus-feature','top1-plus-feature']]
    return summarize_view_roles(records,pairs,draws=4000,seed=20261022)
