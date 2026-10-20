"""Actual generated-support native PBR condition and twelve-step replay proof."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import time

import numpy as np
from PIL import Image
import torch


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.plan.read_text())
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if gpu not in {'0','1','2','3'} or gpu not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.6,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    output=Path(plan['output'])
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from pixal3d.pipelines import samplers
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from pixal3d.modules.direct_ray_retargeting import DirectRayRetargetingConditioner
    from .retarget_only_pbr_contract import build_retarget_only
    from .generated_pbr_conditioning_v2 import install_generated_pbr_conditioning
    from .cascade_from_ss_artifact import load_stage
    from .cascade_artifact_inputs import prepared_views
    from .frozen_projection_streaming import install_frozen_projection_streaming
    install_frozen_projection_streaming(max_assets=1)
    cfg=json.loads(Path(plan['PBR_config']).read_text())
    direct_cfg=json.loads(Path(plan['Direct_config']).read_text())
    r_cfg=json.loads(Path(plan['R_config']).read_text())
    encoder_args=dict(cfg['trainer']['args']['image_cond_model']['args'])
    encoder_args['model_name']=plan['DINO_local_path']
    extractor=DinoV3ProjFeatureExtractor(**encoder_args).eval().requires_grad_(False)
    if extractor.use_naf_upsample:extractor._load_naf()
    extractor.cuda()
    extractor.forward=torch.autocast('cuda',dtype=torch.bfloat16)(extractor.forward)
    conditioner_cfg=direct_cfg['trainer']['args']['deform_conditioner']
    r,_=build_retarget_only(conditioner_cfg)
    d=DirectRayRetargetingConditioner(**conditioner_cfg)
    for model,path in [(r,plan['R_initial']),(d,plan['Direct_initial'])]:
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        model.cuda().eval().requires_grad_(False)
        if any(torch.count_nonzero(value) for name,value in model.state_dict().items()
               if name in ['fusion_gate','retargeter.net.2.weight','retargeter.net.2.bias']):
            raise ValueError('untouched initial zero-output conditioner required')
    pipeline_cfg=json.loads(Path(plan['pipeline_config']).read_text())['args']
    spec=pipeline_cfg['tex_slat_sampler']
    pipe=SimpleNamespace(device=torch.device('cuda'),low_vram=False,
        image_cond_model_tex_1024=extractor,
        shape_slat_normalization=pipeline_cfg['shape_slat_normalization'],
        tex_slat_normalization=pipeline_cfg['tex_slat_normalization'],
        tex_slat_sampler=getattr(samplers,spec['name'])(**spec['args']),
        tex_slat_sampler_params=spec['params'])
    native=MethodType(module.Pixal3DImageTo3DPipeline.get_proj_cond_shape,pipe)
    pipe.get_proj_cond_shape=native
    module.pil_views_to_tensor=prepared_views
    row=plan['row']
    images=[Image.open(path).copy() for path in plan['RGB_paths']]
    with np.load(plan['generated_prediction']) as archive:
        coords=torch.from_numpy(archive['shape_coords'].copy()).int().cuda()
        shape=SparseTensor(feats=torch.from_numpy(archive['shape_feats'].copy()).cuda(),coords=coords)
    cameras=dict(camera_angle_x=row['camera_angle_x'],
        distance=[float(np.linalg.norm(np.asarray(t)[:3,3])) for t in row['transform_matrix']],
        mesh_scale=row['mesh_scale'],transform_matrix=row['transform_matrix'],grid_resolution_override=64)
    condition_checks={}
    conditions={}
    events={}
    with torch.inference_mode():
        conditions['native']=native(extractor,images,coords,**cameras)
        for name,model,physical in [('R-zero',r,False),('Direct-zero',d,True)]:
            pipe.get_proj_cond_shape=native
            events[name]=[]
            data=r_cfg['dataset']['args']
            install_generated_pbr_conditioning(pipe,conditioner=model,row=row,path_map=data['path_map'],
                match_root=data['match_root'],expected_q2c_sha256=data['expected_q2c_sha256'],
                physical_deformation_query=physical,events=events[name])
            conditions[name]=pipe.get_proj_cond_shape(extractor,images,coords,**cameras)
            checks={}
            for branch in ['cond','neg_cond']:
                for key in ['global','proj']:
                    a,b=conditions['native'][branch][key],conditions[name][branch][key]
                    a,b=(a.feats,b.feats) if key=='proj' else (a,b)
                    checks[branch+'/'+key]=dict(exact=torch.equal(a,b),maximum_difference=float((a.float()-b.float()).abs().max()))
            condition_checks[name]=checks
            (output/'condition-checks.json').write_text(json.dumps(condition_checks,indent=2)+'\n')
            if not all(row['exact'] for row in checks.values()):
                raise ValueError('actual native pipeline condition parity failed: '+name)
        # A declared temporary gate perturbation verifies delivery; it is not
        # a trained checkpoint and is never used for a quality comparison.
        perturbed=copy.deepcopy(r)
        perturbed.fusion_gate.fill_(.02)
        pipe.get_proj_cond_shape=native
        events['R-synthetic-gate']=[]
        data=r_cfg['dataset']['args']
        install_generated_pbr_conditioning(pipe,conditioner=perturbed,row=row,path_map=data['path_map'],
            match_root=data['match_root'],expected_q2c_sha256=data['expected_q2c_sha256'],
            physical_deformation_query=False,events=events['R-synthetic-gate'])
        conditions['R-synthetic-gate']=pipe.get_proj_cond_shape(extractor,images,coords,**cameras)
        difference=conditions['R-synthetic-gate']['cond']['proj'].feats.float()-conditions['native']['cond']['proj'].feats.float()
        delivery=dict(DINO_LR_maximum_change=float(difference[:,:1024].abs().max()),
            native_NAF_maximum_change=float(difference[:,1024:].abs().max()),
            native_global_exact=torch.equal(conditions['R-synthetic-gate']['cond']['global'],conditions['native']['cond']['global']),
            synthetic_gate_only=True,trained_model_performance_claim=False)
        if delivery['DINO_LR_maximum_change']<=0 or delivery['native_NAF_maximum_change']!=0 or not delivery['native_global_exact']:
            raise ValueError('native LR-only semantic delivery proof failed')
        extractor.cpu()
        flow=load_stage(cfg,plan['PBR_checkpoint']).cuda()
        samples={}
        noise_hashes={}
        for name,condition in conditions.items():
            torch.manual_seed(plan['replay_seed'])
            first=[]
            handle=flow.register_forward_pre_hook(lambda model,arguments:first.append(digest(arguments[0].feats)) if not first else None)
            try:
                sample=module.Pixal3DImageTo3DPipeline.sample_tex_slat(pipe,condition,flow,shape)
            finally:handle.remove()
            samples[name]=sample.feats.detach().cpu()
            noise_hashes[name]=first[0]
        if len(set(noise_hashes.values()))!=1:
            raise ValueError('native PBR replay actual initial noise must match in all four arms')
        for name in ['R-zero','Direct-zero']:
            if not torch.equal(samples['native'],samples[name]):
                raise ValueError('actual twelve-step native PBR replay zero identity failed: '+name)
        changed=float((samples['native']-samples['R-synthetic-gate']).abs().max())
        if changed<=0:raise ValueError('nonzero conditioner must reach the actual native sampler')
    report=dict(status='PASS_ACTUAL_GENERATED_SUPPORT_NATIVE_PBR_CONDITION_AND_SAMPLER',
        actual_generated_tokens=len(coords),target_ray_queries=1024,condition_checks=condition_checks,
        zero_R_and_Direct_native_twelve_step_replay_exact=True,actual_initial_noise_SHA256=noise_hashes,
        synthetic_gate_delivery=delivery,synthetic_gate_sampler_maximum_change=changed,
        sampler_class=spec['name'],sampler_params=spec['params'],
        native_sampler_noise_draw_device='CPU then copied to assigned GPU',CPU_and_CUDA_RNG_preserved_during_conditioning=True,
        transport_events=events,GT_sparse_coords_or_projection_cache_used=False,
        generated_shape_is_model_prediction=True,seconds=time.monotonic()-started,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,scientific_training_updates=0,
        trained_PBR1905_performance_not_measured=True,model_adopted=False,full_goal_complete=False)
    (output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],peak_GiB=report['peak_GiB'],tokens=len(coords))),flush=True)


if __name__=='__main__':
    main()
