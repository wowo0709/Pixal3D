"""Run the original Pixal3D single-view stack on edited target RGB only.

The input and RNG come from the completed frozen TEST508 four-view campaign.
Only camera 8 is passed to the official pretrained stage models. Ground-truth
geometry is loaded after generation solely for the same CD/F-Score protocol.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def make_pipeline(first: dict):
    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline
    from pixal3d.pipelines import pixal3d_image_to_3d as module
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from .cascade_artifact_inputs import prepared_views

    root = Path(first['official_snapshot'])
    config = json.loads((root / 'pipeline.json').read_text())['args']
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(str(root))
    expected = {
        'sparse_structure_flow_model': 'ss_flow_img_dit_1_3B_64_bf16',
        'shape_slat_flow_model_512': 'slat_flow_img2shape_dit_1_3B_512_bf16',
        'shape_slat_flow_model_1024': 'slat_flow_img2shape_dit_1_3B_1024_bf16',
        'tex_slat_flow_model_1024': 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16',
    }
    if any(Path(config['models'][name]).name != stem for name, stem in expected.items()):
        raise ValueError('official Pixal3D single-view model mapping changed')
    from safetensors import safe_open
    for name, stem in expected.items():
        model = pipeline.models[name]
        with safe_open(str(root / 'ckpts' / (stem + '.safetensors')), framework='pt', device='cpu') as saved:
            keys = set(saved.keys())
            current = model.state_dict()
            if keys != set(current) or any(tuple(saved.get_slice(k).get_shape()) != tuple(current[k].shape) for k in keys):
                raise ValueError('official single-view stage has missing or mismatched tensor: ' + name)
        model.requires_grad_(False).eval()
        model.forward = torch.autocast('cuda', dtype=torch.bfloat16)(model.forward)
    stage_map = {
        'ss64': 'image_cond_model_ss',
        'shape512': 'image_cond_model_shape_512',
        'shape1024': 'image_cond_model_shape_1024',
        'pbr1024': 'image_cond_model_tex_1024',
    }
    for stage, attr in stage_map.items():
        native = json.loads(Path(first['stage_configs'][stage]).read_text())
        args = dict(native['trainer']['args']['image_cond_model']['args'])
        args['model_name'] = first['DINO_local_path']
        if args['image_size'] != native['dataset']['args']['image_size']:
            raise ValueError('native image/extractor size mismatch: ' + stage)
        extractor = DinoV3ProjFeatureExtractor(**args).eval().requires_grad_(False)
        if extractor.use_naf_upsample:
            extractor._load_naf()
        extractor.forward = torch.autocast('cuda', dtype=torch.bfloat16)(extractor.forward)
        setattr(pipeline, attr, extractor)
    module.pil_views_to_tensor = prepared_views
    pipeline.to(torch.device('cuda:0'))
    return pipeline


def run_case(pipeline, case: dict, output: Path):
    from .surface_metrics import compare_surfaces, sample_surface
    from data_toolkit.utils import transform_mesh

    if output.exists():
        report = output / 'generation.json'
        if report.is_file() and json.loads(report.read_text()).get('status') == 'PASS_ORIGINAL_PIXAL3D_SINGLE_VIEW_GENERATION':
            return
        raise FileExistsError('incomplete output requires explicit review: ' + str(output))
    row = case['row']
    sid = row['sample_id']
    target_rgb = Path(case['RGB_paths'][0])
    if not target_rgb.is_file() or sha256(target_rgb) != case['target_RGB_SHA256']:
        raise ValueError('edited target RGB changed: ' + sid)
    if case['input_states'] != ['M1', 'M0', 'M0', 'M0'] or row['conditioning_camera_ids'] != [8, 9, 10, 11]:
        raise ValueError('the source campaign does not have the declared target8 input')
    if len(case['RGB_paths']) != 4 or case['RGB_paths'][0] != case['RGB_path_map'][row['target_anchor']]:
        raise ValueError('target camera input differs from frozen four-view record')
    camera = dict(camera_angle_x=[row['camera_angle_x'][0]],
        distance=[float(np.linalg.norm(np.asarray(row['transform_matrix'][0])[:3, 3]))],
        transform_matrix=[row['transform_matrix'][0]], mesh_scale=row['mesh_scale'])
    output.mkdir(parents=True)
    with Image.open(target_rgb) as image:
        target_image = image.copy()
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        meshes, latents = pipeline.run([target_image], camera, seed=int(case['seed']),
            preprocess_image=False, return_latent=True, pipeline_type='1024_cascade', max_num_tokens=32768)
    if len(meshes) != 1 or latents[2] != 1024:
        raise ValueError('one actual 1024 cascade mesh required')
    mesh = meshes[0]
    shape, texture, resolution = latents
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError('empty single-view prediction')
    np.savez_compressed(output/'prediction.npz', vertices=mesh.vertices.cpu().numpy(),
        faces=mesh.faces.cpu().numpy(), coords=mesh.coords.cpu().numpy(),
        attrs=mesh.attrs.cpu().numpy(), origin=mesh.origin.cpu().numpy(),
        voxel_size=np.asarray(mesh.voxel_size), voxel_shape=np.asarray(mesh.voxel_shape),
        shape_coords=shape.coords.cpu().numpy(), shape_feats=shape.feats.float().cpu().numpy(),
        texture_feats=texture.feats.float().cpu().numpy())
    generated_seconds = time.monotonic() - start
    with np.load(case['source_geometry']) as source:
        vertices, faces = source['V'].copy(), source['F'].copy()
    target = np.load(case['target_vertices'])
    if vertices.shape != target.shape or target.shape[1] != 3:
        raise ValueError('paired source/target topology differs')
    basis = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    transformed = transform_mesh(basis, dict(transform_matrix=row['transform_matrix'][0]))
    rotation = (transformed[1:] - transformed[0]).T.numpy()
    if not np.allclose(transformed[0].numpy(), 0, atol=1e-5) or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5):
        raise ValueError('native anchor frame must be a pure rotation')
    prediction_world = mesh.vertices.float().cpu().numpy() @ rotation / row['mesh_scale']
    center = (vertices.min(0) + vertices.max(0))/2
    extent = float(np.ptp(vertices,axis=0).max())
    pred_points = sample_surface((prediction_world-center)/extent,mesh.faces.cpu().numpy(),seed=case['surface_seed'])
    target_points = sample_surface((target-center)/extent,faces,seed=case['surface_seed'])
    geometry = compare_surfaces(pred_points,target_points,threshold=.02)
    np.savez_compressed(output/'geometry-evaluation.npz',prediction_world=prediction_world,
        target_world=target,target_faces=faces,source_center=center,source_extent=extent,
        anchor_rotation=rotation)
    result = dict(status='PASS_ORIGINAL_PIXAL3D_SINGLE_VIEW_GENERATION',sample_id=sid,
        source_cluster_id=row['source_cluster_id'],input_states=['M1'],input_camera_ids=[8],
        original_single_view_stage_weights=True,original_multiview_stage_weights=False,
        deformation_finetuning_updates=0,target_RGB_path=str(target_rgb),target_RGB_SHA256=case['target_RGB_SHA256'],
        seed=case['seed'],surface_seed=case['surface_seed'],GT_geometry_or_coords_used_in_generation=False,
        GT_loaded_only_after_generation=True,actual_resolution=resolution,
        generation_seconds=generated_seconds,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        metrics=geometry,source_defined_normalization=True,per_output_alignment=False,
        sampling='official Pixal3D pipeline.json 12-step defaults',full_evaluation_complete=False)
    (output/'generation.json').write_text(json.dumps(result,indent=2)+'\n')
    del meshes,latents,mesh,shape,texture
    gc.collect();torch.cuda.empty_cache()
    print(json.dumps(dict(sample_id=sid,status=result['status'],seconds=generated_seconds,
        CD=geometry['chamfer_L1'],FScore=geometry['surface_FScore'])),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in {'0','1','2','3'} or visible not in os.environ.get('MATCHING_V2_ALLOWED_GPUS','').split(','):
        raise ValueError('one authorized physical GPU0-3 required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.8,0)
    assignment=json.loads(args.assignment.read_text())
    if not assignment or len({c['row']['sample_id'] for c in assignment})!=len(assignment):
        raise ValueError('nonempty duplicate-free frozen assignment required')
    pipeline=make_pipeline(assignment[0])
    failures=[]
    for case in assignment:
        sid=case['row']['sample_id']
        try:
            run_case(pipeline,case,args.output_root/sid/'geometry')
        except Exception as exc:
            path=args.output_root/sid/'generation-failure.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(dict(sample_id=sid,error=repr(exc),traceback=traceback.format_exc()),indent=2)+'\n')
            failures.append(sid)
            print(json.dumps(dict(sample_id=sid,status='FAILED_SINGLE_VIEW_CASE',error=repr(exc))),flush=True)
            gc.collect();torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f'{len(failures)} single-view cases failed; inspect generation-failure.json files')


if __name__=='__main__':
    main()
