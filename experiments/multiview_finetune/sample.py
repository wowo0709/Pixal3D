import json
import os
import sys
import time
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import argparse
from configure import REPO as SOURCE, STAGES
from paths import single_visible_device, official_snapshot, checkpoint_path
parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True, help='Checkpoint root with one directory per stage')
parser.add_argument('--output-root', type=Path, help='Samples directory; default: RESULT/samples')
parser.add_argument('--download-report', type=Path, default=SOURCE / 'data/pixal3d/checkpoints/download_report.json')
parser.add_argument('--official-snapshot', type=Path, help='Pixal3D snapshot with pipeline.json and ckpts/*.json; otherwise use download report')
parser.add_argument('--step', type=int, default=3, help='Checkpoint update (default: three-update debug checkpoint)')
parser.add_argument('--stage-steps', type=int, nargs=4, metavar=('SS64', 'SHAPE512', 'SHAPE1024', 'PBR1024'),
                    help='Override shared --step with explicit per-stage updates in recipe order')
parser.add_argument('--configs-dir', type=Path, help='Explicit stage JSON directory; default: per-stage native config.json')
parser.add_argument('--dino', help='Override saved DINO identifier/local path when relocating a run')
parser.add_argument('--ema', help='EMA suffix, e.g. 0.9999; default: raw weights')
parser.add_argument('--memory-budget-gib', type=float, help='Optional PyTorch allocator budget')
parser.add_argument('--min-free-gib', type=float, help='Optional preflight minimum free memory')
parser.add_argument('--batch', type=Path, required=True)
parser.add_argument('--asset', required=True)
parser.add_argument('--views', type=int, nargs='+', choices=[1,4], required=True)
parser.add_argument('--anchor', type=int, choices=[0,1], default=0,
                    help='Cached anchor frame to put first; use its exact prepared scale')
args = parser.parse_args()
DATA = args.batch
single_visible_device()
sys.path.insert(0, str(SOURCE))
import numpy as np
import torch
from PIL import Image
from pixal3d import models
from experiments.multiview_finetune.model_loading import load_run_denoiser
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
from pixal3d.pipelines import samplers
from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
from pixal3d.utils import render_utils
from inference import load_calibrated_manifest
import o_voxel

torch.set_num_threads(4)
free, total = torch.cuda.mem_get_info()
if args.min_free_gib is not None and free < args.min_free_gib * 2**30:
    raise RuntimeError(f'Sampling requires {args.min_free_gib} GiB free; currently {free / 2**30:.2f} GiB')
if args.memory_budget_gib is not None:
    torch.cuda.set_per_process_memory_fraction(args.memory_budget_gib * 2**30 / total)
root = args.result
stage_steps = dict(zip(STAGES, args.stage_steps or [args.step] * 4))
downloads = json.loads(args.download_report.read_text())
official = official_snapshot(args.official_snapshot, args.download_report)
pipeline_args = json.loads((official / 'pipeline.json').read_text())['args']
stage_keys = {'ss64':'sparse_structure_flow_model', 'shape512':'shape_slat_flow_model_512',
    'shape1024':'shape_slat_flow_model_1024', 'pbr1024':'tex_slat_flow_model_1024'}
loaded_models = {}
checkpoint_paths = {}
for stage, key in stage_keys.items():
    stem = STAGES[stage][2]
    config_path = args.configs_dir / f'{stage}.json' if args.configs_dir else root / stage / 'config.json'
    config = json.loads(config_path.read_text())
    checkpoint = checkpoint_path(root / stage, stage_steps[stage], args.ema)
    model = load_run_denoiser(config, checkpoint)
    loaded_models[key] = model
    checkpoint_paths[stage] = str(checkpoint)
    print('LOADED_FINETUNED', stage, checkpoint, flush=True)
for stage, key in [('ss','sparse_structure_decoder'),('shape','shape_slat_decoder'),('pbr','tex_slat_decoder')]:
    loaded_models[key] = models.from_pretrained(downloads['files'][stage + '_decoder'])
    print('LOADED_DECODER', stage, flush=True)
sampler_objects = {}
sampler_params = {}
for name in ['sparse_structure','shape_slat','tex_slat']:
    cfg = pipeline_args[name + '_sampler']
    sampler_objects[name + '_sampler'] = getattr(samplers, cfg['name'])(**cfg['args'])
    sampler_params[name + '_sampler_params'] = cfg['params']
pipeline = Pixal3DImageTo3DPipeline(loaded_models, **sampler_objects, **sampler_params,
    shape_slat_normalization=pipeline_args['shape_slat_normalization'],
    tex_slat_normalization=pipeline_args['tex_slat_normalization'], low_vram=True,
    default_pipeline_type='1024_cascade')
for stage, attr in [('ss64','image_cond_model_ss'),('shape512','image_cond_model_shape_512'),
    ('shape1024','image_cond_model_shape_1024'),('pbr1024','image_cond_model_tex_1024')]:
    config_path = args.configs_dir / f'{stage}.json' if args.configs_dir else root / stage / 'config.json'
    cfg = json.loads(config_path.read_text())['trainer']['args']['image_cond_model']['args']
    if args.dino is not None:
        cfg['model_name'] = args.dino
    extractor = DinoV3ProjFeatureExtractor(**cfg).eval()
    if extractor.use_naf_upsample:
        extractor._load_naf()
    # Keep BF16 AMP local to frozen DINO/NAF. The sparse VAE torso is FP16;
    # global BF16 AMP promotes its BF16 + FP16 residuals to FP32.
    extractor.forward = torch.autocast('cuda', dtype=torch.bfloat16)(extractor.forward)
    setattr(pipeline, attr, extractor)
pipeline.to(torch.device('cuda:0'))

def generate(views):
    out = (args.output_root or root / 'samples') / (args.asset[:12] + '-K' + str(views))
    out.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    sha = args.asset
    manifest = DATA / 'renders_cond' / sha / 'transforms.json'
    scale = json.loads((DATA / 'ss_latents/ss_enc_conv3d_16l8_fp16_64_view' / sha / f'view{args.anchor:02d}_scale.json').read_text())['total_scale']
    images, cameras = load_calibrated_manifest(manifest, mesh_scale=scale)
    indices = ([args.anchor] + [i for i in range(len(images)) if i != args.anchor])[:views]
    images = [images[i] for i in indices]
    for key in ['camera_angle_x','distance','transform_matrix']:
        cameras[key] = [cameras[key][i] for i in indices]
    # Prepared frames already have calibrated intrinsics. Match the training loader's
    # black alpha composite, without the CLI's extra bounding-box crop.
    rgb_images = []
    for image in images:
        rgba = image.convert('RGBA')
        rgb = Image.new('RGB', rgba.size, (0,0,0))
        rgb.paste(rgba, mask=rgba.getchannel('A'))
        rgb_images.append(rgb)
    contact = Image.new('RGB', (512*views,512))
    for i, image in enumerate(rgb_images):
        contact.paste(image.resize((512,512)), (i*512,0))
    contact.save(out / 'input_views.png')
    evidence = {'asset':sha, 'input_manifest':str(manifest), 'view_indices':indices,
        'cached_anchor':args.anchor, 'scale_method':'exact prepared scale for the first cached anchor',
        'mesh_scale':scale, 'checkpoints':checkpoint_paths, 'pipeline_type':'1024_cascade',
        'seed':42, 'sampler_steps':12, 'preprocess_image':False,
        'render_convention':'Y-up anchor frame; front/right/back/left; distance=2; FOV=40 degrees',
        'torch_allocator_budget_gib':args.memory_budget_gib, 'checkpoint_steps':stage_steps,
        'weights':'raw' if args.ema is None else 'ema' + args.ema,
        'scope':'Functional sample from a held-out mixed-source batch; checkpoint provenance is recorded above.'}
    (out / 'sample_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('GENERATING', json.dumps(evidence), flush=True)
    with torch.inference_mode():
        meshes, (shape, texture, resolution) = pipeline.run(rgb_images, camera_params=cameras,
            seed=42, preprocess_image=False, return_latent=True,
            sparse_structure_sampler_params={'steps':12}, shape_slat_sampler_params={'steps':12}, tex_slat_sampler_params={'steps':12},
            pipeline_type='1024_cascade', max_num_tokens=32768)
    mesh = meshes[0]
    assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
    for values in [mesh.vertices, mesh.attrs, shape.feats, texture.feats]:
        assert torch.isfinite(values).all()
    torch.save({'shape_feats':shape.feats.cpu(), 'shape_coords':shape.coords.cpu(),
        'texture_feats':texture.feats.cpu(), 'texture_coords':texture.coords.cpu(),
        'resolution':resolution}, out / 'generated_latents.pt')
    print('GENERATED_MESH', len(mesh.vertices), len(mesh.faces), resolution, flush=True)
    import cv2
    from pixal3d.renderers import EnvMap
    environment = cv2.cvtColor(cv2.imread(str(SOURCE / 'assets/hdri/studio.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.from_numpy(environment).float().cuda())
    renders = render_utils.render_proj_aligned_video(mesh, camera_angle_x=np.deg2rad(40),
        distance=2, resolution=512, num_frames=4, envmap=envmap)
    assert any(np.any(frame > 0) for frame in renders['mask']), 'Generated mesh rendered blank'
    for key in ['base_color','normal']:
        grid = Image.new('RGB', (512*4,512))
        for i, frame in enumerate(renders[key]):
            grid.paste(Image.fromarray(frame), (i*512,0))
        grid.save(out / (key + '_views.png'))
    print('RENDERED', list(renders), flush=True)
    glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout, grid_size=resolution,
        aabb=[[-.5,-.5,-.5],[.5,.5,.5]], decimation_target=100000, texture_size=1024,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
    glb.export(str(out / 'finetuned_multiview.glb'), extension_webp=True)
    torch.cuda.synchronize()
    evidence.update(actual_resolution=resolution, vertices=len(mesh.vertices), faces=len(mesh.faces),
        shape_tokens=len(shape.feats), texture_tokens=len(texture.feats),
        glb_bytes=(out / 'finetuned_multiview.glb').stat().st_size,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        elapsed_seconds=time.monotonic()-began, complete=True)
    (out / 'sample_manifest.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print('SAMPLE_COMPLETE', json.dumps(evidence), flush=True)

for views in args.views:
    generate(views)
    torch.cuda.empty_cache()
