"""Generate saved single-view/multi-view comparisons using this checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

SOURCE = Path(__file__).resolve().parents[1]
STAGE_MODELS = {
    'ss64': ('sparse_structure_flow_model', 'ss_flow_img_dit_1_3B_64_bf16'),
    'shape512': ('shape_slat_flow_model_512', 'slat_flow_img2shape_dit_1_3B_512_bf16'),
    'shape1024': ('shape_slat_flow_model_1024', 'slat_flow_img2shape_dit_1_3B_1024_bf16'),
    'pbr1024': ('tex_slat_flow_model_1024', 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16'),
}


def require_single_visible_gpu():
    """Use logical cuda:0 for one explicitly assigned physical ID or UUID."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if not visible or ',' in visible or visible == '-1':
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES to exactly one assigned GPU ID or UUID')
    return 'cuda:0'


def resolve_weight_paths(weights_root, official_root=None, dino_model=None):
    """Resolve report-relative files; retain absolute external weight pointers."""
    weights_root = Path(weights_root).resolve()
    downloads = json.loads((weights_root / 'download_report.json').read_text())

    def report_path(value):
        path = Path(value)
        return str(path if path.is_absolute() else weights_root / path)

    for stage, (_, stem) in STAGE_MODELS.items():
        downloads['files'][stem] = {key: report_path(value)
                                    for key, value in downloads['files'][stem].items()}
    for stage in ['ss', 'shape', 'pbr']:
        downloads['files'][stage + '_decoder'] = report_path(downloads['files'][stage + '_decoder'])
    downloads['dino'] = str(Path(dino_model).resolve()) if dino_model else report_path(downloads['dino'])
    if official_root is None:
        hf_home = Path(os.environ.get('HF_HOME',
                       str(Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'huggingface')))
        hub = Path(os.environ.get('HF_HUB_CACHE', os.environ.get('HUGGINGFACE_HUB_CACHE', str(hf_home / 'hub'))))
        official = hub / 'models--TencentARC--Pixal3D/snapshots' / downloads['revisions']['TencentARC/Pixal3D']
    else:
        official = Path(official_root).resolve()
        for _, stem in STAGE_MODELS.values():
            downloads['files'][stem]['safetensors'] = str(official / 'ckpts' / (stem + '.safetensors'))
    return official, downloads


def load_local_decoder(prefix):
    """Load the native decoder config/weights locally without HF fallback."""
    prefix = str(prefix)
    config = json.loads(Path(prefix + '.json').read_text())
    from safetensors.torch import load_file
    from pixal3d import models
    model = getattr(models, config['name'])(**config['args'])
    model.load_state_dict(load_file(prefix + '.safetensors', device='cpu'), strict=False)
    return model.eval().requires_grad_(False)


def fingerprint(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--finetuned-only', action='store_true')
    parser.add_argument('--weights-root', type=Path, default=SOURCE / 'data/pixal3d/checkpoints',
                        help='Directory containing download_report.json')
    parser.add_argument('--official-root', type=Path,
                        help='TencentARC/Pixal3D snapshot with pipeline.json and ckpts/')
    parser.add_argument('--dino-model', type=Path, help='Override the report DINO snapshot directory')
    parser.add_argument('--finetuned-root', type=Path, required=True,
                        help='Run root containing <stage>/ckpts/denoiser_stepNNNNNNN.pt')
    parser.add_argument('--checkpoint-step', type=int, default=10)
    parser.add_argument('--memory-budget-gib', type=float,
                        help='Optional PyTorch allocator limit; other libraries allocate separately')
    args = parser.parse_args()
    device = require_single_visible_gpu()
    args.selection = args.selection.resolve()
    args.output_dir = args.output_dir.resolve()
    args.finetuned_root = args.finetuned_root.resolve()
    official, downloads = resolve_weight_paths(args.weights_root, args.official_root, args.dino_model)
    source_revision = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    sys.path.insert(0, str(SOURCE))
    os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
    from result_viewer.conditioning import prepare_conditioning

    import cv2
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from pixal3d import models
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline, samplers
    from pixal3d.renderers import EnvMap
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    from pixal3d.utils import render_utils
    from inference import IMAGE_COND_CONFIGS
    import o_voxel

    torch.set_num_threads(4)
    if args.memory_budget_gib is not None:
        _, total = torch.cuda.mem_get_info(0)
        torch.cuda.set_per_process_memory_fraction(args.memory_budget_gib * 2**30 / total, 0)
    selection = json.loads(args.selection.read_text())
    cases = selection['cases'][args.start_index:args.start_index + args.count]
    # Relative prepared manifests are owned by the selection, not the process cwd.
    for case in cases:
        path = Path(case['input_manifest'])
        case['input_manifest'] = str(path if path.is_absolute() else args.selection.parent / path)
    pipeline_args = json.loads((official / 'pipeline.json').read_text())['args']
    stage_keys = {stage: key for stage, (key, _) in STAGE_MODELS.items()}
    loaded, checkpoints = {}, {}
    for stage, key in stage_keys.items():
        stem = STAGE_MODELS[stage][1]
        config_path = official / 'ckpts' / (stem + '.json')
        config = json.loads(config_path.read_text())
        checkpoint = Path(downloads['files'][stem]['safetensors'])
        model = getattr(models, config['name'])(**config['args'])
        incompatible = model.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=False)
        assert set(incompatible.missing_keys) <= {'rope_phases'} and not incompatible.unexpected_keys
        loaded[key] = model.eval().requires_grad_(False)
        checkpoints[stage] = {'path': str(checkpoint), 'sha256': fingerprint(checkpoint),
            'config': str(config_path), 'config_sha256': fingerprint(config_path)}
        print('LOADED_PRETRAINED', stage, flush=True)
    for stage, key in [('ss', 'sparse_structure_decoder'), ('shape', 'shape_slat_decoder'), ('pbr', 'tex_slat_decoder')]:
        loaded[key] = load_local_decoder(downloads['files'][stage + '_decoder'])
    sampler_objects, sampler_params = {}, {}
    for name in ['sparse_structure', 'shape_slat', 'tex_slat']:
        config = pipeline_args[name + '_sampler']
        sampler_objects[name + '_sampler'] = getattr(samplers, config['name'])(**config['args'])
        sampler_params[name + '_sampler_params'] = {**config['params'], 'steps': 12}
    pipeline = Pixal3DImageTo3DPipeline(loaded, **sampler_objects, **sampler_params,
        shape_slat_normalization=pipeline_args['shape_slat_normalization'],
        tex_slat_normalization=pipeline_args['tex_slat_normalization'], low_vram=True,
        default_pipeline_type='1024_cascade')
    extractor_configs = {}
    for stage, attr in [('ss', 'image_cond_model_ss'), ('shape_512', 'image_cond_model_shape_512'),
                        ('shape_1024', 'image_cond_model_shape_1024'), ('tex_1024', 'image_cond_model_tex_1024')]:
        config = {**IMAGE_COND_CONFIGS[stage], 'model_name': downloads['dino']}
        extractor_configs[stage] = config
        extractor = DinoV3ProjFeatureExtractor(**config).eval().requires_grad_(False)
        if extractor.use_naf_upsample:
            extractor._load_naf()
        # Never apply global BF16 autocast to the native FP16 sparse decoders.
        extractor.forward = torch.autocast('cuda', dtype=torch.bfloat16)(extractor.forward)
        setattr(pipeline, attr, extractor)
    pipeline.to(torch.device(device))
    environment = cv2.cvtColor(cv2.imread(str(SOURCE / 'assets/hdri/studio.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
    envmap = EnvMap(torch.from_numpy(environment).float().cuda())
    progress_path = args.output_dir / f'progress-{args.start_index:02d}.json'
    completed = []
    expected_outputs = len(cases) * (1 if args.finetuned_only else 3)

    def generate(case, mode, fine_tuned):
        indices = case['view_indices'][:1] if mode == 'pretrained-sv' else case['view_indices']
        images, cameras = prepare_conditioning(case['input_manifest'], indices, case['mesh_scale'])
        if mode == 'pretrained-sv':
            cameras.pop('transform_matrix')  # Original single-view conditioning branch.
        out = args.output_dir / case['id'] / mode
        out.mkdir(parents=True)
        evidence = {**case, 'view_indices': indices, 'camera_params': cameras,
            'mode': mode, 'fine_tuned': fine_tuned, 'optimizer_updates': args.checkpoint_step if fine_tuned else 0,
            'weight_origin': f'{args.checkpoint_step}-step fine-tune from official single-view weights' if fine_tuned else 'Official TencentARC/Pixal3D SINGLE-VIEW pretrained weights',
            'checkpoints': checkpoints, 'revisions': downloads['revisions'],
            'source': str(SOURCE), 'source_revision': source_revision,
            'input_manifest_sha256': fingerprint(case['input_manifest']),
            'extractor_configs': extractor_configs, 'pipeline_type': '1024_cascade', 'seed': 42,
            'sampler_steps': 12, 'sampler_params': sampler_params, 'max_num_tokens': 32768,
            'preprocess_image': False, 'background': 'Black alpha composite; calibrated frames unchanged',
            'scope': selection['scope'],
            'render_convention': 'Y-up anchor frame; front/right/back/left, distance=2, FOV=40 degrees',
            'glb_convention': 'o_voxel exporter axis conversion; viewer applies +90 degrees about X to restore native Y-up',
            'complete': False}
        manifest = out / 'sample_manifest.json'
        manifest.write_text(json.dumps(evidence, indent=2) + '\n')
        contact = Image.new('RGB', (512 * len(images), 512))
        for index, image in enumerate(images):
            contact.paste(image.resize((512, 512)), (index * 512, 0))
        contact.save(out / 'input_views.png')
        print('GENERATING', case['id'], mode, 'views', indices, flush=True)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            meshes, (shape, texture, resolution) = pipeline.run(images, camera_params=cameras,
                seed=42, preprocess_image=False, return_latent=True,
                pipeline_type='1024_cascade', max_num_tokens=32768)
        mesh = meshes[0]
        assert len(mesh.vertices) > 0 and len(mesh.faces) > 0
        for values in [mesh.vertices, mesh.attrs, shape.feats, texture.feats]:
            assert torch.isfinite(values).all()
        torch.save({'shape_feats': shape.feats.cpu(), 'shape_coords': shape.coords.cpu(),
            'texture_feats': texture.feats.cpu(), 'texture_coords': texture.coords.cpu(), 'resolution': resolution},
            out / 'generated_latents.pt')
        renders = render_utils.render_proj_aligned_video(mesh, camera_angle_x=np.deg2rad(40),
            distance=2, resolution=512, num_frames=4, envmap=envmap)
        assert any(np.any(frame > 0) for frame in renders['mask']), 'Generated mesh rendered blank'
        for key in ['base_color', 'normal']:
            grid = Image.new('RGB', (512 * 4, 512))
            for index, frame in enumerate(renders[key]):
                grid.paste(Image.fromarray(frame), (index * 512, 0))
            grid.save(out / (key + '_views.png'))
        glb = o_voxel.postprocess.to_glb(vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout, grid_size=resolution,
            aabb=[[-.5, -.5, -.5], [.5, .5, .5]], decimation_target=100000, texture_size=1024,
            remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True)
        output_glb = out / 'model.glb'
        glb.export(str(output_glb), extension_webp=True)
        torch.cuda.synchronize()
        evidence.update(actual_resolution=resolution, vertices=len(mesh.vertices), faces=len(mesh.faces),
            shape_tokens=len(shape.feats), texture_tokens=len(texture.feats), glb_bytes=output_glb.stat().st_size,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, seconds=time.monotonic() - started,
            output_sha256={name: fingerprint(out / name) for name in ['base_color_views.png', 'normal_views.png',
                'input_views.png', 'generated_latents.pt', 'model.glb']}, complete=True)
        manifest.write_text(json.dumps(evidence, indent=2) + '\n')
        completed.append({'case': case['id'], 'mode': mode, 'seconds': evidence['seconds'],
                          'peak_allocated_gib': evidence['peak_allocated_gib']})
        progress_path.write_text(json.dumps({'status': 'running', 'expected_outputs': expected_outputs,
            'completed': completed}, indent=2) + '\n')
        print('SAMPLE_COMPLETE', json.dumps(completed[-1]), flush=True)
        torch.cuda.empty_cache()

    if not args.finetuned_only:
        for case in cases:
            generate(case, 'pretrained-sv', False)
            generate(case, 'pretrained-mv', False)
    # Reuse fixed decoders/extractors and replace only the four denoisers.
    checkpoints = {}
    for stage, key in stage_keys.items():
        checkpoint = args.finetuned_root / stage / 'ckpts' / f'denoiser_step{args.checkpoint_step:07d}.pt'
        # The first sampling pass lazily creates inference-tensor RoPE buffers.
        # Their in-place replacement must remain inside inference mode.
        with torch.inference_mode():
            incompatible = pipeline.models[key].cpu().load_state_dict(
                torch.load(checkpoint, map_location='cpu', weights_only=True), strict=False)
        assert set(incompatible.missing_keys) <= {'rope_phases'} and not incompatible.unexpected_keys
        checkpoints[stage] = {'path': str(checkpoint), 'sha256': fingerprint(checkpoint)}
        print('LOADED_FINETUNED', stage, flush=True)
    for case in cases:
        generate(case, 'finetuned-mv', True)
    progress_path.write_text(json.dumps({'status': 'complete', 'expected_outputs': expected_outputs,
        'completed': completed}, indent=2) + '\n')
    print('ALL_CASES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
