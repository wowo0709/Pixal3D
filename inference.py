import os
import argparse
import json
import math
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import cv2
from PIL import Image

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("ATTN_BACKEND", "flash_attn")
os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'autotune_cache.json')
os.environ["FLEX_GEMM_AUTOTUNER_VERBOSE"] = '1'

from pixal3d.pipelines import Pixal3DImageTo3DPipeline
import o_voxel

# ============================================================================
# Constants & Defaults
# ============================================================================

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
MODEL_PATH = "TencentARC/Pixal3D"

IMAGE_COND_CONFIGS = {
    "ss": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 16,
    },
    "shape_512": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "model_name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}

# ============================================================================
# Model Loading
# ============================================================================

def build_image_cond_model(config: dict):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import DinoV3ProjFeatureExtractor
    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    return model


def load_moge_model(device="cuda", model_name=MOGE_MODEL_NAME):
    from moge.model.v2 import MoGeModel
    moge_model = MoGeModel.from_pretrained(model_name)
    moge_model = moge_model.to(device)
    moge_model.eval()
    return moge_model


def init_pipeline(model_path=MODEL_PATH, device="cuda", low_vram=False):
    print(f"[Pipeline] Loading from {model_path}...")
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)

    print("[ImageCond] Building DinoV3ProjFeatureExtractor models...")
    pipeline.image_cond_model_ss = build_image_cond_model(IMAGE_COND_CONFIGS["ss"])
    pipeline.image_cond_model_shape_512 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_512"])
    pipeline.image_cond_model_shape_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["shape_1024"])
    pipeline.image_cond_model_tex_1024 = build_image_cond_model(IMAGE_COND_CONFIGS["tex_1024"])

    if low_vram:
        # Low-VRAM mode: models stay on CPU, loaded to GPU on-demand per stage.
        # Peak VRAM = one flow model + one DinoV3, not all ~18 GB at once.
        print("[NAF] Pre-downloading NAF upsampler weights (CPU only)...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        pipeline._device = torch.device(device)
        pipeline.low_vram = True
        print("[Pipeline] Low-VRAM mode enabled.")
    else:
        # Standard mode: all models loaded to GPU at once (faster, needs more VRAM).
        pipeline.low_vram = False
        pipeline.cuda()
        pipeline.image_cond_model_ss.cuda()
        pipeline.image_cond_model_shape_512.cuda()
        pipeline.image_cond_model_shape_1024.cuda()
        pipeline.image_cond_model_tex_1024.cuda()
        print("[NAF] Pre-loading NAF upsampler model...")
        for attr in ['image_cond_model_ss', 'image_cond_model_shape_512',
                     'image_cond_model_shape_1024', 'image_cond_model_tex_1024']:
            m = getattr(pipeline, attr, None)
            if m is not None and getattr(m, 'use_naf_upsample', False):
                m._load_naf()
        print("[Pipeline] Standard mode (all models on GPU).")

    return pipeline


FLOW_MODEL_KEYS = (
    "sparse_structure_flow_model",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "tex_slat_flow_model_1024",
)


def load_flow_overrides(pipeline, overrides):
    unknown = set(overrides) - set(FLOW_MODEL_KEYS)
    if unknown:
        raise ValueError(f"unknown flow checkpoint overrides: {sorted(unknown)}")
    for model_key in FLOW_MODEL_KEYS:
        checkpoint = overrides.get(model_key)
        if checkpoint is None:
            continue
        checkpoint = Path(checkpoint)
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        incompatible = pipeline.models[model_key].load_state_dict(
            state_dict, strict=False
        )
        missing = set(incompatible.missing_keys)
        if missing - {"rope_phases"} or incompatible.unexpected_keys:
            raise RuntimeError(
                f"incompatible {model_key} checkpoint {checkpoint}: "
                f"missing={sorted(missing)} "
                f"unexpected={incompatible.unexpected_keys}"
            )
        print(f"[Pipeline] Loaded {model_key}: {checkpoint}")


# ============================================================================
# Camera Estimation
# ============================================================================

def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / torch.tan(torch.tensor(camera_angle_x / 2.0))
    f_pixels = focal_length * resolution / 32.0
    return float(f_pixels.item())


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    gp = grid_point.to(torch.float32) @ rotation_matrix.T
    gp = gp / mesh_scale / 2
    xw, yw, zw = gp[0].item(), gp[1].item(), gp[2].item()
    xt, yt = float(target_point[0].item()), float(target_point[1].item())
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    y_ndc = -(yt - image_resolution / 2.0)
    distance_x = f_pixels * xw / x_ndc - yw
    return {"distance_from_x": float(distance_x), "f_pixels": float(f_pixels)}


def get_camera_params_wild_moge(image_path, moge_model, device="cuda", mesh_scale=1.0, extend_pixel=0, image_resolution=512):
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        output = moge_model.infer(image_tensor)
    intrinsics = output["intrinsics"].squeeze().cpu().numpy()
    fx_normalized = intrinsics[0, 0]
    fx = fx_normalized * width
    camera_angle_x = 2 * math.atan(width / (2 * fx))

    grid_point = torch.tensor([-1.0, 0.0, 0.0])
    distance = distance_from_fov(
        camera_angle_x, grid_point,
        torch.tensor([0 - extend_pixel, image_resolution - 1 + extend_pixel]),
        mesh_scale, image_resolution
    )["distance_from_x"]
    return {'camera_angle_x': camera_angle_x, 'distance': distance, 'mesh_scale': mesh_scale}


def load_calibrated_manifest(path, *, mesh_scale):
    path = Path(path).resolve()
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("transforms.json must contain a JSON object")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 8:
        raise ValueError("transforms.json must contain between 1 and 8 frames")

    if mesh_scale is None:
        warnings.warn(
            "mesh_scale was not provided; assuming canonical unit scale (1.0)",
            UserWarning,
            stacklevel=2,
        )
        mesh_scale = 1.0
    try:
        mesh_scale = float(mesh_scale)
    except (TypeError, ValueError):
        raise ValueError("mesh_scale must be finite and positive") from None
    if not np.isfinite(mesh_scale) or mesh_scale <= 0:
        raise ValueError("mesh_scale must be finite and positive")

    images = []
    angles = []
    distances = []
    transforms = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"frame {index} must be a JSON object")
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            raise ValueError(
                f"file_path must be a non-empty string for frame {index}"
            )
        image_path = (path.parent / file_path).resolve()
        if not image_path.is_relative_to(path.parent):
            raise ValueError(
                "frame file_path must remain inside the manifest directory"
            )
        if not image_path.is_file():
            raise FileNotFoundError(
                f"missing calibrated frame {index}: {image_path}"
            )

        angle = frame.get("camera_angle_x", metadata.get("camera_angle_x"))
        try:
            angle = float(angle)
        except (TypeError, ValueError):
            raise ValueError(
                f"camera_angle_x must be finite for frame {index}"
            ) from None
        if not np.isfinite(angle):
            raise ValueError(f"camera_angle_x must be finite for frame {index}")

        try:
            transform = np.asarray(
                frame.get("transform_matrix"), dtype=np.float32
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"transform_matrix must be finite [4, 4] for frame {index}"
            ) from None
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(
                f"transform_matrix must be finite [4, 4] for frame {index}"
            )

        with Image.open(image_path) as source:
            images.append(source.convert("RGBA"))
        angles.append(angle)
        distances.append(float(np.linalg.norm(transform[:3, 3])))
        transforms.append(transform)

    return images, {
        "camera_angle_x": angles,
        "distance": distances,
        "mesh_scale": mesh_scale,
        "transform_matrix": np.stack(transforms),
    }


# ============================================================================
# Main Inference
# ============================================================================

def run_inference(
    image_path: Optional[str],
    output_path: str,
    seed: int = 42,
    ss_guidance_strength: float = 7.5,
    ss_guidance_rescale: float = 0.7,
    ss_sampling_steps: int = 12,
    ss_rescale_t: float = 5.0,
    shape_slat_guidance_strength: float = 7.5,
    shape_slat_guidance_rescale: float = 0.5,
    shape_slat_sampling_steps: int = 12,
    shape_slat_rescale_t: float = 3.0,
    tex_slat_guidance_strength: float = 1.0,
    tex_slat_guidance_rescale: float = 0.0,
    tex_slat_sampling_steps: int = 12,
    tex_slat_rescale_t: float = 3.0,
    mesh_scale: float = 1.0,
    extend_pixel: int = 0,
    image_resolution: int = 512,
    max_num_tokens: int = 49152,
    model_path: str = MODEL_PATH,
    manual_fov: float = -1.0,
    low_vram: bool = False,
    resolution: int = -1,
    transforms_path: Optional[str] = None,
    flow_checkpoints: Optional[dict] = None,
):
    # Load models
    pipeline = init_pipeline(model_path, low_vram=low_vram)
    load_flow_overrides(pipeline, flow_checkpoints or {})

    if transforms_path is not None:
        print(f"[Inference] Loading calibrated views: {transforms_path}")
        images, camera_params = load_calibrated_manifest(
            transforms_path, mesh_scale=mesh_scale
        )
        image_preprocessed = [
            pipeline.preprocess_image(view) for view in images
        ]
    else:
        if image_path is None:
            raise ValueError("image_path is required without transforms_path")
        if mesh_scale is None:
            mesh_scale = 1.0

        # Preprocess image first — rembg loads to GPU for this call, then offloads.
        # MoGe is loaded afterwards so both never occupy VRAM at the same time.
        print(f"[Inference] Processing image: {image_path}")
        img = Image.open(image_path)
        image_preprocessed = pipeline.preprocess_image(img)

        # Save preprocessed image for MoGe
        tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            f"_tmp_preprocessed_{int(time.time() * 1000)}.png",
        )
        image_preprocessed.save(tmp_path)

        # Camera estimation
        if manual_fov > 0:
            # Use manually specified FOV (in radians)
            camera_angle_x = float(manual_fov)
            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            distance = distance_from_fov(
                camera_angle_x,
                grid_point,
                torch.tensor(
                    [
                        0 - extend_pixel,
                        image_resolution - 1 + extend_pixel,
                    ]
                ),
                mesh_scale,
                image_resolution,
            )["distance_from_x"]
            camera_params = {
                "camera_angle_x": camera_angle_x,
                "distance": distance,
                "mesh_scale": mesh_scale,
            }
            print(
                f"[Inference] Using manual FOV: "
                f"{math.degrees(manual_fov):.2f}° "
                f"({manual_fov:.4f} rad), distance={distance:.4f}"
            )
        else:
            print("[MoGe-2] Loading model for camera estimation...")
            moge_model = load_moge_model(device="cuda")
            print("[Inference] Estimating camera parameters...")
            camera_params = get_camera_params_wild_moge(
                tmp_path,
                moge_model,
                device="cuda",
                mesh_scale=mesh_scale,
                extend_pixel=extend_pixel,
                image_resolution=image_resolution,
            )
            print(
                f"  camera_angle_x={camera_params['camera_angle_x']:.4f}, "
                f"distance={camera_params['distance']:.4f}"
            )
            # MoGe is only needed for camera estimation; free its VRAM for inference.
            moge_model.cpu()
            del moge_model
            torch.cuda.empty_cache()
        os.remove(tmp_path)

    # Run pipeline
    print("[Inference] Running 3D generation pipeline...")
    torch.manual_seed(seed)

    ss_sampler_override = {
        "steps": ss_sampling_steps, "guidance_strength": ss_guidance_strength,
        "guidance_rescale": ss_guidance_rescale, "rescale_t": ss_rescale_t,
    }
    shape_sampler_override = {
        "steps": shape_slat_sampling_steps, "guidance_strength": shape_slat_guidance_strength,
        "guidance_rescale": shape_slat_guidance_rescale, "rescale_t": shape_slat_rescale_t,
    }
    tex_sampler_override = {
        "steps": tex_slat_sampling_steps, "guidance_strength": tex_slat_guidance_strength,
        "guidance_rescale": tex_slat_guidance_rescale, "rescale_t": tex_slat_rescale_t,
    }

    pipeline_type = f"{resolution if resolution > 0 else (1024 if low_vram else 1536)}_cascade"
    print(f"[Inference] Using pipeline_type={pipeline_type}")
    mesh_list, (shape_slat, tex_slat, res) = pipeline.run(
        image_preprocessed,
        camera_params=camera_params,
        seed=seed,
        sparse_structure_sampler_params=ss_sampler_override,
        shape_slat_sampler_params=shape_sampler_override,
        tex_slat_sampler_params=tex_sampler_override,
        preprocess_image=False,
        return_latent=True,
        pipeline_type=pipeline_type,
        max_num_tokens=max_num_tokens,
    )

    mesh = mesh_list[0]

    # Extract GLB
    print("[Inference] Extracting GLB...")
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=pipeline.pbr_attr_layout,
        grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1000000, texture_size=4096,
        remesh=True, remesh_band=1, remesh_project=0, use_tqdm=True,
    )

    # Apply rotation
    rot = np.array([
        [-1,  0,  0,  0],
        [ 0,  0, -1,  0],
        [ 0, -1,  0,  0],
        [ 0,  0,  0,  1],
    ], dtype=np.float64)
    glb.apply_transform(rot)

    # Export
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    glb.export(output_path, extension_webp=True)
    print(f"[Done] GLB saved to: {output_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Pixal3D Inference: Image to GLB")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--image",
        help="Single input image; camera may be estimated with MoGe-2",
    )
    inputs.add_argument(
        "--transforms",
        help="Calibrated transforms.json with the first frame as anchor",
    )
    parser.add_argument("--output", type=str, default="./output.glb", help="Output GLB file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fov", type=float, default=-1.0,
                        help="Manual camera FOV in radians (e.g. 0.2). "
                             "If not set, FOV is auto-estimated via MoGe-2. "
                             "Try 0.2 rad if you notice distortion.")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Model path or HuggingFace repo")
    parser.add_argument("--low_vram", action="store_true",
                        help="Enable low-VRAM mode: models stay on CPU and are loaded to GPU on-demand per stage. "
                             "Reduces peak VRAM from ~18GB to ~10-12GB at the cost of slower inference.")
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Pipeline resolution (1024 or 1536). Default: 1024 if --low_vram, else 1536.")
    parser.add_argument("--ss_ckpt")
    parser.add_argument("--shape512_ckpt")
    parser.add_argument("--shape1024_ckpt")
    parser.add_argument("--pbr1024_ckpt")
    parser.add_argument("--mesh_scale", type=float)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    flow_checkpoints = {
        "sparse_structure_flow_model": args.ss_ckpt,
        "shape_slat_flow_model_512": args.shape512_ckpt,
        "shape_slat_flow_model_1024": args.shape1024_ckpt,
        "tex_slat_flow_model_1024": args.pbr1024_ckpt,
    }
    flow_checkpoints = {
        model_key: checkpoint
        for model_key, checkpoint in flow_checkpoints.items()
        if checkpoint is not None
    }

    run_inference(
        image_path=args.image,
        transforms_path=args.transforms,
        output_path=args.output,
        seed=args.seed,
        manual_fov=args.fov,
        model_path=args.model_path,
        low_vram=args.low_vram,
        resolution=args.resolution,
        flow_checkpoints=flow_checkpoints,
        mesh_scale=args.mesh_scale,
    )


if __name__ == "__main__":
    main()
