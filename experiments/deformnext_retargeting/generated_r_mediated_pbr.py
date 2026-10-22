"""New source-ray PBR conditioning on actual predicted sparse support.

Read image descriptors only. Never open GT latent/mesh/projection caches or
legacy target-patch matching indices. Reuse the actual native sparse trainer
consumer rather than a second implementation of feature delivery.
"""
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import numpy as np
from PIL import Image
import torch

from .generated_pbr_conditioning import resolve_path
from .prepare_rgb_manifest import image_fingerprint
from .projection_maps import anchor_patch_indices


IMAGE_ROW_FIELDS = ('sample_id', 'camera_angle_x', 'transform_matrix', 'mesh_scale',
    'conditioning_input_schema', 'conditioning_input_sha256', 'target_anchor_kind',
    'conditioning_camera_ids', 'target_anchor', 'source_images', 'transport_cache', 'transport_cache_pbr')


def image_only_row(row):
    """Discard GT/data-label paths before entering generated-shape inference."""
    return {key: row[key] for key in IMAGE_ROW_FIELDS}


def actual_target_patch_ids(row, coords):
    xyz = coords.detach().cpu().numpy() if isinstance(coords, torch.Tensor) else np.asarray(coords)
    if (xyz.ndim != 2 or xyz.shape[1] != 4 or not len(xyz) or xyz.dtype.kind not in 'iu'
            or xyz[:, 0].any() or ((xyz[:, 1:] < 0) | (xyz[:, 1:] >= 64)).any()
            or len(np.unique(xyz, axis=0)) != len(xyz)):
        raise ValueError('one nonempty unique actual generated grid64 integer[N,4] support required')
    angles = np.asarray(row['camera_angle_x'], dtype=np.float32)
    transforms = np.asarray(row['transform_matrix'], dtype=np.float32)
    scale = float(row['mesh_scale'])
    if (angles.shape != (4,) or transforms.shape != (4, 4, 4)
            or not np.isfinite(angles).all() or not np.isfinite(transforms).all()
            or (angles <= 0).any() or (angles >= np.pi).any()
            or not np.isfinite(scale) or scale <= 0
            or np.linalg.norm(transforms[0, :3, 3]) <= 0):
        raise ValueError('explicit finite native four-camera calibration and positive mesh scale required')
    # Preserve the target-address convention actually used by the TRAIN cache.
    # Source-camera ray pooling has its own nearest32 membership rule.
    return torch.from_numpy(anchor_patch_indices(row, xyz[:, 1:], 64)).long()


def load_generated_r_transport(row, coords, *, path_map):
    from pixal3d.modules.r9_matching import frame_raw
    if (row.get('conditioning_input_schema') != 'deformnext-edited-rgb-input-v1'
            or row.get('target_anchor_kind') != 'rgb' or len(row['source_images']) != 3
            or row.get('conditioning_camera_ids') != [8, 9, 10, 11]):
        raise ValueError('edited RGB target8 plus three original RGB source9/10/11 required')
    resolve = lambda path: resolve_path(path, path_map)
    expected = row['conditioning_input_sha256']
    if image_fingerprint(row, resolve) != expected:
        raise ValueError('actual RGB files or cameras differ from image-descriptor cache inputs')
    patch_ids = actual_target_patch_ids(row, coords)
    tensors = {}
    specs = [
        ('transport_cache', 'DEFORMNEXT_OMEGA_L12_PIXAL_VALUES_SS64_V1',
         ('target_keys', 'source_keys', 'source_valid')),
        ('transport_cache_pbr', 'DEFORMNEXT_PIXAL_VALUES_PBR1024_V1',
         ('source_values', 'target_query')),
    ]
    for field, schema, names in specs:
        with np.load(resolve(row[field])) as cache:
            if str(cache['schema']) != schema or str(cache['conditioning_input_sha256']) != expected:
                raise ValueError('actual RGB descriptor schema/input identity differs')
            if field == 'transport_cache' and int(cache['omega_completed_updates']) != 23597:
                raise ValueError('frozen Omega epoch7/23597 image descriptors required')
            for name in names:
                tensors[name] = torch.from_numpy(cache[name].copy())
    if (tensors['target_keys'].shape != (1024, 2048)
            or tensors['source_keys'].shape != (3, 1024, 2048)
            or tensors['source_values'].shape != (3, 1024, 2048)
            or tensors['target_query'].shape != (1024, 1024)
            or tensors['source_valid'].shape != (3, 1024) or tensors['source_valid'].dtype != torch.bool
            or any(not torch.isfinite(value).all() for value in tensors.values())):
        raise ValueError('exact finite Omega FRAME+GLOBAL and DINO+NAF2048 image inputs required')
    tensors['target_keys'] = frame_raw(tensors['target_keys'])
    tensors['source_keys'] = frame_raw(tensors['source_keys'])
    with Image.open(resolve(row['target_anchor'])) as image:
        alpha = np.asarray(image.convert('RGBA').resize((512, 512), Image.Resampling.NEAREST))[..., 3]
    tensors['target_valid'] = torch.from_numpy((alpha[8::16, 8::16] > 127).copy()).flatten()
    return {**{key: value.unsqueeze(0) for key, value in tensors.items()}, 'patch_ids': patch_ids}


@torch.no_grad()
def encode_generated_r_pbr(extractor, images, camera, coords, *, conditioner, transport):
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    if (extractor.grid_resolution != 64 or extractor.image_size != 1024
            or images.shape != (1, 4, 3, 1024, 1024)
            or conditioner.value_dim != 2048 or conditioner.projection_mode != 'native-bilinear'
            or conditioner.appearance_mode not in {'top1-only', 'top1-plus-feature', 'raw-topk-plus-feature'}
            or set(transport) != {'target_keys', 'source_keys', 'source_values', 'source_valid',
                                  'target_query', 'target_valid', 'patch_ids'}
            or transport['patch_ids'].shape != (len(coords),)):
        raise ValueError('own native PBR2048 R-only source schema at actual generated support required')
    facade = SimpleNamespace(image_cond_model=extractor, _image_cond_module=extractor,
        image_attn_mode='proj', deform_conditioner=conditioner)
    facade.encode_image_proj = MethodType(ImageConditionedProjMixin.encode_image_proj, facade)
    # Match the actual source module's training/stage-evaluation precision.
    # Keep this context local: native sparse VAE decoders must retain FP16.
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=images.device.type == 'cuda'):
        condition, negative = RMediatedPBRNativeTrainer._encode_deform_views(
            facade, images, dict(camera, coords=coords), transport)
    for values in [condition, negative]:
        if set(values) != {'global', 'proj'} or values['proj'].shape != (len(coords), 2048):
            raise ValueError('native sparse PBR projection must align with every actual generated coordinate')
        values['proj'] = SparseTensor(feats=values['proj'], coords=coords)
    return dict(cond=condition, neg_cond=negative), facade


@contextmanager
def generated_r_pbr_conditioning(pipeline, *, conditioner, row, path_map, events):
    """Intercept texture only and restore the pipeline hook even after failure."""
    from .cascade_artifact_inputs import prepared_views
    original = pipeline.get_proj_cond_shape

    @torch.no_grad()
    def get_condition(self, extractor, image, coords, camera_angle_x=.8575560450553894,
                      distance=2., mesh_scale=1., grid_resolution_override=None, transform_matrix=None):
        if extractor is not self.image_cond_model_tex_1024:
            return original(extractor, image, coords, camera_angle_x=camera_angle_x,
                distance=distance, mesh_scale=mesh_scale, grid_resolution_override=grid_resolution_override,
                transform_matrix=transform_matrix)
        if grid_resolution_override not in (None, 64) or len(image) != 4:
            raise ValueError('explicit native PBR grid64 and edited RGB plus source3 inputs required')
        if extractor.use_naf_upsample and extractor.naf_model is None:
            raise ValueError('load the frozen NAF weights before resetting paired sampler noise')
        for provided, value in zip(image, [row['target_anchor'], *row['source_images']]):
            with Image.open(resolve_path(value, path_map)) as reference:
                a, b = provided.convert('RGBA'), reference.convert('RGBA')
                if a.size != b.size or a.tobytes() != b.tobytes():
                    raise ValueError('actual pipeline RGB pixels differ from the descriptor input files')
        device = self.device
        camera = dict(camera_angle_x=torch.as_tensor(camera_angle_x, dtype=torch.float32, device=device).reshape(1, 4),
            distance=torch.as_tensor(distance, dtype=torch.float32, device=device).reshape(1, 4),
            mesh_scale=torch.as_tensor(mesh_scale, dtype=torch.float32, device=device).reshape(1),
            transform_matrix=torch.as_tensor(transform_matrix, dtype=torch.float32, device=device).reshape(1, 4, 4, 4))
        declared = torch.as_tensor(row['transform_matrix'], dtype=torch.float32, device=device)
        if (not torch.equal(camera['transform_matrix'][0], declared)
                or not torch.equal(camera['camera_angle_x'][0], torch.as_tensor(row['camera_angle_x'], dtype=torch.float32, device=device))
                or not torch.allclose(camera['distance'][0], torch.linalg.vector_norm(declared[:, :3, 3], dim=-1), rtol=1e-6, atol=1e-6)
                or camera['mesh_scale'][0] != float(torch.tensor(row['mesh_scale'], dtype=torch.float32))):
            raise ValueError('actual generated-support cameras differ from declared input cameras')
        transport = {key: value.to(device) for key, value in load_generated_r_transport(row, coords, path_map=path_map).items()}
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
        if self.low_vram: extractor.to(device)
        try:
            result, facade = encode_generated_r_pbr(extractor, prepared_views(image, 1024, device),
                camera, coords, conditioner=conditioner, transport=transport)
        finally:
            if self.low_vram: extractor.cpu()
        if (not torch.equal(cpu_rng, torch.get_rng_state())
                or cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
            raise ValueError('R-mediated conditioning consumed paired CPU/CUDA PBR sampler noise')
        events.append(dict(actual_generated_tokens=len(coords), source_camera_ray_queries=3072,
            target_patch_addresses_recomputed_from_actual_coords=True,
            unique_target_patch_addresses=int(torch.unique(transport['patch_ids']).numel()),
            valid_source_ray_queries=int(facade._last_deform_condition.query_valid.sum()),
            GT_latent_mesh_projection_or_legacy_matching_cache_opened=False,
            native_sparse_training_consumer_reused=True, R_only_delivery=True,
            sampler_CPU_and_CUDA_RNG_unchanged=True))
        return result

    pipeline.get_proj_cond_shape = MethodType(get_condition, pipeline)
    try:
        yield
    finally:
        pipeline.get_proj_cond_shape = original
