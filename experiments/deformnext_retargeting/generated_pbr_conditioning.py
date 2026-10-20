"""Native PBR retargeting on generated sparse support, without GT coordinates.

Transport descriptors come only from the declared edited-RGB cache. Sparse
target-ray addresses are recomputed from the actual generated coordinates.
The fusion operation is the unchanged native training mixin.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace
import json

import numpy as np
from PIL import Image
import torch

from .projection_maps import anchor_patch_indices


def resolve_path(value, path_map):
    for source, destination in path_map.items():
        if value == source or value.startswith(source.rstrip('/')+'/'):
            return Path(destination+value[len(source):])
    path = Path(value)
    if not path.is_absolute():
        raise ValueError('explicit absolute cache paths or a declared path map required')
    return path


def load_generated_transport(row, coords, *, path_map, match_root,
                             expected_q2c_sha256, physical_deformation_query):
    from pixal3d.modules.r9_matching import MATCHER_CONTRACT, ATTENTION_CONTRACT, frame_raw
    xyz = coords.detach().cpu().numpy() if isinstance(coords,torch.Tensor) else np.asarray(coords)
    if (xyz.ndim!=2 or xyz.shape[1]!=4 or len(xyz)==0 or xyz.dtype.kind not in 'iu'
            or xyz[:,0].any()):
        raise ValueError('one actual generated PBR support with integer [N,4] coordinates required')
    if row['conditioning_input_schema']!='deformnext-edited-rgb-input-v1' or row['target_anchor_kind']!='rgb':
        raise ValueError('edited RGB target plus three source RGB inputs required')
    expected=row['conditioning_input_sha256']
    fields=['source_values','source_valid','target_query']
    if physical_deformation_query:
        fields+=['target_keys','source_keys']
    tensors={}
    raw={}
    # Deliberately never read spatial_indices, patch_grid, patch_ids or the
    # legacy deformation field from either cache.
    for name,schema in [('transport_cache','DEFORMNEXT_OMEGA_L12_PIXAL_VALUES_SS64_V1'),
                        ('transport_cache_pbr','DEFORMNEXT_PIXAL_VALUES_PBR1024_V1')]:
        with np.load(resolve_path(row[name],path_map)) as cache:
            if str(cache['schema'])!=schema or str(cache['conditioning_input_sha256'])!=expected:
                raise ValueError('transport cache schema or edited-input identity differs')
            for key in fields:
                if key in cache:
                    tensors[key]=torch.from_numpy(cache[key].copy())
            if physical_deformation_query and name=='transport_cache':
                raw={key:torch.from_numpy(cache[key].copy()) for key in ['target_keys','source_keys']}
    cache_path=resolve_path(str(Path(match_root)/(row['sample_id']+'.npz')),path_map)
    with np.load(cache_path) as cache:
        if (str(cache['schema'])!='deformnext-match-r9-v5'
                or json.loads(str(cache['matcher_contract']))!=MATCHER_CONTRACT
                or json.loads(str(cache['attention_contract']))!=ATTENTION_CONTRACT
                or str(cache['aggregation'])!='global-categorical-v1'
                or str(cache['proposal_policy'])!='GLOBAL64_VALID_OR_NATIVE'
                or str(cache['q2c_sha256'])!=expected_q2c_sha256
                or str(cache['conditioning_input_sha256'])!=expected
                or int(cache['topk'])!=64):
            raise ValueError('native64 frozen matching contract differs')
        for key in ['semantic_indices','matching_logits','candidate_valid']:
            tensors[key]=torch.from_numpy(cache[key].copy())
    tensors['source_valid']=tensors['source_valid'].bool()
    tensors['candidate_valid']=tensors['candidate_valid'].bool()
    tensors['semantic_indices']=tensors['semantic_indices'].long()
    if (tensors['source_values'].shape!=(3,1024,2048)
            or tensors['source_valid'].shape!=(3,1024)
            or tensors['target_query'].shape!=(1024,1024)
            or tensors['semantic_indices'].shape!=(1024,64)
            or tensors['matching_logits'].shape!=(1024,64)
            or tensors['candidate_valid'].shape!=(1024,64)):
        raise ValueError('exact native PBR values/query/candidate shapes required')
    if physical_deformation_query:
        if raw['target_keys'].shape!=(1024,2048) or raw['source_keys'].shape!=(3,1024,2048):
            raise ValueError('physical RAW2048 context channels required')
        tensors['target_keys']=frame_raw(tensors['target_keys'])
        tensors['source_keys']=frame_raw(tensors['source_keys'])
        tensors['target_context']=raw['target_keys']
        tensors['source_context']=raw['source_keys']
    with Image.open(resolve_path(row['target_anchor'],path_map)) as image:
        alpha=np.asarray(image.convert('RGBA').resize((512,512),Image.Resampling.NEAREST))[...,3]
    tensors['target_valid']=torch.from_numpy((alpha[8::16,8::16]>127).copy()).flatten()
    output={key:value.unsqueeze(0) for key,value in tensors.items()}
    output['patch_ids']=torch.from_numpy(anchor_patch_indices(row,xyz[:,1:],64)).long()
    return output


@torch.no_grad()
def encode_generated_pbr(extractor, images, camera_info, coords, *, conditioner, transport):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from pixal3d.modules.sparse import SparseTensor
    if (extractor.grid_resolution!=64 or images.shape!=(1,4,3,1024,1024)
            or coords.ndim!=2 or coords.shape[1]!=4 or torch.any(coords[:,0]!=0)
            or torch.any((coords[:,1:]<0)|(coords[:,1:]>=64))):
        raise ValueError('native PBR1024 four-view images and actual grid64 support required')
    if conditioner is not None and (conditioner.value_dim!=2048 or conditioner.projection_mode!='nearest'):
        raise ValueError('native nearest-ray PBR2048 conditioner required')
    facade=SimpleNamespace(image_cond_model=extractor,_image_cond_module=extractor,
        image_attn_mode='proj',deform_conditioner=conditioner,
        deform_conditioner_config=None if conditioner is None else dict(implementation=conditioner.implementation,arm=conditioner.arm))
    facade.encode_image_proj=MethodType(ImageConditionedProjMixin.encode_image_proj,facade)
    facade._inject_deform_transport=MethodType(ImageConditionedProjMixin._inject_deform_transport,facade)
    information=dict(camera_info,coords=coords)
    condition,negative=ImageConditionedProjMixin._encode_deform_views(facade,images,information,transport)
    if condition['proj'].shape!=(len(coords),2048) or negative['proj'].shape!=condition['proj'].shape:
        raise ValueError('native PBR conditioning must align with generated support')
    for values in [condition,negative]:
        values['proj']=SparseTensor(feats=values['proj'],coords=coords)
    return dict(cond=condition,neg_cond=negative),getattr(facade,'_last_deform_condition',None)


def install_generated_pbr_conditioning(pipeline, *, conditioner, row, path_map,
                                      match_root, expected_q2c_sha256,
                                      physical_deformation_query, events):
    """Intercept only texture conditioning; preserve native Shape stages."""
    if conditioner is None or not isinstance(physical_deformation_query,bool):
        raise ValueError('explicit physical R-only or Direct conditioner required')
    from .cascade_artifact_inputs import prepared_views
    original=pipeline.get_proj_cond_shape

    @torch.no_grad()
    def get_condition(self,image_cond_model,image,coords,camera_angle_x=.8575560450553894,
                      distance=2.,mesh_scale=1.,grid_resolution_override=None,transform_matrix=None):
        if image_cond_model is not self.image_cond_model_tex_1024:
            return original(image_cond_model,image,coords,camera_angle_x=camera_angle_x,
                distance=distance,mesh_scale=mesh_scale,grid_resolution_override=grid_resolution_override,
                transform_matrix=transform_matrix)
        if grid_resolution_override not in (None,64) or len(image)!=4:
            raise ValueError('PBR1024 generated-support inference only')
        expected_paths=[row['target_anchor'],*row['source_images']]
        for provided,path in zip(image,expected_paths):
            with Image.open(resolve_path(path,path_map)) as reference:
                a,b=provided.convert('RGBA'),reference.convert('RGBA')
                if a.size!=b.size or a.tobytes()!=b.tobytes():
                    raise ValueError('actual pipeline RGB pixels differ from the transport cache inputs')
        device=self.device
        extractor=image_cond_model
        if self.low_vram:extractor.to(device)
        cameras=dict(camera_angle_x=torch.as_tensor(camera_angle_x,dtype=torch.float32,device=device).reshape(1,4),
            distance=torch.as_tensor(distance,dtype=torch.float32,device=device).reshape(1,4),
            mesh_scale=torch.as_tensor(mesh_scale,dtype=torch.float32,device=device).reshape(1),
            transform_matrix=torch.as_tensor(transform_matrix,dtype=torch.float32,device=device).reshape(1,4,4,4))
        if (not torch.equal(cameras['camera_angle_x'][0].cpu(),torch.as_tensor(row['camera_angle_x'],dtype=torch.float32))
                or not torch.equal(cameras['transform_matrix'][0].cpu(),torch.as_tensor(row['transform_matrix'],dtype=torch.float32))
                or not torch.allclose(cameras['distance'][0].cpu(),
                    torch.linalg.vector_norm(torch.as_tensor(row['transform_matrix'],dtype=torch.float32)[:,:3,3],dim=-1),rtol=1e-6,atol=1e-6)
                or float(cameras['mesh_scale'][0])!=float(torch.tensor(row['mesh_scale'],dtype=torch.float32))):
            raise ValueError('actual pipeline cameras differ from the declared input cameras')
        values=load_generated_transport(row,coords,path_map=path_map,match_root=match_root,
            expected_q2c_sha256=expected_q2c_sha256,physical_deformation_query=physical_deformation_query)
        values={key:value.to(device) for key,value in values.items()}
        rng=torch.cuda.get_rng_state().clone()
        try:
            result,output=encode_generated_pbr(extractor,prepared_views(image,1024,device),cameras,
                coords,conditioner=conditioner,transport=values)
        finally:
            if self.low_vram:extractor.cpu()
        if not torch.equal(rng,torch.cuda.get_rng_state()):
            raise ValueError('PBR conditioning must not consume the sampler CUDA random stream')
        events.append(dict(actual_generated_tokens=len(coords),target_ray_queries=1024,
            unique_target_ray_addresses=len(torch.unique(values['patch_ids'])),
            valid_matching_rays=int(output.query_valid.sum()),
            GT_sparse_coords_or_projection_cache_used=False,
            extra_conditioner_applied=True,physical_deformation_query=physical_deformation_query,
            native_fusion_reused=True,sampler_CUDA_RNG_unchanged=True))
        return result

    pipeline.get_proj_cond_shape=MethodType(get_condition,pipeline)
    return original
