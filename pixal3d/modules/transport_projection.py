"""Native SS patch-map projection and support-preserving transport."""
import torch


def project_native_patchmap(extractor, patches, camera):
    from ..trainers.flow_matching.mixins.image_conditioned_proj import compute_multiview_projection_matrices
    projection,_=compute_multiview_projection_matrices(camera['transform_matrix'],camera['distance'],
        extractor.fixed_projection_transform)
    if patches.ndim!=3 or patches.shape[1]!=1024:
        raise ValueError('native SS transport requires a 32x32 patch map')
    return extractor.proj_grid(patches.reshape(patches.shape[0],32,32,-1),
        camera['camera_angle_x'][:,0],camera['distance'][:,0],camera['mesh_scale'],projection[:,0])


def support_weighted_projection(project, values, residual, usage):
    """Interpolate the entire update, including hard-rejected adapter values.

    Returning normalized evidence plus its mass preserves the exact native
    fallback and avoids mixing zero padding or rejected values into evidence.
    """
    if values.shape!=residual.shape or usage.shape!=values.shape[:-1]:
        raise ValueError('transport projection shapes differ')
    packed=torch.cat((values*usage[...,None],residual*usage[...,None],usage[...,None]),-1)
    projected=project(packed);width=values.shape[-1]
    source,adapter,mass=projected.split((width,width,1),-1)
    mass=mass.clamp(0,1)
    return source/mass.clamp_min(1e-8),adapter/mass.clamp_min(1e-8),mass>0,mass
