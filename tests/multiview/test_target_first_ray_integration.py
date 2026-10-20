"""Target encoder and native per-source projection wiring, without CUDA."""
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from pixal3d.trainers.flow_matching.target_first_ray import (
    TargetFirstRayImageConditionedProjFlowMatchingCFGTrainer as Trainer)


class Grid:
    front_view_transform_matrix = torch.eye(4)

    def __init__(self):
        self.calls = []

    def __call__(self, values, angle, distance, scale, matrix):
        self.calls.append((angle.clone(), distance.clone(), scale.clone(), matrix.clone()))
        return values.mean((1, 2))[:, None].expand(-1, 4096, -1)


class Source(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(()))
        self.arguments = None

    def forward(self, **kwargs):
        self.arguments = kwargs
        delta = torch.ones_like(kwargs['source_values']) * self.scale
        return SimpleNamespace(map_delta=delta, attention=object(),
                               use_probability=torch.ones(1, 3072))


def fixture():
    grid, module = Grid(), Source()
    extractor = SimpleNamespace(proj_grid=grid, fixed_projection_transform=torch.eye(4))
    history = []

    def encode(images, **camera):
        history.append(dict(images=images.clone(), camera=camera))
        value = images.mean((1, 2, 3))
        positive = {'global': value[:, None, None].expand(-1, 5, 1024).clone(),
                    'proj': value[:, None, None].expand(-1, 4096, 1024).clone()}
        negative = {k:torch.zeros_like(v) for k,v in positive.items()}
        history[-1].update(positive=positive, negative=negative)
        return positive, negative

    facade = SimpleNamespace(image_cond_model=extractor, _image_cond_module=extractor,
        deform_conditioner=module, training_models={'deform_conditioner':module},
        image_attn_mode='proj', encode_image_proj=encode)
    images = torch.randn(1, 4, 3, 4, 4)
    camera = dict(transform_matrix=torch.eye(4).repeat(1, 4, 1, 1),
                  distance=torch.ones(1, 4)*2, camera_angle_x=torch.ones(1, 4), mesh_scale=torch.ones(1))
    camera['transform_matrix'][:, :, 2, 3] = 2
    transport = dict(target_keys=torch.zeros(1,1024,1024), target_query=torch.zeros(1,1024,1024),
        source_keys=torch.zeros(1,3,1024,1024), source_values=torch.zeros(1,3,1024,1024),
        source_valid=torch.ones(1,3,1024,dtype=torch.bool), target_valid=torch.zeros(1,1024,dtype=torch.bool),
        patch_grid=(torch.arange(4096)%1024).reshape(1,16,16,16))
    membership = (torch.arange(4096)%1024).reshape(1,1,4096).expand(-1,3,-1).clone()
    return facade, images, camera, transport, membership, history


def test_zero_source_path_equals_single_target_encoder_and_retains_its_negative_condition():
    facade, images, camera, transport, membership, history = fixture()
    with patch('pixal3d.trainers.flow_matching.target_first_ray.native_ray_membership',return_value=membership):
        actual, negative = Trainer._encode_deform_views(facade,images,camera,transport)
    assert len(history) == 1
    assert torch.equal(history[0]['images'],images[:,0])
    assert actual['global'] is history[0]['positive']['global']
    assert torch.equal(actual['proj'],history[0]['positive']['proj'])
    assert negative is history[0]['negative']
    assert len(facade._image_cond_module.proj_grid.calls) == 3


def test_target_foreground_does_not_filter_native_ray_members_or_mutate_anchor():
    facade, images, camera, transport, membership, history = fixture()
    with torch.no_grad():
        facade.deform_conditioner.scale.fill_(.1)
    with patch('pixal3d.trainers.flow_matching.target_first_ray.native_ray_membership',return_value=membership):
        actual, _ = Trainer._encode_deform_views(facade,images,camera,transport)
    arguments = facade.deform_conditioner.arguments
    assert torch.equal(arguments['ray_ids'],membership)  # all target_valid bits are false
    assert arguments['native_voxels'] is history[0]['positive']['proj']
    assert actual['global'] is history[0]['positive']['global']
    assert not torch.equal(actual['proj'],history[0]['positive']['proj'])
    assert 'spatial_indices' not in arguments
    actual['proj'].square().mean().backward()
    assert facade.deform_conditioner.scale.grad.abs() > 0


def test_absent_transport_uses_target_carrier_and_does_not_project_source_maps():
    facade, images, camera, _, _, history = fixture()
    actual, negative = Trainer._encode_deform_views(facade,images,camera,None)
    assert actual is history[0]['positive']
    assert negative is history[0]['negative']
    assert not facade._image_cond_module.proj_grid.calls
