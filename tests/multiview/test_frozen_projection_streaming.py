"""Streaming must preserve camera/asset pairing and forbid trainable encoders."""
from types import SimpleNamespace

import pytest
import torch

from experiments.deformnext_retargeting.frozen_projection_streaming import stream_projection


def test_streaming_preserves_camera_order_and_native_output_structure():
    encoder = torch.nn.Linear(1, 1).requires_grad_(False)
    model = SimpleNamespace(model=encoder, naf_model=None)
    images = torch.arange(7.).reshape(7, 1, 1, 1)
    camera = torch.arange(7.)*10
    distance = camera+100
    scale = camera+200
    transforms = torch.arange(7*16.).reshape(7,4,4)
    calls = []

    def native(self, image, fov, dist, size, transform):
        calls.append(len(image))
        global_features = torch.stack([image.flatten(), fov, dist, size], dim=-1)
        return global_features, transform+image.reshape(-1, 1, 1)

    expected = native(model, images, camera, distance, scale, transforms)
    calls.clear()
    actual = stream_projection(native, model, images, camera, distance, scale,
                               transforms, max_assets=3)
    assert calls == [3, 3, 1]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert all(not value.requires_grad for value in actual)
    with pytest.raises(ValueError, match='camera arrays'):
        stream_projection(native, model, images, camera[:-1], distance, scale, transforms)


def test_trainable_encoder_cannot_be_silently_chunked_without_gradients():
    model = SimpleNamespace(model=torch.nn.Linear(1, 1), naf_model=None)
    with pytest.raises(ValueError, match='frozen image encoder'):
        stream_projection(lambda *args: None, model, torch.zeros(2, 3, 4, 4))
