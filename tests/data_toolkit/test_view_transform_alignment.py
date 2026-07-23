import numpy as np
import pytest
import torch

from data_toolkit.utils import sphere_normalize_torch, transform_mesh

# o_voxel imports flex_gemm, which probes a GPU name during module import even
# though this test exercises only CPU transform helpers.
_original_get_device_name = torch.cuda.get_device_name
torch.cuda.get_device_name = lambda *_args, **_kwargs: "A100"
try:
    from data_toolkit.voxelize_pbr_view import transform_pbr_dump
finally:
    torch.cuda.get_device_name = _original_get_device_name


def _geometry_transform(vertices, frame):
    vertices = torch.from_numpy(vertices).float().contiguous()
    vertices, _, sphere_radius = sphere_normalize_torch(vertices)
    vertices = transform_mesh(vertices, frame)
    box_scale = 0.49999 / vertices.abs().max().item()
    return vertices * box_scale, box_scale / sphere_radius.item()


def _quantized_support(vertices, resolution):
    coordinates = np.floor((vertices + 0.5) * resolution).astype(np.int64)
    return np.unique(coordinates, axis=0)


def test_pbr_transform_matches_geometry_transform_and_quantized_support():
    vertices = np.array(
        [
            [0.93265724, 0.55689943, 1.8179625],
            [0.28840002, 1.1539986, 1.6804935],
            [0.3404041, 0.9606351, 1.6952579],
            [1.0833015, 0.35972893, 1.5453503],
            [-0.93265724, -0.55689943, -1.8179625],
            [-0.28840002, -1.1539986, -1.6804935],
            [-0.3404041, -0.9606351, -1.6952579],
            [-1.0833015, -0.35972893, -1.5453503],
        ],
        dtype=np.float32,
    )
    frame = {
        "transform_matrix": [
            [1.0, 0.0, 0.0, 1.7],
            [0.0, 1.0, 0.0, 2.3],
            [0.0, 0.0, 1.0, 1.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
    dump = {
        "objects": [
            {
                "vertices": vertices,
                "normals": None,
                "mat_ids": np.zeros(len(vertices), dtype=np.int32),
            }
        ],
        "materials": [{}],
    }

    expected_vertices, expected_scale = _geometry_transform(vertices, frame)
    transformed_dump, total_scale = transform_pbr_dump(dump, frame)
    actual_vertices = transformed_dump["objects"][0]["vertices"]
    expected_vertices = expected_vertices.numpy()

    np.testing.assert_array_equal(actual_vertices, expected_vertices)
    np.testing.assert_array_equal(
        _quantized_support(actual_vertices, resolution=1024),
        _quantized_support(expected_vertices, resolution=1024),
    )
    assert total_scale == pytest.approx(expected_scale)
